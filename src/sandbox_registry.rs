//! Cross-teardown registry of live Claude Code sessions, so they can be
//! restored after the thing they were running in goes away.
//!
//! Two flavors, in two files under the host-shared `~/.local/share/claudectl`
//! mount:
//!   - `local-sessions.json` — laptop sessions, restored with `claude --resume`
//!     by `--restore-sessions` (e.g. after a Ghostty restart-to-update).
//!   - `sandbox-sessions.json` — `sbx`-sandbox sessions, keyed by sandbox name,
//!     restored with `sc --resume` by `--restore-sbx-sessions` after `sbx rm`.
//!     Sandbox transcripts (`~/.claude`) and this registry both live on
//!     host-shared bind mounts, so both survive `sbx rm`.
//!
//! On every hook, `hook_state::record_hook_event` reconciles to claudectl's
//! current live-session set — routing on the sandbox marker
//! ([`current_sandbox`]): [`replace_sandbox_slice`] inside a sandbox, else
//! [`replace_local`]. A session is tracked iff its process is alive; the writer
//! never deletes a session file. An abrupt teardown never fires `SessionEnd`, so
//! the set stays frozen at its last live state — exactly what the matching
//! restore command brings back.
//!
//! Each file has its own `.lock` sidecar; writes are serialized with an
//! advisory `flock` and committed via temp-file + atomic rename, so concurrent
//! hook processes never corrupt or tear a file.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Env var `sc` sets inside the sandbox to mark that Claude is running there.
/// Its mere presence — not its value — gates registry writes, mirroring the
/// `var_os(...).is_some()` convention the sandbox launcher uses elsewhere.
pub const ENV_SANDBOX_MARKER: &str = "LINERA_SANDBOX";
/// Env var carrying the sandbox's name (`sbx` container). Matches `sc`'s
/// `SANDBOX_NAME`, which defaults to `linera-agent` for the shared sandbox.
pub const ENV_SANDBOX_NAME: &str = "SANDBOX_NAME";
/// Default when `SANDBOX_NAME` is unset — kept in sync with `sc`.
const DEFAULT_SANDBOX_NAME: &str = "linera-agent";

fn current_version() -> u32 {
    1
}

/// One resumable session recorded in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Claude Code session id — the argument to `sc --resume <id>`.
    pub session_id: String,
    /// Host working directory the session was launched from, so restore can
    /// reopen it in the right place. Empty if the hook payload omitted `cwd`.
    #[serde(default)]
    pub cwd: String,
    /// Absolute path to the session's JSONL transcript on the shared
    /// `~/.claude` mount. Restore skips entries whose transcript no longer
    /// exists (unresumable). Empty if the payload omitted it.
    #[serde(default)]
    pub transcript: String,
    /// Unix epoch milliseconds at `SessionStart`. Zero if unknown.
    #[serde(default)]
    pub started_at_ms: u64,
    /// The session's `/rename` display name, read from its (container-local)
    /// session JSON on each reconcile so restore can show it after `sbx rm`
    /// destroys that JSON. `None` until the session is named.
    #[serde(default)]
    pub name: Option<String>,
}

/// The on-disk registry: a map of sandbox name -> its live sessions.
#[derive(Debug, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default = "current_version")]
    pub version: u32,
    #[serde(default)]
    pub sandboxes: BTreeMap<String, Vec<SessionEntry>>,
}

impl Default for Registry {
    // Hand-written (not derived): a derived `Default` would set `version` to 0,
    // which a write would then persist — the `serde(default)` only fills the
    // field when it's *absent* on read, not for `Default::default()`.
    fn default() -> Self {
        Registry {
            version: current_version(),
            sandboxes: BTreeMap::new(),
        }
    }
}

/// The local (laptop) session registry — a flat list, since there's only ever
/// one machine's worth of sessions (no sandbox names to key by). Stored in its
/// own file (`local-sessions.json`) so host hooks and the in-sandbox writer
/// never share a file or lock.
#[derive(Debug, Serialize, Deserialize)]
pub struct LocalRegistry {
    #[serde(default = "current_version")]
    pub version: u32,
    #[serde(default)]
    pub sessions: Vec<SessionEntry>,
}

impl Default for LocalRegistry {
    fn default() -> Self {
        LocalRegistry {
            version: current_version(),
            sessions: Vec::new(),
        }
    }
}

/// The sandbox this process is running inside, or `None` on the host.
///
/// Returns `Some(name)` only when the sandbox marker env var is present.
pub fn current_sandbox() -> Option<String> {
    std::env::var_os(ENV_SANDBOX_MARKER)?;
    let name = std::env::var(ENV_SANDBOX_NAME)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_SANDBOX_NAME.to_string());
    Some(name)
}

fn registry_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join(".local/share/claudectl")
}

/// Path to the sandbox registry file (`sandbox-sessions.json`). Honors
/// `CLAUDECTL_SANDBOX_REGISTRY` (tests, to avoid stomping the real file).
pub fn sandbox_registry_path() -> PathBuf {
    std::env::var_os("CLAUDECTL_SANDBOX_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|| registry_dir().join("sandbox-sessions.json"))
}

/// Path to the local (laptop) registry file (`local-sessions.json`). Honors
/// `CLAUDECTL_LOCAL_REGISTRY` (tests).
pub fn local_registry_path() -> PathBuf {
    std::env::var_os("CLAUDECTL_LOCAL_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|| registry_dir().join("local-sessions.json"))
}

/// Read the sandbox registry. A missing or unparseable file yields an empty
/// registry — callers treat "no registry" and "empty registry" identically, and
/// a corrupt file must never block a restore attempt or a hook.
pub fn load() -> Registry {
    match fs::read(sandbox_registry_path()) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => Registry::default(),
    }
}

/// Read the local registry (same missing/corrupt tolerance as [`load`]).
pub fn load_local() -> LocalRegistry {
    match fs::read(local_registry_path()) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => LocalRegistry::default(),
    }
}

/// Reconcile the machine's local (laptop) sessions to `entries` — the current
/// live set on the host. The whole write model: a hook reconciles on every
/// event, so the flat `local-sessions.json` always mirrors what claudectl shows
/// (idle sessions included, dead ones dropped). Writes only when the set
/// changes. The caller picks this vs [`replace_sandbox_slice`] on the sandbox
/// marker ([`current_sandbox`]), so a sandbox can never mis-route here — even
/// one named "host".
pub fn replace_local(entries: Vec<SessionEntry>) -> io::Result<()> {
    let path = local_registry_path();
    with_lock(&path, || {
        let mut registry = load_local();
        if registry.sessions == entries {
            return Ok(());
        }
        registry.sessions = entries;
        write_atomic(&path, &serialize(&registry)?)
    })
}

/// Reconcile one sandbox's slice (keyed by `sandbox` name) of
/// `sandbox-sessions.json` to its current live set, same write model as
/// [`replace_local`]. Empty `entries` removes the key. Other sandboxes' slices
/// are untouched, and an abrupt `sbx rm` fires no further hooks, so the slice
/// freezes at its last live state — exactly what `--restore-sbx-sessions`
/// restores.
pub fn replace_sandbox_slice(sandbox: &str, entries: Vec<SessionEntry>) -> io::Result<()> {
    let path = sandbox_registry_path();
    with_lock(&path, || {
        let mut registry = load();
        let unchanged = match (registry.sandboxes.get(sandbox), entries.is_empty()) {
            (None, true) => true,
            (Some(existing), false) => *existing == entries,
            _ => false,
        };
        if unchanged {
            return Ok(());
        }
        if entries.is_empty() {
            registry.sandboxes.remove(sandbox);
        } else {
            registry.sandboxes.insert(sandbox.to_string(), entries);
        }
        write_atomic(&path, &serialize(&registry)?)
    })
}

/// Pretty JSON with a trailing newline.
fn serialize<T: Serialize>(value: &T) -> io::Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Commit `bytes` to `path` via temp-file + atomic rename, so a reader never
/// observes a half-written file.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

/// Run `body` while holding an exclusive advisory lock on `path`'s sidecar lock
/// file, so concurrent hook processes serialize their read-modify-write cycles.
/// The lock releases when the file descriptor closes at the end of scope.
fn with_lock<T>(path: &Path, body: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_path = path.with_extension("lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    let fd = lock_file.as_raw_fd();
    // SAFETY: `fd` is a valid, open descriptor owned by `lock_file` for the
    // duration of this call; `flock` only reads it. LOCK_EX blocks until the
    // lock is acquired.
    if unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let result = body();
    // Best-effort unlock; dropping `lock_file` releases it regardless.
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes every test that mutates process env vars — set_var/remove_var
    /// are process-global, and Rust runs tests on parallel threads. Recovers
    /// from poisoning so one panicking test doesn't cascade into the rest.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    /// Point the registry at a throwaway file for the duration of a test, while
    /// holding the env lock so no other test observes our `CLAUDECTL_*` vars.
    struct TempRegistry {
        dir: std::path::PathBuf,
        _lock: MutexGuard<'static, ()>,
    }

    impl TempRegistry {
        fn new(tag: &str) -> Self {
            let lock = env_guard();
            // Include the pid: `cargo test` runs the lib and bin test binaries
            // as separate processes in parallel, and a tag-only path would let
            // them race on the same temp files. `ENV_LOCK` only serializes
            // within one process.
            let dir = std::env::temp_dir()
                .join(format!("claudectl-reg-test-{}-{tag}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            // SAFETY: env access here is serialized by the held `ENV_LOCK`.
            unsafe {
                std::env::set_var("CLAUDECTL_SANDBOX_REGISTRY", dir.join("sandbox.json"));
                std::env::set_var("CLAUDECTL_LOCAL_REGISTRY", dir.join("local.json"));
            }
            TempRegistry { dir, _lock: lock }
        }
    }

    impl Drop for TempRegistry {
        fn drop(&mut self) {
            // SAFETY: still holding `ENV_LOCK` via `_lock`.
            unsafe {
                std::env::remove_var("CLAUDECTL_SANDBOX_REGISTRY");
                std::env::remove_var("CLAUDECTL_LOCAL_REGISTRY");
            }
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn entry(id: &str, cwd: &str) -> SessionEntry {
        SessionEntry {
            session_id: id.to_string(),
            cwd: cwd.to_string(),
            transcript: format!("/tmp/{id}.jsonl"),
            started_at_ms: 42,
            name: None,
        }
    }

    #[test]
    fn missing_file_loads_empty() {
        let _guard = TempRegistry::new("missing");
        let registry = load();
        assert!(registry.sandboxes.is_empty());
    }

    #[test]
    fn replace_sandbox_slice_sets_and_roundtrips() {
        let _guard = TempRegistry::new("roundtrip");
        replace_sandbox_slice("linera-agent", vec![entry("aaa", "/work/a")]).unwrap();
        let registry = load();
        let slice = registry.sandboxes.get("linera-agent").unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0], entry("aaa", "/work/a"));
        assert_eq!(registry.version, 1);
    }

    #[test]
    fn replace_sandbox_slice_overwrites_the_whole_slice() {
        let _guard = TempRegistry::new("overwrite");
        replace_sandbox_slice("linera-agent", vec![entry("aaa", "/a"), entry("bbb", "/b")])
            .unwrap();
        // New live set: "aaa" ended, "ccc" started.
        replace_sandbox_slice("linera-agent", vec![entry("bbb", "/b"), entry("ccc", "/c")])
            .unwrap();
        let registry = load();
        let ids: Vec<_> = registry
            .sandboxes
            .get("linera-agent")
            .unwrap()
            .iter()
            .map(|e| e.session_id.as_str())
            .collect();
        assert_eq!(
            ids,
            ["bbb", "ccc"],
            "slice reflects exactly the new live set"
        );
    }

    #[test]
    fn replace_sandbox_slice_empty_removes_the_sandbox() {
        let _guard = TempRegistry::new("empty");
        replace_sandbox_slice("linera-agent", vec![entry("aaa", "/a")]).unwrap();
        replace_sandbox_slice("linera-agent", vec![]).unwrap();
        assert!(!load().sandboxes.contains_key("linera-agent"));
    }

    #[test]
    fn replace_sandbox_slice_keeps_sandboxes_independent() {
        let _guard = TempRegistry::new("independent");
        replace_sandbox_slice("linera-agent", vec![entry("aaa", "/a")]).unwrap();
        replace_sandbox_slice("pm-task", vec![entry("bbb", "/b")]).unwrap();
        replace_sandbox_slice("linera-agent", vec![]).unwrap();
        let registry = load();
        assert!(!registry.sandboxes.contains_key("linera-agent"));
        assert_eq!(registry.sandboxes.get("pm-task").unwrap().len(), 1);
    }

    #[test]
    fn session_entry_with_name_roundtrips() {
        let _guard = TempRegistry::new("name-roundtrip");
        let mut named = entry("aaa", "/a");
        named.name = Some("faucet-migration".to_string());
        replace_sandbox_slice("linera-agent", vec![named]).unwrap();
        let registry = load();
        assert_eq!(
            registry.sandboxes.get("linera-agent").unwrap()[0]
                .name
                .as_deref(),
            Some("faucet-migration")
        );
    }

    #[test]
    fn local_and_sandbox_registries_are_independent() {
        let _guard = TempRegistry::new("local-routing");
        // Local and sandbox sessions live in separate files, written by
        // separate functions — the host and a sandbox never share a slice.
        replace_sandbox_slice("linera-agent", vec![entry("sbx", "/s")]).unwrap();
        replace_local(vec![entry("loc", "/l")]).unwrap();

        // The local sessions land in the flat local file, versioned like the sandbox one.
        let local = load_local();
        assert_eq!(local.version, current_version());
        assert_eq!(local.sessions.len(), 1);
        assert_eq!(local.sessions[0].session_id, "loc");

        // ...and never appear in the sandbox registry, not even under a "host" key
        // (a sandbox literally named "host" still routes to the sandbox file).
        let sandbox = load();
        assert!(!sandbox.sandboxes.contains_key("host"));
        assert_eq!(sandbox.sandboxes.get("linera-agent").unwrap().len(), 1);

        // Emptying the local set clears its file without touching the sandbox slice.
        replace_local(vec![]).unwrap();
        assert!(load_local().sessions.is_empty());
        assert_eq!(load().sandboxes.get("linera-agent").unwrap().len(), 1);
    }

    #[test]
    fn current_sandbox_gated_on_marker() {
        let _lock = env_guard();
        // SAFETY: env access is serialized by the held `ENV_LOCK`.
        unsafe {
            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
            assert_eq!(current_sandbox(), None);

            std::env::set_var(ENV_SANDBOX_MARKER, "1");
            assert_eq!(current_sandbox(), Some(DEFAULT_SANDBOX_NAME.to_string()));

            std::env::set_var(ENV_SANDBOX_NAME, "pm-task");
            assert_eq!(current_sandbox(), Some("pm-task".to_string()));

            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
        }
    }
}
