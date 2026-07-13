//! Cross-teardown registry of live Claude Code sessions running inside an
//! `sbx` agent sandbox.
//!
//! Sandbox sessions (launched via `sc`) run inside an ephemeral `sbx` microVM,
//! but their Claude Code transcripts (`~/.claude`) and this registry
//! (`~/.local/share/claudectl`) live on host-shared bind mounts, so both
//! survive `sbx rm`. On every `SessionStart` / `SessionEnd` hook that fires
//! *inside a sandbox*, `hook_state::record_hook_event` upserts / removes the
//! session here, keyed by the sandbox name.
//!
//! The payoff: an abrupt `sbx rm` never fires `SessionEnd`, so that sandbox's
//! slice stays frozen at its last live state — exactly the set of sessions
//! `claudectl --restore-sessions` brings back, one `sc --resume <id>` window
//! each.
//!
//! Writes are serialized with an advisory `flock` and committed via
//! temp-file + atomic rename, so concurrent hook processes (many sessions
//! starting/ending at once) never corrupt or tear the file.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

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
    /// The session's `/rename` display name, captured from its (container-local)
    /// session JSON so restore can show it after `sbx rm` destroys that JSON.
    /// `None` until the session is named. See [`set_name`].
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
    // which `upsert` would then persist — the `serde(default)` only fills the
    // field when it's *absent* on read, not for `Default::default()`.
    fn default() -> Self {
        Registry {
            version: current_version(),
            sandboxes: BTreeMap::new(),
        }
    }
}

/// The sandbox this process is running inside, or `None` on the host.
///
/// Returns `Some(name)` only when the sandbox marker env var is present, so
/// host Claude sessions (which fire the same hooks) never touch the registry.
pub fn current_sandbox() -> Option<String> {
    std::env::var_os(ENV_SANDBOX_MARKER)?;
    let name = std::env::var(ENV_SANDBOX_NAME)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_SANDBOX_NAME.to_string());
    Some(name)
}

/// Path to the shared registry file. Honors `CLAUDECTL_SANDBOX_REGISTRY`
/// (used by tests to avoid stomping the real file); otherwise the
/// host-shared `~/.local/share/claudectl` mount.
pub fn registry_path() -> PathBuf {
    if let Some(path) = std::env::var_os("CLAUDECTL_SANDBOX_REGISTRY") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join(".local/share/claudectl/sandbox-sessions.json")
}

/// Read the registry. A missing or unparseable file yields an empty registry —
/// callers treat "no registry" and "empty registry" identically, and a
/// corrupt file should never block a restore attempt or a hook.
pub fn load() -> Registry {
    match fs::read(registry_path()) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => Registry::default(),
    }
}

/// Replace a sandbox's entire slice with `entries` — the current live-session
/// set for that sandbox. This is the whole write model: the in-sandbox writer
/// reconciles the registry to claudectl's live session list on each hook event,
/// so the file always mirrors what the sandbox's claudectl UI shows (idle
/// sessions included, dead ones dropped). An empty `entries` removes the
/// sandbox key. Writes only when the slice actually changes, so it's cheap to
/// call on every event.
pub fn replace_slice(sandbox: &str, entries: Vec<SessionEntry>) -> io::Result<()> {
    with_lock(|| {
        let mut registry = load();
        let current = registry.sandboxes.get(sandbox);
        let unchanged = match (current, entries.is_empty()) {
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
        write_atomic(&registry)
    })
}

/// Serialize `registry` and commit it via temp-file + atomic rename, so a
/// reader never observes a half-written file.
fn write_atomic(registry: &Registry) -> io::Result<()> {
    let path = registry_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(registry)?;
    bytes.push(b'\n');
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, &path)
}

/// Run `body` while holding an exclusive advisory lock on a sidecar lock file,
/// so concurrent hook processes serialize their read-modify-write cycles.
/// The lock releases when the file descriptor closes at the end of scope.
fn with_lock<T>(body: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
    let path = registry_path();
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
            unsafe { std::env::set_var("CLAUDECTL_SANDBOX_REGISTRY", dir.join("registry.json")) };
            TempRegistry { dir, _lock: lock }
        }
    }

    impl Drop for TempRegistry {
        fn drop(&mut self) {
            // SAFETY: still holding `ENV_LOCK` via `_lock`.
            unsafe { std::env::remove_var("CLAUDECTL_SANDBOX_REGISTRY") };
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
    fn replace_slice_sets_and_roundtrips() {
        let _guard = TempRegistry::new("roundtrip");
        replace_slice("linera-agent", vec![entry("aaa", "/work/a")]).unwrap();
        let registry = load();
        let slice = registry.sandboxes.get("linera-agent").unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0], entry("aaa", "/work/a"));
        assert_eq!(registry.version, 1);
    }

    #[test]
    fn replace_slice_overwrites_the_whole_slice() {
        let _guard = TempRegistry::new("overwrite");
        replace_slice("linera-agent", vec![entry("aaa", "/a"), entry("bbb", "/b")]).unwrap();
        // New live set: "aaa" ended, "ccc" started.
        replace_slice("linera-agent", vec![entry("bbb", "/b"), entry("ccc", "/c")]).unwrap();
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
    fn replace_slice_empty_removes_the_sandbox() {
        let _guard = TempRegistry::new("empty");
        replace_slice("linera-agent", vec![entry("aaa", "/a")]).unwrap();
        replace_slice("linera-agent", vec![]).unwrap();
        assert!(!load().sandboxes.contains_key("linera-agent"));
    }

    #[test]
    fn replace_slice_keeps_sandboxes_independent() {
        let _guard = TempRegistry::new("independent");
        replace_slice("linera-agent", vec![entry("aaa", "/a")]).unwrap();
        replace_slice("pm-task", vec![entry("bbb", "/b")]).unwrap();
        replace_slice("linera-agent", vec![]).unwrap();
        let registry = load();
        assert!(!registry.sandboxes.contains_key("linera-agent"));
        assert_eq!(registry.sandboxes.get("pm-task").unwrap().len(), 1);
    }

    #[test]
    fn session_entry_with_name_roundtrips() {
        let _guard = TempRegistry::new("name-roundtrip");
        let mut named = entry("aaa", "/a");
        named.name = Some("faucet-migration".to_string());
        replace_slice("linera-agent", vec![named]).unwrap();
        let registry = load();
        assert_eq!(
            registry.sandboxes.get("linera-agent").unwrap()[0]
                .name
                .as_deref(),
            Some("faucet-migration")
        );
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
