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
//! On every hook, `hook_state::record_hook_event` reconciles — routing on the
//! sandbox marker ([`current_sandbox`]): [`replace_sandbox_slice`] inside a
//! sandbox, else the local path. The writer never deletes a session file.
//! `SessionEnd` is the one event that does not reconcile the live set: it fires
//! during teardown (a Ghostty quit DOES fire it, for every session at once),
//! when the live set is collapsing toward empty. It does, however, take one
//! targeted action — see [`forget_session`] below.
//!
//! The local file is a merge ([`merge_live_keep_all`]), not a mirror. A session
//! that is no longer live survives in it while we can show its terminal died
//! with it ([`is_restore_worthy`]). Two — and only two — things may drop an
//! entry: the reaper ([`retain_restorable`]), which settle-samples the owner to
//! confirm a departed session's terminal died too; and [`forget_session`], the
//! SessionEnd path for a session the user *deliberately* closed (a prompt-level
//! exit reason that can't fire during a terminal quit — see
//! `hook_state::is_deliberate_user_close`), which is durable and immediate and
//! so closes the reaper's timing gap. The general reconcile still never forgets:
//! a hook fires from one session while a different terminal may be mid-quit, and
//! one unconfirmed look then would delete exactly the restore set that quit is
//! about to need. Leaving pruning to nobody was the previous bug (closed
//! sessions lived on forever and got resurrected days later); leaving it to
//! everybody was the one before that.
//!
//! The sandbox slice keeps the plain mirror model: `sbx rm` fires no hooks at
//! all, so its slice freezes on its own.
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

use crate::terminal_owner::TerminalOwner;

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
    /// Pid of the session process when this entry was written.
    ///
    /// The owner below is cached against it: a running process can't change its
    /// ancestry, but a session id can outlive the process — `claude --resume`
    /// keeps the id and gets a new pid under a *different* terminal. Keying the
    /// cache on the pid is what stops a restored session inheriting the dead
    /// terminal it was restored from.
    #[serde(default)]
    pub pid: Option<u32>,
    /// The terminal instance this session ran under, recorded while it was
    /// alive (see [`crate::terminal_owner`]). Once the session is gone this is
    /// the only thing that distinguishes "the user closed it" (terminal still
    /// up) from "the terminal died under it" (restore material).
    /// `None` for entries written before this was tracked, and for sessions
    /// whose owner could not be resolved.
    #[serde(default)]
    pub owner_pid: Option<u32>,
    /// Start time of `owner_pid`, so a recycled pid can't impersonate the
    /// terminal that is actually gone.
    #[serde(default)]
    pub owner_started_at: Option<String>,
}

impl SessionEntry {
    /// The recorded terminal instance, if this entry carries one.
    pub fn owner(&self) -> Option<TerminalOwner> {
        Some(TerminalOwner {
            pid: self.owner_pid?,
            started_at: self.owner_started_at.clone()?,
        })
    }

    /// Does this entry describe the running process `(pid, started_at_ms)`?
    ///
    /// Both must match. A live session keeps one pid for its whole life, so a
    /// match means "same process, owner already resolved" — reuse it and skip
    /// `ps`. But a pid is recycled once its process dies, and `claude --resume`
    /// reuses the session id under a brand-new process; comparing the start time
    /// too means neither a recycled pid nor a resume can make a stale entry
    /// vouch for a stranger (which would pin the new session to a dead terminal).
    pub fn matches_process(&self, pid: u32, started_at_ms: u64) -> bool {
        self.pid == Some(pid) && self.started_at_ms == started_at_ms
    }
}

/// The hook write: take the live set in, keep everything else untouched.
///
/// Live sessions win — they carry fresh names, transcripts and owners. Every
/// departed session's entry is kept verbatim; hooks never forget. A hook fires
/// from one session while a *different* terminal may be mid-quit, and one
/// unconfirmed look then could delete exactly the restore set that quit needs,
/// so the decision to forget is the reaper's alone (see [`retain_restorable`]).
pub fn merge_live_keep_all(
    previous: &[SessionEntry],
    live: Vec<SessionEntry>,
) -> Vec<SessionEntry> {
    let mut entries = live;
    let live_ids: std::collections::HashSet<&str> = entries
        .iter()
        .map(|entry| entry.session_id.as_str())
        .collect();

    let retained: Vec<SessionEntry> = previous
        .iter()
        .filter(|entry| !live_ids.contains(entry.session_id.as_str()))
        .cloned()
        .collect();

    entries.extend(retained);
    entries
}

/// The reaper write: keep live sessions untouched, drop the ones the user
/// closed. Purely subtractive — it never adds or re-attributes, so it can't
/// clobber an owner a hook just recorded.
///
/// An entry is kept when any of these hold:
/// - its session is in `live_ids` — the union of two live scans taken around a
///   settle delay, so a session briefly missing from one scan (a torn pointer
///   read, or one that started mid-pass) still counts as live;
/// - `process_alive(entry)` — its recorded session process exists *right now*.
///   The scans and table are frozen before the registry lock is taken, but this
///   closure runs inside it, and a hook may have registered a brand-new session
///   in between; without a now-check the reaper would prune that live session's
///   entry, and an idle session would never re-register. Start-time-blind by
///   design: a recycled pid keeps a departed entry one process-lifetime longer,
///   which errs toward keeping;
/// - [`is_restore_worthy`] — departed, and its terminal died with it.
///
/// `owner_alive` must be evaluated against a process table sampled *after* the
/// settle delay: only then has a terminal that co-died with its sessions had
/// time to exit, so its owner reads gone and its sessions are correctly kept.
pub fn retain_restorable<P, F>(
    current: &[SessionEntry],
    live_ids: &std::collections::HashSet<String>,
    process_alive: P,
    owner_alive: F,
) -> Vec<SessionEntry>
where
    P: Fn(&SessionEntry) -> bool,
    F: Fn(&TerminalOwner) -> bool,
{
    current
        .iter()
        .filter(|entry| {
            live_ids.contains(&entry.session_id)
                || process_alive(entry)
                || is_restore_worthy(entry, &owner_alive)
        })
        .cloned()
        .collect()
}

/// Is a departed session's entry worth keeping for `--restore-sessions`?
///
/// Only when its terminal is gone too: that is what "the terminal died under
/// it" looks like after the fact. If the terminal is still running, the session
/// ended without it — the user closed it — and resurrecting it days later is
/// the bug this rule exists to prevent.
///
/// Entries with no recorded owner are dropped: that covers registries written
/// before owners were tracked (a one-time clear-out of exactly the stale entries
/// that caused the bug) and sessions we could not attribute, where guessing
/// "restore it" is the harmful direction.
pub fn is_restore_worthy<F>(entry: &SessionEntry, owner_alive: F) -> bool
where
    F: Fn(&TerminalOwner) -> bool,
{
    entry.owner().is_some_and(|owner| !owner_alive(&owner))
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

/// Rewrite the machine's local (laptop) session registry under the file lock.
///
/// `update` is handed the current contents and returns what they should become.
/// It runs *inside* the lock on purpose: the reconcile is a read-modify-write
/// now that entries can outlive their session, and computing it from a read
/// taken outside the lock would let two concurrent hooks race — one writing back
/// a merge that resurrects the very entry the other just pruned. Writes only
/// when the result differs.
///
/// The caller picks this vs [`replace_sandbox_slice`] on the sandbox marker
/// ([`current_sandbox`]), so a sandbox can never mis-route here — even one named
/// "host".
pub fn update_local<F>(update: F) -> io::Result<()>
where
    F: FnOnce(&[SessionEntry]) -> Vec<SessionEntry>,
{
    let path = local_registry_path();
    with_lock(&path, || {
        let mut registry = load_local();
        let entries = update(&registry.sessions);
        if registry.sessions == entries {
            return Ok(());
        }
        registry.sessions = entries;
        write_atomic(&path, &serialize(&registry)?)
    })
}

/// Reconcile one sandbox's slice (keyed by `sandbox` name) of
/// `sandbox-sessions.json` to its current live set.
///
/// Unlike the local file ([`update_local`]) this stays a plain mirror: `sbx rm`
/// is abrupt and fires no further hooks, so the slice freezes at its last live
/// state on its own — exactly what `--restore-sbx-sessions` restores — and
/// container pids say nothing about which host terminal owned anything.
/// Empty `entries` removes the key; other sandboxes' slices are untouched.
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

/// Durably forget `session_id` from the restore registry for the current scope
/// — the current sandbox's slice inside a sandbox, else the host-local registry.
///
/// Called when a SessionEnd `reason` proves the user *deliberately* closed the
/// session (see `hook_state::is_deliberate_user_close`). Unlike the reaper's
/// timer-based prune — which can only tell "the user closed it" from "the
/// terminal died under it" while the terminal is still alive, and so misses a
/// close that is followed by a terminal quit before its next tick — this is
/// immediate and permanent: once forgotten here, no later terminal death can
/// resurrect the session through `--restore-sessions`. A no-op if the id is
/// absent (idempotent), so a duplicate SessionEnd or a race with the reaper is
/// harmless.
pub fn forget_session(session_id: &str) -> io::Result<()> {
    match current_sandbox() {
        Some(sandbox) => {
            let remaining: Vec<SessionEntry> = load()
                .sandboxes
                .remove(&sandbox)
                .unwrap_or_default()
                .into_iter()
                .filter(|entry| entry.session_id != session_id)
                .collect();
            replace_sandbox_slice(&sandbox, remaining)
        }
        None => update_local(|current| {
            current
                .iter()
                .filter(|entry| entry.session_id != session_id)
                .cloned()
                .collect()
        }),
    }
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
pub(crate) mod tests {
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
    pub(crate) struct TempRegistry {
        dir: std::path::PathBuf,
        /// `Some(previous)` when the test also pointed `HOME` at the temp dir.
        saved_home: Option<Option<std::ffi::OsString>>,
        _lock: MutexGuard<'static, ()>,
    }

    impl TempRegistry {
        pub(crate) fn new(tag: &str) -> Self {
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
            TempRegistry {
                dir,
                saved_home: None,
                _lock: lock,
            }
        }

        /// Like [`TempRegistry::new`], but also points `HOME` at the temp dir,
        /// so code that derives paths from it — `discovery::live_sessions`
        /// reading `~/.claude/sessions`, hook state under `~/.claudectl` —
        /// sees an isolated, empty view instead of the real machine's.
        pub(crate) fn with_home(tag: &str) -> Self {
            let mut fixture = Self::new(tag);
            fixture.saved_home = Some(std::env::var_os("HOME"));
            // SAFETY: env access serialized by the `ENV_LOCK` held in `_lock`.
            unsafe {
                std::env::set_var("HOME", &fixture.dir);
            }
            fixture
        }
    }

    impl Drop for TempRegistry {
        fn drop(&mut self) {
            // SAFETY: still holding `ENV_LOCK` via `_lock`.
            unsafe {
                std::env::remove_var("CLAUDECTL_SANDBOX_REGISTRY");
                std::env::remove_var("CLAUDECTL_LOCAL_REGISTRY");
                if let Some(previous) = self.saved_home.take() {
                    match previous {
                        Some(home) => std::env::set_var("HOME", home),
                        None => std::env::remove_var("HOME"),
                    }
                }
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
            pid: None,
            owner_pid: None,
            owner_started_at: None,
        }
    }

    /// An entry recorded while running under terminal instance `owner_pid`.
    fn owned_entry(id: &str, owner_pid: u32) -> SessionEntry {
        SessionEntry {
            owner_pid: Some(owner_pid),
            owner_started_at: Some(format!("start-of-{owner_pid}")),
            ..entry(id, "/work")
        }
    }

    fn ids(entries: &[SessionEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.session_id.as_str()).collect()
    }

    /// Session ids the reaper saw live across its two scans.
    fn live_set(ids: &[&str]) -> std::collections::HashSet<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    /// Owner-liveness against a fixed set of running terminal instances, keyed
    /// the way `owned_entry` records them.
    fn terminals_running(pids: &[u32]) -> impl Fn(&TerminalOwner) -> bool + '_ {
        move |candidate: &TerminalOwner| {
            pids.iter().any(|pid| {
                candidate.pid == *pid && candidate.started_at == format!("start-of-{pid}")
            })
        }
    }

    // ---- hook merge: adds and refreshes, never forgets ------------------

    #[test]
    fn hook_merge_keeps_every_departed_entry_regardless_of_owner() {
        // A hook must never prune — not a hand-closed session, not a legacy one.
        // It can't tell them apart safely mid-quit, so it leaves the verdict to
        // the reaper. Live wins; everything absent from live is kept verbatim.
        let previous = [
            owned_entry("closed-by-hand", 15601),
            entry("legacy-no-owner", "/w"),
        ];
        let merged = merge_live_keep_all(&previous, vec![]);
        assert_eq!(ids(&merged), ["closed-by-hand", "legacy-no-owner"]);
    }

    #[test]
    fn hook_merge_lets_the_live_copy_win() {
        let previous = [SessionEntry {
            name: Some("stale-name".to_string()),
            ..owned_entry("aaa", 111)
        }];
        let live = vec![SessionEntry {
            name: Some("fresh-name".to_string()),
            ..owned_entry("aaa", 222)
        }];
        let merged = merge_live_keep_all(&previous, live);
        assert_eq!(merged.len(), 1, "a session must not be listed twice");
        assert_eq!(merged[0].name.as_deref(), Some("fresh-name"));
        assert_eq!(merged[0].owner_pid, Some(222));
    }

    #[test]
    fn hook_merge_lets_a_new_session_coexist_with_restorable_ones() {
        // Starting a session before --restore-sessions must not wipe the restore
        // set: the merge adds, it doesn't replace.
        let previous = [owned_entry("from-dead-terminal", 111)];
        let live = vec![owned_entry("brand-new", 222)];
        let merged = merge_live_keep_all(&previous, live);
        assert_eq!(ids(&merged), ["brand-new", "from-dead-terminal"]);
    }

    // ---- reaper subtraction: keeps live, drops hand-closed --------------

    #[test]
    fn reaper_keeps_sessions_whose_terminal_died_with_them() {
        // A Ghostty quit: sessions gone, app gone (owner not alive in the
        // post-settle sample). The whole point of the registry — these stay.
        let current = [owned_entry("aaa", 15601), owned_entry("bbb", 15601)];
        let kept = retain_restorable(&current, &live_set(&[]), |_| false, terminals_running(&[]));
        assert_eq!(ids(&kept), ["aaa", "bbb"]);
    }

    #[test]
    fn reaper_drops_sessions_closed_under_a_terminal_that_is_still_running() {
        // /exit or ⌘W: sessions gone, Ghostty still there. The user closed them.
        let current = [owned_entry("aaa", 15601), owned_entry("bbb", 15601)];
        let kept = retain_restorable(
            &current,
            &live_set(&[]),
            |_| false,
            terminals_running(&[15601]),
        );
        assert!(kept.is_empty());
    }

    #[test]
    fn reaper_never_touches_a_live_session() {
        // In either scan's live set → left alone, whatever its owner looks like.
        // This is what keeps the reaper from clobbering a hook's fresh entry.
        let current = [owned_entry("live", 15601)];
        let kept = retain_restorable(
            &current,
            &live_set(&["live"]),
            |_| false,
            terminals_running(&[15601]),
        );
        assert_eq!(ids(&kept), ["live"]);
    }

    #[test]
    fn reaper_judges_each_terminal_separately() {
        let current = [owned_entry("from-dead", 111), owned_entry("from-live", 222)];
        let kept = retain_restorable(
            &current,
            &live_set(&[]),
            |_| false,
            terminals_running(&[222]),
        );
        assert_eq!(ids(&kept), ["from-dead"]);
    }

    #[test]
    fn reaper_keeps_a_dead_session_whose_owner_pid_was_recycled() {
        // Same pid, different instance: the terminal is gone even though
        // something answers to its pid now — the start time doesn't match.
        let current = [SessionEntry {
            owner_started_at: Some("some-older-boot".to_string()),
            ..owned_entry("aaa", 15601)
        }];
        let kept = retain_restorable(
            &current,
            &live_set(&[]),
            |_| false,
            terminals_running(&[15601]),
        );
        assert_eq!(ids(&kept), ["aaa"], "recycled pid must not count as alive");
    }

    #[test]
    fn reaper_drops_departed_entries_that_carry_no_owner() {
        // Legacy entries (pre-owner) can't be judged → dropped once departed.
        let current = [entry("legacy", "/work")];
        let kept = retain_restorable(&current, &live_set(&[]), |_| false, terminals_running(&[]));
        assert!(kept.is_empty());
    }

    #[test]
    fn reaper_keeps_a_session_live_in_only_one_scan() {
        // Union of the two scans: a session that flickered out of one scan (a
        // torn pointer read, or one that started mid-pass) is still "live" and
        // must not be pruned — even though its terminal is running.
        let current = [owned_entry("flickered", 15601)];
        let kept = retain_restorable(
            &current,
            &live_set(&["flickered"]),
            |_| false,
            terminals_running(&[15601]),
        );
        assert_eq!(ids(&kept), ["flickered"]);
    }

    #[test]
    fn reaper_keeps_an_entry_whose_process_is_alive_right_now() {
        // A session that registered itself after both scans (its hook won the
        // flock first) is in neither scan and its terminal is alive — but its
        // process is running. Pruning it would silently drop a live session
        // that may never fire another hook. The now-check inside the locked
        // closure is what closes that window.
        let current = [owned_entry("just-started", 15601)];
        let kept = retain_restorable(
            &current,
            &live_set(&[]),
            |_| true,
            terminals_running(&[15601]),
        );
        assert_eq!(ids(&kept), ["just-started"]);
    }

    #[test]
    fn update_local_computes_from_the_locked_contents() {
        // The merge is a read-modify-write, so it has to see what's actually on
        // disk at write time — reading first and writing later would let a
        // concurrent hook's prune be undone by this one's stale view.
        let _guard = TempRegistry::new("update-local-reads-under-lock");
        update_local(|_| vec![entry("aaa", "/a")]).unwrap();
        update_local(|previous| {
            assert_eq!(ids(previous), ["aaa"], "closure must see the live file");
            let mut next = previous.to_vec();
            next.push(entry("bbb", "/b"));
            next
        })
        .unwrap();
        assert_eq!(ids(&load_local().sessions), ["aaa", "bbb"]);
    }

    #[test]
    fn missing_file_loads_empty() {
        let _guard = TempRegistry::new("missing");
        let registry = load();
        assert!(registry.sandboxes.is_empty());
    }

    #[test]
    fn forget_session_removes_only_the_named_host_entry() {
        let _guard = TempRegistry::new("forget-session");
        update_local(|_| vec![entry("aaa", "/a"), entry("bbb", "/b")]).unwrap();
        forget_session("aaa").unwrap();
        assert_eq!(
            ids(&load_local().sessions),
            ["bbb"],
            "only 'aaa' is forgotten"
        );
        // Idempotent: forgetting an absent id leaves the registry untouched.
        forget_session("zzz").unwrap();
        assert_eq!(ids(&load_local().sessions), ["bbb"]);
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
        update_local(|_| vec![entry("loc", "/l")]).unwrap();

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
        update_local(|_| vec![]).unwrap();
        assert!(load_local().sessions.is_empty());
        assert_eq!(load().sandboxes.get("linera-agent").unwrap().len(), 1);
    }

    #[test]
    fn session_end_hook_preserves_the_registry() {
        let _fixture = TempRegistry::with_home("session-end-preserves");
        // SAFETY: env access is serialized by the `ENV_LOCK` held by `_fixture`.
        unsafe {
            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
        }
        update_local(|_| vec![entry("aaa", "/work/a")]).unwrap();

        // Terminal-app quit: by the time a dying session's SessionEnd hook
        // fires, its pointer file is gone and the live set is collapsing to
        // empty. The registry must keep its last live state — that is exactly
        // what `--restore-sessions` reads seconds later.
        let end = serde_json::json!({"hook_event_name": "SessionEnd", "session_id": "aaa"});
        crate::hook_state::record_hook_event(&end).unwrap();
        assert_eq!(load_local().sessions, vec![entry("aaa", "/work/a")]);

        // Nor may any other hook event forget it. Hooks fire from one session
        // while another terminal may be mid-quit; a hook that pruned on a single
        // look could delete a quitting terminal's whole restore set.
        let stop = serde_json::json!({"hook_event_name": "Stop", "session_id": "aaa"});
        crate::hook_state::record_hook_event(&stop).unwrap();
        assert_eq!(load_local().sessions, vec![entry("aaa", "/work/a")]);
    }

    #[test]
    fn a_hook_records_the_live_set_only_into_its_own_file() {
        // P4 at the seam this commit rewrote: with the sandbox marker set, a
        // hook write must touch the sandbox slice only — the host's file is not
        // ours to mirror from inside a container, even from a sandbox named
        // "host".
        let _fixture = TempRegistry::with_home("hook-routing");
        // SAFETY: env access is serialized by the `ENV_LOCK` held by `_fixture`.
        unsafe {
            std::env::set_var(ENV_SANDBOX_MARKER, "1");
            std::env::set_var(ENV_SANDBOX_NAME, "host");
        }
        update_local(|_| vec![owned_entry("host-side", 15601)]).unwrap();

        let routed =
            crate::hook_state::record_live_sessions(&crate::terminal_owner::OwnerCheck::lazy());

        // SAFETY: still holding `ENV_LOCK` via `_fixture`.
        unsafe {
            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
        }
        routed.unwrap();
        assert_eq!(
            ids(&load_local().sessions),
            ["host-side"],
            "a sandbox hook must not rewrite the host registry"
        );
        assert!(
            !load().sandboxes.contains_key("host"),
            "and its writes land under the sandbox name, not a bogus 'host' key"
        );
    }

    #[test]
    fn matches_process_guards_pid_reuse_and_resume() {
        // `claude --resume` keeps the session id but runs a new process under a
        // new terminal. Matching on pid AND start time means a stale entry can't
        // make a resumed (or pid-recycled) session inherit a dead terminal.
        let entry = SessionEntry {
            pid: Some(100),
            started_at_ms: 42,
            ..owned_entry("aaa", 15601)
        };
        assert!(entry.matches_process(100, 42), "same process");
        assert!(
            !entry.matches_process(100, 99),
            "same pid, different start time — a recycled pid"
        );
        assert!(
            !entry.matches_process(200, 42),
            "different pid — a resumed session"
        );
        assert!(
            !SessionEntry { pid: None, ..entry }.matches_process(100, 42),
            "an entry with no recorded pid vouches for nothing"
        );
    }

    #[test]
    fn a_hook_keeps_what_the_reaper_would_drop() {
        // The two writes over identical input, side by side: a hand-closed
        // session (terminal 15601 still running) survives the hook merge but not
        // the reaper's subtraction.
        let previous = [owned_entry("closed-by-hand", 15601)];
        let kept = merge_live_keep_all(&previous, vec![]);
        assert_eq!(ids(&kept), ["closed-by-hand"]);

        let pruned = retain_restorable(
            &previous,
            &live_set(&[]),
            |_| false,
            terminals_running(&[15601]),
        );
        assert!(pruned.is_empty());
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
