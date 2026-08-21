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
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::terminal_owner::TerminalOwner;

/// Env var `sc` sets inside the sandbox to mark that Claude is running there.
/// Its mere presence — not its value — gates registry writes, mirroring the
/// `var_os(...).is_some()` convention the sandbox launcher uses elsewhere.
pub const ENV_SANDBOX_MARKER: &str = "LINERA_SANDBOX";
/// Env var carrying the sandbox's name (`sbx` container). Matches `sc`'s
/// `SANDBOX_NAME`, which defaults to `linera-agent` for the shared sandbox.
pub const ENV_SANDBOX_NAME: &str = "SANDBOX_NAME";
/// Env var carrying the sandbox's id, set by `sbx` alongside `SANDBOX_NAME`.
/// A second chance at the name when only one of the two was inherited.
pub const ENV_SANDBOX_VM_ID: &str = "SANDBOX_VM_ID";
/// Default when `SANDBOX_NAME` is unset — kept in sync with `sc`.
const DEFAULT_SANDBOX_NAME: &str = "linera-agent";
/// Directory the sandbox bootstrap creates for per-pid session sidecars.
///
/// Used here purely as a "am I inside a sandbox" marker. It is a property of
/// the machine rather than of one process's environment, so unlike
/// [`ENV_SANDBOX_MARKER`] no exec can drop it. Absent on hosts — the same
/// assumption `process::read_terminal_sidecar` already relies on.
const SANDBOX_MARKER_DIR: &str = "/var/lib/sandbox-sessions";
/// The UTS hostname, which `sbx` sets to the sandbox name. Namespace-scoped,
/// so it survives any exec. **Not** `/etc/hostname`, which reads
/// `localhost.localdomain` inside these sandboxes and would silently become a
/// bogus slice name.
const UTS_HOSTNAME_PATH: &str = "/proc/sys/kernel/hostname";

fn current_version() -> u32 {
    1
}

/// One resumable session recorded in the registry.
///
/// `Default` exists for test fixtures, which build entries caring about two or
/// three fields at a time. Production code constructs this literally, in one
/// place ([`crate::hook_state::record_live_sessions`]), so a field added here
/// still has to be answered for where it actually matters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Host-side terminal-surface id for a session running inside a sandbox,
    /// copied from the per-pid `terminal.json` sidecar the agent-sandbox
    /// wrapper writes at launch.
    ///
    /// Load-bearing for cross-sandbox Tab. Everything else this entry carries
    /// is *container*-scoped: `pid` belongs to the sandbox's own pid namespace
    /// and `cwd` is a path inside it, so a host reading this slice has nothing
    /// it can route on. Without this field the Ghostty matcher fell all the way
    /// through to "every surface whose working directory is `$HOME`" and took
    /// the first one — a coin flip that only looked like it worked because
    /// *named* sessions got rescued by the title disambiguator further down.
    #[serde(default)]
    pub host_terminal_id: Option<String>,
    /// Host-side tty for the same session, from the same sidecar. The fallback
    /// for terminals that match by tty rather than a surface id (iTerm2,
    /// Terminal.app, tmux, WezTerm).
    #[serde(default)]
    pub host_tty: Option<String>,
    /// When `SessionEnd` fired for this session, if it has.
    ///
    /// This file serves two consumers whose retention needs are opposites. The
    /// dashboard wants the *live* set. `--restore-sbx-sessions` wants exactly
    /// the sessions that died with their terminal, and deliberately keeps them
    /// — `session_end_forgets_the_entry_only_on_a_deliberate_close` asserts an
    /// entry closed by reason `other` is *kept*, because that is the restore
    /// material.
    ///
    /// Before this field the two were indistinguishable, so once the renderer
    /// started reading membership from here a closed terminal left its row on
    /// screen until some *unrelated* session in the same sandbox happened to
    /// fire a hook and trigger the wholesale reconcile — a minute or more, and
    /// unbounded when the rest of the sandbox is idle.
    ///
    /// Marking instead of deleting keeps both consumers honest: the view skips
    /// these, restore still finds them. Cleared when the id is seen live again,
    /// so a `--resume` un-departs the session rather than stranding it hidden.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub departed_at_ms: Option<u64>,
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

/// Carry stored identity forward into incoming entries that lost theirs.
///
/// An incoming empty field does not mean "the session lost this property" — it
/// means the discovery pass had no source for it. When Claude Code deletes a
/// session's pointer file mid-session (it does, routinely), the next discovery
/// falls back to the process table, which knows only the pid and the
/// `--resume` uuid. Everything else comes back blank. The registry copy is the
/// durable one, so a blank must never overwrite it; a non-empty incoming value
/// always wins, being genuinely fresher.
///
/// **`name` alone was covered here until 2026-08-06, and the other two fields
/// have exactly the same failure.** With `cwd` blanked, `transcript` is
/// recomputed as `~/.claude/projects/-/<id>.jsonl` — the `-` being what an
/// empty cwd renders as — which does not exist. Every affected row then loses
/// its title *and* reads `Unreadable`, showing nothing but a pid. Observed on
/// 18 sessions at once: `cwd: ""`, `name: null`, and a transcript path under
/// `projects/-/` while the real transcript sat in `projects/-Users-ndr/`.
fn backfill_missing_identity(previous: &[SessionEntry], entries: &mut [SessionEntry]) {
    for entry in entries.iter_mut() {
        let Some(known) = previous
            .iter()
            .find(|prev| prev.session_id == entry.session_id)
        else {
            continue;
        };
        if entry.name.is_none() {
            entry.name = known.name.clone();
        }
        // `cwd` and `transcript` move together: `transcript` is derived from
        // `cwd`, so a rediscovered entry that lost the first has a wrong,
        // non-existent value for the second rather than an empty one. Restoring
        // the stored cwd without the stored transcript would leave the row
        // pointing at a file that is not there.
        if entry.cwd.is_empty() && !known.cwd.is_empty() {
            entry.cwd = known.cwd.clone();
            entry.transcript = known.transcript.clone();
        }
    }
}

/// The hook write: take the live set in, keep everything else untouched.
///
/// Live sessions win — they carry fresh names, transcripts and owners (an
/// incoming entry that lost its *name* is the one exception; see
/// [`backfill_missing_identity`]). Every
/// departed session's entry is kept verbatim; hooks never forget. A hook fires
/// from one session while a *different* terminal may be mid-quit, and one
/// unconfirmed look then could delete exactly the restore set that quit needs,
/// so the decision to forget is the reaper's alone (see [`retain_restorable`]).
pub fn merge_live_keep_all(
    previous: &[SessionEntry],
    live: Vec<SessionEntry>,
) -> Vec<SessionEntry> {
    let mut entries = live;
    backfill_missing_identity(previous, &mut entries);
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
    release_superseded_pids(&mut entries);
    entries
}

/// A pid belongs to exactly one conversation at a time. Claude Code rotates
/// `sessionId` under a running process (/clear, compaction), so a slice
/// accumulates one entry per sid all claiming the same pid — and every
/// pid-alive gate downstream (the pointerless re-add, restore's resumed
/// check) then treats each superseded conversation as live too, one display
/// row each (2026-07-28: one tab showed as three rows). First claimant
/// keeps the pid: entries are ordered live-scan first, so the sid that
/// currently owns the process wins. Later entries with the same pid split
/// two ways:
/// - same `started_at_ms` as the claimant → the same process, i.e. a
///   conversation the user ENDED by rotating past it. Dropped: keeping it
///   made `--restore-sessions` spawn a second window resuming the
///   superseded conversation while its successor was still running, and
///   the reaper would prune it within one pass anyway (terminal still up,
///   no live process).
/// - different `started_at_ms` → an unrelated conversation whose recorded
///   pid was recycled. Released to pid-less: it stays a restore candidate
///   (its terminal is typically gone), it just no longer counts as alive.
fn release_superseded_pids(entries: &mut Vec<SessionEntry>) {
    let mut claimed: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    entries.retain_mut(|entry| {
        let Some(pid) = entry.pid else { return true };
        match claimed.get(&pid) {
            None => {
                claimed.insert(pid, entry.started_at_ms);
                true
            }
            Some(claimant_started) if *claimant_started == entry.started_at_ms => false,
            Some(_) => {
                entry.pid = None;
                true
            }
        }
    });
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

/// One origin's collected inventory. `is_current` marks the sandbox the alias
/// resolves to; every other running sandbox is superseded and draining.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SandboxOrigin {
    #[serde(default)]
    pub is_current: bool,
    /// Untyped on purpose. These rows are a *persisted* format: a snapshot on
    /// disk may have been written by an older claudectl, and a reader that
    /// refused to deserialize an unexpected field would render nothing at all
    /// rather than the fields it does understand. `from_snapshot_value` picks
    /// what it needs and defaults the rest.
    #[serde(default)]
    pub sessions: Vec<serde_json::Value>,
}

/// Host-collected snapshot of every sandbox's sessions (`sandboxes.json`).
///
/// Distinct from [`Registry`], which is written by *hooks from inside* each
/// sandbox and only ever describes the writer's own slice. This file is written
/// by a single host-side collector that can see all sandboxes at once, so it is
/// the only place cross-sandbox liveness is authoritative.
///
/// `collected_at_ms` is load-bearing, not decoration: a reader that cannot tell
/// a fresh snapshot from one abandoned by a dead collector would render a
/// confidently wrong session list. Consumers must degrade on staleness rather
/// than trust it silently.
#[derive(Debug, Serialize, Deserialize)]
pub struct SandboxSnapshot {
    #[serde(default = "current_version")]
    pub version: u32,
    #[serde(default)]
    pub collected_at_ms: u64,
    #[serde(default)]
    pub sandboxes: BTreeMap<String, SandboxOrigin>,
}

impl Default for SandboxSnapshot {
    // Hand-written for the same reason as `Registry`'s: a derived impl would
    // persist version 0.
    fn default() -> Self {
        SandboxSnapshot {
            version: current_version(),
            collected_at_ms: 0,
            sandboxes: BTreeMap::new(),
        }
    }
}

/// How many collector intervals a snapshot may fall behind before its
/// measurements are treated as expired.
///
/// Two rather than one: the reaper is a timer, not a guarantee. A single tick
/// can be skipped by a slow `sbx exec`, a laptop asleep between fires, or a
/// run that lost the race with the orphan-scan cache — none of which mean the
/// collector is dead, and all of which would make a one-interval bound flap
/// the vitals off and on for a perfectly healthy fleet.
const STALE_AFTER_INTERVALS: u64 = 2;

impl SandboxSnapshot {
    /// Age of this snapshot, or `None` if it carries no collection time.
    ///
    /// `None` is not zero: it means a writer that predates `collected_at_ms`,
    /// or a default-constructed value. Callers must treat it as "unknown", the
    /// same way they treat expired — never as "fresh".
    pub fn age(&self, now_ms: u64) -> Option<Duration> {
        (self.collected_at_ms != 0)
            .then(|| Duration::from_millis(now_ms.saturating_sub(self.collected_at_ms)))
    }

    /// Whether the collected measurements may still be shown.
    ///
    /// The field this reads has been written and documented as load-bearing
    /// since the snapshot was introduced, but until now nothing read it, so a
    /// dead collector rendered an arbitrarily old CPU and memory column with
    /// full confidence and no indication. A reader that cannot tell a fresh
    /// snapshot from an abandoned one has to assume the worst.
    ///
    /// `collector_interval` is the reaper's configured period, so raising
    /// `--reaper-interval` widens this automatically instead of silently
    /// expiring every snapshot the moment it exceeds a hardcoded guess.
    pub fn is_fresh(&self, now_ms: u64, collector_interval: Duration) -> bool {
        match self.age(now_ms) {
            Some(age) => age < collector_interval * STALE_AFTER_INTERVALS as u32,
            None => false,
        }
    }
}

/// Path to the host-collected snapshot (`sandboxes.json`). Honors
/// `CLAUDECTL_SANDBOX_SNAPSHOT` (tests, to avoid stomping the real file).
pub fn sandbox_snapshot_path() -> PathBuf {
    std::env::var_os("CLAUDECTL_SANDBOX_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| registry_dir().join("sandboxes.json"))
}

/// Read the host-collected snapshot. Missing or unparseable yields the default
/// — same posture as [`load`]: a corrupt or absent file must degrade to "no
/// foreign origins", never block rendering the local ones.
pub fn load_snapshot() -> SandboxSnapshot {
    match fs::read(sandbox_snapshot_path()) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => SandboxSnapshot::default(),
    }
}

/// Replace the snapshot wholesale. Unlike the hook-written registries there is
/// no merge: the collector observes every sandbox in one pass, so its view is
/// complete by construction and a merge could only resurrect a reaped sandbox.
pub fn write_snapshot(snapshot: &SandboxSnapshot) -> io::Result<()> {
    let path = sandbox_snapshot_path();
    with_lock(&path, || write_atomic(&path, &serialize(snapshot)?))
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
/// This decides which registry a hook writes — the sandbox's own slice, or the
/// **host's** file. Getting it wrong in the `None` direction is the dangerous
/// one, and the env marker alone got it wrong that way: it **fails open**.
/// `LINERA_SANDBOX` is set by the bootstrap, but `sbx exec` does not inherit
/// it, so a `claudectl-hook` invoked that way concluded it was the laptop and
/// wrote sandbox sessions into the host-local registry — precisely what the
/// marker's doc claims it prevents ("a sandbox can never write the host's
/// file"). Observed live 2026-08-05: `registry: live set = 19 sessions,
/// 0 with host routing, scope=host-local` from inside a sandbox.
///
/// So presence is now established by either signal: the env marker, or a
/// filesystem marker that no exec can drop. Absence of both still means host,
/// which keeps the host's behaviour unchanged.
pub fn current_sandbox() -> Option<String> {
    if !in_sandbox() {
        return None;
    }
    Some(registry_sandbox_name(
        env_value(ENV_SANDBOX_NAME).as_deref(),
        env_value(ENV_SANDBOX_VM_ID).as_deref(),
        read_uts_hostname().as_deref(),
    ))
}

fn in_sandbox() -> bool {
    std::env::var_os(ENV_SANDBOX_MARKER).is_some() || marker_dir().is_dir()
}

/// The filesystem sandbox marker, overridable by `CLAUDECTL_SANDBOX_MARKER_DIR`.
///
/// The override exists for the same reason `CLAUDECTL_SANDBOX_REGISTRY` and
/// friends do: a test that wants to exercise *host* behaviour must be able to
/// say so while running inside a sandbox, where the real marker is present and
/// unremovable. Clearing `LINERA_SANDBOX` used to be enough for that; adding a
/// filesystem signal would otherwise have made host behaviour untestable here —
/// exactly the "passes on CI, fails in the sandbox" scope mismatch documented
/// on `TempRegistry`.
fn marker_dir() -> PathBuf {
    std::env::var_os("CLAUDECTL_SANDBOX_MARKER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(SANDBOX_MARKER_DIR))
}

/// A non-empty, trimmed env var, or `None`.
fn env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn read_uts_hostname() -> Option<String> {
    fs::read_to_string(UTS_HOSTNAME_PATH)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Pick the sandbox's name from the sources available to this exec.
///
/// Pure so the precedence is testable without mutating process-global env,
/// which is shared across the test binary's threads. The explicit-beats-
/// inferred-beats-default ordering itself is [`crate::reaper::resolve_sandbox_name`];
/// what lives here is only which sources feed it and which are disqualified.
///
/// The hostname is inferred rather than explicit because a name the launcher
/// set deliberately beats one read off the machine. It is still preferred over
/// [`DEFAULT_SANDBOX_NAME`]: defaulting writes a slice under a name that may
/// belong to a *different* sandbox, and a wrong slice is worse than the
/// host-local misroute this function exists to fix. Loopback names are
/// disqualified — inside these sandboxes `/etc/hostname` says
/// `localhost.localdomain`, and a registry keyed on that would collide every
/// sandbox that fell back to it into one slice.
fn registry_sandbox_name(
    name_env: Option<&str>,
    vm_id_env: Option<&str>,
    uts_hostname: Option<&str>,
) -> String {
    crate::reaper::resolve_sandbox_name(
        name_env.or(vm_id_env),
        uts_hostname.filter(|host| !is_loopback_name(host)),
        DEFAULT_SANDBOX_NAME,
    )
}

fn is_loopback_name(host: &str) -> bool {
    host == "localhost" || host.starts_with("localhost.")
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
    try_load().unwrap_or_default()
}

/// The same read, keeping the difference between "there is nothing recorded"
/// and "we could not find out".
///
/// **Any read-modify-write must use this one.** [`load`] answers both cases
/// with an empty registry, which is right for a renderer and catastrophic as
/// the base of a write-back: [`replace_sandbox_slice`] serialises the whole
/// file, so one unreadable read while a hook saves its own slice deletes every
/// other sandbox's sessions from disk.
///
/// That is not hypothetical. On 2026-08-17 the registry went from 42 sessions
/// across two sandboxes to the 1 session of the sandbox whose hook happened to
/// fire, and 41 live sessions vanished from the dashboard. A torn read is the
/// mechanism to expect here: the file is rewritten in place by the host
/// collector while in-sandbox hooks read it over virtiofs, where fresh
/// metadata with stale pages is a known hazard, and a half-written 20 KB JSON
/// fails to parse.
///
/// A *missing* file is not a failure — that is a first run, and an empty
/// registry is the honest answer.
pub fn try_load() -> io::Result<Registry> {
    let bytes = match fs::read(sandbox_registry_path()) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Registry::default()),
        Err(e) => return Err(e),
    };
    serde_json::from_slice(&bytes).map_err(|e| {
        io::Error::other(format!(
            "registry at {} is unreadable ({e}); refusing to treat {} bytes we cannot parse \
             as an empty registry",
            sandbox_registry_path().display(),
            bytes.len()
        ))
    })
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
pub fn replace_sandbox_slice(sandbox: &str, mut entries: Vec<SessionEntry>) -> io::Result<()> {
    let path = sandbox_registry_path();
    with_lock(&path, || {
        // Never `load()` here: this write serialises the *whole* file, so a
        // base we could not read would publish this sandbox's slice as the
        // entire registry and delete every other sandbox with it.
        let mut registry = try_load()?;
        // A replace is wholesale, so without this a single tick whose live
        // set was assembled from the process table alone (registry read
        // missed, pointers long gone) would blank EVERY session title in the
        // sandbox at once. Backfill before the unchanged-comparison so a
        // no-op-after-backfill write is still skipped.
        if let Some(existing) = registry.sandboxes.get(sandbox) {
            backfill_missing_identity(existing, &mut entries);
        }
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

/// Every session id the registry holds, across all sandboxes.
pub fn known_session_ids() -> std::collections::HashSet<String> {
    load()
        .sandboxes
        .values()
        .flatten()
        .map(|entry| entry.session_id.clone())
        .collect()
}

/// Add sessions the host can see that the registry has no record of.
///
/// Additive only, and that is the point. A host `ps` sweep proves a session
/// EXISTS; it never proves one does not, because a session launched with a
/// prompt rather than `--resume` carries no id in its argv. Every removal path
/// in this module is driven by evidence from inside the sandbox; this one only
/// ever adds, so it cannot participate in losing anything.
///
/// It closes the gap that leaves live sessions invisible: a slice is rewritten
/// only when a hook fires inside its sandbox, so a session idle since it was
/// resumed never triggers one, and a slice lost while it was idle stays lost.
/// The host's process table has no such blind spot.
///
/// An id the registry already holds is left exactly as it is — the in-sandbox
/// writer knows the pid and the `/rename` name, and this caller does not.
pub fn adopt_host_visible(
    seen: impl IntoIterator<Item = (String, SessionEntry)>,
) -> io::Result<()> {
    let path = sandbox_registry_path();
    with_lock(&path, || {
        let mut registry = try_load()?;
        let known: std::collections::HashSet<String> = registry
            .sandboxes
            .values()
            .flatten()
            .map(|entry| entry.session_id.clone())
            .collect();

        let mut added = 0usize;
        for (sandbox, entry) in seen {
            if entry.session_id.is_empty() || known.contains(&entry.session_id) {
                continue;
            }
            registry.sandboxes.entry(sandbox).or_default().push(entry);
            added += 1;
        }
        if added == 0 {
            return Ok(());
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
            let remaining: Vec<SessionEntry> = try_load()?
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

/// Stamp `session_id` as departed in the current scope, keeping the entry.
///
/// The counterpart to [`forget_session`], for the `SessionEnd` reasons that are
/// *not* a deliberate user close. Those must stay on disk — a session whose
/// terminal died is precisely what `--restore-sbx-sessions` brings back — but
/// they must stop rendering as live the moment the hook fires, rather than
/// waiting for an unrelated session in the same sandbox to trigger a reconcile.
///
/// Idempotent, and it does not overwrite an existing stamp: the first
/// `SessionEnd` is the departure, and a duplicate delivery must not move the
/// clock. A no-op when the id is absent.
pub fn mark_session_departed(session_id: &str, at_ms: u64) -> io::Result<()> {
    let stamp = |entries: &[SessionEntry]| -> Vec<SessionEntry> {
        entries
            .iter()
            .map(|entry| {
                if entry.session_id == session_id && entry.departed_at_ms.is_none() {
                    SessionEntry {
                        departed_at_ms: Some(at_ms),
                        ..entry.clone()
                    }
                } else {
                    entry.clone()
                }
            })
            .collect()
    };
    match current_sandbox() {
        Some(sandbox) => {
            let slice = try_load()?.sandboxes.remove(&sandbox).unwrap_or_default();
            // Nothing recorded for this sandbox yet ⇒ nothing to stamp. Writing
            // an empty slice here would remove the key and, with it, any
            // restore material a concurrent writer had just added.
            if slice.is_empty() {
                return Ok(());
            }
            replace_sandbox_slice(&sandbox, stamp(&slice))
        }
        None => update_local(|current| stamp(current)),
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

    pub(crate) fn env_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    /// Point the registry at a throwaway file for the duration of a test, while
    /// holding the env lock so no other test observes our `CLAUDECTL_*` vars.
    pub(crate) struct TempRegistry {
        dir: std::path::PathBuf,
        /// `Some(previous)` when the test also pointed `HOME` at the temp dir.
        saved_home: Option<Option<std::ffi::OsString>>,
        /// Previous `LINERA_SANDBOX`, restored on drop. See [`TempRegistry::new`].
        saved_sandbox_marker: Option<std::ffi::OsString>,
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
            // Pin the scope to the HOST by clearing the sandbox marker.
            //
            // `current_sandbox()` routes every registry write to either the
            // local file or a sandbox slice, and these fixtures seed the local
            // one. Left inherited, the same test asserts against a file the
            // code never wrote — passing on CI and failing inside an
            // agent-sandbox, where `LINERA_SANDBOX=1` is always set. That is
            // what made `session_end_forgets_the_entry_only_on_a_deliberate_close`
            // look like an unexplained environment flake for weeks; it was a
            // scope mismatch, not flakiness. Tests that want sandbox scope set
            // the variable themselves after constructing the fixture (see
            // `reaper.rs`), and Drop restores whatever was here.
            let saved_sandbox_marker = std::env::var_os(ENV_SANDBOX_MARKER);
            // SAFETY: env access here is serialized by the held `ENV_LOCK`.
            unsafe {
                std::env::set_var("CLAUDECTL_SANDBOX_REGISTRY", dir.join("sandbox.json"));
                std::env::set_var("CLAUDECTL_LOCAL_REGISTRY", dir.join("local.json"));
                std::env::set_var("CLAUDECTL_SANDBOX_SNAPSHOT", dir.join("sandboxes.json"));
                std::env::remove_var(ENV_SANDBOX_MARKER);
                // Clearing the env marker alone no longer means "host": the
                // filesystem marker is present in every sandbox and cannot be
                // removed. Point it somewhere that doesn't exist so this
                // fixture keeps meaning host scope in both places it runs.
                std::env::set_var(
                    "CLAUDECTL_SANDBOX_MARKER_DIR",
                    dir.join("no-sandbox-marker"),
                );
            }
            TempRegistry {
                dir,
                saved_home: None,
                saved_sandbox_marker,
                _lock: lock,
            }
        }

        /// Like [`TempRegistry::new`], but also points `HOME` at the temp dir,
        /// so code that derives paths from it — `discovery::live_sessions`
        /// reading `~/.claude/sessions`, hook state under
        /// `~/.local/share/claudectl/state` — sees an isolated, empty view
        /// instead of the real machine's.
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
                std::env::remove_var("CLAUDECTL_SANDBOX_SNAPSHOT");
                std::env::remove_var("CLAUDECTL_SANDBOX_MARKER_DIR");
                match self.saved_sandbox_marker.take() {
                    Some(value) => std::env::set_var(ENV_SANDBOX_MARKER, value),
                    None => std::env::remove_var(ENV_SANDBOX_MARKER),
                }
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

    /// `SANDBOX_NAME` is what the launcher set deliberately, so it wins.
    #[test]
    fn sandbox_name_prefers_the_explicit_env_over_everything() {
        assert_eq!(
            registry_sandbox_name(Some("linera-agent-abc"), Some("vm-id"), Some("host-name")),
            "linera-agent-abc"
        );
    }

    /// `sbx exec` inherits neither `SANDBOX_NAME` nor `LINERA_SANDBOX`. The UTS
    /// hostname is the only name left, and it is what `sbx` sets — falling
    /// through to `DEFAULT_SANDBOX_NAME` here would write a slice under a name
    /// that may belong to a different sandbox.
    #[test]
    fn sandbox_name_falls_back_to_the_uts_hostname_when_no_env_survived() {
        assert_eq!(
            registry_sandbox_name(None, None, Some("linera-agent-cd708d9d80bc")),
            "linera-agent-cd708d9d80bc"
        );
        assert_eq!(
            registry_sandbox_name(None, Some("linera-agent-vm"), Some("ignored")),
            "linera-agent-vm",
            "an inherited vm id still beats the inferred hostname"
        );
    }

    /// `/etc/hostname` reads `localhost.localdomain` inside these sandboxes.
    /// Keying a registry slice on that would collide every sandbox that fell
    /// back to it into one shared slice.
    #[test]
    fn sandbox_name_rejects_loopback_hostnames() {
        assert_eq!(
            registry_sandbox_name(None, None, Some("localhost.localdomain")),
            DEFAULT_SANDBOX_NAME
        );
        assert_eq!(
            registry_sandbox_name(None, None, Some("localhost")),
            DEFAULT_SANDBOX_NAME
        );
        assert_eq!(
            registry_sandbox_name(None, None, None),
            DEFAULT_SANDBOX_NAME,
            "no source at all still yields the documented default"
        );
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
            ..Default::default()
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
    fn regression_rotation_drops_the_superseded_sid_entry() {
        // 2026-07-28: Claude Code rotated sessionId under a running process
        // (/clear); the superseded sid's entry kept the pid, so every
        // pid-alive gate downstream treated one process as several live
        // conversations (one display row each), and --restore-sessions
        // offered to resume a conversation the user had deliberately ended.
        // Same pid + same started_at = the same process: the superseded
        // conversation is over — drop it.
        let previous = [SessionEntry {
            pid: Some(3_276_188),
            started_at_ms: 1_785_275_675_303,
            name: Some("pm-control-chain-liveness-alerting".into()),
            ..entry("aaaa-superseded", "/Users/ndr")
        }];
        let live = vec![SessionEntry {
            pid: Some(3_276_188),
            started_at_ms: 1_785_275_675_303,
            ..entry("bbbb-current", "/Users/ndr")
        }];
        let merged = merge_live_keep_all(&previous, live);
        assert_eq!(ids(&merged), ["bbbb-current"]);
        assert_eq!(merged[0].pid, Some(3_276_188), "current sid keeps the pid");
    }

    #[test]
    fn recycled_pid_releases_but_keeps_the_unrelated_entry() {
        // Different started_at = a different process: the stored entry's pid
        // was merely recycled by an unrelated session. The old conversation
        // was never ended by the user — it must stay a restore candidate,
        // just no longer counted as alive.
        let previous = [SessionEntry {
            pid: Some(4242),
            started_at_ms: 100,
            name: Some("old-unrelated-conversation".into()),
            ..entry("aaaa-old", "/Users/ndr")
        }];
        let live = vec![SessionEntry {
            pid: Some(4242),
            started_at_ms: 900,
            ..entry("bbbb-new", "/Users/ndr")
        }];
        let merged = merge_live_keep_all(&previous, live);
        assert_eq!(ids(&merged), ["bbbb-new", "aaaa-old"]);
        assert_eq!(merged[0].pid, Some(4242));
        assert_eq!(merged[1].pid, None, "recycled pid is released, entry kept");
        assert_eq!(
            merged[1].name.as_deref(),
            Some("old-unrelated-conversation")
        );
    }

    #[test]
    fn no_two_entries_share_a_pid_after_any_merge() {
        // The invariant behind the display's one-row-per-pid guarantee: no
        // matter how polluted the inputs are (old binaries recorded dup-pid
        // slices), a merge result never has two entries claiming one pid.
        let previous = [
            SessionEntry {
                pid: Some(7),
                ..entry("p1", "/w")
            },
            SessionEntry {
                pid: Some(7),
                ..entry("p2", "/w")
            },
            SessionEntry {
                pid: Some(8),
                ..entry("p3", "/w")
            },
        ];
        let live = vec![SessionEntry {
            pid: Some(7),
            ..entry("l1", "/w")
        }];
        let merged = merge_live_keep_all(&previous, live);
        let mut claimed = std::collections::HashSet::new();
        for e in &merged {
            if let Some(pid) = e.pid {
                assert!(claimed.insert(pid), "pid {pid} claimed twice");
            }
        }
        assert_eq!(
            merged.iter().find(|e| e.session_id == "l1").unwrap().pid,
            Some(7)
        );
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

    /// `entry(..)` with a display name attached.
    fn named_entry(id: &str, cwd: &str, name: &str) -> SessionEntry {
        SessionEntry {
            name: Some(name.to_string()),
            ..entry(id, cwd)
        }
    }

    #[test]
    fn regression_merge_keeps_title_when_incoming_entry_lost_it() {
        // 2026-07-28 incident: a session's registry entry was pruned and the
        // next hook tick rediscovered it from the process table — which only
        // knows the `--resume` uuid, so the incoming entry had `name: None`.
        // The wholesale merge let that None overwrite the stored title
        // ("resume-old-sessions-audit" became null, blank in the TUI). A
        // stored name must survive a name-less re-record.
        let previous = vec![named_entry(
            "3c20ad09",
            "/Users/ndr",
            "resume-old-sessions-audit",
        )];
        let merged = merge_live_keep_all(&previous, vec![entry("3c20ad09", "/Users/ndr")]);
        assert_eq!(
            merged[0].name.as_deref(),
            Some("resume-old-sessions-audit"),
            "a name-less live re-record must not blank the stored title"
        );
    }

    #[test]
    fn merge_takes_incoming_rename_over_stored_name() {
        // The backfill is for None only: a genuinely fresher name (pointer
        // file or transcript recovery saw a /rename) always wins.
        let previous = vec![named_entry("s1", "/w", "old-name")];
        let merged = merge_live_keep_all(&previous, vec![named_entry("s1", "/w", "new-name")]);
        assert_eq!(merged[0].name.as_deref(), Some("new-name"));
    }

    #[test]
    fn regression_sandbox_replace_keeps_titles_when_live_set_lost_them() {
        // Same incident shape, sandbox flavor — and worse: the sandbox write
        // is a wholesale slice replace, so ONE tick assembled from the
        // process table alone would blank every session title in the sandbox
        // at once. The stored titles must survive.
        let _guard = TempRegistry::new("name-backfill");
        replace_sandbox_slice(
            "linera-agent",
            vec![
                named_entry("aaa", "/a", "title-a"),
                named_entry("bbb", "/b", "title-b"),
            ],
        )
        .unwrap();
        // Next tick: same live set, but discovery had no name source.
        replace_sandbox_slice("linera-agent", vec![entry("aaa", "/a"), entry("bbb", "/b")])
            .unwrap();
        let registry = load();
        let names: Vec<_> = registry
            .sandboxes
            .get("linera-agent")
            .unwrap()
            .iter()
            .map(|e| (e.session_id.clone(), e.name.clone()))
            .collect();
        assert_eq!(
            names,
            [
                ("aaa".to_string(), Some("title-a".to_string())),
                ("bbb".to_string(), Some("title-b".to_string())),
            ],
            "a name-less replace must not blank stored sandbox titles"
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
    fn adopt_adds_a_session_the_registry_never_saw() {
        let _guard = TempRegistry::new("adopt-new");
        replace_sandbox_slice("sbx-a", vec![entry("aaa", "/a")]).unwrap();
        adopt_host_visible([("sbx-a".to_string(), entry("bbb", "/b"))]).unwrap();
        assert_eq!(ids(load().sandboxes.get("sbx-a").unwrap()), ["aaa", "bbb"]);
    }

    /// The in-sandbox writer knows the pid and the `/rename` name; a host sweep
    /// knows neither. Overwriting would downgrade a good entry to a poorer one.
    #[test]
    fn adopt_never_overwrites_an_entry_the_registry_already_holds() {
        let _guard = TempRegistry::new("adopt-existing");
        let mut rich = entry("aaa", "/a");
        rich.pid = Some(4242);
        rich.name = Some("named-by-the-sandbox".to_string());
        replace_sandbox_slice("sbx-a", vec![rich]).unwrap();

        adopt_host_visible([("sbx-a".to_string(), entry("aaa", "/somewhere-else"))]).unwrap();

        let slice = load().sandboxes.remove("sbx-a").unwrap();
        assert_eq!(slice.len(), 1, "no duplicate row");
        assert_eq!(slice[0].pid, Some(4242), "pid preserved");
        assert_eq!(slice[0].name.as_deref(), Some("named-by-the-sandbox"));
    }

    #[test]
    fn adopt_never_removes_anything() {
        let _guard = TempRegistry::new("adopt-additive");
        replace_sandbox_slice("sbx-a", vec![entry("aaa", "/a")]).unwrap();
        replace_sandbox_slice("sbx-b", vec![entry("bbb", "/b")]).unwrap();
        adopt_host_visible(std::iter::empty()).unwrap();
        let registry = load();
        assert_eq!(ids(registry.sandboxes.get("sbx-a").unwrap()), ["aaa"]);
        assert_eq!(ids(registry.sandboxes.get("sbx-b").unwrap()), ["bbb"]);
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
    fn regression_an_unreadable_registry_never_deletes_another_sandbox() {
        // 2026-08-17, 18:22:26Z. Two sandboxes, 42 sessions. One sandbox's hook
        // saved its own 1-session slice; the read that write-back was built on
        // came up empty, and `replace_sandbox_slice` serialised the result as
        // the whole file. The sibling's key — 41 live sessions, every one of
        // them still writing its transcript — was gone from disk, and every row
        // disappeared from the dashboard at once.
        //
        // The read failing is not the bug and cannot be prevented here: the
        // file is rewritten in place by the host collector while in-sandbox
        // hooks read it over virtiofs. Publishing a write built on that read is
        // the bug.
        let _fixture = TempRegistry::new("unreadable-registry-write-back");
        let path = sandbox_registry_path();

        replace_sandbox_slice("sandbox-a", vec![entry("aaa", "/work/a")]).unwrap();
        replace_sandbox_slice("sandbox-b", vec![entry("bbb", "/work/b")]).unwrap();
        assert_eq!(load().sandboxes.len(), 2, "two sandboxes recorded");

        // A torn read: the trailing half of the JSON never made it to the page
        // cache, so it parses as nothing.
        let intact = fs::read(&path).unwrap();
        fs::write(&path, &intact[..intact.len() / 2]).unwrap();
        let truncated = fs::read(&path).unwrap();

        let result = replace_sandbox_slice("sandbox-a", vec![entry("aaa", "/work/a")]);
        let err = result.expect_err("a registry we cannot parse is not an empty registry");
        assert!(
            err.to_string().contains("unreadable"),
            "the error must say the file could not be read, got: {err}"
        );
        assert_eq!(
            fs::read(&path).unwrap(),
            truncated,
            "the write must be abandoned, not applied on top of a base we never read"
        );

        // And once the file is readable again, the other sandbox is still there
        // — the entries were never the thing at risk, the write-back was.
        fs::write(&path, &intact).unwrap();
        replace_sandbox_slice("sandbox-a", vec![entry("aaa", "/work/a2")]).unwrap();
        let registry = load();
        assert_eq!(registry.sandboxes.len(), 2, "sandbox-b survived");
        assert_eq!(registry.sandboxes["sandbox-b"][0].session_id, "bbb");
    }

    #[test]
    fn a_registry_that_has_never_been_written_is_legitimately_empty() {
        // The other half: absence is a real answer. Treating a first run as a
        // failure would mean no sandbox could ever record its first session.
        let _fixture = TempRegistry::new("registry-first-run");
        assert!(!sandbox_registry_path().exists());
        assert!(
            try_load()
                .expect("a missing registry is empty, not unreadable")
                .sandboxes
                .is_empty()
        );
        replace_sandbox_slice("sandbox-a", vec![entry("aaa", "/work/a")]).unwrap();
        assert_eq!(load().sandboxes["sandbox-a"].len(), 1);
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
        // Everything restore reads is intact. The entry additionally carries a
        // departure stamp, which is what lets the dashboard stop rendering it
        // immediately without deleting the restore material — the two used to
        // be indistinguishable, so a dead row lingered until an unrelated hook
        // forced a reconcile.
        let stored = load_local().sessions;
        assert_eq!(stored.len(), 1);
        assert_eq!(
            (
                stored[0].session_id.as_str(),
                stored[0].cwd.as_str(),
                stored[0].transcript.as_str(),
                stored[0].started_at_ms
            ),
            ("aaa", "/work/a", "/tmp/aaa.jsonl", 42),
            "restore material must survive SessionEnd verbatim"
        );
        assert!(stored[0].departed_at_ms.is_some(), "and be marked departed");

        // Nor may any other hook event forget it. Hooks fire from one session
        // while another terminal may be mid-quit; a hook that pruned on a single
        // look could delete a quitting terminal's whole restore set.
        let stop = serde_json::json!({"hook_event_name": "Stop", "session_id": "aaa"});
        crate::hook_state::record_hook_event(&stop).unwrap();
        let after_stop = load_local().sessions;
        assert_eq!(after_stop.len(), 1);
        assert_eq!(after_stop[0].session_id, "aaa");
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
    fn snapshot_round_trips_through_disk_with_current_flag_intact() {
        let _reg = TempRegistry::new("snapshot");
        let mut snapshot = SandboxSnapshot {
            collected_at_ms: 1_784_953_050_628,
            ..Default::default()
        };
        snapshot.sandboxes.insert(
            "linera-agent-a3f11b28c4d0".to_string(),
            SandboxOrigin {
                is_current: true,
                sessions: vec![serde_json::json!({"session_id": "s1"})],
            },
        );
        snapshot.sandboxes.insert(
            "linera-agent-251d6f7c9065".to_string(),
            SandboxOrigin::default(),
        );
        write_snapshot(&snapshot).unwrap();

        let bytes = fs::read(sandbox_snapshot_path()).unwrap();
        let back: SandboxSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.version, current_version());
        assert_eq!(back.collected_at_ms, 1_784_953_050_628);
        // Exactly one origin is current; a superseded one must not inherit it.
        assert!(back.sandboxes["linera-agent-a3f11b28c4d0"].is_current);
        assert!(!back.sandboxes["linera-agent-251d6f7c9065"].is_current);
        assert_eq!(
            back.sandboxes["linera-agent-a3f11b28c4d0"].sessions.len(),
            1
        );
        assert!(
            back.sandboxes["linera-agent-251d6f7c9065"]
                .sessions
                .is_empty()
        );
    }

    #[test]
    fn current_sandbox_gated_on_marker() {
        let _lock = env_guard();
        let saved_vm_id = std::env::var_os(ENV_SANDBOX_VM_ID);
        // SAFETY: env access is serialized by the held `ENV_LOCK`.
        unsafe {
            // Host scope now needs BOTH markers absent. Inside a sandbox the
            // filesystem one is real and unremovable, so point it at a path
            // that doesn't exist — otherwise this test asserts host behaviour
            // while standing in a sandbox and fails there but not on CI.
            std::env::set_var("CLAUDECTL_SANDBOX_MARKER_DIR", "/nonexistent/no-marker");
            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
            // The real `SANDBOX_VM_ID` leaks in from the surrounding sandbox
            // and would answer the name lookup below with a live sandbox name.
            std::env::remove_var(ENV_SANDBOX_VM_ID);
            assert_eq!(current_sandbox(), None);

            // Presence, not the exact name: with no name env left, the name now
            // comes from the machine's UTS hostname, which differs between a
            // sandbox and a CI runner. Precedence is pinned by the pure
            // `registry_sandbox_name` tests instead, where it doesn't depend on
            // where the suite happens to run.
            std::env::set_var(ENV_SANDBOX_MARKER, "1");
            assert!(current_sandbox().is_some());

            std::env::set_var(ENV_SANDBOX_NAME, "pm-task");
            assert_eq!(current_sandbox(), Some("pm-task".to_string()));

            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
            std::env::remove_var("CLAUDECTL_SANDBOX_MARKER_DIR");
            if let Some(value) = saved_vm_id {
                std::env::set_var(ENV_SANDBOX_VM_ID, value);
            }
        }
    }

    /// The regression this whole change exists for: an exec that did not
    /// inherit `LINERA_SANDBOX` concluded it was the laptop and wrote sandbox
    /// sessions into the host's registry — the one thing the marker's doc
    /// promises cannot happen.
    #[test]
    fn a_sandbox_that_lost_the_env_marker_is_still_a_sandbox() {
        let _lock = env_guard();
        let saved_vm_id = std::env::var_os(ENV_SANDBOX_VM_ID);
        let marker = std::env::temp_dir().join("claudectl-marker-present");
        fs::create_dir_all(&marker).unwrap();
        // SAFETY: env access is serialized by the held `ENV_LOCK`.
        unsafe {
            std::env::set_var("CLAUDECTL_SANDBOX_MARKER_DIR", &marker);
            std::env::remove_var(ENV_SANDBOX_MARKER);
            std::env::remove_var(ENV_SANDBOX_NAME);
            std::env::set_var(ENV_SANDBOX_VM_ID, "linera-agent-cd708d9d80bc");

            assert_eq!(
                current_sandbox(),
                Some("linera-agent-cd708d9d80bc".to_string()),
                "the filesystem marker must survive an exec that dropped the env one"
            );

            std::env::remove_var("CLAUDECTL_SANDBOX_MARKER_DIR");
            std::env::remove_var(ENV_SANDBOX_VM_ID);
            if let Some(value) = saved_vm_id {
                std::env::set_var(ENV_SANDBOX_VM_ID, value);
            }
        }
        let _ = fs::remove_dir_all(&marker);
    }
}

#[cfg(test)]
mod identity_backfill_tests {
    use super::*;

    fn entry(session_id: &str, cwd: &str, name: Option<&str>) -> SessionEntry {
        SessionEntry {
            session_id: session_id.to_string(),
            cwd: cwd.to_string(),
            transcript: if cwd.is_empty() {
                format!("/home/u/.claude/projects/-/{session_id}.jsonl")
            } else {
                format!("/home/u/.claude/projects/-Users-ndr-work/{session_id}.jsonl")
            },
            name: name.map(str::to_string),
            pid: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn regression_a_blank_rediscovery_never_overwrites_stored_identity() {
        // The write that broke 18 sessions at once. Claude Code deleted their
        // pointer files, discovery fell back to the process table, and the
        // resulting entries carried a pid and a uuid and nothing else. The
        // wholesale slice replace then persisted those blanks over good data,
        // so every row lost its title and pointed at `projects/-/<id>.jsonl`.
        //
        // `name` alone was guarded here before 2026-08-06; `cwd` and the
        // `transcript` derived from it were not.
        let stored = vec![entry("abc", "/Users/ndr/work", Some("my-title"))];
        let mut rediscovered = vec![entry("abc", "", None)];

        backfill_missing_identity(&stored, &mut rediscovered);

        assert_eq!(rediscovered[0].name.as_deref(), Some("my-title"));
        assert_eq!(rediscovered[0].cwd, "/Users/ndr/work");
        assert_eq!(
            rediscovered[0].transcript, stored[0].transcript,
            "transcript is derived from cwd, so restoring one without the \
             other leaves the row pointing at a file that is not there"
        );
    }

    #[test]
    fn a_fresher_value_still_wins() {
        // The guard must not freeze identity: a session that genuinely moved,
        // or was renamed, has to be able to say so. Only *blanks* are refused.
        let stored = vec![entry("abc", "/Users/ndr/old", Some("old-title"))];
        let mut fresh = vec![entry("abc", "/Users/ndr/work", Some("new-title"))];

        backfill_missing_identity(&stored, &mut fresh);

        assert_eq!(fresh[0].name.as_deref(), Some("new-title"));
        assert_eq!(fresh[0].cwd, "/Users/ndr/work");
    }

    #[test]
    fn an_unknown_session_is_left_alone() {
        let stored = vec![entry("abc", "/Users/ndr/work", Some("t"))];
        let mut incoming = vec![entry("zzz", "", None)];
        backfill_missing_identity(&stored, &mut incoming);
        assert_eq!(incoming[0].cwd, "", "nothing stored to recover from");
        assert!(incoming[0].name.is_none());
    }
}
