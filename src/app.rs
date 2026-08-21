use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::widgets::TableState;

use crate::discovery;
use crate::helpers::{
    create_aggregate_session, dirs_home, fire_notification, fire_webhook, kill_process,
};
use crate::hooks::{HookEvent, HookRegistry};
use crate::launch::{self, LaunchRequest};
use crate::monitor;
use crate::process;
use crate::session::{ClaudeSession, SessionStatus};
use crate::terminals;
use crate::theme::Theme;

/// Output of `do_refresh_io`. Captures everything the I/O-heavy pass
/// needs to hand back to the main thread for status / conflict / budget
/// post-processing. All fields are owned values so the struct is `Send`
/// and can be returned across `tokio::task::spawn_blocking`.
pub struct RefreshIoOutput {
    pub sessions: Vec<ClaudeSession>,
    pub new_pids: Vec<u32>,
    pub scan_elapsed: std::time::Duration,
    pub ps_elapsed: std::time::Duration,
    pub jsonl_elapsed: std::time::Duration,
}

/// Run the I/O-heavy half of an App refresh and return the enriched
/// session list plus timing data. This is a pure function over
/// `prev_sessions`: no borrows of `App`, no shared state, all inputs and
/// outputs are owned values. Designed to run on
/// `tokio::task::spawn_blocking` once we wire up async refresh —
/// extracting it as a free fn first keeps the diff small while letting
/// us audit "what's actually slow" in isolation.
///
/// Side effects: shells out via `discovery`, `process`, `monitor` —
/// each does its own `std::fs` / `std::process::Command` calls. Total
/// cost on a typical box (~38 sessions): ~30 ms steady-state.
/// Whether this session still needs its transcript located.
///
/// Not just "has no path": a registry entry can carry a path that does not
/// exist. `transcript` is stored as `projects/<cwd-slug>/<id>.jsonl`, so an
/// entry whose cwd was blanked carries `projects/-/<id>.jsonl` — present,
/// wrong, and pointing at nothing. Treating that as resolved is what left
/// those rows reading `Unreadable` forever: the recovery path was never
/// reached because a value was already there.
fn needs_transcript_resolution(session: &ClaudeSession) -> bool {
    match session.jsonl_path.as_ref() {
        None => true,
        Some(path) => !path.exists(),
    }
}

pub fn do_refresh_io(prev_sessions: Vec<ClaudeSession>) -> RefreshIoOutput {
    let scan_start = std::time::Instant::now();
    let discovered = discovery::scan_sessions();
    let scan_elapsed = scan_start.elapsed();

    // Captured before the merge consumes `prev_sessions`. Native rows carry
    // their own previous sample through the merge; foreign rows are rebuilt
    // from scratch every tick and would otherwise never have a pair to
    // difference, leaving every sandbox session's CPU permanently unknown.
    let prev_cpu_samples: std::collections::HashMap<String, crate::cpu::CpuSample> = prev_sessions
        .iter()
        .filter_map(|session| Some((session.session_id.clone(), session.cpu_sample?)))
        .collect();

    let (mut sessions, new_pids) = merge_discovered_sessions(prev_sessions, discovered);

    let ps_start = std::time::Instant::now();
    process::fetch_and_enrich(&mut sessions);
    let ps_elapsed = ps_start.elapsed();

    for session in &mut sessions {
        if needs_transcript_resolution(session) {
            discovery::resolve_jsonl_paths(std::slice::from_mut(session));
        }
    }
    discovery::scan_subagents(&mut sessions);
    discovery::resolve_worktree_ids(&mut sessions);

    // Snapshot previous cost for burn rate BEFORE reading new JSONL data.
    for session in &mut sessions {
        session.prev_cost_usd = session.cost_usd;
    }

    let jsonl_start = std::time::Instant::now();
    for session in &mut sessions {
        monitor::update_tokens(session);
    }
    let jsonl_elapsed = jsonl_start.elapsed();

    // Foreign sessions join after the local `ps` pass, never before:
    // `fetch_and_enrich` addresses sessions by pid in *our* namespace and would
    // decorate a foreign row with whatever unrelated local process holds that
    // number.
    //
    // But they DO go through the transcript monitor, on the same code path as
    // local rows. Their transcripts live on the host-shared `~/.claude` mount,
    // so everything transcript-derived — tokens, cost, context, last message,
    // status — is ours to compute, live, this tick. Taking those values from
    // the collecting sandbox instead was the mistake behind three separate
    // "column renders blank" bugs: it made a second implementation of data we
    // already had, and pinned it to whatever claudectl version happens to be
    // baked into that sandbox's image. Only what is genuinely per-VM (pid,
    // cpu, mem) still comes from the snapshot.
    let mut foreign = foreign_sessions(&sessions, &prev_cpu_samples);
    for session in &mut foreign {
        if needs_transcript_resolution(session) {
            discovery::resolve_jsonl_paths(std::slice::from_mut(session));
        }
        monitor::update_tokens(session);
    }
    sessions.extend(foreign);

    RefreshIoOutput {
        sessions,
        new_pids,
        scan_elapsed,
        ps_elapsed,
        jsonl_elapsed,
    }
}

/// Sessions from every origin that is not this one.
///
/// Membership comes from the hook-written registry, not the collected
/// snapshot. Both files sit on the host-shared mount, but they move at wildly
/// different speeds: hooks write the registry as sessions start and stop,
/// while the snapshot is rewritten only by the reaper's timer — 300 s apart by
/// default, and measured at 311 s on a live host. Reading membership from the
/// snapshot meant a session took up to five minutes to appear after it started
/// and up to five more to disappear after it was closed, no matter how fast
/// the TUI refreshed. Refreshing faster than the writer changes nothing, so
/// the fix is to read the file that is already current.
///
/// The snapshot keeps exactly the job it is fast enough for: overlaying `cpu`
/// and `mem_mb`, the only two facts about another VM's process that the host
/// cannot compute for itself. Everything else on the row is either identity
/// (the registry carries it) or transcript-derived (recomputed live from the
/// shared `~/.claude` mount by the caller, immediately after this returns).
///
/// Excludes our own sandbox's slice — those were discovered natively above —
/// and de-dupes on `session_id`, which now matters for more than mislabelling:
/// one session id legitimately appears under two sandbox keys once a session
/// has been resumed in a newer sandbox, and only one row should render.
fn foreign_sessions(
    local: &[ClaudeSession],
    prev_cpu_samples: &std::collections::HashMap<String, crate::cpu::CpuSample>,
) -> Vec<ClaudeSession> {
    let here = crate::sandbox_registry::current_sandbox();
    // Before reading the registry, fold in anything the host can see that the
    // registry lost. A slice is only rewritten when a hook fires inside its
    // sandbox, so an idle session's row never comes back on its own.
    crate::reaper::adopt_host_visible_sessions();
    let registry = crate::sandbox_registry::load();
    let snapshot = crate::sandbox_registry::load_snapshot();
    let running = running_sandbox_filter(&snapshot);
    // One `ps` sweep for every sandbox we are about to consider, taken before
    // the selection loop so the loop itself stays free of process calls.
    let names: Vec<String> = registry.sandboxes.keys().cloned().collect();
    let open_ttys = crate::reaper::open_host_ttys_by_sandbox(&names);
    foreign_sessions_from(
        &registry,
        &snapshot,
        &running,
        here.as_deref(),
        local,
        now_ms(),
        collector_interval(),
        open_ttys.as_ref(),
        prev_cpu_samples,
    )
}

/// The pure half of [`foreign_sessions`], with every input passed in.
///
/// Split out so the selection rules can be tested against a constructed
/// registry: the wrapper's reads all resolve through process-global state
/// (`HOME`, the `CLAUDECTL_*` overrides, the wall clock) and one of them shells
/// out to `sbx`, none of which a test of "which rows render" should have to
/// arrange.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is one ambient input this function exists to exclude"
)]
pub(crate) fn foreign_sessions_from(
    registry: &crate::sandbox_registry::Registry,
    snapshot: &crate::sandbox_registry::SandboxSnapshot,
    running: &RunningFilter,
    here: Option<&str>,
    local: &[ClaudeSession],
    now_ms: u64,
    collector_interval: std::time::Duration,
    terminals: Option<&crate::reaper::OpenTerminals>,
    prev_cpu_samples: &std::collections::HashMap<String, crate::cpu::CpuSample>,
) -> Vec<ClaudeSession> {
    let vitals = snapshot_vitals_at(snapshot, now_ms, collector_interval);

    let mut seen: std::collections::HashSet<String> = local
        .iter()
        .map(|session| session.session_id.clone())
        .collect();

    let mut out = Vec::new();
    for (name, entries) in &registry.sandboxes {
        if here == Some(name.as_str()) {
            continue;
        }
        // A slice whose sandbox is gone is not stale data to be cleaned up —
        // it is the payload `--restore-sbx-sessions` replays after `sbx rm`,
        // frozen deliberately at its last live state because a removed sandbox
        // fires no further hooks. It must survive on disk and simply not
        // render, so the filter belongs here and never in the registry writer.
        if !running.allows(name) {
            continue;
        }
        let open_ttys = terminals.and_then(|sweep| sweep.by_sandbox.get(name.as_str()));
        let collector_saw = collector_live_ids(snapshot, name, now_ms, collector_interval);
        for entry in entries {
            // A stamped entry is a session whose `SessionEnd` has fired. It
            // stays on disk as `--restore-sbx-sessions` material and stops
            // rendering here, which is the whole point of stamping rather than
            // deleting: before this, a closed terminal left its row on screen
            // until an unrelated session in the same sandbox happened to fire
            // a hook and trigger the wholesale reconcile.
            if entry.departed_at_ms.is_some() {
                continue;
            }
            if collector_says_gone(entry, collector_saw.as_ref(), snapshot.collected_at_ms) {
                continue;
            }
            if terminal_is_gone(
                entry,
                open_ttys,
                terminals.is_some_and(|sweep| sweep.wrapper_argv_parsed),
            ) {
                continue;
            }
            let Some(mut session) = ClaudeSession::from_registry_entry(name, entry) else {
                continue;
            };
            if !seen.insert(session.session_id.clone()) {
                continue;
            }
            if let Some(vitals) = vitals.get(session.session_id.as_str()) {
                // Foreign rows are rebuilt from the registry every tick — they
                // never pass through `merge_discovered_sessions`, which is what
                // carries a native session's previous sample forward. So the
                // previous sample is threaded in explicitly, keyed by session
                // id (pids are not unique across sandbox namespaces).
                session.cpu_rate_percent = prev_cpu_samples
                    .get(session.session_id.as_str())
                    .copied()
                    .zip(vitals.cpu_sample)
                    .and_then(|(prev, cur)| crate::cpu::cpu_rate_percent(prev, cur));
                session.cpu_sample = vitals.cpu_sample;
                session.mem_mb = vitals.mem_mb;
                session.has_child_process = vitals.has_child;
                session.child_observed_at_ms = vitals.observed_at_ms;
            }
            out.push(session);
        }
    }
    out
}

/// Session ids the collector actually observed alive in `sandbox`, or `None`
/// when its snapshot is too old to be evidence of anything.
///
/// This is the only liveness signal that requires **nothing** to be running
/// inside the sandbox. The other two both do, which is why both shipped and
/// changed nothing on a real machine:
///
/// - `departed_at_ms` is stamped by `claudectl-hook`, baked into the sandbox
///   image. Measured on a live host: **0 of 46** registry entries stamped,
///   because the installed hook predated the field.
/// - `host_tty` is recorded by `record_live_sessions`, also in-sandbox. Same
///   host: **21 of 46** entries had it, so the terminal check silently failed
///   open for more than half of them — including the session being reported.
///
/// The collector, by contrast, runs on the host and probes each sandbox with a
/// real in-VM `ps`, then writes what it saw to `sandboxes.json`. Whatever the
/// image contains, that file is the host's own record of who was alive. It is
/// only as fresh as the collector interval, so this bounds the staleness of a
/// dead row instead of leaving it unbounded — it does not replace the stamp,
/// which is what makes removal immediate.
fn collector_live_ids<'a>(
    snapshot: &'a crate::sandbox_registry::SandboxSnapshot,
    sandbox: &str,
    now_ms: u64,
    collector_interval: std::time::Duration,
) -> Option<std::collections::HashSet<&'a str>> {
    if !snapshot.is_fresh(now_ms, collector_interval) {
        return None;
    }
    let origin = snapshot.sandboxes.get(sandbox)?;
    Some(
        origin
            .sessions
            .iter()
            .filter_map(|value| value.get("session_id").and_then(serde_json::Value::as_str))
            .collect(),
    )
}

/// Whether the collector positively observed this sandbox and did **not** find
/// this session alive.
///
/// Fails open at every step, and the `started_at_ms` guard is the load-bearing
/// one: a session that started *after* the last collection is missing from the
/// snapshot because the collector never had a chance to see it, not because it
/// is dead. Without that check this would hide every newly started session for
/// up to a full interval — reintroducing the appearance lag that #36 fixed, in
/// exchange for fixing the removal lag. Both halves have to hold at once.
fn collector_says_gone(
    entry: &crate::sandbox_registry::SessionEntry,
    collector_saw: Option<&std::collections::HashSet<&str>>,
    collected_at_ms: u64,
) -> bool {
    let Some(saw) = collector_saw else {
        return false;
    };
    // The collector only reports sessions it could probe by pid; an entry
    // without one could never appear, so its absence proves nothing.
    if entry.pid.is_none() {
        return false;
    }
    // Unknown or post-collection start ⇒ the collector cannot have seen it.
    if entry.started_at_ms == 0 || entry.started_at_ms >= collected_at_ms {
        return false;
    }
    !saw.contains(entry.session_id.as_str())
}

/// Whether this session's terminal is provably gone from the host.
///
/// The second, independent liveness signal, and the only one that works
/// without deploying anything into the sandbox. `SessionEnd` stamping
/// (`departed_at_ms`) is written by `claudectl-hook` *inside* the VM, so it
/// only takes effect once the sandbox image ships a new binary — measured on a
/// live host, every one of 47 registry entries across 6 sandboxes was unstamped
/// because the installed hook predated the field. Meanwhile every `sc` session
/// is launched by an `sbx exec` carrying `SANDBOX_HOST_TTY=` in its argv, which
/// vanishes from the host process table the instant the window closes.
///
/// **Fails open at every step.** Each of these returns "still alive":
///
/// - `ps` could not be run or failed ⇒ `open` is `None`.
/// - The entry predates `host_tty`, or the session was not launched by the
///   wrapper ⇒ nothing to match on.
/// - The sweep parsed no `sbx exec … SANDBOX_HOST_TTY=` line *anywhere* on the
///   host ⇒ "nothing attached" is indistinguishable from an argv format we no
///   longer parse, and hiding rows on an uncorroborated signal is worse than
///   the lag this removes.
///
/// It deliberately does **not** fail open merely because *this* sandbox's set is
/// empty, which is what the first version did. Closing the window of a sandbox
/// holding one session is precisely what empties its set, so that guard turned
/// the signal off for the common shape — one session per ephemeral sandbox — and
/// left removal to the 300 s collector. Measured on 2026-08-10: a row outlived
/// its session by ~96 s. `wrapper_argv_parsed` corroborates the parse instead,
/// which is the thing the emptiness check was really standing in for.
fn terminal_is_gone(
    entry: &crate::sandbox_registry::SessionEntry,
    open: Option<&std::collections::HashSet<String>>,
    wrapper_argv_parsed: bool,
) -> bool {
    let Some(open) = open else {
        return false;
    };
    if open.is_empty() && !wrapper_argv_parsed {
        return false;
    }
    let Some(tty) = entry.host_tty.as_deref().filter(|tty| !tty.is_empty()) else {
        return false;
    };
    !open.contains(tty)
}

/// Which sandboxes may render, and on what authority.
pub(crate) enum RunningFilter {
    /// `sbx` answered: exactly these names are running. An empty set is a real
    /// answer meaning "none", not a failure.
    Known(std::collections::HashSet<String>),
    /// `sbx` could not be asked — no binary on PATH, which is the ordinary
    /// case for an in-sandbox claudectl, where the host bridges deliberately
    /// keep `sbx` unreachable. Fall back to the collector's last observed
    /// running set: staler, but it is the same question answered earlier.
    Collected(std::collections::HashSet<String>),
}

impl RunningFilter {
    fn allows(&self, name: &str) -> bool {
        match self {
            Self::Known(names) | Self::Collected(names) => names.contains(name),
        }
    }
}

fn running_sandbox_filter(snapshot: &crate::sandbox_registry::SandboxSnapshot) -> RunningFilter {
    match crate::reaper::running_sandboxes() {
        Some(names) => RunningFilter::Known(names.into_iter().collect()),
        None => RunningFilter::Collected(snapshot.sandboxes.keys().cloned().collect()),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The reaper period this host is actually running, read from the installed
/// scheduler unit rather than assumed.
///
/// The default is only a default: `--install-reaper --reaper-interval` accepts
/// anything from 10 s to an hour, and a host running a 30-minute reaper would
/// otherwise have every snapshot judged expired the moment it aged past ten
/// minutes. Falling back to the CLI default when the unit cannot be read keeps
/// this honest on a host where the reaper was never installed — there, no
/// snapshot is being written at all, and expiring it is the correct outcome.
fn collector_interval() -> std::time::Duration {
    std::time::Duration::from_secs(
        crate::reaper::installed_interval_seconds()
            .unwrap_or(crate::reaper::DEFAULT_INTERVAL_SECONDS),
    )
}

/// The per-VM measurements the host cannot take itself.
pub(crate) struct SnapshotVitals {
    /// Cumulative CPU seconds as of the snapshot's `collected_at_ms`, not a
    /// percentage: the host differences consecutive snapshots into a rate, the
    /// same way it does for native pids. `None` when the collector predates
    /// this field, which leaves the rate unknown rather than inventing one.
    pub cpu_sample: Option<crate::cpu::CpuSample>,
    pub mem_mb: f64,
    /// Whether the collector saw this claude process parenting anything.
    /// `None` when it did not report — an older probe, or a sandbox it could
    /// not place — and absence must never be read as "no children".
    pub has_child: Option<bool>,
    /// When the observation above was taken, so a reader can require it to be
    /// *newer* than the tool call it is being used to judge.
    pub observed_at_ms: u64,
}

/// Index the snapshot's collected rows by session id so a registry-built row
/// can pick up its CPU and memory. Rows the collector has not seen yet simply
/// find nothing and keep `from_raw`'s defaults — an unmeasured session renders
/// with blank vitals rather than being withheld from the list entirely.
///
/// Returns nothing at all once the snapshot has expired. CPU and memory are
/// instantaneous samples; minutes-old ones are not "slightly stale", they are
/// measurements of a moment that has passed, and a row showing 7.5% for a
/// process that has since gone quiet is a confident lie. Blank cells say "we
/// don't know", which is the truth. Session membership is unaffected — it
/// comes from the registry and stays live regardless.
///
/// Clock and collector period are passed in rather than read from the process,
/// so the expiry rule can be tested without sleeping or installing a scheduler
/// unit; [`foreign_sessions`] supplies the ambient values.
pub(crate) fn snapshot_vitals_at(
    snapshot: &crate::sandbox_registry::SandboxSnapshot,
    now_ms: u64,
    collector_interval: std::time::Duration,
) -> std::collections::HashMap<String, SnapshotVitals> {
    let mut out = std::collections::HashMap::new();
    if !snapshot.is_fresh(now_ms, collector_interval) {
        return out;
    }
    for origin in snapshot.sandboxes.values() {
        for value in &origin.sessions {
            let Some(id) = value.get("session_id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            out.insert(
                id.to_string(),
                SnapshotVitals {
                    cpu_sample: value
                        .get("cputime_secs")
                        .and_then(serde_json::Value::as_f64)
                        .map(|cputime_secs| crate::cpu::CpuSample {
                            cputime_secs,
                            sampled_at_ms: snapshot.collected_at_ms,
                        }),
                    mem_mb: value
                        .get("mem_mb")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                    has_child: value.get("has_child").and_then(serde_json::Value::as_bool),
                    observed_at_ms: snapshot.collected_at_ms,
                },
            );
        }
    }
    out
}

/// Refresh-driven application state. Owned exclusively by `App.data` and
/// swapped atomically each refresh cycle. Render and key-event paths read
/// snapshots via `App::data_snapshot()`; mutations (refresh) go through
/// `App::replace_data()` which builds a fresh `AppData` and atomically
/// replaces the shared `Arc`.
///
/// Splitting these fields off from `App` is the foundation for moving
/// refresh I/O onto a background tokio task: the task owns `AppData`
/// mutations, the render thread reads snapshots concurrently, and no
/// torn reads are possible because the swap is at the whole-state level.
#[derive(Default, Clone)]
pub struct AppData {
    pub sessions: Vec<ClaudeSession>,
    pub ledger_today: crate::usage_ledger::UsageSummary,
    pub ledger_week: crate::usage_ledger::UsageSummary,
    pub ledger_month: crate::usage_ledger::UsageSummary,
}

pub const SORT_COLUMNS: &[&str] = &[
    "Status", "Context", "Cost", "$/hr", "Elapsed", "Last", "Name",
];

/// Default path for the persisted park list.
pub fn parked_path() -> std::path::PathBuf {
    dirs_home().join(".claudectl").join("parked.json")
}

/// Load the parked-session set from `path`. Returns an empty set on any
/// failure (missing file, malformed JSON, I/O error) — parking is best-effort
/// convenience, not critical state.
pub fn load_parked_from(path: &std::path::Path) -> HashSet<String> {
    let Ok(bytes) = std::fs::read(path) else {
        return HashSet::new();
    };
    // Accept either {"parked": [...]} or a bare ["..."] array.
    if let Ok(obj) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        if let Some(arr) = obj.get("parked").and_then(|v| v.as_array()) {
            return arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
        if let Some(arr) = obj.as_array() {
            return arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
    }
    HashSet::new()
}

/// Persist the parked-session set to `path`. Best-effort: creates parent
/// directories if needed and writes atomically (temp + rename). Failures
/// are silent — the in-memory set still works for the current session.
pub fn save_parked_to(path: &std::path::Path, parked: &HashSet<String>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut ids: Vec<&String> = parked.iter().collect();
    ids.sort(); // deterministic file output
    let payload = serde_json::json!({ "parked": ids });
    let tmp_path = path.with_extension("json.tmp");
    if std::fs::write(&tmp_path, payload.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp_path, path);
    }
}

/// Merge the latest `discovered` sessions with the previous set, preserving
/// accumulated per-session state (jsonl_offset, tokens, cost, cpu_history,
/// …) across ticks. Ephemeral fields (`elapsed`, `started_at`) and the
/// user-visible `session_name` (rewritten by Claude Code's `/rename`) are
/// refreshed from the discovered copy.
///
/// Returns the merged list and the PIDs that are brand new this tick.
pub fn merge_discovered_sessions(
    existing: Vec<ClaudeSession>,
    discovered: Vec<ClaudeSession>,
) -> (Vec<ClaudeSession>, Vec<u32>) {
    let mut existing: HashMap<u32, ClaudeSession> =
        existing.into_iter().map(|s| (s.pid, s)).collect();
    let mut new_pids = Vec::new();
    let merged = discovered
        .into_iter()
        .map(|new| {
            if let Some(mut prev) = existing.remove(&new.pid) {
                prev.elapsed = new.elapsed;
                prev.started_at = new.started_at;
                // A row born before its discovery source settled (first-tick
                // race) can carry an empty cwd; later scans know it. Backfill —
                // cwd feeds transcript resolution and the terminal-switch tab
                // lookup, so a permanently empty cwd keeps both broken.
                if prev.cwd.is_empty() && !new.cwd.is_empty() {
                    prev.cwd = new.cwd;
                    prev.project_name = new.project_name;
                }
                // Claude Code can rotate `sessionId` under the same OS PID
                // (/clear, compaction, resume-into-new-file). The transcript
                // file path changes but `do_refresh_io` skips re-resolution
                // while jsonl_path.is_some(), so without this the TUI keeps
                // reading the abandoned transcript and "Last" never advances.
                //
                // An EMPTY new id is not a rotation: it is a ps-backstop row
                // whose `--resume` uuid is unknown (discovery lost both the
                // pointer and the registry entry). Adopting it would erase a
                // known identity and reset the JSONL offset, double-counting
                // the transcript on re-read. Keep what we know.
                if !new.session_id.is_empty() && prev.session_id != new.session_id {
                    // A rotation is a NEW conversation: the old transcript's
                    // explicit /rename title must not outlive it, or a
                    // recycled pid / post-/clear row wears a dead session's
                    // title forever. Adopt the discovered name wholesale and
                    // drop the hold — the new transcript re-parses from
                    // offset 0 this same pass, so a carried-over custom-title
                    // record re-establishes the explicit title immediately.
                    prev.session_name = new.session_name;
                    prev.name_is_explicit = new.name_is_explicit;
                    prev.session_id = new.session_id;
                    prev.jsonl_path = None;
                    prev.jsonl_offset = 0;
                } else if !new.session_name.is_empty() && !prev.name_is_explicit {
                    // An empty discovered name carries no information
                    // (ps-backstop rows have none) — never erase a name the
                    // TUI already knows, e.g. one recovered from the
                    // transcript's custom-title.
                    //
                    // An EXPLICIT title (transcript custom-title, via
                    // monitor) outranks every scan source: a registry entry
                    // recorded before the rename re-supplies the stale name
                    // every tick, and letting it win here is what made a
                    // fresh `/rename` revert seconds later. Newer
                    // custom-title records still update the title through
                    // the monitor path.
                    prev.session_name = new.session_name;
                }
                prev
            } else {
                new_pids.push(new.pid);
                new
            }
        })
        .collect();
    (merged, new_pids)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusFilter {
    All,
    NeedsInput,
    Compacting,
    Processing,
    WaitingInput,
    Unknown,
    Idle,
    Finished,
}

impl StatusFilter {
    pub fn next(self) -> Self {
        match self {
            Self::All => Self::NeedsInput,
            Self::NeedsInput => Self::Compacting,
            Self::Compacting => Self::Processing,
            Self::Processing => Self::WaitingInput,
            Self::WaitingInput => Self::Unknown,
            Self::Unknown => Self::Idle,
            Self::Idle => Self::Finished,
            Self::Finished => Self::All,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all" => Some(Self::All),
            "needsinput" | "needs-input" => Some(Self::NeedsInput),
            "compacting" => Some(Self::Compacting),
            "processing" => Some(Self::Processing),
            "waiting" | "waitinginput" | "waiting-input" => Some(Self::WaitingInput),
            "unknown" => Some(Self::Unknown),
            "idle" => Some(Self::Idle),
            "finished" => Some(Self::Finished),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::NeedsInput => "Needs Input",
            Self::Compacting => "Compacting",
            Self::Processing => "Processing",
            Self::WaitingInput => "Waiting",
            Self::Unknown => "Unknown",
            Self::Idle => "Idle",
            Self::Finished => "Finished",
        }
    }

    fn matches(self, status: SessionStatus) -> bool {
        match self {
            Self::All => true,
            Self::NeedsInput => status == SessionStatus::NeedsInput,
            Self::Compacting => status == SessionStatus::Compacting,
            Self::Processing => status == SessionStatus::Processing,
            Self::WaitingInput => status == SessionStatus::WaitingInput,
            Self::Unknown => status == SessionStatus::Unknown,
            Self::Idle => status == SessionStatus::Idle,
            Self::Finished => status == SessionStatus::Finished,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusFilter {
    All,
    Attention,
    OverBudget,
    HighContext,
    UnknownTelemetry,
    Conflict,
}

impl FocusFilter {
    pub fn next(self) -> Self {
        match self {
            Self::All => Self::Attention,
            Self::Attention => Self::OverBudget,
            Self::OverBudget => Self::HighContext,
            Self::HighContext => Self::UnknownTelemetry,
            Self::UnknownTelemetry => Self::Conflict,
            Self::Conflict => Self::All,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all" => Some(Self::All),
            "attention" => Some(Self::Attention),
            "overbudget" | "over-budget" => Some(Self::OverBudget),
            "highcontext" | "high-context" => Some(Self::HighContext),
            "unknowntelemetry" | "unknown-telemetry" => Some(Self::UnknownTelemetry),
            "conflict" | "conflicts" => Some(Self::Conflict),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Attention => "Attention",
            Self::OverBudget => "Over Budget",
            Self::HighContext => "High Context",
            Self::UnknownTelemetry => "Unknown Telemetry",
            Self::Conflict => "Conflict",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchField {
    Cwd,
    Prompt,
    Resume,
}

impl LaunchField {
    fn next(self) -> Self {
        match self {
            Self::Cwd => Self::Prompt,
            Self::Prompt => Self::Resume,
            Self::Resume => Self::Resume,
        }
    }

    fn prev(self) -> Self {
        match self {
            Self::Cwd => Self::Cwd,
            Self::Prompt => Self::Cwd,
            Self::Resume => Self::Prompt,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Cwd => "cwd",
            Self::Prompt => "prompt",
            Self::Resume => "resume",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchForm {
    pub field: LaunchField,
    pub cwd: String,
    pub prompt: String,
    pub resume: String,
}

impl Default for LaunchForm {
    fn default() -> Self {
        Self {
            field: LaunchField::Cwd,
            cwd: ".".into(),
            prompt: String::new(),
            resume: String::new(),
        }
    }
}

impl LaunchForm {
    pub fn active_buffer(&self) -> &str {
        match self.field {
            LaunchField::Cwd => &self.cwd,
            LaunchField::Prompt => &self.prompt,
            LaunchField::Resume => &self.resume,
        }
    }

    fn active_buffer_mut(&mut self) -> &mut String {
        match self.field {
            LaunchField::Cwd => &mut self.cwd,
            LaunchField::Prompt => &mut self.prompt,
            LaunchField::Resume => &mut self.resume,
        }
    }

    fn advance(&mut self) {
        self.field = self.field.next();
    }

    fn retreat(&mut self) {
        self.field = self.field.prev();
    }

    fn is_last_field(&self) -> bool {
        self.field == LaunchField::Resume
    }

    pub fn status_hint(&self) -> String {
        format!(
            "New session [{}] Enter next, Tab move, Ctrl+Enter launch, Esc cancel",
            self.field.label()
        )
    }

    fn request(&self) -> Result<LaunchRequest, String> {
        launch::prepare(
            &self.cwd,
            Some(self.prompt.as_str()),
            Some(self.resume.as_str()),
        )
    }

    pub fn summary(&self) -> String {
        let cwd = compact_value(&self.cwd, ".");
        let prompt = if self.prompt.trim().is_empty() {
            "skip".to_string()
        } else {
            "set".to_string()
        };
        let resume = compact_value(&self.resume, "skip");
        format!("cwd={cwd} | prompt={prompt} | resume={resume}")
    }
}

fn compact_value(value: &str, empty_label: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return empty_label.to_string();
    }

    const MAX_LEN: usize = 24;
    if trimmed.chars().count() <= MAX_LEN {
        trimmed.to_string()
    } else {
        let prefix: String = trimmed.chars().take(MAX_LEN - 1).collect();
        format!("{prefix}…")
    }
}

pub struct App {
    /// Refresh-driven state behind a shared atomic. Reads use
    /// `data_snapshot()` (cheap Arc-clone, lock held for ns); writes
    /// build a fresh `AppData` and atomic-swap with `replace_data()`.
    pub data: Arc<RwLock<Arc<AppData>>>,
    /// Sender end of the channel that `refresh_nonblocking` uses to
    /// ship `RefreshIoOutput` from a `spawn_blocking` worker back to
    /// the main thread. Cloned per-spawn so each worker gets its own
    /// handle; the receiver lives on `App`.
    pub refresh_tx: tokio::sync::mpsc::UnboundedSender<RefreshIoOutput>,
    /// Receiver end of the refresh channel. Drained on each tick via
    /// `try_recv` (non-blocking) so the main thread never waits for
    /// a refresh worker to finish.
    pub refresh_rx: tokio::sync::mpsc::UnboundedReceiver<RefreshIoOutput>,
    /// True while a `do_refresh_io` worker is in flight on the tokio
    /// blocking pool. Prevents a slow refresh from kicking off
    /// duplicates each tick — the next tick reuses the existing
    /// worker's eventual result. Cleared once the result is recv'd.
    pub refresh_in_flight: bool,
    pub table_state: TableState,
    pub should_quit: bool,
    pub status_msg: String,
    pub pending_kill: Option<u32>,
    pub input_mode: bool,
    pub input_buffer: String,
    pub input_target_pid: Option<u32>,
    pub notify: bool,
    pub prev_statuses: HashMap<u32, SessionStatus>,
    pub show_help: bool,
    pub sort_column: usize,
    /// When true, the final sort result is reversed relative to the column's
    /// natural direction. Toggled by capital `S`; reset to false whenever
    /// the user cycles to a different column with lowercase `s`.
    pub sort_reversed: bool,
    /// Session IDs the user has parked. Parked sessions stay visible but
    /// sort to a separate section below all non-parked rows, regardless of
    /// the active sort column or direction. Persisted to
    /// `~/.claudectl/parked.json` so parking survives claudectl restarts.
    pub parked: HashSet<String>,
    pub auto_approve: HashSet<u32>,
    pub pending_auto_approve: Option<u32>,
    pub finished_at: HashMap<u32, std::time::Instant>, // When PIDs were first seen as Finished
    pub debug: bool,
    pub debug_timings: DebugTimings,
    pub grouped_view: bool,
    pub detail_panel: bool, // Show expanded detail for selected session
    pub webhook_url: Option<String>,
    pub webhook_filter: Option<Vec<String>>, // Only fire on these status names
    pub launch_mode: bool,                   // Capturing launch wizard fields
    pub launch_form: LaunchForm,
    pub search_mode: bool,
    pub search_buffer: String,
    pub search_query: String,
    pub status_filter: StatusFilter,
    pub focus_filter: FocusFilter,
    pub budget_usd: Option<f64>,     // Per-session budget
    pub kill_on_budget: bool,        // Auto-kill when budget exceeded
    pub budget_warned: HashSet<u32>, // PIDs that have been warned at 80%
    pub budget_killed: HashSet<u32>, // PIDs that have been killed
    pub theme: Theme,
    pub ledger_refresh_tick: u32, // Cheap rollup refresh every N ticks
    pub ledger_scan_tick: u32,    // Expensive scan_and_append every N ticks
    pub hooks: HookRegistry,
    pub daily_limit: Option<f64>,
    pub weekly_limit: Option<f64>,
    pub daily_alert_fired: bool, // Prevent repeated alerts per app session
    pub weekly_alert_fired: bool,
    pub context_warn_threshold: u8, // 0-100, fires on_context_high hook
    pub context_warned: HashSet<u32>, // PIDs that have been warned (reset if context drops below threshold)
    pub needs_input_since: HashMap<u32, std::time::Instant>, // When each PID entered NeedsInput
    pub conflict_pids: HashSet<u32>,  // PIDs that share a working directory with another session
    pub conflict_alerted: HashSet<String>, // cwds that have already triggered a conflict alert
    pub file_conflict_pids: HashSet<u32>, // PIDs involved in file-level conflicts
    pub file_conflicts: HashMap<String, Vec<u32>>, // file path → PIDs that modified it
    pub file_conflict_alerted: HashSet<String>, // Files already alerted
    pub file_conflicts_enabled: bool, // Config: detect file-level conflicts
    pub auto_deny_file_conflicts: bool, // Config: auto-deny conflicting writes
    pub demo_mode: bool,
    pub demo_tick: u32,
    pub session_recordings: HashMap<u32, String>, // pid -> output_path for active recordings
    pub rules: Vec<crate::rules::AutoRule>,
    pub auto_actions_fired: HashMap<u32, std::time::Instant>, // Debounce: pid -> last action time
    pub last_rule_action: Option<String>,                     // Last auto-action status for display
    pub health_thresholds: crate::config::HealthThresholds,
    pub brain_config: Option<crate::config::BrainConfig>,
    pub brain_engine: Option<crate::brain::engine::BrainEngine>,
    pub idle_config: crate::config::IdleConfig,
    pub last_user_interaction: std::time::Instant,
    pub idle_mode_active: bool,
    pub idle_tasks_launched: Vec<String>,
    pub idle_report: Vec<String>,
}

#[derive(Default, Clone)]
pub struct DebugTimings {
    pub scan_ms: f64,
    pub ps_ms: f64,
    pub jsonl_ms: f64,
    pub total_ms: f64,
    // Rolling averages (last 10 ticks)
    history: Vec<(f64, f64, f64, f64)>,
}

impl DebugTimings {
    pub fn record(&mut self, scan: f64, ps: f64, jsonl: f64, total: f64) {
        self.scan_ms = scan;
        self.ps_ms = ps;
        self.jsonl_ms = jsonl;
        self.total_ms = total;
        self.history.push((scan, ps, jsonl, total));
        if self.history.len() > 10 {
            self.history.remove(0);
        }
    }

    pub fn avg_total_ms(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|h| h.3).sum::<f64>() / self.history.len() as f64
    }

    pub fn format(&self) -> String {
        format!(
            "tick: {:.1}ms (avg {:.1}ms) | scan: {:.1}ms | ps: {:.1}ms | jsonl: {:.1}ms",
            self.total_ms,
            self.avg_total_ms(),
            self.scan_ms,
            self.ps_ms,
            self.jsonl_ms,
        )
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Cheap shared-state read: clones the inner Arc (a refcount bump,
    /// not a deep clone) while the lock is held for nanoseconds. Use
    /// for any read that touches refresh-driven fields. Iterate the
    /// returned snapshot for the lifetime of one frame; do not interleave
    /// with `replace_data` and expect consistency across calls.
    pub fn data_snapshot(&self) -> Arc<AppData> {
        Arc::clone(&self.data.read().expect("AppData read lock poisoned"))
    }

    /// Atomic-swap the shared `Arc<AppData>` to a freshly built one.
    /// Holds the write lock only long enough to assign — readers see
    /// either the old state or the new one, never a partial mix.
    pub fn replace_data(&self, new: AppData) {
        *self.data.write().expect("AppData write lock poisoned") = Arc::new(new);
    }

    /// Mutate the shared AppData in place via a closure. Uses
    /// `Arc::make_mut` so callers see a `&mut AppData` directly: if no
    /// other readers hold the Arc the mutation is in-place; otherwise
    /// it triggers one copy. Holds the write lock for the duration of
    /// the closure, so keep the body short — for long refresh work,
    /// snapshot + build a fresh `AppData` locally + `replace_data`.
    pub fn modify_data<F: FnOnce(&mut AppData)>(&self, f: F) {
        let mut guard = self.data.write().expect("AppData write lock poisoned");
        let inner: &mut AppData = Arc::make_mut(&mut *guard);
        f(inner);
    }

    /// Common idiom for sort / reorder paths: snapshot, clone the
    /// AppData, run a mutator on the sessions vec, atomic-swap. Holds
    /// no locks across the closure; safe for the closure body to
    /// reborrow `&self` (e.g. to call `App::apply_sort`).
    pub fn with_sessions<F: FnOnce(&mut Vec<ClaudeSession>)>(&self, f: F) {
        let snap = self.data_snapshot();
        let mut new_data = (*snap).clone();
        f(&mut new_data.sessions);
        self.replace_data(new_data);
    }

    pub fn new() -> Self {
        let (refresh_tx, refresh_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = Self {
            data: Arc::new(RwLock::new(Arc::new(AppData::default()))),
            refresh_tx,
            refresh_rx,
            refresh_in_flight: false,
            table_state: TableState::default(),
            should_quit: false,
            status_msg: String::new(),
            pending_kill: None,
            input_mode: false,
            input_buffer: String::new(),
            input_target_pid: None,
            notify: false,
            prev_statuses: HashMap::new(),
            show_help: false,
            sort_column: 0,
            sort_reversed: false,
            parked: load_parked_from(&parked_path()),
            auto_approve: HashSet::new(),
            pending_auto_approve: None,
            finished_at: HashMap::new(),
            debug: false,
            debug_timings: DebugTimings::default(),
            grouped_view: false,
            detail_panel: false,
            webhook_url: None,
            webhook_filter: None,
            launch_mode: false,
            launch_form: LaunchForm::default(),
            search_mode: false,
            search_buffer: String::new(),
            search_query: String::new(),
            status_filter: StatusFilter::All,
            focus_filter: FocusFilter::All,
            budget_usd: None,
            kill_on_budget: false,
            budget_warned: HashSet::new(),
            budget_killed: HashSet::new(),
            theme: Theme::from_mode(crate::theme::ThemeMode::Dark),
            ledger_refresh_tick: 0,
            ledger_scan_tick: 0,
            hooks: HookRegistry::new(),
            daily_limit: None,
            weekly_limit: None,
            daily_alert_fired: false,
            weekly_alert_fired: false,
            context_warn_threshold: 75,
            context_warned: HashSet::new(),
            needs_input_since: HashMap::new(),
            conflict_pids: HashSet::new(),
            conflict_alerted: HashSet::new(),
            file_conflict_pids: HashSet::new(),
            file_conflicts: HashMap::new(),
            file_conflict_alerted: HashSet::new(),
            file_conflicts_enabled: true,
            auto_deny_file_conflicts: false,
            demo_mode: false,
            demo_tick: 0,
            session_recordings: HashMap::new(),
            rules: Vec::new(),
            auto_actions_fired: HashMap::new(),
            last_rule_action: None,
            health_thresholds: crate::config::HealthThresholds::default(),
            brain_config: None,
            brain_engine: None,
            idle_config: crate::config::IdleConfig::default(),
            last_user_interaction: std::time::Instant::now(),
            idle_mode_active: false,
            idle_tasks_launched: Vec::new(),
            idle_report: Vec::new(),
        };
        app.refresh();
        // Seed rollups from any rows already in the CSV — instant via
        // the in-memory ledger cache. Don't block startup on
        // scan_and_append: a full first-time scan can take 1–3 s on a
        // heavy ~/.claude/projects tree. Kick it off on a background
        // thread instead; the title bar renders with the previously
        // persisted totals immediately, and the background scan
        // updates them within a few seconds.
        app.refresh_ledger_rollups();
        crate::usage_ledger::scan_and_append_background();
        if app.visible_session_count() > 0 {
            app.table_state.select(Some(0));
        }
        app
    }

    pub fn refresh(&mut self) {
        let tick_start = std::time::Instant::now();

        if self.demo_mode {
            self.refresh_demo();
            if self.debug {
                let total_elapsed = tick_start.elapsed();
                self.debug_timings
                    .record(0.0, 0.0, 0.0, total_elapsed.as_secs_f64() * 1000.0);
            }
            return;
        }

        // Run the I/O-heavy phase synchronously, then apply the
        // post-I/O App-state mutations. Splitting these makes the
        // async path (`refresh_nonblocking`) easy: it spawns
        // `do_refresh_io` on a worker and routes the result through a
        // channel, then calls `apply_refresh_output` on the main thread.
        let prev_sessions = self.data_snapshot().sessions.clone();
        let out = do_refresh_io(prev_sessions);
        self.apply_refresh_output(out, tick_start);
    }

    /// Non-blocking refresh: drain any completed worker results, apply
    /// them on the main thread, and kick off a new worker if none is
    /// in flight. The TUI loop calls this every tick instead of
    /// `refresh()` so the I/O latency doesn't block render or key
    /// handling. Falls back to a synchronous refresh when called
    /// outside a tokio runtime (e.g. one-shot CLI commands, tests),
    /// preserving existing behaviour for those entry points.
    ///
    /// Returns `true` when this call applied at least one
    /// `RefreshIoOutput` to App state (i.e. fresh data is now
    /// visible). Tests use the return value to drive a poll loop;
    /// production callers can ignore it.
    pub fn refresh_nonblocking(&mut self) -> bool {
        if self.demo_mode {
            self.refresh();
            return true;
        }

        // Drain ALL completed results, applying each. Keeping a stale
        // result in the channel would hide the latest one; we want the
        // main thread to converge on the freshest snapshot fast.
        let mut applied_any = false;
        while let Ok(out) = self.refresh_rx.try_recv() {
            self.apply_refresh_output(out, std::time::Instant::now());
            self.refresh_in_flight = false;
            applied_any = true;
        }

        // Don't kick off a new worker until the previous one's result
        // has been applied — keeps memory bounded under bursty I/O.
        if self.refresh_in_flight {
            return applied_any;
        }

        // Kick a new refresh on the tokio blocking pool. If we're not
        // in a runtime context (tests, one-shot CLI), fall back to a
        // synchronous refresh on the calling thread.
        let prev_sessions = self.data_snapshot().sessions.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let tx = self.refresh_tx.clone();
            handle.spawn_blocking(move || {
                let out = do_refresh_io(prev_sessions);
                let _ = tx.send(out);
            });
            self.refresh_in_flight = true;
        } else if !applied_any {
            // No runtime AND we didn't already apply a result this tick:
            // run synchronously so the caller still gets fresh data.
            let out = do_refresh_io(prev_sessions);
            self.apply_refresh_output(out, std::time::Instant::now());
            applied_any = true;
        }
        applied_any
    }

    /// Apply an already-computed `RefreshIoOutput` to App state.
    /// All the burn-rate / budget / context / status-transition /
    /// conflict-detection / file-conflict logic runs here on the main
    /// thread. The I/O has already happened; this is pure CPU + App
    /// field mutation.
    fn apply_refresh_output(&mut self, out: RefreshIoOutput, tick_start: std::time::Instant) {
        let RefreshIoOutput {
            mut sessions,
            new_pids,
            scan_elapsed,
            ps_elapsed,
            jsonl_elapsed,
        } = out;

        // Compute burn rate from cost delta (skip first tick where prev_cost is 0)
        for session in &mut sessions {
            if session.prev_cost_usd > 0.001 {
                let delta = session.cost_usd - session.prev_cost_usd;
                if delta > 0.001 {
                    session.burn_rate_per_hr = delta * 1800.0;
                } else {
                    // Decay burn rate toward zero when no new cost
                    session.burn_rate_per_hr *= 0.5;
                    if session.burn_rate_per_hr < 0.01 {
                        session.burn_rate_per_hr = 0.0;
                    }
                }
            }
        }

        // Budget enforcement
        if let Some(budget) = self.budget_usd {
            for session in &sessions {
                let pct = session.cost_usd / budget * 100.0;

                // Warn at 80%
                if (80.0..100.0).contains(&pct) && !self.budget_warned.contains(&session.pid) {
                    self.budget_warned.insert(session.pid);
                    self.status_msg = format!(
                        "BUDGET WARNING: {} at {:.0}% (${:.2}/${:.2})",
                        session.display_name(),
                        pct,
                        session.cost_usd,
                        budget
                    );
                    fire_notification(&format!("{} budget {:.0}%", session.display_name(), pct));
                    self.hooks.fire(HookEvent::BudgetWarning, session);
                }

                // Kill at 100%
                if pct >= 100.0 && !self.budget_killed.contains(&session.pid) {
                    self.budget_killed.insert(session.pid);
                    if self.kill_on_budget {
                        let _ = kill_process(session.pid);
                        self.status_msg = format!(
                            "BUDGET EXCEEDED: Killed {} (${:.2}/${:.2})",
                            session.display_name(),
                            session.cost_usd,
                            budget
                        );
                    } else {
                        self.status_msg = format!(
                            "BUDGET EXCEEDED: {} at ${:.2}/{:.2} — use --kill-on-budget to auto-kill",
                            session.display_name(),
                            session.cost_usd,
                            budget
                        );
                    }
                    fire_notification(&format!("{} exceeded budget!", session.display_name()));
                    self.hooks.fire(HookEvent::BudgetExceeded, session);
                }
            }
        }

        // Context threshold warnings
        if self.context_warn_threshold > 0 {
            let threshold = self.context_warn_threshold as f64;
            for session in &sessions {
                let pct = session.context_percent();
                if pct >= threshold && !self.context_warned.contains(&session.pid) {
                    self.context_warned.insert(session.pid);
                    self.status_msg = format!(
                        "CONTEXT HIGH: {} at {:.0}% of context window",
                        session.display_name(),
                        pct
                    );
                    fire_notification(&format!(
                        "{} context at {:.0}%",
                        session.display_name(),
                        pct
                    ));
                    self.hooks.fire(HookEvent::ContextHigh, session);
                } else if pct < threshold && self.context_warned.contains(&session.pid) {
                    // Reset warning if context dropped (e.g., after /compact)
                    self.context_warned.remove(&session.pid);
                }
            }
        }

        // Record activity for sparkline and cache decay score
        for session in &mut sessions {
            session.record_activity();
            session.decay_score =
                crate::health::compute_decay_score(session, &self.health_thresholds);
        }

        // Track when sessions first appear as Finished, remove after 30s
        let now = std::time::Instant::now();
        for session in &sessions {
            if session.status == SessionStatus::Finished
                && !self.finished_at.contains_key(&session.pid)
            {
                self.finished_at.insert(session.pid, now);
                // Record to history on first Finished detection
                crate::history::record_session(session);
            }
        }
        sessions.retain(|s| {
            if s.status == SessionStatus::Finished {
                if let Some(&t) = self.finished_at.get(&s.pid) {
                    return now.duration_since(t).as_secs() < 30;
                }
            }
            true
        });
        // Clean up old finished_at entries. The in-memory map is the ONLY
        // thing cleaned here: claudectl never deletes Claude Code's session
        // pointer files (`~/.claude/sessions/{pid}.json`) or their sidecars.
        // "Finished" only means the pid is absent from THIS process's `ps`
        // view — a TUI running inside the sandbox sees every host session as
        // Finished (host pids don't exist in the sandbox PID namespace) and
        // used to delete live sessions' pointer files through the shared
        // mount, silently emptying the `--restore-sessions` registry. Dead
        // sidecars are swept by the reaper, which checks liveness in the
        // right namespace.
        self.finished_at
            .retain(|_, t| now.duration_since(*t).as_secs() < 60);

        // Sort
        self.apply_sort(&mut sessions);

        // Notifications and webhooks: check for status transitions
        for session in &sessions {
            let prev = self.prev_statuses.get(&session.pid).copied();
            let changed = prev.is_some() && prev != Some(session.status);

            if !changed {
                continue;
            }

            crate::logger::log(
                "DEBUG",
                &format!(
                    "session {}: status {} -> {}",
                    session.display_name(),
                    prev.unwrap(),
                    session.status
                ),
            );

            // Desktop notification on NeedsInput
            if self.notify && session.status == SessionStatus::NeedsInput {
                fire_notification(&session.project_name);
            }

            // Webhook on status change
            if let Some(ref url) = self.webhook_url {
                let new_status = session.status.to_string();
                let should_fire = match &self.webhook_filter {
                    Some(filter) => filter.iter().any(|f| f.eq_ignore_ascii_case(&new_status)),
                    None => true,
                };
                if should_fire {
                    crate::logger::log(
                        "DEBUG",
                        &format!(
                            "webhook fired for {} -> {}",
                            session.display_name(),
                            new_status
                        ),
                    );
                    fire_webhook(
                        url,
                        session,
                        prev.map(|p| p.to_string()).unwrap_or_default(),
                    );
                }
            }

            // Event hooks
            self.hooks.fire_with_status(
                HookEvent::StatusChange,
                session,
                &prev.unwrap().to_string(),
                &session.status.to_string(),
            );

            match session.status {
                SessionStatus::NeedsInput => {
                    self.hooks.fire(HookEvent::NeedsInput, session);
                }
                SessionStatus::Finished => {
                    self.hooks.fire(HookEvent::Finished, session);
                }
                SessionStatus::Idle => {
                    self.hooks.fire(HookEvent::Idle, session);
                }
                _ => {}
            }
        }

        // Fire hooks for newly discovered sessions
        for session in sessions.iter().filter(|s| new_pids.contains(&s.pid)) {
            self.hooks.fire(HookEvent::SessionStart, session);
        }

        // Track NeedsInput wait times
        let now_instant = std::time::Instant::now();
        for session in &sessions {
            if session.status == SessionStatus::NeedsInput {
                // Record when it first entered NeedsInput
                self.needs_input_since
                    .entry(session.pid)
                    .or_insert(now_instant);
            } else {
                // Clear if no longer NeedsInput
                self.needs_input_since.remove(&session.pid);
            }
        }
        // Clean up entries for sessions that no longer exist
        let active_pids: HashSet<u32> = sessions.iter().map(|s| s.pid).collect();
        self.needs_input_since
            .retain(|pid, _| active_pids.contains(pid));

        // Conflict detection: find sessions sharing the same git worktree
        // Uses worktree_id (git show-toplevel) so different worktrees don't false-positive
        self.conflict_pids.clear();
        let mut wt_sessions: HashMap<&str, Vec<u32>> = HashMap::new();
        for session in &sessions {
            if session.status != SessionStatus::Finished {
                let key = session.worktree_id.as_deref().unwrap_or(&session.cwd);
                wt_sessions.entry(key).or_default().push(session.pid);
            }
        }
        for (wt, pids) in &wt_sessions {
            if pids.len() >= 2 {
                for &pid in pids {
                    self.conflict_pids.insert(pid);
                }
                // Fire hook once per worktree conflict (not on every tick)
                if !self.conflict_alerted.contains(*wt) {
                    self.conflict_alerted.insert(wt.to_string());
                    let project = sessions
                        .iter()
                        .find(|s| s.pid == pids[0])
                        .map(|s| s.display_name())
                        .unwrap_or("unknown");
                    self.status_msg =
                        format!("CONFLICT: {} sessions sharing {}", pids.len(), project);
                    fire_notification(&format!("{} sessions in {}", pids.len(), project));
                    if let Some(session) = sessions.iter().find(|s| s.pid == pids[0]) {
                        self.hooks.fire(HookEvent::ConflictDetected, session);
                    }
                }
            }
        }
        // Clear alerts for worktrees that no longer have conflicts
        self.conflict_alerted.retain(|wt| {
            wt_sessions
                .get(wt.as_str())
                .map(|pids| pids.len() >= 2)
                .unwrap_or(false)
        });

        // File-level conflict detection: find files edited by multiple sessions
        self.file_conflict_pids.clear();
        self.file_conflicts.clear();
        // Reset has_file_conflict on all sessions
        for session in &mut sessions {
            session.has_file_conflict = false;
        }

        if self.file_conflicts_enabled {
            // Build file → PIDs map from files_modified across active sessions
            let mut file_pids: HashMap<String, Vec<u32>> = HashMap::new();
            for session in &sessions {
                if session.status == SessionStatus::Finished {
                    continue;
                }
                for file in session.files_modified.keys() {
                    file_pids.entry(file.clone()).or_default().push(session.pid);
                }
                // Also consider pending file edits (predictive conflict)
                if let Some(ref pending) = session.pending_file_path {
                    file_pids
                        .entry(pending.clone())
                        .or_default()
                        .push(session.pid);
                }
            }

            // Deduplicate PIDs per file (a session may appear twice if it both modified and is pending)
            for pids in file_pids.values_mut() {
                pids.sort_unstable();
                pids.dedup();
            }

            // Record conflicts where 2+ sessions touch the same file
            for (file, pids) in &file_pids {
                if pids.len() >= 2 {
                    for &pid in pids {
                        self.file_conflict_pids.insert(pid);
                    }
                    self.file_conflicts.insert(file.clone(), pids.clone());

                    // Mark sessions with pending file conflicts
                    for session in &mut sessions {
                        if let Some(ref pending) = session.pending_file_path {
                            if pending == file && pids.contains(&session.pid) {
                                session.has_file_conflict = true;
                            }
                        }
                    }

                    // Fire alert once per conflicting file
                    if !self.file_conflict_alerted.contains(file) {
                        self.file_conflict_alerted.insert(file.clone());
                        let names: Vec<&str> = pids
                            .iter()
                            .filter_map(|pid| {
                                sessions
                                    .iter()
                                    .find(|s| s.pid == *pid)
                                    .map(|s| s.display_name())
                            })
                            .collect();
                        let short = file.rsplit('/').next().unwrap_or(file);
                        self.status_msg =
                            format!("FILE CONFLICT: {} edited by {}", short, names.join(", "));
                        fire_notification(&format!("File conflict: {short}"));
                        if let Some(session) = sessions.iter().find(|s| s.pid == pids[0]) {
                            self.hooks.fire(HookEvent::ConflictDetected, session);
                        }
                    }
                }
            }

            // Clear alerts for files no longer in conflict
            self.file_conflict_alerted
                .retain(|f| self.file_conflicts.contains_key(f));
        }

        // Update prev_statuses
        self.prev_statuses = sessions.iter().map(|s| (s.pid, s.status)).collect();

        // Atomic-swap the new sessions list into shared AppData.
        // Preserve ledger summaries from the previous snapshot so a
        // refresh that doesn't touch the ledger doesn't accidentally
        // zero it out.
        let prev = self.data_snapshot();
        self.replace_data(AppData {
            sessions,
            ledger_today: prev.ledger_today.clone(),
            ledger_week: prev.ledger_week.clone(),
            ledger_month: prev.ledger_month.clone(),
        });
        self.normalize_selection();

        // Record debug timings
        if self.debug {
            let total_elapsed = tick_start.elapsed();
            self.debug_timings.record(
                scan_elapsed.as_secs_f64() * 1000.0,
                ps_elapsed.as_secs_f64() * 1000.0,
                jsonl_elapsed.as_secs_f64() * 1000.0,
                total_elapsed.as_secs_f64() * 1000.0,
            );
        }
    }

    fn apply_sort(&self, sessions: &mut [ClaudeSession]) {
        match self.sort_column {
            0 => sessions.sort_by(|a, b| {
                a.status.sort_key().cmp(&b.status.sort_key()).then_with(|| {
                    // Within NeedsInput, sort by longest waiting first
                    if a.status == SessionStatus::NeedsInput {
                        let a_wait = self.wait_duration(a.pid).unwrap_or_default();
                        let b_wait = self.wait_duration(b.pid).unwrap_or_default();
                        b_wait.cmp(&a_wait)
                    } else {
                        b.elapsed.cmp(&a.elapsed)
                    }
                })
            }),
            1 => sessions.sort_by(|a, b| {
                b.context_percent()
                    .partial_cmp(&a.context_percent())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            2 => sessions.sort_by(|a, b| {
                b.cost_usd
                    .partial_cmp(&a.cost_usd)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            3 => sessions.sort_by(|a, b| {
                b.burn_rate_per_hr
                    .partial_cmp(&a.burn_rate_per_hr)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            4 => sessions.sort_by_key(|s| std::cmp::Reverse(s.elapsed)),
            5 => sessions.sort_by(|a, b| {
                // Last user interaction: most recent first. Sessions with no
                // recorded user message (ts == 0) sort to the bottom.
                let key = |s: &ClaudeSession| {
                    (
                        s.last_user_message_ts == 0,
                        std::cmp::Reverse(s.last_user_message_ts),
                    )
                };
                key(a).cmp(&key(b))
            }),
            6 => sessions.sort_by(|a, b| {
                // Sort key: unnamed sessions last, then ascending
                // case-insensitive session_name, then project_name as tiebreak
                // so two sessions with the same name group by project.
                let key = |s: &ClaudeSession| {
                    (
                        s.session_name.is_empty(),
                        s.session_name.to_lowercase(),
                        s.project_name.to_lowercase(),
                    )
                };
                key(a).cmp(&key(b))
            }),
            _ => {}
        }
        // User-toggled direction override flips whatever the column's natural
        // sort produced. Stable sort + `reverse()` preserves tiebreak order
        // from the original sort (mirrored).
        if self.sort_reversed {
            sessions.reverse();
        }
        // Parked sessions always drop to the bottom, regardless of column or
        // direction. Stable partition preserves their relative sort order
        // among themselves.
        if !self.parked.is_empty() {
            // `sort_by_key` is stable, so this partitions into
            // (non-parked, parked) without disturbing order within each half.
            sessions.sort_by_key(|s| self.is_parked(&s.session_id));
        }
    }

    pub fn cycle_sort(&mut self) {
        self.sort_column = (self.sort_column + 1) % SORT_COLUMNS.len();
        // Each new column starts in its natural direction — avoids surprise
        // ordering carried over from a prior column's `S` toggle.
        self.sort_reversed = false;
        self.status_msg = format!("Sort: {}", SORT_COLUMNS[self.sort_column]);
        self.with_sessions(|s| self.apply_sort(s));
    }

    pub fn is_parked(&self, session_id: &str) -> bool {
        self.parked.contains(session_id)
    }

    /// Add or remove a session_id from the parked set. Used by tests and by
    /// the key handler via `toggle_park_selected`.
    pub fn toggle_park(&mut self, session_id: &str) {
        if self.parked.contains(session_id) {
            self.parked.remove(session_id);
        } else {
            self.parked.insert(session_id.to_string());
        }
    }

    /// Toggle park state on whichever session is currently selected, then
    /// re-apply the sort so the row moves between sections. Does NOT write
    /// to disk — callers that want persistence pair this with `save_parked`
    /// (the `p` key handler does).
    pub fn toggle_park_selected(&mut self) {
        let Some(session) = self.selected_session() else {
            return;
        };
        let session_id = session.session_id.clone();
        let display_name = session.display_name().to_string();
        let was_parked = self.is_parked(&session_id);
        self.toggle_park(&session_id);
        self.status_msg = if was_parked {
            format!("Unparked {display_name}")
        } else {
            format!("Parked {display_name}")
        };
        self.with_sessions(|s| self.apply_sort(s));
        self.normalize_selection();
    }

    /// Persist the current parked set to the default disk location
    /// (`~/.claudectl/parked.json`). Best-effort — failures are silent.
    pub fn save_parked(&self) {
        save_parked_to(&parked_path(), &self.parked);
    }

    pub fn toggle_sort_direction(&mut self) {
        self.sort_reversed = !self.sort_reversed;
        let label = SORT_COLUMNS[self.sort_column];
        self.status_msg = if self.sort_reversed {
            format!("Sort: {label} (reversed)")
        } else {
            format!("Sort: {label}")
        };
        self.with_sessions(|s| self.apply_sort(s));
    }

    fn refresh_demo(&mut self) {
        self.demo_tick += 1;
        let mut sessions = crate::demo::generate_sessions(self.demo_tick);

        // Track NeedsInput wait times (same as real mode)
        let now_instant = std::time::Instant::now();
        for session in &sessions {
            if session.status == SessionStatus::NeedsInput {
                self.needs_input_since
                    .entry(session.pid)
                    .or_insert(now_instant);
            } else {
                self.needs_input_since.remove(&session.pid);
            }
        }

        // Conflict detection using worktree_id
        self.conflict_pids.clear();
        let mut wt_sessions: HashMap<&str, Vec<u32>> = HashMap::new();
        for session in &sessions {
            if session.status != SessionStatus::Finished {
                let key = session.worktree_id.as_deref().unwrap_or(&session.cwd);
                wt_sessions.entry(key).or_default().push(session.pid);
            }
        }
        for pids in wt_sessions.values() {
            if pids.len() >= 2 {
                for &pid in pids {
                    self.conflict_pids.insert(pid);
                }
            }
        }

        // Scripted demo events: rules, brain, routing, health alerts
        if let Some(event) = crate::demo::demo_event(self.demo_tick) {
            self.status_msg = event.message.clone();
            match event.kind {
                crate::demo::EventKind::RuleAction => {
                    self.last_rule_action = Some(event.message);
                }
                crate::demo::EventKind::BrainSuggestion | crate::demo::EventKind::BrainOverride => {
                    // Show brain activity via status message
                }
                crate::demo::EventKind::Route | crate::demo::EventKind::HealthAlert => {}
            }
        }

        // Inject fake brain pending suggestions so the status bar shows brain activity
        if let Some(ref mut engine) = self.brain_engine {
            engine.pending.clear();
            // At certain phases, show a pending suggestion for a NeedsInput session
            let phase = self.demo_tick % 24;
            if (9..=12).contains(&phase) {
                // Find a NeedsInput session to attach the suggestion to
                if let Some(s) = sessions
                    .iter()
                    .find(|s| s.status == SessionStatus::NeedsInput)
                {
                    engine.pending.insert(
                        s.pid,
                        crate::brain::client::BrainSuggestion {
                            action: crate::rules::RuleAction::Approve,
                            message: s.pending_tool_input.clone(),
                            reasoning: "Safe build command, no side effects".into(),
                            confidence: 0.92,
                            suggested_at: 0,
                        },
                    );
                }
            }
            if (14..=16).contains(&phase) {
                if let Some(s) = sessions
                    .iter()
                    .find(|s| s.status == SessionStatus::NeedsInput)
                {
                    engine.pending.insert(
                        s.pid,
                        crate::brain::client::BrainSuggestion {
                            action: crate::rules::RuleAction::Deny,
                            message: s.pending_tool_input.clone(),
                            reasoning: "Destructive operation, needs manual review".into(),
                            confidence: 0.87,
                            suggested_at: 0,
                        },
                    );
                }
            }
        }

        // Compute decay scores for demo sessions (same as real refresh path)
        for session in &mut sessions {
            session.decay_score =
                crate::health::compute_decay_score(session, &self.health_thresholds);
        }

        let prev = self.data_snapshot();
        self.replace_data(AppData {
            sessions,
            ledger_today: prev.ledger_today.clone(),
            ledger_week: prev.ledger_week.clone(),
            ledger_month: prev.ledger_month.clone(),
        });
        self.normalize_selection();
    }

    pub fn tick(&mut self) {
        self.status_msg.clear();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Refresh `elapsed` on every session in place. Cheap; runs every
        // tick. Uses `modify_data` so we don't deep-clone the AppData
        // for what's just a per-session integer write.
        self.modify_data(|d| {
            for session in &mut d.sessions {
                let elapsed_ms = now_ms.saturating_sub(session.started_at);
                session.elapsed = std::time::Duration::from_millis(elapsed_ms);
            }
        });

        // Non-blocking refresh: applies any completed worker result and
        // kicks the next one off without waiting. The main thread
        // returns immediately so the render+key loop stays smooth even
        // when `do_refresh_io` takes ~30 ms (or much more on a slow
        // disk).
        self.refresh_nonblocking();
        self.run_auto_actions();

        // Check idle mode transition
        self.check_idle_mode();

        // Two cadences for the ledger, both non-blocking:
        // - Rollup refresh: sub-ms thanks to the in-memory ledger cache;
        //   keeps the title bar's day/week/month totals current. Runs
        //   every 3 ticks ≈ 6 s.
        // - Background scan_and_append: kicked off on a worker thread
        //   every 15 ticks ≈ 30 s. The main thread never blocks on it,
        //   so a slow JSONL walk no longer freezes the TUI. New rows
        //   become visible to subsequent rollup refreshes once the
        //   worker finishes (typically <30 ms steady-state on this box).
        self.ledger_refresh_tick += 1;
        if self.ledger_refresh_tick >= 3 {
            self.ledger_refresh_tick = 0;
            self.refresh_ledger_rollups();
            self.check_aggregate_budgets();
        }
        self.ledger_scan_tick += 1;
        if self.ledger_scan_tick >= 15 {
            self.ledger_scan_tick = 0;
            crate::usage_ledger::scan_and_append_background();
        }
    }

    /// Recompute rolling 24-hour / 7-day / 30-day summaries from the
    /// in-memory ledger cache. Cheap (microseconds), so safe to run
    /// every tick without freezing the TUI.
    fn refresh_ledger_rollups(&mut self) {
        let now = crate::usage_ledger::now_ms();
        let day_cutoff = now.saturating_sub(86_400_000);
        let week_cutoff = now.saturating_sub(7 * 86_400_000);
        let month_cutoff = now.saturating_sub(30 * 86_400_000);
        let today = crate::usage_ledger::load_summary(day_cutoff);
        let week = crate::usage_ledger::load_summary(week_cutoff);
        let month = crate::usage_ledger::load_summary(month_cutoff);
        let snap = self.data_snapshot();
        self.replace_data(AppData {
            sessions: snap.sessions.clone(),
            ledger_today: today,
            ledger_week: week,
            ledger_month: month,
        });
    }

    /// Get how long a session has been waiting for input, if applicable.
    pub fn wait_duration(&self, pid: u32) -> Option<std::time::Duration> {
        self.needs_input_since
            .get(&pid)
            .map(|since| since.elapsed())
    }

    /// Format wait duration as a compact string (e.g., "2m 34s").
    pub fn format_wait_time(&self, pid: u32) -> Option<String> {
        let dur = self.wait_duration(pid)?;
        let secs = dur.as_secs();
        if secs < 60 {
            Some(format!("{secs}s"))
        } else {
            Some(format!("{}m {}s", secs / 60, secs % 60))
        }
    }

    /// Compute budget exhaustion ETA based on current burn rate.
    /// Returns (spent, limit, eta_string, urgency) where urgency is 0=safe, 1=warn, 2=critical.
    pub fn budget_eta(&self) -> Option<(f64, f64, String, u8)> {
        let snap = self.data_snapshot();
        let live_cost: f64 = snap.sessions.iter().map(|s| s.cost_usd).sum();
        let total_burn: f64 = snap.sessions.iter().map(|s| s.burn_rate_per_hr).sum();

        // Prefer daily limit, fall back to per-session budget
        let (spent, limit) = if let Some(daily) = self.daily_limit {
            (snap.ledger_today.cost_usd + live_cost, daily)
        } else if let Some(budget) = self.budget_usd {
            // For per-session budget, show the session closest to limit
            if let Some(session) = snap.sessions.iter().max_by(|a, b| {
                (a.cost_usd / budget)
                    .partial_cmp(&(b.cost_usd / budget))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                (session.cost_usd, budget)
            } else {
                return None;
            }
        } else {
            return None;
        };

        let remaining = limit - spent;
        if remaining <= 0.0 {
            return Some((spent, limit, "exceeded".into(), 2));
        }
        if total_burn < 0.01 {
            return Some((spent, limit, "safe".into(), 0));
        }

        let hours_left = remaining / total_burn;
        let mins_left = (hours_left * 60.0) as u64;
        let eta_str = if mins_left >= 120 {
            format!("{}h {}m", mins_left / 60, mins_left % 60)
        } else {
            format!("{}m", mins_left)
        };

        let urgency = if mins_left <= 30 {
            2
        } else if mins_left <= 120 {
            1
        } else {
            0
        };
        Some((spent, limit, eta_str, urgency))
    }

    fn check_aggregate_budgets(&mut self) {
        let snap = self.data_snapshot();
        // Also include cost from currently live sessions. The ledger only
        // captures messages that have already been written to JSONL; a
        // mid-flight streaming response isn't there yet.
        let live_cost: f64 = snap.sessions.iter().map(|s| s.cost_usd).sum();

        // Daily limit check
        if let Some(daily_limit) = self.daily_limit {
            let today_total = snap.ledger_today.cost_usd + live_cost;
            let pct = today_total / daily_limit * 100.0;

            if pct >= 80.0 && !self.daily_alert_fired {
                self.daily_alert_fired = true;
                self.status_msg = format!(
                    "DAILY BUDGET: ${:.2}/${:.2} ({:.0}%)",
                    today_total, daily_limit, pct
                );
                fire_notification(&format!("Daily budget at {:.0}%", pct));

                // Fire hooks with a synthetic session containing aggregate data
                let mut dummy = create_aggregate_session(today_total, daily_limit, "daily");
                self.hooks.fire(HookEvent::BudgetWarning, &dummy);

                if pct >= 100.0 {
                    dummy.cost_usd = today_total;
                    self.hooks.fire(HookEvent::BudgetExceeded, &dummy);
                }
            }
        }

        // Weekly limit check
        if let Some(weekly_limit) = self.weekly_limit {
            let week_total = snap.ledger_week.cost_usd + live_cost;
            let pct = week_total / weekly_limit * 100.0;

            if pct >= 80.0 && !self.weekly_alert_fired {
                self.weekly_alert_fired = true;
                self.status_msg = format!(
                    "WEEKLY BUDGET: ${:.2}/${:.2} ({:.0}%)",
                    week_total, weekly_limit, pct
                );
                fire_notification(&format!("Weekly budget at {:.0}%", pct));

                let mut dummy = create_aggregate_session(week_total, weekly_limit, "weekly");
                self.hooks.fire(HookEvent::BudgetWarning, &dummy);

                if pct >= 100.0 {
                    dummy.cost_usd = week_total;
                    self.hooks.fire(HookEvent::BudgetExceeded, &dummy);
                }
            }
        }
    }

    fn check_idle_mode(&mut self) {
        if !self.idle_config.enabled {
            return;
        }
        let idle_threshold = std::time::Duration::from_secs(self.idle_config.after_idle_mins * 60);
        let was_idle = self.idle_mode_active;
        self.idle_mode_active = self.last_user_interaction.elapsed() > idle_threshold;

        if self.idle_mode_active && !was_idle {
            crate::logger::log("IDLE", "Entering idle mode");
        }
    }

    /// Check if currently in idle mode (used by other systems like lifecycle restart).
    #[allow(dead_code)]
    pub fn is_idle(&self) -> bool {
        self.idle_mode_active
    }

    fn run_auto_actions(&mut self) {
        let snap = self.data_snapshot();
        // In demo mode, events are scripted in refresh_demo() — skip real execution
        if self.demo_mode {
            return;
        }

        // Legacy per-PID auto-approve (toggled with 'a' key)
        let legacy_pids: Vec<u32> = snap
            .sessions
            .iter()
            .filter(|s| s.status == SessionStatus::NeedsInput && self.auto_approve.contains(&s.pid))
            .map(|s| s.pid)
            .collect();

        for pid in legacy_pids {
            if let Some(session) = snap.sessions.iter().find(|s| s.pid == pid) {
                crate::brain::decisions::log_observation(
                    session.pid,
                    session.display_name(),
                    session.pending_tool_name.as_deref(),
                    session.pending_tool_input.as_deref(),
                    "user_approve",
                    Some(session),
                );
                match terminals::approve_session(session) {
                    Ok(()) => self.status_msg = format!("Auto-approved {}", session.display_name()),
                    Err(e) => self.status_msg = format!("Auto-approve error: {e}"),
                }
            }
        }

        // Built-in file conflict auto-deny: deny writes to files being edited by another session
        if self.auto_deny_file_conflicts {
            let conflict_candidates: Vec<(u32, String, String)> = snap
                .sessions
                .iter()
                .filter(|s| {
                    s.status == SessionStatus::NeedsInput
                        && s.has_file_conflict
                        && s.pending_file_path.is_some()
                })
                .filter_map(|s| {
                    let file = s.pending_file_path.as_ref()?;
                    let other_pids = self.file_conflicts.get(file)?;
                    let other_name = other_pids
                        .iter()
                        .filter(|&&p| p != s.pid)
                        .find_map(|pid| {
                            snap.sessions
                                .iter()
                                .find(|o| o.pid == *pid)
                                .map(|o| format!("{} (PID {})", o.display_name(), o.pid))
                        })
                        .unwrap_or_else(|| "another session".into());
                    Some((s.pid, file.clone(), other_name))
                })
                .collect();

            for (pid, file, other) in conflict_candidates {
                // Debounce
                if let Some(last) = self.auto_actions_fired.get(&pid) {
                    if last.elapsed().as_secs() < 5 {
                        continue;
                    }
                }
                if let Some(session) = snap.sessions.iter().find(|s| s.pid == pid) {
                    // Log passive observation: conflict auto-deny
                    crate::brain::decisions::log_observation(
                        session.pid,
                        session.display_name(),
                        session.pending_tool_name.as_deref(),
                        session.pending_tool_input.as_deref(),
                        "conflict_deny",
                        Some(session),
                    );
                    let short = file.rsplit('/').next().unwrap_or(&file);
                    let msg = format!("File {short} is being edited by {other}");
                    match terminals::send_input(session, &msg) {
                        Ok(()) => {
                            let status = format!(
                                "File conflict: denied {} edit to {short}",
                                session.display_name()
                            );
                            crate::logger::log("CONFLICT", &status);
                            self.status_msg = status;
                        }
                        Err(e) => {
                            self.status_msg = format!("File conflict deny error: {e}");
                        }
                    }
                    self.auto_actions_fired
                        .insert(pid, std::time::Instant::now());
                }
            }
        }

        // Rule-based auto-actions
        if !self.rules.is_empty() {
            let candidates: Vec<u32> = snap
                .sessions
                .iter()
                .filter(|s| {
                    matches!(
                        s.status,
                        SessionStatus::NeedsInput | SessionStatus::WaitingInput
                    )
                })
                .filter(|s| !self.auto_approve.contains(&s.pid)) // Legacy takes priority
                .map(|s| s.pid)
                .collect();

            for pid in candidates {
                // Debounce: don't re-fire within 3 seconds for same PID
                if let Some(last) = self.auto_actions_fired.get(&pid) {
                    if last.elapsed().as_secs() < 3 {
                        continue;
                    }
                }

                let session = match snap.sessions.iter().find(|s| s.pid == pid) {
                    Some(s) => s,
                    None => continue,
                };

                let result = crate::rules::evaluate(&self.rules, session);
                let Some(rule_match) = result else {
                    continue;
                };

                // Log passive observation: static rule fired
                let obs_action = format!("rule_{}", rule_match.action.label());
                crate::brain::decisions::log_observation(
                    session.pid,
                    session.display_name(),
                    session.pending_tool_name.as_deref(),
                    session.pending_tool_input.as_deref(),
                    &obs_action,
                    Some(session),
                );

                let msg = crate::rules::execute(&rule_match, session);
                match msg {
                    Ok(status) => {
                        crate::logger::log("AUTO", &status);
                        self.last_rule_action = Some(status.clone());
                        self.status_msg = status;
                    }
                    Err(e) => {
                        self.status_msg = format!("Rule error: {e}");
                    }
                }

                self.auto_actions_fired
                    .insert(pid, std::time::Instant::now());
            }
        } // end if !self.rules.is_empty()

        // Brain inference (opt-in, runs after rules)
        if let Some(ref mut engine) = self.brain_engine {
            // Collect deny-only rules for override checking
            let deny_rules: Vec<_> = self
                .rules
                .iter()
                .filter(|r| r.action == crate::rules::RuleAction::Deny)
                .cloned()
                .collect();

            let actions = engine.tick(&snap.sessions, &deny_rules);
            for (_pid, msg) in actions {
                crate::logger::log("BRAIN", &msg);
                self.status_msg = msg;
            }

            engine.cleanup(&snap.sessions);

            // Deliver pending mailbox messages to sessions waiting for input
            let deliveries = crate::brain::mailbox::deliver_pending(&snap.sessions);
            for (_pid, msg) in deliveries {
                crate::logger::log("MAILBOX", &msg);
                self.status_msg = msg;
            }
        }
    }

    pub fn handle_auto_approve(&mut self) {
        let Some(session) = self.selected_session() else {
            return;
        };
        let pid = session.pid;
        let name = session.display_name().to_string();

        if self.pending_auto_approve == Some(pid) {
            if self.auto_approve.contains(&pid) {
                self.auto_approve.remove(&pid);
                self.status_msg = format!("Auto-approve OFF for {name}");
            } else {
                self.auto_approve.insert(pid);
                self.status_msg = format!("Auto-approve ON for {name}");
            }
            self.pending_auto_approve = None;
        } else {
            self.pending_auto_approve = Some(pid);
            let action = if self.auto_approve.contains(&pid) {
                "disable"
            } else {
                "enable"
            };
            self.status_msg = format!("Press a again to {action} auto-approve for {name}");
        }
    }

    pub fn cancel_pending_auto_approve(&mut self) {
        self.pending_auto_approve = None;
    }

    pub fn next(&mut self) {
        let len = self.visible_session_count();
        if len == 0 {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) if i >= len - 1 => 0,
            Some(i) => i + 1,
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    pub fn previous(&mut self) {
        let len = self.visible_session_count();
        if len == 0 {
            return;
        }
        let i = match self.table_state.selected() {
            Some(0) => len - 1,
            Some(i) => i - 1,
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    pub fn selected_session(&self) -> Option<ClaudeSession> {
        // One snapshot for both the ordering and the lookup. Taking two (as
        // this used to, via `visible_session_indices()` and then
        // `data_snapshot()`) let a refresh land in between and resolve the
        // index against a list it was never computed for.
        let snap = self.data_snapshot();
        let selected = self.table_state.selected()?;
        let session_idx = *self.ordered_indices(&snap).get(selected)?;
        snap.sessions.get(session_idx).cloned()
    }

    pub fn handle_kill(&mut self) {
        let Some(session) = self.selected_session() else {
            return;
        };
        let pid = session.pid;
        let name = session.display_name().to_string();

        if self.pending_kill == Some(pid) {
            match kill_process(pid) {
                Ok(()) => {
                    self.status_msg = format!("Killed {name} (PID {pid})");
                    self.auto_approve.remove(&pid);
                    // Don't delete session file yet — let the Finished tombstone show for 30s.
                    // The file will be cleaned up when the tombstone expires.
                    self.refresh();
                }
                Err(e) => self.status_msg = format!("Kill failed: {e}"),
            }
            self.pending_kill = None;
        } else {
            self.pending_kill = Some(pid);
            self.status_msg = format!("Kill {name} (PID {pid})? Press d again to confirm");
        }
    }

    pub fn cancel_pending_kill(&mut self) {
        if self.pending_kill.is_some() {
            self.pending_kill = None;
            self.status_msg = "Kill cancelled".into();
        }
    }

    /// Handle a key event. Returns false if the application should quit.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        self.last_user_interaction = std::time::Instant::now();

        // Transition out of idle mode on any key press
        if self.idle_mode_active {
            self.idle_mode_active = false;
            if !self.idle_report.is_empty() {
                let report = self.idle_report.join("; ");
                self.status_msg = format!("Idle report: {report}");
                self.idle_report.clear();
            }
            self.idle_tasks_launched.clear();
        }

        // Help overlay: any key dismisses
        if self.show_help {
            self.show_help = false;
            return true;
        }

        // Launch mode: capture directory for new session
        if self.launch_mode {
            self.handle_launch_key(key);
            return true;
        }

        if self.search_mode {
            self.handle_search_key(key);
            return true;
        }

        // Input mode: capture text for sending to a session
        if self.input_mode {
            self.handle_input_key(key);
            return true;
        }

        // Normal mode
        self.handle_normal_key(key);
        !self.should_quit
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                if let Some(pid) = self.input_target_pid {
                    let snap = self.data_snapshot();
                    if let Some(session) = snap.sessions.iter().find(|s| s.pid == pid) {
                        // Log passive observation: user sent manual input
                        crate::brain::decisions::log_observation(
                            session.pid,
                            session.display_name(),
                            session.pending_tool_name.as_deref(),
                            session.pending_tool_input.as_deref(),
                            "user_input",
                            Some(session),
                        );
                        let text = format!("{}\n", self.input_buffer);
                        match terminals::send_input(session, &text) {
                            Ok(()) => {
                                self.status_msg = format!("Sent to {}", session.display_name())
                            }
                            Err(e) => self.status_msg = format!("Error: {e}"),
                        }
                    }
                }
                self.input_mode = false;
                self.input_buffer.clear();
                self.input_target_pid = None;
            }
            KeyCode::Esc => {
                self.input_mode = false;
                self.input_buffer.clear();
                self.input_target_pid = None;
                self.status_msg = "Input cancelled".into();
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            _ => {}
        }
    }

    fn handle_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                self.search_query = self.search_buffer.trim().to_string();
                self.search_mode = false;
                self.normalize_selection();
                if self.search_query.is_empty() {
                    self.status_msg = "Search cleared".into();
                } else {
                    self.status_msg = format!("Search: {}", self.search_query);
                }
            }
            KeyCode::Esc => {
                self.search_mode = false;
                self.search_buffer.clear();
                self.status_msg = "Search cancelled".into();
            }
            KeyCode::Backspace => {
                self.search_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.search_buffer.push(c);
            }
            _ => {}
        }
    }

    fn handle_normal_key(&mut self, key: KeyEvent) {
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) => {
                self.should_quit = true;
            }
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            (KeyCode::Esc, _) => {
                // Stepwise unwind: cancel pending actions / close panels /
                // clear filters, one layer per press. Use `z` to clear
                // everything at once.
                if self.pending_kill.is_some() {
                    self.cancel_pending_kill();
                    self.status_msg = "Kill cancelled".into();
                } else if self.pending_auto_approve.is_some() {
                    self.cancel_pending_auto_approve();
                    self.status_msg = "Auto-approve cancelled".into();
                } else if self.detail_panel {
                    self.detail_panel = false;
                } else if !self.search_query.trim().is_empty() {
                    self.search_query.clear();
                    self.search_buffer.clear();
                    self.normalize_selection();
                    self.status_msg = "Search cleared".into();
                } else if self.status_filter != StatusFilter::All
                    || self.focus_filter != FocusFilter::All
                {
                    self.status_filter = StatusFilter::All;
                    self.focus_filter = FocusFilter::All;
                    self.normalize_selection();
                    self.status_msg = "Filters cleared".into();
                }
            }
            (KeyCode::Char('j'), _) | (KeyCode::Down, _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.next();
            }
            (KeyCode::Char('k'), _) | (KeyCode::Up, _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.previous();
            }
            (KeyCode::Char('r'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.refresh();
            }
            (KeyCode::Char('R'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.toggle_session_recording();
            }
            (KeyCode::Char('d'), _) | (KeyCode::Char('x'), _) => {
                self.cancel_pending_auto_approve();
                self.handle_kill();
            }
            (KeyCode::Char('y'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.handle_approve();
            }
            (KeyCode::Char('b'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.handle_brain_accept();
            }
            (KeyCode::Char('B'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.handle_brain_reject();
            }
            (KeyCode::Char('i'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.enter_input_mode();
            }
            (KeyCode::Char('c'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.handle_compact();
            }
            (KeyCode::Char('?'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.show_help = !self.show_help;
            }
            (KeyCode::Char('s'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.cycle_sort();
            }
            (KeyCode::Char('S'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.toggle_sort_direction();
            }
            (KeyCode::Char('p'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.toggle_park_selected();
                self.save_parked();
            }
            (KeyCode::Char('f'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.cycle_status_filter();
            }
            (KeyCode::Char('v'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.cycle_focus_filter();
            }
            (KeyCode::Char('z'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.clear_filters();
            }
            (KeyCode::Char('/'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.enter_search_mode();
            }
            (KeyCode::Char('a'), _) => {
                self.cancel_pending_kill();
                self.handle_auto_approve();
            }
            (KeyCode::Char('n'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.enter_launch_mode();
            }
            (KeyCode::Char('g'), _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.grouped_view = !self.grouped_view;
                self.status_msg = if self.grouped_view {
                    "Grouped by project".into()
                } else {
                    "Flat view".into()
                };
            }
            (KeyCode::Enter, _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.detail_panel = !self.detail_panel;
            }
            (KeyCode::Tab, _) => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
                self.handle_switch_terminal();
            }
            _ => {
                self.cancel_pending_kill();
                self.cancel_pending_auto_approve();
            }
        }
    }

    fn handle_launch_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.submit_launch_form();
            }
            KeyCode::Enter => {
                if self.launch_form.is_last_field() {
                    self.submit_launch_form();
                } else {
                    self.launch_form.advance();
                    self.status_msg = self.launch_form.status_hint();
                }
            }
            KeyCode::Tab | KeyCode::Down => {
                self.launch_form.advance();
                self.status_msg = self.launch_form.status_hint();
            }
            KeyCode::BackTab | KeyCode::Up => {
                self.launch_form.retreat();
                self.status_msg = self.launch_form.status_hint();
            }
            KeyCode::Esc => {
                self.launch_mode = false;
                self.launch_form = LaunchForm::default();
                self.status_msg = "Launch cancelled".into();
            }
            KeyCode::Backspace => {
                self.launch_form.active_buffer_mut().pop();
            }
            KeyCode::Char(c) => {
                self.launch_form.active_buffer_mut().push(c);
            }
            _ => {}
        }
    }

    fn enter_launch_mode(&mut self) {
        self.launch_mode = true;
        self.launch_form = LaunchForm::default();
        self.status_msg = self.launch_form.status_hint();
    }

    fn submit_launch_form(&mut self) {
        let request = match self.launch_form.request() {
            Ok(request) => request,
            Err(err) => {
                self.launch_form.field = LaunchField::Cwd;
                self.status_msg = format!("Launch failed: {err}");
                return;
            }
        };

        match launch::launch(&request) {
            Ok(target) => {
                self.launch_mode = false;
                self.launch_form = LaunchForm::default();
                self.status_msg = format!(
                    "Launched session in {target} at {}{}",
                    request.cwd_path.display(),
                    request.option_summary()
                );
            }
            Err(err) => {
                self.status_msg = format!("Launch failed: {err}");
            }
        }
    }

    fn enter_search_mode(&mut self) {
        self.search_mode = true;
        self.search_buffer = self.search_query.clone();
    }

    pub fn clear_filters(&mut self) {
        self.status_filter = StatusFilter::All;
        self.focus_filter = FocusFilter::All;
        self.search_query.clear();
        self.search_buffer.clear();
        self.search_mode = false;
        self.normalize_selection();
        self.status_msg = "Filters cleared".into();
    }

    pub fn cycle_status_filter(&mut self) {
        self.status_filter = self.status_filter.next();
        self.normalize_selection();
        self.status_msg = format!("Status filter: {}", self.status_filter.label());
    }

    pub fn cycle_focus_filter(&mut self) {
        self.focus_filter = self.focus_filter.next();
        self.normalize_selection();
        self.status_msg = format!("Focus filter: {}", self.focus_filter.label());
    }

    pub fn has_active_filters(&self) -> bool {
        self.status_filter != StatusFilter::All
            || self.focus_filter != FocusFilter::All
            || !self.search_query.trim().is_empty()
    }

    pub fn filter_summary(&self) -> String {
        let mut parts = Vec::new();
        if self.status_filter != StatusFilter::All {
            parts.push(format!("status={}", self.status_filter.label()));
        }
        if self.focus_filter != FocusFilter::All {
            parts.push(format!("focus={}", self.focus_filter.label()));
        }
        if !self.search_query.trim().is_empty() {
            parts.push(format!("search=\"{}\"", self.search_query));
        }
        if parts.is_empty() {
            "filters: none".to_string()
        } else {
            format!("filters: {}", parts.join(" | "))
        }
    }

    /// Sessions passing the current filters, as indices into `snap.sessions`,
    /// in snapshot order and *before* any grouped-view reordering.
    ///
    /// Private on purpose: everything that navigates or renders must go
    /// through [`Self::ordered_indices`], which additionally applies the
    /// render order. This one exists only so [`Self::project_groups`] can
    /// aggregate without recursing back through the ordering it feeds.
    fn filtered_indices(&self, snap: &AppData) -> Vec<usize> {
        snap.sessions
            .iter()
            .enumerate()
            .filter_map(|(idx, session)| self.matches_filters(session).then_some(idx))
            .collect()
    }

    /// The visible sessions as snapshot indices, **in the exact order the
    /// table renders them**.
    ///
    /// This is the single source of visible ordering, and that is load-bearing:
    /// `table_state.selected()` is an index into this list, and the table
    /// highlights the row at that position. When the render walked a different
    /// order than navigation did — which grouped view always did, since it
    /// emits sessions grouped by project while navigation ran in snapshot
    /// order — the highlighted row and the session that Tab/kill/approve
    /// actually acted on were two different sessions.
    fn ordered_indices(&self, snap: &AppData) -> Vec<usize> {
        let mut indices = self.filtered_indices(snap);
        if self.grouped_view {
            // Rank each project by its position in `project_groups()` (the
            // order the grouped render emits them in), then sort by that rank.
            // `sort_by_key` is stable, so sessions keep snapshot order within
            // a group — which is what the grouped render produces too.
            let groups = self.project_groups_in(snap);
            let ranks: HashMap<&str, usize> = groups
                .iter()
                .enumerate()
                .map(|(rank, group)| (group.name.as_str(), rank))
                .collect();
            indices.sort_by_key(|idx| {
                ranks
                    .get(snap.sessions[*idx].project_name.as_str())
                    .copied()
                    .unwrap_or(usize::MAX)
            });
        }
        indices
    }

    pub fn visible_session_indices(&self) -> Vec<usize> {
        let snap = self.data_snapshot();
        self.ordered_indices(&snap)
    }

    /// Owned snapshot of the sessions matching the current filters, in render
    /// order. Returns `Vec<ClaudeSession>` (cloned) instead of references
    /// because the underlying `AppData` lives behind an `Arc<RwLock<...>>` —
    /// we can't hand out references whose lifetime is tied to `&self` once
    /// the snapshot Arc is dropped at end of call. Callers that just need to
    /// iterate immutably can take `.iter()` on the result.
    pub fn visible_sessions(&self) -> Vec<ClaudeSession> {
        let snap = self.data_snapshot();
        self.ordered_indices(&snap)
            .into_iter()
            .filter_map(|idx| snap.sessions.get(idx).cloned())
            .collect()
    }

    pub fn visible_session_count(&self) -> usize {
        self.visible_session_indices().len()
    }

    fn normalize_selection(&mut self) {
        let len = self.visible_session_count();
        if len == 0 {
            self.table_state.select(None);
        } else if self.table_state.selected().is_none() {
            self.table_state.select(Some(0));
        } else if let Some(sel) = self.table_state.selected() {
            if sel >= len {
                self.table_state.select(Some(len - 1));
            }
        }
    }

    fn matches_filters(&self, session: &ClaudeSession) -> bool {
        self.status_filter.matches(session.status)
            && self.matches_focus_filter(session)
            && self.matches_search_query(session)
    }

    fn matches_focus_filter(&self, session: &ClaudeSession) -> bool {
        let over_budget = self
            .budget_usd
            .map(|budget| session.has_usage_metrics() && session.cost_usd >= budget)
            .unwrap_or(false);
        let high_context = session.has_usage_metrics()
            && session.context_percent() >= self.context_warn_threshold as f64;
        let unknown_telemetry = !session.has_usage_metrics();
        let conflict = self.conflict_pids.contains(&session.pid);

        match self.focus_filter {
            FocusFilter::All => true,
            FocusFilter::Attention => {
                session.status == SessionStatus::NeedsInput
                    || over_budget
                    || high_context
                    || unknown_telemetry
                    || conflict
            }
            FocusFilter::OverBudget => over_budget,
            FocusFilter::HighContext => high_context,
            FocusFilter::UnknownTelemetry => unknown_telemetry,
            FocusFilter::Conflict => conflict,
        }
    }

    fn matches_search_query(&self, session: &ClaudeSession) -> bool {
        let query = self.search_query.trim();
        if query.is_empty() {
            return true;
        }

        let query = query.to_ascii_lowercase();
        let fields = [
            session.display_name().to_string(),
            session.project_name.clone(),
            session.model.clone(),
            session.cwd.clone(),
            session.session_id.clone(),
        ];

        fields
            .iter()
            .any(|field| field.to_ascii_lowercase().contains(&query))
    }

    fn handle_approve(&mut self) {
        if let Some(session) = self.selected_session() {
            if session.status == SessionStatus::NeedsInput {
                // Log passive observation: user approved without brain involvement
                crate::brain::decisions::log_observation(
                    session.pid,
                    session.display_name(),
                    session.pending_tool_name.as_deref(),
                    session.pending_tool_input.as_deref(),
                    "user_approve",
                    Some(&session),
                );
                match terminals::approve_session(&session) {
                    Ok(()) => self.status_msg = format!("Approved {}", session.display_name()),
                    Err(e) => self.status_msg = format!("Error: {e}"),
                }
            } else {
                self.status_msg = "Session is not waiting for input".into();
            }
        }
    }

    fn handle_brain_accept(&mut self) {
        // Clone session data first to avoid borrow conflict with brain_engine
        let Some(session) = self.selected_session() else {
            return;
        };
        let pid = session.pid;
        let Some(ref mut engine) = self.brain_engine else {
            self.status_msg = "Brain is not enabled".into();
            return;
        };
        // Get suggestion before accept (for logging)
        let suggestion = engine.pending.get(&pid).cloned();
        if suggestion.is_none() {
            self.status_msg = "No brain suggestion pending for this session".into();
            return;
        }
        if let Some(msg) = engine.accept(pid, &session) {
            if let Some(ref sg) = suggestion {
                crate::brain::decisions::log_decision(
                    pid,
                    session.display_name(),
                    session.pending_tool_name.as_deref(),
                    session.pending_tool_input.as_deref(),
                    sg,
                    "accept",
                    Some(&session),
                    crate::brain::decisions::DecisionType::Session,
                );
            }
            crate::logger::log("BRAIN", &format!("Accepted: {msg}"));
            self.status_msg = msg;
        }
    }

    fn handle_brain_reject(&mut self) {
        let Some(session) = self.selected_session() else {
            return;
        };
        let pid = session.pid;
        let Some(ref mut engine) = self.brain_engine else {
            self.status_msg = "Brain is not enabled".into();
            return;
        };
        if let Some(suggestion) = engine.reject(pid) {
            crate::brain::decisions::log_decision(
                pid,
                session.display_name(),
                session.pending_tool_name.as_deref(),
                session.pending_tool_input.as_deref(),
                &suggestion,
                "reject",
                Some(&session),
                crate::brain::decisions::DecisionType::Session,
            );
            let msg = format!(
                "Rejected brain suggestion: {} ({})",
                suggestion.action.label(),
                suggestion.reasoning,
            );
            crate::logger::log("BRAIN", &msg);
            self.status_msg = msg;
        } else {
            self.status_msg = "No brain suggestion pending for this session".into();
        }
    }

    fn toggle_session_recording(&mut self) {
        // If any recordings are active, R stops ALL of them
        if !self.session_recordings.is_empty() {
            let count = self.session_recordings.len();
            let paths: Vec<String> = self.session_recordings.values().cloned().collect();
            self.session_recordings.clear();
            self.status_msg = if count == 1 {
                format!("Recording stopped → {}", paths[0])
            } else {
                format!("{count} recordings stopped")
            };
            return;
        }

        // No recordings active — start recording the selected session
        let info = self
            .selected_session()
            .map(|s| (s.pid, s.display_name().to_string(), s.jsonl_path.is_some()));
        let Some((pid, name, has_jsonl)) = info else {
            return;
        };

        if !has_jsonl {
            self.status_msg = "Cannot record — no JSONL file for this session".into();
            return;
        }
        let path = format!("{}-{}.gif", name, pid);
        self.session_recordings.insert(pid, path.clone());
        self.status_msg = format!("Recording {name} → {path} (R to stop)");
    }

    fn handle_compact(&mut self) {
        if let Some(session) = self.selected_session() {
            match session.status {
                SessionStatus::WaitingInput | SessionStatus::Idle => {
                    match terminals::send_input(&session, "/compact\n") {
                        Ok(()) => {
                            self.status_msg = format!("Sent /compact to {}", session.display_name())
                        }
                        Err(e) => self.status_msg = format!("Compact error: {e}"),
                    }
                }
                SessionStatus::NeedsInput => {
                    self.status_msg =
                        "Cannot compact — session is waiting for permission approval".into();
                }
                SessionStatus::Compacting => {
                    self.status_msg = "Already compacting".into();
                }
                SessionStatus::Processing => {
                    self.status_msg =
                        "Cannot compact — session is processing (wait until idle)".into();
                }
                SessionStatus::Unknown => {
                    self.status_msg =
                        "Cannot compact — transcript telemetry is unavailable for this session"
                            .into();
                }
                SessionStatus::Finished => {
                    self.status_msg = "Cannot compact — session has finished".into();
                }
            }
        }
    }

    fn enter_input_mode(&mut self) {
        let info = self
            .selected_session()
            .map(|s| (s.pid, s.display_name().to_string()));
        if let Some((pid, name)) = info {
            self.input_mode = true;
            self.input_buffer.clear();
            self.input_target_pid = Some(pid);
            self.status_msg = format!("Input to {name} (Enter to send, Esc to cancel): ");
        }
    }

    fn handle_switch_terminal(&mut self) {
        if let Some(session) = self.selected_session() {
            match terminals::switch_to_terminal(&session) {
                Ok(()) => {
                    self.status_msg = format!("Switched to {}", session.display_name());
                }
                Err(e) => {
                    self.status_msg = format!("Error: {e}");
                }
            }
        } else {
            self.status_msg = "No session selected".into();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectGroup {
    pub name: String,
    pub session_count: usize,
    pub active_count: usize,
    pub total_cost: f64,
    pub avg_context_pct: f64,
}

impl App {
    pub fn project_groups(&self) -> Vec<ProjectGroup> {
        self.project_groups_in(&self.data_snapshot())
    }

    /// Groups computed against a caller-supplied snapshot, so
    /// [`Self::ordered_indices`] can rank projects using the very same data it
    /// is ordering — two `data_snapshot()` calls can straddle a refresh and
    /// rank an ordering against a list it was not built from.
    fn project_groups_in(&self, snap: &AppData) -> Vec<ProjectGroup> {
        // Deliberately built from the *unordered* filtered set: `ordered_indices`
        // asks this function for the group order, so going through
        // `visible_sessions()` here would recurse. Group stats don't depend on
        // the order sessions arrive in, so nothing is lost.
        let visible: Vec<&ClaudeSession> = self
            .filtered_indices(snap)
            .into_iter()
            .filter_map(|idx| snap.sessions.get(idx))
            .collect();
        let mut groups: HashMap<String, Vec<&ClaudeSession>> = HashMap::new();
        for s in &visible {
            groups.entry(s.project_name.clone()).or_default().push(s);
        }

        let mut result: Vec<ProjectGroup> = groups
            .into_iter()
            .map(|(name, sessions)| {
                let active_count = sessions
                    .iter()
                    .filter(|s| {
                        matches!(
                            s.status,
                            SessionStatus::Processing | SessionStatus::NeedsInput
                        )
                    })
                    .count();
                let total_cost: f64 = sessions.iter().map(|s| s.cost_usd).sum();
                let avg_context_pct = if sessions.is_empty() {
                    0.0
                } else {
                    sessions.iter().map(|s| s.context_percent()).sum::<f64>()
                        / sessions.len() as f64
                };
                ProjectGroup {
                    name,
                    session_count: sessions.len(),
                    active_count,
                    total_cost,
                    avg_context_pct,
                }
            })
            .collect();

        result.sort_by(|a, b| {
            b.total_cost
                .partial_cmp(&a.total_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }
}

#[cfg(test)]
mod foreign_session_tests {
    use super::*;
    use crate::sandbox_registry::{Registry, SandboxOrigin, SandboxSnapshot, SessionEntry};

    fn entry(session_id: &str, name: &str, pid: u32) -> SessionEntry {
        SessionEntry {
            session_id: session_id.to_string(),
            cwd: "/Users/ndr/repos/linera-infra".to_string(),
            transcript: String::new(),
            started_at_ms: 1_700_000_000_000,
            name: Some(name.to_string()),
            pid: Some(pid),
            owner_pid: None,
            owner_started_at: None,
            ..Default::default()
        }
    }

    fn registry_of(slices: &[(&str, Vec<SessionEntry>)]) -> Registry {
        let mut registry = Registry::default();
        for (name, entries) in slices {
            registry
                .sandboxes
                .insert((*name).to_string(), entries.clone());
        }
        registry
    }

    fn running(names: &[&str]) -> RunningFilter {
        RunningFilter::Known(names.iter().map(|n| (*n).to_string()).collect())
    }

    fn ids(sessions: &[ClaudeSession]) -> Vec<&str> {
        sessions.iter().map(|s| s.session_id.as_str()).collect()
    }

    /// No previous CPU sample — the first tick after startup. Rows still
    /// render; their CPU rate is simply not known yet.
    fn no_prev_cpu() -> std::collections::HashMap<String, crate::cpu::CpuSample> {
        std::collections::HashMap::new()
    }

    #[test]
    fn a_session_renders_from_the_registry_alone() {
        // The reported bug's second half: a session started in a sandbox the
        // collector has not yet visited never appeared at all, because the
        // snapshot was the only membership source and it is rewritten by a
        // 300 s timer. An empty snapshot must not hide a live session.
        let registry = registry_of(&[("linera-agent-5e85", vec![entry("250a74d3", "new", 386)])]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-5e85"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["250a74d3"]);
        assert_eq!(rows[0].display_name(), "new");
    }

    #[test]
    fn a_removed_sandboxs_slice_is_not_rendered_but_is_not_required_to_be_gone() {
        // `sbx rm` fires no hooks, so the slice freezes at its last live state
        // — deliberately, because that frozen copy is what
        // `--restore-sbx-sessions` replays. It must stay on disk and simply
        // not render, which is why the filter lives here and not in the writer.
        let registry = registry_of(&[
            ("linera-agent-dead", vec![entry("f5bb6dba", "reaped", 9302)]),
            ("linera-agent-live", vec![entry("250a74d3", "current", 386)]),
        ]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["250a74d3"]);
        assert_eq!(
            registry.sandboxes["linera-agent-dead"].len(),
            1,
            "the restore payload must survive being filtered out of the view"
        );
    }

    #[test]
    fn negative_control_the_same_slice_renders_once_its_sandbox_is_running() {
        // Proves the test above filters on liveness rather than passing for
        // some unrelated reason: same registry, same row, sandbox now running.
        let registry =
            registry_of(&[("linera-agent-dead", vec![entry("f5bb6dba", "reaped", 9302)])]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-dead"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["f5bb6dba"]);
    }

    #[test]
    fn a_departed_session_stops_rendering_immediately() {
        // The reported bug: closing a terminal window left the row on screen
        // for close to a minute. `SessionEnd` fires at once, but for a
        // non-deliberate reason the entry is deliberately KEPT as
        // `--restore-sbx-sessions` material, so nothing removed it until an
        // unrelated session in the same sandbox happened to fire a hook and
        // trigger the wholesale reconcile.
        let mut departed = entry("f5bb6dba", "closed-terminal", 9302);
        departed.departed_at_ms = Some(1_785_814_692_000);
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![departed, entry("250a74d3", "still-here", 386)],
        )]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["250a74d3"]);
    }

    #[test]
    fn negative_control_the_same_entry_renders_before_it_departs() {
        // Proves the test above keys on the stamp and not on something
        // incidental to the fixture: identical registry, stamp removed.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![
                entry("f5bb6dba", "closed-terminal", 9302),
                entry("250a74d3", "still-here", 386),
            ],
        )]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["f5bb6dba", "250a74d3"]);
    }

    #[test]
    fn a_departed_entry_is_still_restore_material() {
        // The stamp hides the row; it must not remove the data. Deleting it
        // would break restore-after-`sbx rm`, which is the reason these entries
        // are retained in the first place.
        let mut departed = entry("f5bb6dba", "closed-terminal", 9302);
        departed.departed_at_ms = Some(1_785_814_692_000);
        let registry = registry_of(&[("linera-agent-live", vec![departed])]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert!(rows.is_empty(), "hidden from the view");
        assert_eq!(
            registry.sandboxes["linera-agent-live"].len(),
            1,
            "but still on disk for --restore-sbx-sessions"
        );
    }

    /// A sweep in which `sandbox` has exactly `open` attached, taken on a host
    /// where the wrapper's argv still parses — the ordinary case.
    fn ttys(sandbox: &str, open: &[&str]) -> crate::reaper::OpenTerminals {
        crate::reaper::OpenTerminals {
            by_sandbox: std::iter::once((
                sandbox.to_string(),
                open.iter().map(|t| (*t).to_string()).collect(),
            ))
            .collect(),
            wrapper_argv_parsed: true,
        }
    }

    fn with_tty(session_id: &str, name: &str, pid: u32, tty: &str) -> SessionEntry {
        let mut e = entry(session_id, name, pid);
        e.host_tty = Some(tty.to_string());
        e
    }

    #[test]
    fn a_session_whose_host_terminal_is_gone_stops_rendering() {
        // The signal that works WITHOUT deploying anything into the sandbox.
        // `departed_at_ms` is written by claudectl-hook inside the VM, so it is
        // inert until the sandbox image ships a new binary; on a live host all
        // 47 entries across 6 sandboxes were unstamped for exactly that reason.
        // The `sbx exec` carrying SANDBOX_HOST_TTY dies with the window, and
        // the host sees that immediately.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![
                with_tty("gone-1", "closed-window", 1, "/dev/ttys055"),
                with_tty("here-1", "still-open", 2, "/dev/ttys001"),
            ],
        )]);
        let open = ttys("linera-agent-live", &["/dev/ttys001"]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            Some(&open),
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["here-1"]);
    }

    #[test]
    fn negative_control_both_render_while_both_terminals_are_open() {
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![
                with_tty("gone-1", "closed-window", 1, "/dev/ttys055"),
                with_tty("here-1", "still-open", 2, "/dev/ttys001"),
            ],
        )]);
        let open = ttys("linera-agent-live", &["/dev/ttys001", "/dev/ttys055"]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            Some(&open),
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["gone-1", "here-1"]);
    }

    #[test]
    fn the_last_window_of_a_sandbox_closing_retires_its_row() {
        // The topology this has to cover: one session per ephemeral sandbox.
        // When its only window closes, the sandbox's tty set goes EMPTY — so a
        // guard that fails open on emptiness never fires for the common case,
        // and the row falls through to the 300 s collector. Measured on
        // 2026-08-10: a row outlived its session by ~96 s, and only went when an
        // unrelated launch happened to reap the drained sandbox.
        //
        // The sweep having parsed the wrapper's argv *somewhere* on the host is
        // what makes the empty set an answer instead of a suspected mismatch.
        let registry = registry_of(&[(
            "linera-agent-alone",
            vec![with_tty("only-1", "just-exited", 1, "/dev/ttys019")],
        )]);
        let open = ttys("linera-agent-alone", &[]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-alone"]),
            None,
            &[],
            NOW,
            INTERVAL,
            Some(&open),
            &no_prev_cpu(),
        );
        assert!(
            rows.is_empty(),
            "the sandbox's only terminal is gone; nothing should render"
        );
        assert_eq!(
            registry.sandboxes["linera-agent-alone"].len(),
            1,
            "but the entry stays on disk as --restore-sbx-sessions material"
        );
    }

    #[test]
    fn liveness_fails_open_when_the_host_cannot_be_asked() {
        // `None` means "no ps" — in-sandbox claudectl, or a failed call. It must
        // never read as "nothing is alive", which would blank the whole view.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![with_tty("gone-1", "closed-window", 1, "/dev/ttys055")],
        )]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["gone-1"]);
    }

    #[test]
    fn liveness_fails_open_when_the_sweep_parsed_no_wrapper_terminal_at_all() {
        // `ps` ran and matched nothing anywhere on the host. That is what an
        // argv format change looks like from here, so it must not be read as
        // "every session is dead" — the guard the emptiness check was really
        // standing in for, now stated directly.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![with_tty("gone-1", "closed-window", 1, "/dev/ttys055")],
        )]);
        let unparsed = crate::reaper::OpenTerminals {
            wrapper_argv_parsed: false,
            ..ttys("linera-agent-live", &[])
        };
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            Some(&unparsed),
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["gone-1"]);
    }

    #[test]
    fn liveness_fails_open_for_an_entry_with_no_recorded_tty() {
        // Entries written before #41, and sessions not launched by the wrapper,
        // carry no host_tty. There is nothing to match on, so they must render.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![
                entry("no-tty", "older-entry", 1),
                with_tty("here-1", "still-open", 2, "/dev/ttys001"),
            ],
        )]);
        let open = ttys("linera-agent-live", &["/dev/ttys001"]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            Some(&open),
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["no-tty", "here-1"]);
    }

    #[test]
    fn one_sandboxs_terminals_never_judge_anothers() {
        // Two sandboxes can hand out the same tty number. Judging a session
        // against the wrong sandbox's set would hide a live row.
        let registry = registry_of(&[
            (
                "linera-agent-a",
                vec![with_tty("a-1", "in-a", 1, "/dev/ttys001")],
            ),
            (
                "linera-agent-b",
                vec![with_tty("b-1", "in-b", 2, "/dev/ttys001")],
            ),
        ]);
        let mut open = ttys("linera-agent-a", &["/dev/ttys001"]);
        open.by_sandbox.insert(
            "linera-agent-b".to_string(),
            std::iter::once("/dev/ttys009".to_string()).collect(),
        );
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-a", "linera-agent-b"]),
            None,
            &[],
            NOW,
            INTERVAL,
            Some(&open),
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["a-1"], "b-1's window is closed; a-1's is not");
    }

    /// A snapshot in which the collector saw exactly `alive` in `sandbox`.
    fn collected(sandbox: &str, alive: &[&str], collected_at_ms: u64) -> SandboxSnapshot {
        let mut snapshot = SandboxSnapshot {
            collected_at_ms,
            ..Default::default()
        };
        snapshot.sandboxes.insert(
            sandbox.to_string(),
            SandboxOrigin {
                is_current: true,
                sessions: alive
                    .iter()
                    .map(|id| serde_json::json!({ "session_id": id, "cpu": 0.0, "mem_mb": 0.0 }))
                    .collect(),
            },
        );
        snapshot
    }

    /// An entry as an OLD sandbox image writes it: no departure stamp, no
    /// `host_tty`. Both of those fields come from binaries baked into the
    /// image, so on any sandbox that has not been rebuilt this is all the
    /// renderer gets.
    fn old_image_entry(session_id: &str, name: &str, pid: u32, started_at_ms: u64) -> SessionEntry {
        let mut e = entry(session_id, name, pid);
        e.host_tty = None;
        e.host_terminal_id = None;
        e.departed_at_ms = None;
        e.started_at_ms = started_at_ms;
        e
    }

    #[test]
    fn a_dead_session_is_removed_even_when_the_sandbox_image_is_old() {
        // THE regression test for this whole saga. Three fixes shipped and none
        // helped a real machine, because each needed a field written by a
        // binary inside the sandbox: #42's `departed_at_ms` (0 of 46 entries
        // had it) and #45's `host_tty` (21 of 46). This asserts a dead row goes
        // away with NEITHER field present — i.e. using only what the host can
        // observe by itself.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![
                old_image_entry(
                    "dead-1",
                    "sandbox-image-prefetch-watcher",
                    35939,
                    NOW - 600_000,
                ),
                old_image_entry("alive-1", "still-running", 2, NOW - 600_000),
            ],
        )]);
        // The collector probed the sandbox 60s ago and found only `alive-1`.
        let snapshot = collected("linera-agent-live", &["alive-1"], NOW - 60_000);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None, // no host ps either — nothing but the snapshot
            &no_prev_cpu(),
        );
        assert_eq!(
            ids(&rows),
            ["alive-1"],
            "a dead session must disappear without any help from inside the sandbox"
        );
    }

    #[test]
    fn a_session_started_after_the_last_collection_still_appears() {
        // The other half, and the one that makes this dangerous to get wrong:
        // a brand-new session is absent from the snapshot because the collector
        // has not run since it started, NOT because it is dead. Hiding it would
        // trade the removal lag for the appearance lag #36 fixed.
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![old_image_entry("brand-new", "just-started", 3, NOW - 5_000)],
        )]);
        let snapshot = collected("linera-agent-live", &[], NOW - 60_000);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["brand-new"]);
    }

    #[test]
    fn a_stale_snapshot_is_not_evidence_of_death() {
        let registry = registry_of(&[(
            "linera-agent-live",
            vec![old_image_entry("dead-1", "gone", 1, NOW - 3_600_000)],
        )]);
        // Older than two collector intervals ⇒ the collector is presumed dead.
        let snapshot = collected("linera-agent-live", &[], NOW - 3_000_000);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["dead-1"], "stale snapshot ⇒ no opinion");
    }

    #[test]
    fn a_sandbox_the_collector_never_visited_is_not_judged() {
        let registry = registry_of(&[(
            "linera-agent-unvisited",
            vec![old_image_entry(
                "a-1",
                "unknown-to-collector",
                1,
                NOW - 600_000,
            )],
        )]);
        let snapshot = collected("linera-agent-other", &[], NOW - 60_000);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-unvisited"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["a-1"]);
    }

    #[test]
    fn an_entry_with_no_pid_is_not_judged_by_the_collector() {
        // The collector can only report sessions it could probe by pid, so a
        // pidless entry could never appear in the snapshot and its absence
        // proves nothing.
        let mut e = old_image_entry("no-pid", "pidless", 1, NOW - 600_000);
        e.pid = None;
        let registry = registry_of(&[("linera-agent-live", vec![e])]);
        let snapshot = collected("linera-agent-live", &[], NOW - 60_000);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["no-pid"]);
    }

    #[test]
    fn removal_needs_no_field_that_a_sandbox_binary_writes() {
        // Stated as a property rather than a scenario, because "the fix relied
        // on an in-sandbox writer" is the mistake that shipped three times.
        // Every field a sandbox binary populates is cleared here; removal must
        // still work. If a future change reintroduces such a dependency, this
        // fails.
        let mut dead = entry("dead-1", "closed", 9);
        dead.departed_at_ms = None; // written by claudectl-hook (in-sandbox)
        dead.host_tty = None; // written by record_live_sessions (in-sandbox)
        dead.host_terminal_id = None; // ditto
        dead.started_at_ms = NOW - 600_000;
        let registry = registry_of(&[("linera-agent-live", vec![dead])]);
        let rows = foreign_sessions_from(
            &registry,
            &collected("linera-agent-live", &[], NOW - 60_000),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert!(
            rows.is_empty(),
            "the host must be able to retire a dead row on its own"
        );
    }

    #[test]
    fn our_own_slice_is_left_to_native_discovery() {
        let registry = registry_of(&[
            ("linera-agent-here", vec![entry("local-1", "mine", 1)]),
            ("linera-agent-other", vec![entry("remote-1", "theirs", 2)]),
        ]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-here", "linera-agent-other"]),
            Some("linera-agent-here"),
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["remote-1"]);
    }

    #[test]
    fn a_session_resumed_in_a_newer_sandbox_renders_once() {
        // One id legitimately sits in two slices once a session has moved to a
        // rolled sandbox, and both can be running at the same time while the
        // older one drains.
        let registry = registry_of(&[
            ("linera-agent-new", vec![entry("a02a0bcd", "moved", 8234)]),
            ("linera-agent-old", vec![entry("a02a0bcd", "moved", 5181)]),
        ]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-new", "linera-agent-old"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["a02a0bcd"]);
    }

    #[test]
    fn a_locally_discovered_session_is_not_duplicated() {
        let registry = registry_of(&[("linera-agent-live", vec![entry("dup-1", "seen", 7)])]);
        let mut local = ClaudeSession::from_raw(crate::session::RawSession {
            pid: 7,
            session_id: "dup-1".into(),
            cwd: "/tmp".into(),
            started_at: 0,
            name: None,
            name_source: None,
        });
        local.project_name = "tmp".into();
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            std::slice::from_ref(&local),
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert!(rows.is_empty());
    }

    #[test]
    fn vitals_are_overlaid_from_the_snapshot_when_it_has_them() {
        let registry = registry_of(&[("linera-agent-live", vec![entry("abc-123", "row", 11036)])]);
        // Stamped, not defaulted: an unstamped snapshot now reads as expired,
        // which is the point of the freshness check and would make this test
        // assert the wrong thing.
        let snapshot = snapshot_collected_at(NOW - 60_000);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(rows[0].mem_mb, 256.0);
        assert_eq!(
            rows[0].cpu_rate_percent, None,
            "one snapshot is one sample; a rate needs two"
        );
        assert_eq!(
            rows[0]
                .cpu_sample
                .expect("the sample is carried forward")
                .cputime_secs,
            7.5,
            "and it must be retained, or the next tick has nothing to difference"
        );
    }

    #[test]
    fn a_sandbox_sessions_cpu_rate_comes_from_two_consecutive_snapshots() {
        // Foreign rows are rebuilt from the registry every tick and never pass
        // through `merge_discovered_sessions`, so the previous sample reaches
        // them only through the explicit hand-off. Without it every sandbox
        // session's CPU stays unknown forever — and `Processing` would never be
        // claimed for one that genuinely is working.
        let registry = registry_of(&[("linera-agent-live", vec![entry("abc-123", "row", 11036)])]);
        let snapshot = snapshot_collected_at(NOW);
        // 2.5 CPU-seconds consumed over the 5 s between collector passes.
        let prev = std::collections::HashMap::from([(
            "abc-123".to_string(),
            crate::cpu::CpuSample {
                cputime_secs: 5.0,
                sampled_at_ms: NOW - 5_000,
            },
        )]);
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &prev,
        );
        let rate = rows[0].cpu_rate_percent.expect("two samples make a rate");
        assert!((rate - 50.0).abs() < 0.01, "got {rate}");
    }

    #[test]
    fn a_session_the_collector_has_not_measured_yet_still_renders() {
        // The whole point of the split: missing vitals cost a row its CPU and
        // MEM cells, never its place in the list.
        let registry =
            registry_of(&[("linera-agent-live", vec![entry("fresh-1", "brand-new", 42)])]);
        let rows = foreign_sessions_from(
            &registry,
            &SandboxSnapshot::default(),
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["fresh-1"]);
        assert_eq!(
            rows[0].cpu_rate_percent, None,
            "unmeasured reads as unknown, not as an idle 0"
        );
        assert_eq!(rows[0].mem_mb, 0.0);
    }

    #[test]
    fn without_sbx_the_collectors_last_running_set_is_used() {
        // In-sandbox claudectl cannot run `sbx` at all — the host bridges are
        // closed allowlists. Falling back to the collector's observed set keeps
        // the view working there; treating "cannot ask" as "nothing running"
        // would blank it entirely.
        let registry = registry_of(&[
            ("linera-agent-live", vec![entry("seen-1", "known", 1)]),
            ("linera-agent-dead", vec![entry("gone-1", "reaped", 2)]),
        ]);
        let mut snapshot = SandboxSnapshot::default();
        snapshot.sandboxes.insert(
            "linera-agent-live".to_string(),
            SandboxOrigin {
                is_current: true,
                sessions: vec![],
            },
        );
        let rows = foreign_sessions_from(
            &registry,
            &snapshot,
            &running_sandbox_filter_for_test(None, &snapshot),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["seen-1"]);
    }

    const INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);
    const NOW: u64 = 1_785_814_692_000;

    fn snapshot_collected_at(collected_at_ms: u64) -> SandboxSnapshot {
        let mut snapshot = SandboxSnapshot {
            collected_at_ms,
            ..Default::default()
        };
        snapshot.sandboxes.insert(
            "linera-agent-live".to_string(),
            SandboxOrigin {
                is_current: true,
                sessions: vec![serde_json::json!({
                    "session_id": "abc-123", "cputime_secs": 7.5, "mem_mb": 256.0,
                })],
            },
        );
        snapshot
    }

    #[test]
    fn a_fresh_snapshot_supplies_vitals() {
        let snapshot = snapshot_collected_at(NOW - 60_000);
        let vitals = snapshot_vitals_at(&snapshot, NOW, INTERVAL);
        assert_eq!(vitals["abc-123"].cpu_sample.unwrap().cputime_secs, 7.5);
    }

    #[test]
    fn an_expired_snapshot_supplies_none() {
        // Past two intervals the collector is presumed dead. CPU and memory are
        // instantaneous samples: showing 11-minute-old ones as current is a
        // confident lie, and a blank cell is the honest answer.
        let snapshot = snapshot_collected_at(NOW - 11 * 60 * 1000);
        assert!(snapshot_vitals_at(&snapshot, NOW, INTERVAL).is_empty());
    }

    #[test]
    fn one_missed_tick_is_not_treated_as_a_dead_collector() {
        // 6 minutes on a 5-minute reaper is a single skipped fire — a slow
        // `sbx exec` or a sleeping laptop. Expiring here would flap the vitals
        // off and on for a healthy fleet.
        let snapshot = snapshot_collected_at(NOW - 6 * 60 * 1000);
        assert!(!snapshot_vitals_at(&snapshot, NOW, INTERVAL).is_empty());
    }

    #[test]
    fn a_snapshot_with_no_collection_time_is_never_trusted() {
        // `collected_at_ms == 0` means a writer that predates the field, or a
        // default-constructed value. Unknown age must read as expired, never
        // as "fresh" — the zero would otherwise compute an enormous age or,
        // worse, be mistaken for "just collected".
        let snapshot = snapshot_collected_at(0);
        assert!(snapshot_vitals_at(&snapshot, NOW, INTERVAL).is_empty());
        assert_eq!(snapshot.age(NOW), None);
    }

    #[test]
    fn the_bound_scales_with_the_configured_collector_interval() {
        // A host running `--reaper-interval 1800` must not have every snapshot
        // judged expired against the 5-minute default.
        let snapshot = snapshot_collected_at(NOW - 20 * 60 * 1000);
        assert!(
            snapshot_vitals_at(&snapshot, NOW, INTERVAL).is_empty(),
            "20 min is expired on a 5-minute reaper"
        );
        assert!(
            !snapshot_vitals_at(&snapshot, NOW, std::time::Duration::from_secs(1800)).is_empty(),
            "the same age is fine on a 30-minute reaper"
        );
    }

    #[test]
    fn a_clock_that_went_backwards_does_not_read_as_fresh() {
        // `saturating_sub` floors the age at zero for a snapshot stamped in the
        // future (NTP step, or a sandbox clock ahead of the host). Age zero is
        // "fresh", which is the safe direction: it shows real measurements
        // rather than blanking a working fleet.
        let snapshot = snapshot_collected_at(NOW + 60_000);
        assert_eq!(snapshot.age(NOW), Some(std::time::Duration::ZERO));
        assert!(!snapshot_vitals_at(&snapshot, NOW, INTERVAL).is_empty());
    }

    #[test]
    fn membership_is_unaffected_by_an_expired_snapshot() {
        // The whole point of the split: staleness costs a row its CPU and MEM
        // cells, never its place in the list. Membership comes from the
        // registry and does not depend on the collector being alive at all.
        let registry = registry_of(&[("linera-agent-live", vec![entry("abc-123", "row", 11036)])]);
        let stale = snapshot_collected_at(NOW - 60 * 60 * 1000);
        let rows = foreign_sessions_from(
            &registry,
            &stale,
            &running(&["linera-agent-live"]),
            None,
            &[],
            NOW,
            INTERVAL,
            None,
            &no_prev_cpu(),
        );
        assert_eq!(ids(&rows), ["abc-123"]);
        assert_eq!(
            rows[0].cpu_rate_percent, None,
            "stale vitals must not be shown"
        );
    }

    /// `running_sandbox_filter` with the `sbx` call's result injected.
    fn running_sandbox_filter_for_test(
        sbx_said: Option<Vec<String>>,
        snapshot: &SandboxSnapshot,
    ) -> RunningFilter {
        match sbx_said {
            Some(names) => RunningFilter::Known(names.into_iter().collect()),
            None => RunningFilter::Collected(snapshot.sandboxes.keys().cloned().collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{RawSession, TelemetryStatus};

    fn make_session(
        pid: u32,
        project: &str,
        model: &str,
        status: SessionStatus,
        cost_usd: f64,
        context_pct: f64,
        telemetry_available: bool,
    ) -> ClaudeSession {
        let raw = RawSession {
            pid,
            session_id: format!("session-{pid}"),
            cwd: format!("/tmp/{project}"),
            started_at: 0,
            name: None,
            name_source: None,
        };
        let mut session = ClaudeSession::from_raw(raw);
        session.project_name = project.to_string();
        session.model = model.to_string();
        session.status = status;
        session.cost_usd = cost_usd;
        session.context_max = 100;
        session.context_tokens = context_pct as u64;
        session.telemetry_status = if telemetry_available {
            TelemetryStatus::Available
        } else {
            TelemetryStatus::MissingTranscript
        };
        session.usage_metrics_available = telemetry_available;
        session
    }

    fn make_test_app() -> App {
        let mut app = App::new();
        let sessions = vec![
            make_session(
                11,
                "blocked-api",
                "sonnet-4.6",
                SessionStatus::NeedsInput,
                2.0,
                40.0,
                true,
            ),
            make_session(
                12,
                "hot-cost",
                "opus-4.6",
                SessionStatus::Processing,
                7.5,
                30.0,
                true,
            ),
            make_session(
                13,
                "high-context",
                "haiku",
                SessionStatus::WaitingInput,
                1.0,
                90.0,
                true,
            ),
            make_session(
                14,
                "unknown-metrics",
                "",
                SessionStatus::Unknown,
                0.0,
                0.0,
                false,
            ),
        ];
        app.replace_data(AppData {
            sessions,
            ..AppData::default()
        });
        app.budget_usd = Some(5.0);
        app.context_warn_threshold = 75;
        app.conflict_pids.insert(13);
        app.normalize_selection();
        app
    }

    /// Build an App with empty AppData, bypassing the host-side
    /// session discovery that `App::new()` does in its constructor.
    /// The data-helper tests below assert on the post-replace state
    /// directly, so we don't want real sessions leaking in.
    fn app_with_empty_data() -> App {
        let app = App::new();
        app.replace_data(AppData::default());
        app
    }

    #[test]
    fn data_snapshot_returns_empty_appdata_after_clear() {
        let app = app_with_empty_data();
        let snap = app.data_snapshot();
        assert!(snap.sessions.is_empty());
        assert_eq!(snap.ledger_today.msg_count, 0);
        assert_eq!(snap.ledger_week.msg_count, 0);
        assert_eq!(snap.ledger_month.msg_count, 0);
    }

    #[test]
    fn replace_data_swaps_atomically_and_takes_effect_for_next_snapshot() {
        let app = app_with_empty_data();
        let snap_before = app.data_snapshot();
        assert!(snap_before.sessions.is_empty());

        let mut new_data = AppData::default();
        new_data.sessions.push(make_session(
            42,
            "regression-app",
            "sonnet",
            SessionStatus::Idle,
            0.5,
            10.0,
            true,
        ));
        new_data.ledger_today.msg_count = 7;
        app.replace_data(new_data);

        let snap_after = app.data_snapshot();
        assert_eq!(snap_after.sessions.len(), 1);
        assert_eq!(snap_after.sessions[0].pid, 42);
        assert_eq!(snap_after.ledger_today.msg_count, 7);

        // The previous snapshot must NOT see the new data — it captured a
        // distinct Arc that is still alive via `snap_before`. This is the
        // whole point of `Arc<RwLock<Arc<AppData>>>`: readers hold an
        // immutable owned snapshot that survives the next swap.
        assert!(snap_before.sessions.is_empty());
        assert_eq!(snap_before.ledger_today.msg_count, 0);
    }

    #[test]
    fn modify_data_mutates_in_place_and_visible_to_subsequent_snapshot() {
        let app = app_with_empty_data();
        app.modify_data(|d| {
            d.ledger_week.fresh_input = 12345;
            d.sessions.push(make_session(
                99,
                "x",
                "sonnet",
                SessionStatus::Idle,
                0.0,
                0.0,
                false,
            ));
        });

        let snap = app.data_snapshot();
        assert_eq!(snap.ledger_week.fresh_input, 12345);
        assert_eq!(snap.sessions.len(), 1);
        assert_eq!(snap.sessions[0].pid, 99);
    }

    /// Perf regression: `do_refresh_io` against an empty session list
    /// should be near-instant. Before commit bf9f59b0 the per-PID
    /// terminal-sidecar read happened on every tick; this test
    /// indirectly catches if that path regresses to per-tick I/O for
    /// already-cached sidecars (in which case 1000 idle calls would
    /// blow past the 5 s bound).
    ///
    /// The bound is intentionally loose because the real I/O cost
    /// depends on the host's `~/.claude/sessions` size — we're checking
    /// "doesn't go quadratic", not "stays under N ms". Calibrated on
    /// the same sandbox where one cold call is ~1.5 s and warm calls
    /// are ~30 ms.
    #[test]
    fn perf_do_refresh_io_does_not_explode_with_repeated_calls() {
        let mut prev = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let out = do_refresh_io(prev);
            prev = out.sessions;
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 30,
            "do_refresh_io × 3 took {} ms (regression — expect <5 s \
             cold + sub-second warm even on a heavy box)",
            elapsed.as_millis()
        );
    }

    #[test]
    fn refresh_nonblocking_falls_back_to_sync_when_no_runtime() {
        // No tokio runtime is active in unit tests by default — the
        // function must therefore run `do_refresh_io` synchronously
        // and return `true` (work was applied) with refresh_in_flight
        // cleared, rather than dispatching to a worker that never runs.
        let mut app = app_with_empty_data();
        assert!(app.refresh_nonblocking());
        assert!(!app.refresh_in_flight);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refresh_nonblocking_kicks_worker_under_runtime() {
        // With a multi-threaded tokio runtime the first call kicks
        // a `do_refresh_io` worker onto the blocking pool. Subsequent
        // calls return false until the worker sends its result back
        // through the channel; once the result is applied,
        // refresh_nonblocking returns true. We poll on the return
        // value rather than `refresh_in_flight` because each call
        // also schedules the NEXT worker, so the flag is true on
        // exit even after a successful drain.
        //
        // Note: `do_refresh_io` reads the real host filesystem
        // (~/.claude/sessions, ~/.claude/projects). On a heavy box
        // the cold pass can take seconds; the 60 s deadline is
        // generous to keep this test reliable on slow CI/sandboxes.
        let mut app = app_with_empty_data();
        let kicked = app.refresh_nonblocking();
        assert!(!kicked, "first call only schedules; nothing applied yet");
        assert!(app.refresh_in_flight, "worker must be scheduled");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            tokio::task::yield_now().await;
            if app.refresh_nonblocking() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!("refresh worker did not complete within 60 s");
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    #[test]
    fn with_sessions_preserves_ledger_fields() {
        let app = app_with_empty_data();
        app.modify_data(|d| {
            d.ledger_month.cost_usd = 999.99;
        });
        // Sort/reorder mutation: must not zero out the ledger.
        app.with_sessions(|s| {
            s.push(make_session(
                1,
                "a",
                "haiku",
                SessionStatus::Idle,
                0.0,
                0.0,
                false,
            ));
        });
        let snap = app.data_snapshot();
        assert!((snap.ledger_month.cost_usd - 999.99).abs() < 1e-9);
        assert_eq!(snap.sessions.len(), 1);
    }

    #[test]
    fn status_filter_returns_only_matching_sessions() {
        let mut app = make_test_app();
        app.status_filter = StatusFilter::NeedsInput;
        let visible: Vec<u32> = app.visible_sessions().iter().map(|s| s.pid).collect();
        assert_eq!(visible, vec![11]);
    }

    #[test]
    fn focus_filter_attention_matches_high_signal_sessions() {
        let mut app = make_test_app();
        app.focus_filter = FocusFilter::Attention;
        let visible: Vec<u32> = app.visible_sessions().iter().map(|s| s.pid).collect();
        assert_eq!(visible, vec![11, 12, 13, 14]);
    }

    #[test]
    fn search_query_matches_project_and_model() {
        let mut app = make_test_app();
        app.search_query = "sonnet".into();
        let visible: Vec<u32> = app.visible_sessions().iter().map(|s| s.pid).collect();
        assert_eq!(visible, vec![11]);

        app.search_query = "unknown-metrics".into();
        let visible: Vec<u32> = app.visible_sessions().iter().map(|s| s.pid).collect();
        assert_eq!(visible, vec![14]);
    }

    #[test]
    fn normalize_selection_clamps_to_filtered_session_count() {
        let mut app = make_test_app();
        app.table_state.select(Some(3));
        app.status_filter = StatusFilter::NeedsInput;
        app.normalize_selection();
        assert_eq!(app.table_state.selected(), Some(0));
        assert_eq!(app.selected_session().map(|s| s.pid), Some(11));
    }

    #[test]
    fn launch_wizard_starts_with_cli_defaults() {
        let mut app = App::new();
        app.enter_launch_mode();

        assert!(app.launch_mode);
        assert_eq!(app.launch_form.field, LaunchField::Cwd);
        assert_eq!(app.launch_form.cwd, ".");
        assert!(app.launch_form.prompt.is_empty());
        assert!(app.launch_form.resume.is_empty());
    }

    #[test]
    fn launch_wizard_moves_between_fields() {
        let mut app = App::new();
        app.enter_launch_mode();

        app.handle_launch_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.launch_form.field, LaunchField::Prompt);

        app.handle_launch_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.launch_form.field, LaunchField::Resume);

        app.handle_launch_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
        assert_eq!(app.launch_form.field, LaunchField::Prompt);
    }

    #[test]
    fn invalid_launch_keeps_wizard_open_and_reports_error() {
        let mut app = App::new();
        app.enter_launch_mode();
        app.launch_form.cwd = "/tmp/claudectl-this-path-should-not-exist".into();
        app.launch_form.field = LaunchField::Resume;

        app.submit_launch_form();

        assert!(app.launch_mode);
        assert_eq!(app.launch_form.field, LaunchField::Cwd);
        assert!(
            app.status_msg
                .starts_with("Launch failed: Directory not found:")
        );
    }

    // ------------------------------------------------------------------
    // apply_sort coverage
    // ------------------------------------------------------------------

    fn named_session(pid: u32, name: &str, project: &str, last_user_ms: u64) -> ClaudeSession {
        let raw = RawSession {
            pid,
            session_id: format!("s-{pid}"),
            cwd: format!("/tmp/{project}"),
            started_at: 0,
            name: None,
            name_source: None,
        };
        let mut s = ClaudeSession::from_raw(raw);
        s.project_name = project.into();
        s.session_name = name.into();
        s.last_user_message_ts = last_user_ms;
        s
    }

    #[test]
    fn apply_sort_by_name_puts_unnamed_last_then_alpha_tiebreak_by_project() {
        let mut app = App::new();
        app.sort_column = 6; // Name
        let mut sessions = vec![
            named_session(1, "", "zeta", 0),
            named_session(2, "Beta-feature", "alpha", 0),
            named_session(3, "alpha-feature", "gamma", 0),
            named_session(4, "", "alpha", 0),
            named_session(5, "alpha-feature", "beta", 0),
        ];
        app.apply_sort(&mut sessions);
        let order: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        // Sort key is (is_empty, lowercased session_name, lowercased project_name):
        //   pid 5: (false, "alpha-feature", "beta")
        //   pid 3: (false, "alpha-feature", "gamma")
        //   pid 2: (false, "beta-feature",  "alpha")
        //   pid 4: (true,  "",              "alpha")  — unnamed, project tiebreak
        //   pid 1: (true,  "",              "zeta")
        assert_eq!(order, vec![5, 3, 2, 4, 1]);
    }

    #[test]
    fn apply_sort_by_last_puts_most_recent_first_with_never_at_bottom() {
        let mut app = App::new();
        app.sort_column = 5; // Last
        let mut sessions = vec![
            named_session(1, "a", "p", 1_000),
            named_session(2, "b", "p", 5_000),
            named_session(3, "c", "p", 0), // never
            named_session(4, "d", "p", 3_000),
        ];
        app.apply_sort(&mut sessions);
        let order: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(order, vec![2, 4, 1, 3]);
    }

    #[test]
    fn apply_sort_by_cost_descending() {
        let mut app = App::new();
        app.sort_column = 2; // Cost
        let mut sessions = vec![
            make_session(1, "a", "m", SessionStatus::Processing, 0.5, 0.0, true),
            make_session(2, "b", "m", SessionStatus::Processing, 10.0, 0.0, true),
            make_session(3, "c", "m", SessionStatus::Processing, 3.0, 0.0, true),
        ];
        app.apply_sort(&mut sessions);
        let order: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(order, vec![2, 3, 1]);
    }

    #[test]
    fn apply_sort_by_elapsed_longest_first() {
        use std::time::Duration;
        let mut app = App::new();
        app.sort_column = 4; // Elapsed
        let mut sessions = vec![
            make_session(1, "a", "m", SessionStatus::Processing, 0.0, 0.0, true),
            make_session(2, "b", "m", SessionStatus::Processing, 0.0, 0.0, true),
            make_session(3, "c", "m", SessionStatus::Processing, 0.0, 0.0, true),
        ];
        sessions[0].elapsed = Duration::from_secs(30);
        sessions[1].elapsed = Duration::from_secs(300);
        sessions[2].elapsed = Duration::from_secs(90);
        app.apply_sort(&mut sessions);
        let order: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(order, vec![2, 3, 1]);
    }

    // ------------------------------------------------------------------
    // Parking coverage
    // ------------------------------------------------------------------

    // Note: there's no "parked_set_defaults_to_empty" test because
    // `App::new()` loads from `~/.claudectl/parked.json`, so the "default"
    // depends on the user's environment. The empty-case contract is covered
    // by `load_parked_from_missing_file_returns_empty` below.

    #[test]
    fn toggle_park_adds_session_id() {
        let mut app = App::new();
        app.toggle_park("abc-123");
        assert!(app.is_parked("abc-123"));
    }

    #[test]
    fn toggle_park_removes_already_parked_session() {
        let mut app = App::new();
        app.toggle_park("abc-123");
        app.toggle_park("abc-123");
        assert!(!app.is_parked("abc-123"));
    }

    #[test]
    fn save_and_load_parked_roundtrip() {
        use std::collections::HashSet;
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_path_buf();
        let to_save: HashSet<String> = ["s1".into(), "s2".into(), "s3".into()]
            .into_iter()
            .collect();
        save_parked_to(&path, &to_save);
        let loaded = load_parked_from(&path);
        assert_eq!(loaded, to_save);
    }

    #[test]
    fn load_parked_from_missing_file_returns_empty() {
        let loaded = load_parked_from(std::path::Path::new("/nonexistent/parked.json"));
        assert!(loaded.is_empty());
    }

    #[test]
    fn load_parked_from_malformed_file_returns_empty() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        use std::io::Write;
        writeln!(tmp, "this is not json").unwrap();
        let loaded = load_parked_from(tmp.path());
        assert!(loaded.is_empty());
    }

    #[test]
    fn apply_sort_puts_parked_at_end_regardless_of_column() {
        let mut app = App::new();
        app.sort_column = 2; // Cost
        app.parked.insert("s-99".into());
        let mut sessions = vec![
            make_session(1, "a", "m", SessionStatus::Processing, 0.5, 0.0, true),
            make_session(99, "b", "m", SessionStatus::Processing, 10.0, 0.0, true), // highest cost
            make_session(3, "c", "m", SessionStatus::Processing, 3.0, 0.0, true),
        ];
        // override session_ids (make_session uses format!("session-{pid}"))
        sessions[1].session_id = "s-99".into();
        app.apply_sort(&mut sessions);
        let order: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(
            order,
            vec![3, 1, 99],
            "Highest-cost session 99 is parked, so it drops below non-parked sessions"
        );
    }

    #[test]
    fn apply_sort_parked_at_end_even_when_reversed() {
        let mut app = App::new();
        app.sort_column = 2; // Cost
        app.sort_reversed = true;
        app.parked.insert("s-99".into());
        let mut sessions = vec![
            make_session(1, "a", "m", SessionStatus::Processing, 0.5, 0.0, true),
            make_session(99, "b", "m", SessionStatus::Processing, 10.0, 0.0, true),
            make_session(3, "c", "m", SessionStatus::Processing, 3.0, 0.0, true),
        ];
        sessions[1].session_id = "s-99".into();
        app.apply_sort(&mut sessions);
        let order: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        // Non-parked reverse-cost: 1 (0.5), 3 (3.0). Parked 99 still at end.
        assert_eq!(order, vec![1, 3, 99]);
    }

    #[test]
    fn toggle_park_selected_parks_the_highlighted_session() {
        // Use toggle_park_selected directly to avoid the disk side-effect
        // of the key handler's save_parked() call. The pure-logic toggle
        // is what we care about here.
        let mut app = make_test_app();
        app.parked.clear(); // isolate from any disk-loaded state
        app.toggle_park_selected();
        assert!(app.is_parked("session-11"));
    }

    #[test]
    fn toggle_park_selected_unparks_when_re_selected() {
        // After parking, the row moves to the parked section at the bottom.
        // Re-select that row, toggle again, and it should unpark.
        let mut app = make_test_app();
        app.parked.clear();
        app.toggle_park_selected();
        assert!(app.is_parked("session-11"));

        // Find PID 11's new index (it's now in the parked section).
        let idx = app
            .data_snapshot()
            .sessions
            .iter()
            .position(|s| s.pid == 11)
            .expect("PID 11 must still be present after parking");
        app.table_state.select(Some(idx));

        app.toggle_park_selected();
        assert!(!app.is_parked("session-11"));
    }

    // ------------------------------------------------------------------
    // Sort direction toggle coverage (S key)
    // ------------------------------------------------------------------

    #[test]
    fn sort_reversed_defaults_to_false() {
        let app = App::new();
        assert!(!app.sort_reversed);
    }

    #[test]
    fn apply_sort_honors_sort_reversed_flag() {
        let mut app = App::new();
        app.sort_column = 5; // Last: default = most recent first
        let mut sessions = vec![
            named_session(1, "a", "p", 1_000),
            named_session(2, "b", "p", 5_000),
            named_session(3, "c", "p", 3_000),
        ];
        app.apply_sort(&mut sessions);
        let natural: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(natural, vec![2, 3, 1]);

        app.sort_reversed = true;
        let mut sessions = vec![
            named_session(1, "a", "p", 1_000),
            named_session(2, "b", "p", 5_000),
            named_session(3, "c", "p", 3_000),
        ];
        app.apply_sort(&mut sessions);
        let reversed: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(reversed, vec![1, 3, 2]);
    }

    #[test]
    fn toggle_sort_direction_flips_flag_and_sets_status() {
        let mut app = make_test_app();
        app.sort_column = 2; // Cost
        assert!(!app.sort_reversed);
        app.toggle_sort_direction();
        assert!(app.sort_reversed);
        assert!(
            app.status_msg.contains("Cost") && app.status_msg.contains("reversed"),
            "status should mention column and direction, got: {}",
            app.status_msg
        );
        app.toggle_sort_direction();
        assert!(!app.sort_reversed);
        assert!(!app.status_msg.contains("reversed"));
    }

    #[test]
    fn cycle_sort_resets_sort_reversed() {
        let mut app = make_test_app();
        app.sort_reversed = true;
        app.cycle_sort();
        assert!(
            !app.sort_reversed,
            "cycling to a new column should reset direction to natural"
        );
    }

    #[test]
    fn capital_s_toggles_sort_direction() {
        let mut app = make_test_app();
        assert!(!app.sort_reversed);
        app.handle_normal_key(KeyEvent::new(KeyCode::Char('S'), KeyModifiers::SHIFT));
        assert!(app.sort_reversed);
    }

    // ------------------------------------------------------------------
    // handle_normal_key Esc unwind coverage
    // ------------------------------------------------------------------

    fn press_esc(app: &mut App) {
        app.handle_normal_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    }

    #[test]
    fn esc_with_no_state_is_noop_not_quit() {
        let mut app = make_test_app();
        press_esc(&mut app);
        assert!(!app.should_quit, "Esc must not quit the TUI in normal mode");
    }

    #[test]
    fn esc_cancels_pending_kill() {
        let mut app = make_test_app();
        app.pending_kill = Some(11);
        press_esc(&mut app);
        assert!(app.pending_kill.is_none());
        assert_eq!(app.status_msg, "Kill cancelled");
        assert!(!app.should_quit);
    }

    #[test]
    fn esc_cancels_pending_auto_approve() {
        let mut app = make_test_app();
        app.pending_auto_approve = Some(11);
        press_esc(&mut app);
        assert!(app.pending_auto_approve.is_none());
        assert_eq!(app.status_msg, "Auto-approve cancelled");
    }

    #[test]
    fn esc_closes_open_detail_panel() {
        let mut app = make_test_app();
        app.detail_panel = true;
        press_esc(&mut app);
        assert!(!app.detail_panel);
    }

    #[test]
    fn esc_clears_committed_search_query() {
        let mut app = make_test_app();
        app.search_query = "needle".into();
        press_esc(&mut app);
        assert_eq!(app.search_query, "");
        assert_eq!(app.status_msg, "Search cleared");
    }

    #[test]
    fn esc_clears_active_filters() {
        let mut app = make_test_app();
        app.status_filter = StatusFilter::NeedsInput;
        app.focus_filter = FocusFilter::Attention;
        press_esc(&mut app);
        assert!(matches!(app.status_filter, StatusFilter::All));
        assert!(matches!(app.focus_filter, FocusFilter::All));
        assert_eq!(app.status_msg, "Filters cleared");
    }

    #[test]
    fn esc_unwinds_one_layer_per_press() {
        // Set up three layers: search query, filter, detail panel.
        // Priority order: detail panel > search > filters.
        let mut app = make_test_app();
        app.detail_panel = true;
        app.search_query = "x".into();
        app.status_filter = StatusFilter::NeedsInput;

        press_esc(&mut app);
        assert!(!app.detail_panel, "first Esc closes detail panel");
        assert_eq!(app.search_query, "x");
        assert!(matches!(app.status_filter, StatusFilter::NeedsInput));

        press_esc(&mut app);
        assert_eq!(app.search_query, "", "second Esc clears search");
        assert!(matches!(app.status_filter, StatusFilter::NeedsInput));

        press_esc(&mut app);
        assert!(
            matches!(app.status_filter, StatusFilter::All),
            "third Esc clears filters"
        );

        press_esc(&mut app);
        assert!(!app.should_quit, "fourth Esc on empty state is no-op");
    }

    #[test]
    fn q_still_quits_in_normal_mode() {
        let mut app = make_test_app();
        app.handle_normal_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(app.should_quit);
    }

    #[test]
    fn ctrl_c_still_quits_in_normal_mode() {
        let mut app = make_test_app();
        app.handle_normal_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }

    // ------------------------------------------------------------------
    // merge_discovered_sessions coverage
    // ------------------------------------------------------------------

    #[test]
    fn merge_inserts_brand_new_pids() {
        let (merged, new_pids) = merge_discovered_sessions(
            vec![],
            vec![
                named_session(100, "a", "proj", 0),
                named_session(101, "b", "proj", 0),
            ],
        );
        let pids: Vec<u32> = merged.iter().map(|s| s.pid).collect();
        assert_eq!(pids, vec![100, 101]);
        assert_eq!(new_pids, vec![100, 101]);
    }

    #[test]
    fn merge_preserves_accumulated_state_for_existing_pid() {
        // The existing session carries accumulated cost and tokens; the
        // newly-discovered copy has zero values (fresh parse). After merge,
        // the accumulated state must survive.
        let mut existing = named_session(42, "old-name", "proj", 10_000);
        existing.cost_usd = 3.50;
        existing.own_input_tokens = 1_000;
        existing.jsonl_offset = 4096;

        let fresh = named_session(42, "old-name", "proj", 0); // discovery scan has no accumulated state
        let (merged, new_pids) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(new_pids, vec![] as Vec<u32>);
        assert_eq!(merged.len(), 1);
        let s = &merged[0];
        assert_eq!(s.cost_usd, 3.50, "cost must be preserved across merge");
        assert_eq!(s.own_input_tokens, 1_000, "tokens preserved");
        assert_eq!(s.jsonl_offset, 4096, "JSONL offset preserved");
        assert_eq!(
            s.last_user_message_ts, 10_000,
            "transcript-derived timestamps preserved"
        );
    }

    #[test]
    fn merge_backfills_empty_cwd_from_later_discovery() {
        // A row born in a first-tick race can have an empty cwd; once a later
        // scan knows it, the merge must adopt it (cwd drives transcript
        // resolution and the terminal-switch tab lookup) — while a non-empty
        // cwd must never be clobbered.
        let mut existing = named_session(42, "s", "proj", 0);
        existing.cwd = String::new();
        existing.project_name = String::new();

        let mut fresh = named_session(42, "s", "proj", 0);
        fresh.cwd = "/Users/ndr/work".into();
        fresh.project_name = "work".into();

        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(merged[0].cwd, "/Users/ndr/work", "empty cwd backfilled");
        assert_eq!(merged[0].project_name, "work");

        let mut known = named_session(7, "s", "proj", 0);
        known.cwd = "/Users/ndr/known".into();
        let mut empty = named_session(7, "s", "proj", 0);
        empty.cwd = String::new();
        let (merged, _) = merge_discovered_sessions(vec![known], vec![empty]);
        assert_eq!(
            merged[0].cwd, "/Users/ndr/known",
            "known cwd never clobbered by an empty one"
        );
    }

    #[test]
    fn merge_keeps_known_name_and_id_over_empty_ps_backstop_row() {
        // A ps-backstop row (pointer + registry both lost) carries no name and
        // possibly no session_id. Merging it must not erase the identity the
        // TUI already holds, nor reset the JSONL offset (which would
        // double-count the transcript on re-read).
        let mut existing = named_session(42, "known-name", "proj", 1_000);
        existing.session_id = "known-id".into();
        existing.jsonl_path = Some(std::path::PathBuf::from("/tmp/known-id.jsonl"));
        existing.jsonl_offset = 4096;

        let mut fresh = named_session(42, "", "proj", 0);
        fresh.session_id = String::new();
        fresh.jsonl_path = None;

        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(merged.len(), 1);
        let s = &merged[0];
        assert_eq!(s.session_name, "known-name", "empty name must not clobber");
        assert_eq!(s.session_id, "known-id", "empty id is not a rotation");
        assert_eq!(
            s.jsonl_offset, 4096,
            "identity kept, so accumulated offset kept"
        );
    }

    #[test]
    fn merge_picks_up_renamed_session_name_from_discovery() {
        // Core regression test for commit 4: Claude Code's /rename rewrites
        // the session JSON mid-run. `scan_sessions` produces a fresh copy
        // with the new name; the merge must overwrite the previous name so
        // the TUI reflects the rename without a restart.
        let existing = named_session(42, "original-name", "proj", 0);
        let mut fresh = named_session(42, "renamed-session", "proj", 0);
        // Simulate discovery: only ephemeral fields differ.
        fresh.cost_usd = 0.0;

        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(merged[0].session_name, "renamed-session");
    }

    #[test]
    fn regression_explicit_title_outranks_scan_name_in_merge() {
        // 2026-07-28 rename-revert: the registry re-supplied a stale name
        // every tick; a title recovered from the transcript's custom-title
        // (name_is_explicit) must win against any scan-supplied name.
        let mut existing = named_session(42, "explicit-title", "proj", 0);
        existing.name_is_explicit = true;
        let fresh = named_session(42, "stale-registry-name", "proj", 0);
        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(merged[0].session_name, "explicit-title");
        assert!(merged[0].name_is_explicit, "flag must persist across ticks");
    }

    #[test]
    fn regression_rotation_releases_the_explicit_title() {
        // A sessionId rotation under the same pid (/clear, compaction,
        // recycled pid) is a NEW conversation: the old transcript's /rename
        // title must not stick to it — held, it would permanently block the
        // new identity's own names (only a fresh /rename could ever fix it).
        let mut existing = named_session(42, "old-explicit-title", "proj", 0);
        existing.session_id = "session-a".into();
        existing.name_is_explicit = true;
        let mut fresh = named_session(42, "new-conversation-name", "proj", 0);
        fresh.session_id = "session-b".into();
        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(merged[0].session_name, "new-conversation-name");
        assert!(
            !merged[0].name_is_explicit,
            "the explicit hold must not survive a rotation"
        );
        assert_eq!(merged[0].session_id, "session-b");
    }

    #[test]
    fn merge_drops_existing_when_pid_no_longer_in_discovery() {
        let existing_a = named_session(1, "a", "proj", 0);
        let existing_b = named_session(2, "b", "proj", 0);
        let (merged, new_pids) = merge_discovered_sessions(
            vec![existing_a, existing_b],
            vec![named_session(1, "a", "proj", 0)],
        );
        let pids: Vec<u32> = merged.iter().map(|s| s.pid).collect();
        assert_eq!(pids, vec![1], "PID 2 is gone, so it must drop from output");
        assert_eq!(new_pids, vec![] as Vec<u32>);
    }

    #[test]
    fn merge_adopts_new_session_id_and_clears_jsonl_path_on_rotation() {
        // Repro for the "Last never updates" bug. Claude Code keeps the
        // same OS PID but rotates `sessionId` in ~/.claude/sessions/<PID>.json
        // (/clear, compaction, resume-into-new-file). Without the fix,
        // merge preserves prev.session_id + prev.jsonl_path, and
        // `do_refresh_io` skips re-resolution because jsonl_path.is_some(),
        // so claudectl keeps reading the abandoned transcript forever.
        let mut existing = named_session(42, "session", "proj", 1_000);
        existing.session_id = "old-session-id".into();
        existing.jsonl_path = Some(std::path::PathBuf::from("/tmp/old-session-id.jsonl"));
        existing.jsonl_offset = 4096;

        let mut fresh = named_session(42, "session", "proj", 0);
        fresh.session_id = "new-session-id".into();
        fresh.jsonl_path = None;
        fresh.jsonl_offset = 0;

        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        assert_eq!(merged.len(), 1);
        let s = &merged[0];
        assert_eq!(
            s.session_id, "new-session-id",
            "merge must adopt the freshly-discovered session_id when it rotates"
        );
        assert!(
            s.jsonl_path.is_none(),
            "stale jsonl_path must be cleared so resolve_jsonl_paths re-runs"
        );
        assert_eq!(
            s.jsonl_offset, 0,
            "jsonl_offset must reset since we'll be reading a new file from byte 0"
        );
    }

    #[test]
    fn merge_preserves_jsonl_path_when_session_id_unchanged() {
        // Counter-test: when session_id matches across discovery,
        // accumulated jsonl_path/offset must still be preserved.
        let mut existing = named_session(42, "session", "proj", 1_000);
        existing.session_id = "stable-id".into();
        existing.jsonl_path = Some(std::path::PathBuf::from("/tmp/stable-id.jsonl"));
        existing.jsonl_offset = 4096;

        let mut fresh = named_session(42, "session", "proj", 0);
        fresh.session_id = "stable-id".into();
        fresh.jsonl_path = None;
        fresh.jsonl_offset = 0;

        let (merged, _) = merge_discovered_sessions(vec![existing], vec![fresh]);
        let s = &merged[0];
        assert_eq!(s.session_id, "stable-id");
        assert_eq!(
            s.jsonl_path,
            Some(std::path::PathBuf::from("/tmp/stable-id.jsonl")),
            "jsonl_path must survive when session_id is unchanged"
        );
        assert_eq!(s.jsonl_offset, 4096, "jsonl_offset must survive too");
    }
}
