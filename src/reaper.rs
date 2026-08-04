//! One-shot orphan reaper for in-sandbox `claude` processes.
//!
//! When a user closes an iTerm2 tab whose `claude` runs inside the
//! `linera-agent` agent-sandbox, Docker exec doesn't propagate SIGHUP to the
//! container-side exec target (moby/moby#9098). The in-VM `claude` survives,
//! its sidecar (`{pid}.terminal.json`) keeps pointing at a host TTY that is
//! no longer attached, and the row sits Idle forever.
//!
//! The reaper detects this by diffing two sets:
//! - Open set: host-side TTYs of currently-running `sbx exec ... <sandbox>`
//!   processes, extracted from `SANDBOX_HOST_TTY=/dev/ttysNNN` in argv.
//! - Sandbox set: per-PID sidecars under the sandbox sessions dir, each
//!   carrying its `host_tty` and a kill(0) liveness check.
//!
//! Any sandbox PID whose sidecar `host_tty` is not in the open set AND whose
//! process is alive is sent SIGHUP. Sidecars whose PID is dead are swept off
//! disk along with their `{pid}.json` companion.
//!
//! Wired as `claudectl --reap-orphans` in `main.rs`. Add `--dry-run` to
//! preview without killing or removing. Also exposes `--install-reaper` /
//! `--uninstall-reaper` to wire a launchd job on macOS.
//!
//! ## Environment overrides
//!
//! - `CLAUDECTL_SANDBOX_NAME` — sbx sandbox to scan. If unset, the reaper
//!   runs `sbx ls` once and uses the single running sandbox if there is
//!   exactly one; otherwise falls back to `linera-agent`.
//! - `CLAUDECTL_SANDBOX_SESSIONS_DIR` — in-sandbox path holding the per-PID
//!   `{pid}.terminal.json` sidecars. Default `/var/lib/sandbox-sessions`.
//!
//! Both env vars are read on every invocation; an empty value falls back to
//! the default (treat empty as unset).

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use crate::sandbox_registry;

/// Process-wide cache for the auto-detected sandbox name. `None` means
/// `sbx ls` could not pick exactly one running sandbox; the resolver then
/// falls back to `DEFAULT_SANDBOX_NAME`. Populated lazily on first call.
static AUTO_SANDBOX_NAME: OnceLock<Option<String>> = OnceLock::new();

const DEFAULT_SANDBOX_NAME: &str = "linera-agent";

fn sandbox_name() -> String {
    let env = std::env::var("CLAUDECTL_SANDBOX_NAME").ok();
    let auto = AUTO_SANDBOX_NAME.get_or_init(detect_running_sandbox_name);
    resolve_sandbox_name(env.as_deref(), auto.as_deref(), DEFAULT_SANDBOX_NAME)
}

/// Pure resolver. Picks the first non-empty source: explicit env override
/// → auto-detected name → default. Tests target this directly so they
/// don't have to mutate process-global env state (which races other tests
/// under parallel cargo test).
pub(crate) fn resolve_sandbox_name(
    env_override: Option<&str>,
    auto_detected: Option<&str>,
    default: &str,
) -> String {
    if let Some(v) = env_override {
        if !v.is_empty() {
            return v.to_string();
        }
    }
    if let Some(v) = auto_detected {
        if !v.is_empty() {
            return v.to_string();
        }
    }
    default.to_string()
}

/// Shell out to `sbx ls --json` once and try to identify a unique running
/// sandbox. Returns `None` for any failure (binary missing, non-zero exit,
/// parse miss), in which case the caller falls back to `DEFAULT_SANDBOX_NAME`.
fn detect_running_sandbox_name() -> Option<String> {
    let output = sbx_list_json().ok()?;
    parse_sbx_ls_for_single_running_sandbox(&output)
}

/// `sbx ls --json` stdout, or an error. One place so the flag can't drift
/// between the single-sandbox resolver and the collector — they must agree on
/// what "running" means, and they only do if they read the same output.
fn sbx_list_json() -> io::Result<String> {
    let output = Command::new("sbx").args(["ls", "--json"]).output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "sbx ls --json failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Pure parser: given `sbx ls --json` stdout, return the name of the unique
/// running sandbox if and only if there is exactly one. Anything else (zero,
/// multiple, stopped-only, malformed) returns `None` so the caller can fall
/// back to the default. We require the running condition because a stopped
/// sandbox is no help to the reaper — it has no in-VM processes to scan.
pub(crate) fn parse_sbx_ls_for_single_running_sandbox(stdout: &str) -> Option<String> {
    let mut running = parse_sbx_ls_running_names(stdout);
    if running.len() == 1 {
        return running.pop();
    }
    None
}

/// One row of `sbx ls --json`. Only the two fields we act on are named; `id`,
/// `agent` and `workspaces` are ignored, and serde skips unknown fields, so a
/// future `sbx` adding columns can't break the parse.
#[derive(serde::Deserialize)]
struct SbxListEntry {
    name: String,
    status: String,
}

#[derive(serde::Deserialize)]
struct SbxListing {
    #[serde(default)]
    sandboxes: Vec<SbxListEntry>,
}

/// Pure parser: every running sandbox name in `sbx ls --json` stdout, in listed
/// order.
///
/// Stopped rows are excluded for the same reason the single-sandbox resolver
/// excludes them — a stopped sandbox has no in-VM processes to scan and nothing
/// to collect from.
///
/// JSON rather than the text table because this now has to enumerate a *set*:
/// column-position parsing was tolerable when the question was "is this one
/// name present", but `PORTS` is empty in the common case, so telling an absent
/// optional column from a shifted one is guesswork the structured form removes.
/// Malformed input yields an empty list — the caller treats that as "no
/// sandboxes", which is the same conservative outcome the text parser gave.
pub(crate) fn parse_sbx_ls_running_names(stdout: &str) -> Vec<String> {
    let Ok(listing) = serde_json::from_str::<SbxListing>(stdout) else {
        return Vec::new();
    };
    listing
        .sandboxes
        .into_iter()
        .filter(|entry| entry.status == "running")
        .map(|entry| entry.name)
        .collect()
}

/// Host-side pointer written by the sandbox wrapper naming the sandbox the
/// `linera-agent` alias currently resolves to. Absent until the wrapper side
/// lands, and absent whenever the wrapper has never created a sandbox.
fn current_sandbox_pointer() -> Option<String> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let path = std::env::var_os("CLAUDECTL_CURRENT_SANDBOX_POINTER")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".cache/sandbox-claude-stage/current-sandbox"));
    let raw = std::fs::read_to_string(path).ok()?;
    let name = raw.trim().to_string();
    (!name.is_empty()).then_some(name)
}

/// Pure resolver for which running sandbox is "current".
///
/// The pointer wins when it names a sandbox that is actually running. A pointer
/// naming a dead sandbox is ignored rather than trusted: it means the wrapper
/// created something that has since been reaped, and marking a corpse current
/// would strand every live sandbox as "superseded".
///
/// With no usable pointer we fall back to "the only running sandbox, if there
/// is exactly one" — which is precisely today's world, so this reads correctly
/// before the wrapper side exists. With several running and no pointer we
/// return `None`: guessing would put a `current` badge on an arbitrary row.
pub(crate) fn resolve_current_sandbox(pointer: Option<&str>, running: &[String]) -> Option<String> {
    if let Some(name) = pointer {
        if running.iter().any(|r| r == name) {
            return Some(name.to_string());
        }
    }
    match running {
        [only] => Some(only.clone()),
        _ => None,
    }
}

/// One live `claude` process seen inside a sandbox — the only facts about a
/// foreign session that the host cannot work out for itself.
///
/// Everything else a row renders is either *identity* (the hook-written
/// registry has it) or *transcript-derived* (the host recomputes it from the
/// shared `~/.claude` mount every tick — see `app::do_refresh_io`). CPU and
/// resident memory are genuinely per-VM measurements, and membership in the
/// probe's output is the liveness verdict.
#[derive(Debug, Clone, PartialEq)]
struct SandboxProc {
    pid: u32,
    cpu_percent: f32,
    mem_mb: f64,
}

/// Collect one running sandbox's sessions: identity from the hook-written
/// registry, CPU/memory/liveness from a `ps` probe run inside that VM.
///
/// Nothing here speaks to the sandbox's own `claudectl`. That binary is baked
/// into the sandbox image, which rebuilds nightly at best, so the host was
/// reading a wire format that could be arbitrarily older than its own — and
/// was, repeatedly, each time surfacing as a column that silently rendered
/// blank for every session in an older sandbox. `ps` has no version to skew
/// against, and the registry is written by hooks running *this* claudectl's
/// schema on the shared mount.
///
/// Errors are per-sandbox and non-fatal: one unreachable sandbox must not cost
/// us the inventory of every other one, so the caller records an empty slice
/// for it and carries on.
fn collect_one_sandbox(
    name: &str,
    registry: &[sandbox_registry::SessionEntry],
) -> io::Result<Vec<serde_json::Value>> {
    let pids: Vec<u32> = registry.iter().filter_map(|entry| entry.pid).collect();
    if pids.is_empty() {
        // Nothing recorded for this sandbox ⇒ nothing to ask it about. Not
        // merely an optimisation: every `sbx exec` is seconds of wrapper
        // startup, paid once per running sandbox on every reaper tick.
        return Ok(Vec::new());
    }
    let live = probe_sandbox_procs(name, &pids)?;
    Ok(sessions_from_registry(registry, &live))
}

/// Run [`PROC_PROBE_SCRIPT`] inside `name` for `pids` and parse what it saw.
///
/// Caller must have established that `name` is *running*: `sbx exec` starts a
/// stopped sandbox, so probing one indiscriminately would resurrect sandboxes
/// the user had deliberately shut down.
fn probe_sandbox_procs(name: &str, pids: &[u32]) -> io::Result<HashMap<u32, SandboxProc>> {
    let mut cmd = Command::new("sbx");
    cmd.args(["exec", name, "bash", "-c", PROC_PROBE_SCRIPT, "--"]);
    cmd.args(pids.iter().map(u32::to_string));
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "sbx exec {name} proc-probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(parse_sandbox_procs(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

/// The in-sandbox process probe, kept as a named constant for the same reason
/// [`sidecar_scan_script`] is its own function: tests execute it for real
/// against live pids with a plain `bash -c`, no `sbx` involved.
///
/// Pids arrive as positional arguments and one `ps` row is printed per live
/// one: `pid %cpu rss command`. `ps` is the entire dependency — no jq, no
/// in-sandbox claudectl — which is precisely what makes the collector immune
/// to image-vs-host version skew.
///
/// `ps` exits non-zero when *none* of the pids exist, which is an ordinary
/// outcome (every recorded session has since exited), so its failure is
/// swallowed and the script always exits 0. Empty stdout then reads as
/// "nothing alive here", which is exactly right; a genuine `sbx` failure still
/// shows up as a non-zero exit from `sbx` itself.
const PROC_PROBE_SCRIPT: &str = r#"
set -u
if [ "$#" -eq 0 ]; then exit 0; fi
IFS=,
pids="$*"
unset IFS
ps -o pid=,%cpu=,rss=,command= -p "$pids" 2>/dev/null || true
exit 0
"#;

/// Pure parser for the probe's `ps` output, keyed by pid.
///
/// Rows whose argv0 is not exactly `claude` are dropped rather than merely
/// left unmatched: pids get recycled inside a container as readily as anywhere
/// else, and a registry entry naming a recycled pid would otherwise report an
/// unrelated process's CPU and memory as a live session's. This is the same
/// claude-only liveness the in-VM scan applies (`discovery`'s `is_live` is
/// membership in the live *claude* process map), so a session reaches the
/// snapshot on exactly the terms it would reach that sandbox's own dashboard.
fn parse_sandbox_procs(text: &str) -> HashMap<u32, SandboxProc> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(pid) = fields.next().and_then(|f| f.parse::<u32>().ok()) else {
            continue;
        };
        let Some(cpu_percent) = fields.next().and_then(|f| f.parse::<f32>().ok()) else {
            continue;
        };
        let Some(rss_kb) = fields.next().and_then(|f| f.parse::<f64>().ok()) else {
            continue;
        };
        let command = fields.collect::<Vec<&str>>().join(" ");
        if !crate::process::is_claude_process(&command) {
            continue;
        }
        out.insert(
            pid,
            SandboxProc {
                pid,
                cpu_percent,
                mem_mb: rss_kb / 1024.0,
            },
        );
    }
    out
}

/// Build one sandbox's snapshot rows from its registry slice, keeping only the
/// sessions the probe found alive.
///
/// The registry is the *primary* identity source here, not a fallback it once
/// was: it is written by in-sandbox hooks running this same claudectl's schema
/// onto the host-shared mount, so `session_id`, `cwd` and `name` are always the
/// current shape — which is the whole point of no longer asking an image-aged
/// binary for them. An entry with no id is skipped: the renderer drops what it
/// cannot identify (`ClaudeSession::from_snapshot_value` returns `None`), so
/// emitting it would only inflate the file.
///
/// A recorded pid that the probe did not return is a departed session. Dropping
/// it is what keeps the snapshot honest between hook fires: `replace_sandbox_slice`
/// mirrors the live set only when some hook in that sandbox fires, and it
/// freezes entirely once `sbx rm` takes the sandbox away.
fn sessions_from_registry(
    entries: &[sandbox_registry::SessionEntry],
    live: &HashMap<u32, SandboxProc>,
) -> Vec<serde_json::Value> {
    entries
        .iter()
        .filter(|entry| !entry.session_id.is_empty())
        .filter_map(|entry| {
            let sample = live.get(&entry.pid?)?;
            Some(serde_json::json!({
                "session_id": entry.session_id,
                "cwd": entry.cwd,
                "session_name": entry.name.clone().unwrap_or_default(),
                "started_at": entry.started_at_ms,
                "pid": sample.pid,
                // Both rounded to two decimals, for the reason `to_json_value`
                // rounds memory: the snapshot is rewritten in full on every
                // collector pass, nothing renders more than two decimals of
                // either, and widening an f32 to JSON otherwise writes its
                // binary error out in full (`3.4` becomes 3.4000000953674316).
                "cpu": (f64::from(sample.cpu_percent) * 100.0).round() / 100.0,
                "mem_mb": (sample.mem_mb * 100.0).round() / 100.0,
            }))
        })
        .collect()
}

/// Collect every running sandbox into the host snapshot and write it.
///
/// This is the only place cross-sandbox liveness is authoritative — the
/// per-sandbox registries are written from inside each sandbox and can only
/// ever describe their own slice.
fn collect_and_write_snapshot(out: &mut impl Write) -> io::Result<()> {
    let running = parse_sbx_ls_running_names(&sbx_list_json()?);
    let current = resolve_current_sandbox(current_sandbox_pointer().as_deref(), &running);

    let mut snapshot = sandbox_registry::SandboxSnapshot {
        collected_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        ..Default::default()
    };
    // Read once, outside the loop: the hook-written registry is the identity
    // source for every sandbox, so re-reading it per sandbox would be the same
    // file parsed N times.
    let registry = sandbox_registry::load();
    let no_entries: Vec<sandbox_registry::SessionEntry> = Vec::new();
    // `running` is a fully materialised list before any `sbx exec` runs. That
    // ordering is deliberate everywhere this pattern appears: `sbx` consumes
    // the stdin it inherits, so an exec driven straight off a streaming
    // enumeration eats the rest of its own input and collects one sandbox.
    let mut tally: Vec<String> = Vec::with_capacity(running.len());
    for name in &running {
        let slice = registry.sandboxes.get(name).unwrap_or(&no_entries);
        let sessions = match collect_one_sandbox(name, slice) {
            Ok(sessions) => sessions,
            Err(e) => {
                // Log which sandbox and why, not just that collection was
                // partial — a snapshot silently missing an origin looks
                // identical to a sandbox with no sessions.
                writeln!(out, "reaper: collect from '{name}' failed: {e}")?;
                Vec::new()
            }
        };
        // Both numbers, not just the result: "0 sessions" is the shape every
        // one of these bugs took, and only the recorded count says whether
        // that means the registry was empty or the probe found nothing alive.
        tally.push(format!("{name}={}/{} live", sessions.len(), slice.len()));
        snapshot.sandboxes.insert(
            name.clone(),
            sandbox_registry::SandboxOrigin {
                is_current: current.as_deref() == Some(name.as_str()),
                sessions,
            },
        );
    }
    sandbox_registry::write_snapshot(&snapshot)?;
    writeln!(
        out,
        "reaper: collected {} sandbox(es) [{}] into {}",
        running.len(),
        tally.join(", "),
        sandbox_registry::sandbox_snapshot_path().display()
    )
}

pub(crate) fn sandbox_sessions_dir() -> String {
    resolve_or_default(
        std::env::var("CLAUDECTL_SANDBOX_SESSIONS_DIR")
            .ok()
            .as_deref(),
        "/var/lib/sandbox-sessions",
    )
}

/// Treat both "unset" and "set-but-empty" as fallback to default. Empty env
/// values almost always come from a typo or an unquoted shell expansion;
/// silently using `""` would make `sbx exec ""` fail with a confusing error.
fn resolve_or_default(value: Option<&str>, default: &str) -> String {
    match value {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => default.to_string(),
    }
}

/// Sidecar entry parsed from inside the sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxSidecar {
    pub pid: u32,
    pub host_tty: String,
    /// True if `kill -0 <pid>` succeeded inside the sandbox at scan time.
    pub alive: bool,
    /// Optional human label from `{pid}.json` (e.g. session name). May be
    /// empty when the companion `{pid}.json` is missing.
    pub name: String,
}

/// Result of orphan-set computation. Two disjoint groups: live processes to
/// SIGHUP, and dead sidecars whose disk artefacts should be swept.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct OrphanPlan {
    /// Sidecars whose `host_tty` is not in the open set AND PID is alive.
    /// These get SIGHUP.
    pub kill: Vec<SandboxSidecar>,
    /// Sidecars whose PID is dead. Just clean their `{pid}.terminal.json`
    /// (and any matching `{pid}.json`).
    pub sweep: Vec<SandboxSidecar>,
}

/// Cap on the number of alive orphans the auto-reaper will kill in a single
/// pass. A spike past this is more likely a parser regression or env
/// corruption than a genuine surge in orphans.
pub const MAX_KILLS_PER_PASS: usize = 10;

/// Decision returned by `decide_action`. Pure; the I/O wrapper in `run()`
/// translates this into stderr/stdout output and `sbx exec` calls.
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    /// Refuse to act. The string is the human-readable reason logged to
    /// stderr.
    Skip(String),
    /// Proceed with this plan. May be empty (no orphans).
    Execute(OrphanPlan),
}

/// Pure decision function. Given the host's open TTY set and the sandbox's
/// sidecar list, decide whether to act and how. Centralises the safety
/// guards so they can be unit-tested without mocking I/O.
pub fn decide_action(open_ttys: &HashSet<String>, sidecars: &[SandboxSidecar]) -> Action {
    let plan = compute_orphans(open_ttys, sidecars);
    let alive_count = sidecars.iter().filter(|s| s.alive).count();

    // Guard 1: 0 open host TTYs but alive sandbox claudes exist => host
    // scan probably failed (transient ps glitch, sbx daemon restart).
    // Refusing is conservative; the next pass will pick up real orphans
    // once the host scan succeeds.
    if open_ttys.is_empty() && alive_count > 0 {
        return Action::Skip(format!(
            "0 open host TTYs but {alive_count} alive sandbox claudes — refusing to act \
             (probable host scan failure). Re-run --dry-run to investigate."
        ));
    }

    // Guard 2: kill set exceeds the safety cap. More likely a bug than a
    // real surge.
    if plan.kill.len() > MAX_KILLS_PER_PASS {
        return Action::Skip(format!(
            "{} kill candidates exceeds safety cap ({}); refusing to act. \
             Run `claudectl --reap-orphans --dry-run` to inspect.",
            plan.kill.len(),
            MAX_KILLS_PER_PASS
        ));
    }

    Action::Execute(plan)
}

/// Pure orphan-detection. No I/O.
///
/// Rules:
/// - sidecar PID is dead → sweep (disk-only orphan).
/// - sidecar PID is alive AND host_tty is in `open_ttys` → current, keep.
/// - sidecar PID is alive AND host_tty is NOT in `open_ttys` → kill.
///
/// In the TTY-reuse case (two sidecars on the same host_tty), both go through
/// these rules independently; the dead one ends up in `sweep`, the alive one
/// in either `kill` or "current" depending on whether the TTY is still open.
pub fn compute_orphans(open_ttys: &HashSet<String>, sidecars: &[SandboxSidecar]) -> OrphanPlan {
    let mut plan = OrphanPlan::default();
    for sc in sidecars {
        if !sc.alive {
            plan.sweep.push(sc.clone());
            continue;
        }
        if !open_ttys.contains(&sc.host_tty) {
            plan.kill.push(sc.clone());
        }
    }
    plan
}

// ── Tick-skip cache ───────────────────────────────────────────────────────
//
// Most ticks have no host-side TTY changes since the previous tick → no new
// orphans can have appeared → previous tick's full pass already handled
// everything. Skipping the in-sandbox scan saves the sbx exec round-trip
// (~2.6s startup overhead per call), which dominates steady-state cost when
// the timer fires every minute.
//
// Cache shape: `<XDG_CACHE_HOME or ~/.cache>/claudectl/reaper-last-state`.
// File body  : sorted, newline-joined open SANDBOX_HOST_TTY values.
// Freshness  : the file's own mtime, ceilinged at MAX_CACHE_AGE.
//
// Cache is written ONLY after a successful scan + decide_action accepted
// the plan. Skipped on dry-run, on safety-guard skips, and on error paths,
// so a transient failure or preview pass doesn't suppress the next real
// tick.

/// Hard ceiling on cache freshness. Past this, the next tick must do a
/// full pass even if the host TTY set hasn't changed — catches in-sandbox
/// `claude` deaths (panic, oom-kill) that wouldn't show up as a host TTY
/// transition.
const MAX_CACHE_AGE: Duration = Duration::from_secs(30 * 60);

/// What the cache decides we should do this tick.
#[derive(Debug, PartialEq, Eq)]
pub enum CacheAction {
    /// State matches the cached snapshot AND the cache is within the
    /// freshness ceiling — skip the in-sandbox scan entirely.
    Skip,
    /// State differs, cache is stale, or there is no cache yet — run the
    /// full pass and (on success) refresh the cache.
    FullPass,
}

/// Stable string representation of the open-TTY set. Sorted so the same
/// set always serialises to the same body (HashSet iteration order isn't
/// stable across runs, and we need byte-exact equality for the cache).
pub fn state_string(open_ttys: &HashSet<String>) -> String {
    let mut sorted: Vec<&str> = open_ttys.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted.join("\n")
}

/// Pure decision: does the current tick's open-TTY state plus the cached
/// state warrant skipping the in-sandbox scan? Skips only when the state
/// matches AND the cache age is within `max_age`.
pub fn decide_cache_action(
    prev_state: Option<&str>,
    new_state: &str,
    cache_age: Option<Duration>,
    max_age: Duration,
) -> CacheAction {
    match (prev_state, cache_age) {
        (Some(prev), Some(age)) if prev == new_state && age <= max_age => CacheAction::Skip,
        _ => CacheAction::FullPass,
    }
}

fn cache_path() -> Option<PathBuf> {
    let dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| home_dir().ok().map(|h| h.join(".cache")))?;
    Some(dir.join("claudectl").join("reaper-last-state"))
}

fn read_cache_state(path: &Path) -> Option<(String, Duration)> {
    let body = std::fs::read_to_string(path).ok()?;
    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    let age = std::time::SystemTime::now().duration_since(mtime).ok()?;
    Some((body, age))
}

fn write_cache_state(path: &Path, state: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, state)
}

/// Entry point for `claudectl --reap-orphans`. Returns `Ok(())` even when no
/// sandboxes/sbx are present — the reaper is a no-op fallback.
pub fn run(dry_run: bool) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    if !dry_run {
        prune_closed_sessions();
    }

    if !sbx_available() {
        writeln!(
            io::stderr(),
            "reaper: `sbx` not in PATH; nothing to do (no sandboxes on this host)",
        )?;
        return Ok(());
    }

    // Publish the cross-sandbox snapshot before the orphan scan. Deliberately
    // independent of it: a collection failure must not cost us orphan reaping,
    // and vice versa. Skipped under --dry-run, which promises no writes.
    if !dry_run {
        if let Err(e) = collect_and_write_snapshot(&mut out) {
            writeln!(io::stderr(), "reaper: snapshot collection failed: {e}")?;
        }
    }

    let open_ttys = match scan_host_open_ttys() {
        Ok(set) => set,
        Err(e) => {
            writeln!(io::stderr(), "reaper: host ps scan failed: {e}")?;
            return Ok(());
        }
    };

    let new_state = state_string(&open_ttys);
    let cache = cache_path();
    if !dry_run {
        if let Some(path) = &cache {
            let prev = read_cache_state(path);
            let action = decide_cache_action(
                prev.as_ref().map(|(s, _)| s.as_str()),
                &new_state,
                prev.as_ref().map(|(_, age)| *age),
                MAX_CACHE_AGE,
            );
            if matches!(action, CacheAction::Skip) {
                return Ok(());
            }
        }
    }

    let sidecars = match scan_sandbox_sidecars() {
        Ok(list) => list,
        Err(e) => {
            writeln!(io::stderr(), "reaper: sandbox sidecar scan failed: {e}")?;
            return Ok(());
        }
    };

    let plan = match decide_action(&open_ttys, &sidecars) {
        Action::Skip(reason) => {
            writeln!(io::stderr(), "reaper: {reason}")?;
            return Ok(());
        }
        Action::Execute(plan) => plan,
    };

    // Refresh the cache as soon as both observations succeeded and the
    // safety guard accepted the plan. Even if the kill/sweep below errors
    // out, the cache reflects "we've seen this state and decided to act";
    // the MAX_CACHE_AGE ceiling bounds how long a survived orphan can
    // linger before the next tick re-checks anyway.
    if !dry_run {
        if let Some(path) = &cache {
            if let Err(e) = write_cache_state(path, &new_state) {
                writeln!(io::stderr(), "reaper: cache write failed: {e}")?;
            }
        }
    }

    if plan.kill.is_empty() && plan.sweep.is_empty() {
        writeln!(out, "no orphans")?;
        return Ok(());
    }

    for orphan in &plan.kill {
        writeln!(
            out,
            "{}reaped: pid={} tty={} name={}",
            if dry_run { "[dry-run] " } else { "" },
            orphan.pid,
            orphan.host_tty,
            if orphan.name.is_empty() {
                "?"
            } else {
                orphan.name.as_str()
            },
        )?;
    }
    for orphan in &plan.sweep {
        writeln!(
            out,
            "{}swept (dead): pid={} tty={} name={}",
            if dry_run { "[dry-run] " } else { "" },
            orphan.pid,
            orphan.host_tty,
            if orphan.name.is_empty() {
                "?"
            } else {
                orphan.name.as_str()
            },
        )?;
    }

    if dry_run {
        return Ok(());
    }

    // Apply the plan in ONE sbx exec instead of two. The dead pids get
    // their `{pid}.json` and `{pid}.terminal.json` removed; the alive pids
    // get SIGHUP. Alive pids' sidecars are NOT swept this pass — if the
    // HUP fails or the process ignores it, the sidecar must remain so the
    // next reaper tick re-detects it.
    let dead_pids: Vec<u32> = plan.sweep.iter().map(|o| o.pid).collect();
    let alive_pids: Vec<u32> = plan.kill.iter().map(|o| o.pid).collect();
    apply_plan(&dead_pids, &alive_pids)?;

    Ok(())
}

/// How long to let the world settle between observing a session gone and
/// judging whether its terminal is gone too.
///
/// A quit tears sessions down before the terminal process itself finishes
/// exiting; sampled in that window a session that died *with* its terminal looks
/// like one the user closed. Waiting this long — after the departure is already
/// observed — lets a co-dying terminal exit, so its owner reads gone and the
/// session is kept. Bounded best-effort: a terminal that lingers longer than
/// this still risks a spurious prune, which is why a session closed within one
/// interval of a quit is a documented limit rather than a guarantee. Off-lock,
/// so a generous value costs only latency on the pruning of genuinely-closed
/// sessions, once per reaper tick.
const SETTLE_DELAY: Duration = Duration::from_secs(3);

/// Drop sessions the user closed from the restore registry.
///
/// The reaper owns this because nothing else can. The registry is otherwise
/// only written by hooks; a hook needs a live session to fire it, and it must
/// not forget anyway (a hook can't observe another terminal mid-quit safely).
/// Close every session and the registry freezes exactly as it was — which is
/// how `--restore-sessions` came to resurrect sessions closed two days earlier.
/// Running on a timer means the verdict lands within one interval of the close,
/// with or without anything else on the machine.
///
/// The pass is **purely subtractive**: it keeps every live session untouched
/// (owners are the hooks' to attribute) and only removes departed ones whose
/// terminal is gone too. It orders its observations so a quit can't fool it —
/// scan, settle, scan again, then sample owners:
///
/// 1. Two live scans around [`SETTLE_DELAY`]; their union is "still live". A
///    session missing from only one scan (a torn read, or one that started
///    mid-pass) is thus left alone rather than pruned.
/// 2. The owner sample is taken *after* the settle, so a terminal that co-died
///    with its sessions has had time to exit and reads gone — its sessions kept.
///
/// Every observation can fail closed: a failed live scan (not merely an empty
/// one) or a failed `ps` aborts the pass without touching the registry, so a
/// transient error never destroys restore data.
///
/// The corollary is a real dependency: **a host that never runs the reaper has
/// no prune path at all**, and the original bug returns. Nothing at restore time
/// can substitute, because the evidence — that the terminal outlived the
/// session — stops existing once that terminal exits. Two consequences worth
/// knowing: a session closed within one interval of a terminal quit may still
/// be restored, and an entry whose terminal really did die lingers until some
/// later restore consumes it.
///
/// Host-only: inside a sandbox the host process table is meaningless.
/// Best-effort — the registry must never take the reaper down.
fn prune_closed_sessions() {
    prune_closed_sessions_after(SETTLE_DELAY)
}

fn prune_closed_sessions_after(settle: Duration) {
    use std::collections::HashSet;

    if crate::sandbox_registry::current_sandbox().is_some() {
        return;
    }

    let live_ids = |sessions: Vec<crate::session::ClaudeSession>| -> HashSet<String> {
        sessions.into_iter().map(|s| s.session_id).collect()
    };

    // Observe departures before judging owners, and fail closed on a scan error
    // (a failed enumeration must never read as "everyone closed").
    let Some(before) = crate::discovery::try_live_sessions().map(live_ids) else {
        return;
    };
    std::thread::sleep(settle);
    let Some(after) = crate::discovery::try_live_sessions().map(live_ids) else {
        return;
    };
    let Some(table) = crate::terminal_owner::ProcessTable::snapshot() else {
        return;
    };

    let live = still_live(before, after);
    if let Err(e) = crate::sandbox_registry::update_local(|current| {
        crate::sandbox_registry::retain_restorable(
            current,
            &live,
            // The scans and the table are frozen by now, but this closure runs
            // inside the registry flock, where it can see an entry a hook wrote
            // for a session that started after both scans. The now-check keeps
            // it: pruning would drop a live session's entry, and an idle
            // session never fires the hook that would re-add it.
            |entry| entry.pid.is_some_and(crate::discovery::pid_alive),
            |owner| table.is_alive(owner),
        )
    }) {
        let _ = writeln!(io::stderr(), "reaper: registry prune failed: {e}");
    }
}

/// The sessions to treat as still live: seen by *either* scan.
///
/// The union — never the intersection — is the torn-read forgiveness the
/// two-scan design exists for: a pointer file mid-rewrite (or a session starting
/// mid-pass) is missing from one scan only, and demanding both would judge it
/// departed and prune it.
fn still_live(
    before: std::collections::HashSet<String>,
    after: std::collections::HashSet<String>,
) -> std::collections::HashSet<String> {
    before.union(&after).cloned().collect()
}

fn sbx_available() -> bool {
    // `sbx --help` exits 0 when the binary is in PATH and runnable. We use
    // it instead of `--version` (which sbx doesn't accept) because we just
    // want a "is this binary present and executable" probe.
    Command::new("sbx")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Parse host `ps -ax -o pid,command` for `sbx exec ... linera-agent` lines
/// and pull `SANDBOX_HOST_TTY=/dev/ttysNNN` out of each.
fn scan_host_open_ttys() -> io::Result<HashSet<String>> {
    let output = Command::new("ps")
        .args(["-ax", "-o", "pid,command"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "ps exited non-zero: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(extract_open_ttys(&text, &sandbox_name()))
}

/// Pure parser: takes `ps -ax -o pid,command` output, returns the set of
/// `SANDBOX_HOST_TTY` values from `sbx exec ... <sandbox>` lines.
fn extract_open_ttys(ps_output: &str, sandbox: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for line in ps_output.lines() {
        if !line.contains("sbx exec") {
            continue;
        }
        if !line.contains(sandbox) {
            continue;
        }
        if let Some(tty) = line
            .split_whitespace()
            .find_map(|tok| tok.strip_prefix("SANDBOX_HOST_TTY="))
        {
            set.insert(tty.to_string());
        }
    }
    set
}

/// Run a single `sbx exec linera-agent bash -c '...'` that walks the sandbox's
/// sessions dir and prints one tab-separated line per `{pid}.terminal.json`:
/// `pid<TAB>host_tty<TAB>alive<TAB>name`.
fn scan_sandbox_sidecars() -> io::Result<Vec<SandboxSidecar>> {
    // The script enumerates every {pid}.terminal.json, extracts host_tty,
    // checks kill -0 on the pid (read from the matching {pid}.json's "pid"
    // field, falling back to the sidecar filename's pid stem), and pulls a
    // best-effort name from {pid}.json's "name" field if present.
    //
    // No jq dependency in the sandbox — use bash + grep + sed. Each output
    // line is `<pid>\t<host_tty>\t<alive>\t<name>` where alive is 0 or 1.
    let script = sidecar_scan_script(&sandbox_sessions_dir());

    let output = Command::new("sbx")
        .args(["exec", &sandbox_name(), "bash", "-c", &script])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "sbx exec sidecar-scan failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(parse_sandbox_sidecars(&text))
}

/// The in-sandbox scan script for `dir`. Its own fn so tests can execute it
/// against a fixture directory with plain `bash -c` (no sbx).
fn sidecar_scan_script(dir: &str) -> String {
    format!(
        r#"
set -u
DIR={dir}
shopt -s nullglob
for sc in "$DIR"/*.terminal.json; do
  fname="${{sc##*/}}"
  pid="${{fname%.terminal.json}}"
  host_tty=$(grep -o '"host_tty"[[:space:]]*:[[:space:]]*"[^"]*"' "$sc" 2>/dev/null \
    | sed 's/.*"host_tty"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' | head -n1)
  [ -z "$host_tty" ] && continue
  if kill -0 "$pid" 2>/dev/null; then alive=1; else alive=0; fi
  name=""
  if [ -f "$DIR/$pid.json" ]; then
    name=$(grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$DIR/$pid.json" 2>/dev/null \
      | sed 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' | head -n1)
    # Claude Code >= 2.1.220 marks its auto-derived placeholder names
    # (e.g. "ndr-5e") with nameSource:"derived" — don't label reap logs
    # with them (mirrors RawSession::title()).
    if grep -q '"nameSource"[[:space:]]*:[[:space:]]*"derived"' "$DIR/$pid.json" 2>/dev/null; then
      name=""
    fi
  fi
  printf '%s\t%s\t%s\t%s\n' "$pid" "$host_tty" "$alive" "$name"
done
"#
    )
}

/// Pure parser for the tab-separated sandbox-side scan output.
fn parse_sandbox_sidecars(text: &str) -> Vec<SandboxSidecar> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split('\t');
        let Some(pid_s) = parts.next() else { continue };
        let Some(host_tty) = parts.next() else {
            continue;
        };
        let Some(alive_s) = parts.next() else {
            continue;
        };
        let name = parts.next().unwrap_or("").to_string();
        let Ok(pid) = pid_s.parse::<u32>() else {
            continue;
        };
        let alive = alive_s == "1";
        out.push(SandboxSidecar {
            pid,
            host_tty: host_tty.to_string(),
            alive,
            name,
        });
    }
    out
}

/// Apply the reaper plan in ONE `sbx exec`: sweep dead-PID sidecar files
/// then SIGHUP alive-PID orphans. No-op when both lists are empty.
///
/// The script gets the dead-pid count as its first positional, lets the
/// shell shift it off, and uses the rest of the count as sweep targets;
/// anything still on the argv after that is the kill set. This avoids a
/// second `sbx exec` round-trip (each is on the order of seconds of
/// startup overhead from inside the sbx wrapper).
///
/// Errors:
/// - `Command::output` failure (sbx binary missing/unspawnable) is
///   propagated — same as the prior split implementation.
/// - Non-zero bash exit (rm trips, kill returns non-zero) is logged to
///   stderr but does NOT fail the run. `rm -f` swallows its own errors
///   already; a non-zero exit here usually means kill couldn't deliver
///   the signal, which is a soft warning at worst — the next reaper pass
///   will re-detect the still-alive orphan and retry.
fn apply_plan(dead_pids: &[u32], alive_pids: &[u32]) -> io::Result<()> {
    if dead_pids.is_empty() && alive_pids.is_empty() {
        return Ok(());
    }

    let dir = sandbox_sessions_dir();
    let script = format!(
        r#"
set -u
DIR={dir}
N_DEAD=$1; shift
i=0
while [ "$i" -lt "$N_DEAD" ]; do
  pid=$1; shift
  rm -f "$DIR/$pid.json" "$DIR/$pid.terminal.json" 2>/dev/null || true
  i=$((i+1))
done
if [ "$#" -gt 0 ]; then
  kill -HUP "$@" 2>&1
fi
"#
    );

    let mut all_args: Vec<String> = Vec::with_capacity(1 + dead_pids.len() + alive_pids.len());
    all_args.push(dead_pids.len().to_string());
    all_args.extend(dead_pids.iter().map(u32::to_string));
    all_args.extend(alive_pids.iter().map(u32::to_string));

    let name = sandbox_name();
    let mut cmd = Command::new("sbx");
    cmd.args(["exec", &name, "bash", "-c", &script, "--"]);
    cmd.args(&all_args);
    match cmd.output() {
        Ok(o) if !o.status.success() => {
            writeln!(
                io::stderr(),
                "reaper: apply_plan returned non-zero: {}",
                String::from_utf8_lossy(&o.stderr)
            )?;
            Ok(())
        }
        Err(e) => Err(e),
        Ok(_) => Ok(()),
    }
}

// ── Install/uninstall (macOS launchd + Linux systemd-user) ────────────────

// Used by macOS install/uninstall and by the plist-renderer tests; absent
// in non-test Linux builds so the binary compiles dead-code-clean there.
#[cfg(any(target_os = "macos", test))]
const LAUNCH_AGENT_LABEL: &str = "linera.claudectl-reaper";

// Used by Linux install/uninstall and by the systemd unit-renderer tests.
#[cfg(any(target_os = "linux", test))]
const SYSTEMD_UNIT_BASENAME: &str = "claudectl-reaper";

/// Hard floor: anything below this hammers `sbx exec` faster than a real
/// reaper pass completes (the in-sandbox bash + grep + kill pipeline takes
/// a second or two). Hard ceiling: anything above an hour means the user
/// is closing tabs faster than the reaper can find them.
pub const MIN_INTERVAL_SECONDS: u64 = 10;
pub const MAX_INTERVAL_SECONDS: u64 = 3600;

fn home_dir() -> io::Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("HOME is not set; cannot locate user home"))
}

#[cfg(target_os = "macos")]
fn plist_path() -> io::Result<PathBuf> {
    Ok(home_dir()?
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{LAUNCH_AGENT_LABEL}.plist")))
}

#[cfg(target_os = "macos")]
fn err_log_path() -> io::Result<PathBuf> {
    Ok(home_dir()?
        .join("Library")
        .join("Logs")
        .join("claudectl-reaper.err.log"))
}

#[cfg(target_os = "linux")]
fn systemd_user_dir() -> io::Result<PathBuf> {
    Ok(home_dir()?.join(".config").join("systemd").join("user"))
}

#[cfg(target_os = "linux")]
fn systemd_service_path() -> io::Result<PathBuf> {
    Ok(systemd_user_dir()?.join(format!("{SYSTEMD_UNIT_BASENAME}.service")))
}

#[cfg(target_os = "linux")]
fn systemd_timer_path() -> io::Result<PathBuf> {
    Ok(systemd_user_dir()?.join(format!("{SYSTEMD_UNIT_BASENAME}.timer")))
}

#[cfg(target_os = "linux")]
fn linux_err_log_path() -> io::Result<PathBuf> {
    // Per XDG Base Directory spec: state goes under $XDG_STATE_HOME, default
    // ~/.local/state. systemd resolves `%h` to the user's HOME at runtime.
    Ok(home_dir()?
        .join(".local")
        .join("state")
        .join("claudectl-reaper.err.log"))
}

/// Pure plist renderer. The XML body is byte-for-byte equivalent to the
/// hand-written plist that's been driving the auto-reaper on Andre's box —
/// changing whitespace here will break the byte-equivalence verification.
#[cfg(any(target_os = "macos", test))]
pub fn build_plist(exe_path: &Path, interval_seconds: u64, err_log: &Path, home: &Path) -> String {
    let exe = exe_path.display();
    let err = err_log.display();
    let home_disp = home.display();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCH_AGENT_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>--reap-orphans</string>
    </array>

    <key>StartInterval</key>
    <integer>{interval_seconds}</integer>

    <key>RunAtLoad</key>
    <false/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>{home_disp}</string>
    </dict>

    <key>StandardOutPath</key>
    <string>/dev/null</string>

    <key>StandardErrorPath</key>
    <string>{err}</string>

    <key>ProcessType</key>
    <string>Background</string>

    <key>Nice</key>
    <integer>5</integer>
</dict>
</plist>
"#
    )
}

#[cfg(target_os = "macos")]
fn current_uid() -> u32 {
    // libc::getuid is FFI but always-succeeds (returns the real UID of the
    // calling process). No errno path.
    // SAFETY: getuid() takes no arguments and has no failure modes per
    // POSIX; it only reads kernel state.
    unsafe { libc::getuid() }
}

/// Best-effort `launchctl bootout`. Failure is expected when nothing is
/// loaded yet; we ignore the error and let the caller continue.
#[cfg(target_os = "macos")]
fn launchctl_bootout(uid: u32) -> Result<bool, io::Error> {
    let target = format!("gui/{uid}/{LAUNCH_AGENT_LABEL}");
    let output = Command::new("launchctl")
        .args(["bootout", &target])
        .output();
    match output {
        Ok(o) => Ok(o.status.success()),
        Err(e) => Err(e),
    }
}

#[cfg(target_os = "macos")]
fn launchctl_bootstrap(uid: u32, plist: &Path) -> io::Result<()> {
    let target = format!("gui/{uid}");
    let output = Command::new("launchctl")
        .args(["bootstrap", &target])
        .arg(plist)
        .output()
        .map_err(|e| io::Error::other(format!("launchctl bootstrap exec failed: {e}")))?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "launchctl bootstrap failed (exit {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Atomic plist write: `<plist>.tmp` then rename. Avoids the half-written
/// file racing against an in-flight launchd reload.
#[cfg(target_os = "macos")]
fn write_plist_atomic(path: &Path, body: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = path.to_path_buf();
    let mut name = tmp
        .file_name()
        .ok_or_else(|| io::Error::other("plist path has no filename"))?
        .to_owned();
    name.push(".tmp");
    tmp.set_file_name(name);
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn validate_interval(interval_seconds: u64) -> io::Result<()> {
    if !(MIN_INTERVAL_SECONDS..=MAX_INTERVAL_SECONDS).contains(&interval_seconds) {
        writeln!(
            io::stderr(),
            "claudectl --install-reaper: --reaper-interval {interval_seconds} \
             out of range [{MIN_INTERVAL_SECONDS}..={MAX_INTERVAL_SECONDS}] seconds"
        )?;
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "interval out of range",
        ));
    }
    Ok(())
}

/// Wire `claudectl --reap-orphans` to the host's user-scoped scheduler at
/// the given interval. macOS uses launchd, Linux uses a systemd user timer;
/// other platforms print a hint and exit 0. Idempotent on both supported
/// platforms.
#[cfg(target_os = "macos")]
pub fn install_launch_agent(interval_seconds: u64) -> io::Result<()> {
    validate_interval(interval_seconds)?;

    let exe = std::env::current_exe()
        .map_err(|e| io::Error::other(format!("cannot resolve current binary path: {e}")))?;
    let home = home_dir()?;
    let plist = plist_path()?;
    let err_log = err_log_path()?;

    if let Some(parent) = err_log.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let body = build_plist(&exe, interval_seconds, &err_log, &home);
    write_plist_atomic(&plist, &body)?;

    let uid = current_uid();
    // bootout-then-bootstrap is the launchctl-blessed reload pattern. We
    // don't propagate bootout failures because the most common cause is
    // "agent isn't loaded", which is fine.
    let _ = launchctl_bootout(uid);
    launchctl_bootstrap(uid, &plist)?;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "installed: {} (interval={}s)",
        plist.display(),
        interval_seconds
    )?;
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn install_launch_agent(interval_seconds: u64) -> io::Result<()> {
    validate_interval(interval_seconds)?;

    let exe = std::env::current_exe()
        .map_err(|e| io::Error::other(format!("cannot resolve current binary path: {e}")))?;
    let unit_dir = systemd_user_dir()?;
    std::fs::create_dir_all(&unit_dir)?;

    let err_log = linux_err_log_path()?;
    if let Some(parent) = err_log.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let service_path = systemd_service_path()?;
    let timer_path = systemd_timer_path()?;
    let service_body = build_systemd_service(&exe);
    let timer_body = build_systemd_timer(interval_seconds);
    std::fs::write(&service_path, service_body)?;
    std::fs::write(&timer_path, timer_body)?;

    // systemctl is the only way to make systemd notice the new units. If
    // it's missing, leave the unit files on disk so the user can wire them
    // manually (e.g. via a non-systemd init or a remote reload).
    if !systemctl_available() {
        writeln!(
            io::stderr(),
            "reaper: `systemctl` not in PATH. Wrote {} and {}, but did not enable the timer. \
             Reload manually once systemctl is available.",
            service_path.display(),
            timer_path.display()
        )?;
        return Err(io::Error::other("systemctl not found"));
    }

    systemctl_user(&["daemon-reload"])?;
    let timer_unit = format!("{SYSTEMD_UNIT_BASENAME}.timer");
    systemctl_user(&["enable", "--now", &timer_unit])?;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "installed: {} (interval={}s)",
        timer_path.display(),
        interval_seconds
    )?;
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn install_launch_agent(interval_seconds: u64) -> io::Result<()> {
    let _ = interval_seconds;
    writeln!(
        io::stderr(),
        "reaper auto-install: not supported on {}; run `claudectl --reap-orphans` \
         from cron or a custom scheduler.",
        std::env::consts::OS
    )?;
    Ok(())
}

/// Reverse of `install_launch_agent`. Tolerates "nothing was installed".
#[cfg(target_os = "macos")]
pub fn uninstall_launch_agent() -> io::Result<()> {
    let plist = plist_path()?;
    let uid = current_uid();
    match launchctl_bootout(uid) {
        Ok(true) => {}
        Ok(false) => {
            writeln!(
                io::stderr(),
                "reaper: launchctl bootout returned non-zero (likely already unloaded)"
            )?;
        }
        Err(e) => {
            writeln!(io::stderr(), "reaper: launchctl bootout exec failed: {e}")?;
        }
    }
    let _ = std::fs::remove_file(&plist);

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "uninstalled: {}", plist.display())?;
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn uninstall_launch_agent() -> io::Result<()> {
    let timer_path = systemd_timer_path()?;
    let service_path = systemd_service_path()?;
    let timer_unit = format!("{SYSTEMD_UNIT_BASENAME}.timer");

    if systemctl_available() {
        // Best-effort disable; ignore exit code because "not loaded" is fine.
        let _ = Command::new("systemctl")
            .args(["--user", "disable", "--now", &timer_unit])
            .output();
    }

    let _ = std::fs::remove_file(&timer_path);
    let _ = std::fs::remove_file(&service_path);

    if systemctl_available() {
        let _ = Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .output();
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "uninstalled: {}", timer_path.display())?;
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn uninstall_launch_agent() -> io::Result<()> {
    writeln!(
        io::stderr(),
        "reaper auto-uninstall: not supported on {}; nothing to do.",
        std::env::consts::OS
    )?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn systemctl_available() -> bool {
    Command::new("systemctl")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn systemctl_user(args: &[&str]) -> io::Result<()> {
    let mut cmd = Command::new("systemctl");
    cmd.arg("--user");
    cmd.args(args);
    let output = cmd
        .output()
        .map_err(|e| io::Error::other(format!("systemctl --user {args:?} exec failed: {e}")))?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "systemctl --user {args:?} failed (exit {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Pure renderer for the systemd `.service` unit. Snapshot-tested. The unit
/// is `Type=oneshot`: each timer firing executes `claudectl --reap-orphans`
/// to completion and exits, mirroring the launchd `StartInterval` model.
/// `StandardError=append:%h/.local/state/...` puts the error log at a
/// predictable path inside the user's HOME (resolved by systemd via `%h`).
#[cfg(any(target_os = "linux", test))]
pub fn build_systemd_service(exe_path: &Path) -> String {
    let exe = exe_path.display();
    format!(
        "[Unit]\n\
Description=claudectl orphan reaper for in-sandbox claude processes\n\
\n\
[Service]\n\
Type=oneshot\n\
ExecStart={exe} --reap-orphans\n\
Nice=5\n\
StandardOutput=null\n\
StandardError=append:%h/.local/state/claudectl-reaper.err.log\n"
    )
}

/// Pure renderer for the systemd `.timer` unit. `OnUnitActiveSec` schedules
/// the next run that many seconds after the previous run completed, which
/// matches launchd's `StartInterval` semantics. `Persistent=true` makes the
/// timer catch up on missed runs after suspend/reboot.
#[cfg(any(target_os = "linux", test))]
pub fn build_systemd_timer(interval_seconds: u64) -> String {
    format!(
        "[Unit]\n\
Description=Periodic claudectl orphan reaper\n\
\n\
[Timer]\n\
Unit={SYSTEMD_UNIT_BASENAME}.service\n\
OnUnitActiveSec={interval_seconds}s\n\
OnBootSec={interval_seconds}s\n\
Persistent=true\n\
\n\
[Install]\n\
WantedBy=timers.target\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- registry prune (end-to-end through the real scan/ps path) ------

    /// Write a live session pointer file for `pid` under `$HOME/.claude/sessions`,
    /// so `discovery::live_sessions()` reports it live.
    fn write_live_pointer(pid: u32, session_id: &str) {
        let dir = home_dir().unwrap().join(".claude").join("sessions");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(format!("{pid}.json")),
            format!(r#"{{"pid":{pid},"sessionId":"{session_id}","cwd":"/work","startedAt":1}}"#),
        )
        .unwrap();
    }

    /// The current process's own owner — a provably-alive terminal instance,
    /// resolved exactly as the reaper would.
    fn own_live_owner() -> crate::terminal_owner::TerminalOwner {
        crate::terminal_owner::ProcessTable::snapshot()
            .unwrap()
            .owner_of(std::process::id())
            .expect("this test process has a resolvable owner")
    }

    fn seed(entries: Vec<crate::sandbox_registry::SessionEntry>) {
        crate::sandbox_registry::update_local(|_| entries).unwrap();
    }

    fn registry_ids() -> Vec<String> {
        crate::sandbox_registry::load_local()
            .sessions
            .into_iter()
            .map(|e| e.session_id)
            .collect()
    }

    #[test]
    fn prune_keeps_live_and_dead_owner_drops_hand_closed() {
        let _fixture = crate::sandbox_registry::tests::TempRegistry::with_home("reaper-prune-e2e");
        // SAFETY: env access is serialized by the ENV_LOCK held by `_fixture`.
        unsafe {
            std::env::remove_var("LINERA_SANDBOX");
            std::env::remove_var("SANDBOX_NAME");
        }
        let live_owner = own_live_owner();
        write_live_pointer(std::process::id(), "live-session");

        seed(vec![
            // Live: pointer file present, this process is its pid → left alone.
            crate::sandbox_registry::SessionEntry {
                session_id: "live-session".into(),
                cwd: "/work".into(),
                transcript: String::new(),
                started_at_ms: 1,
                name: None,
                pid: Some(std::process::id()),
                owner_pid: Some(live_owner.pid),
                owner_started_at: Some(live_owner.started_at.clone()),
            },
            // Departed, owner (this test's terminal) still alive → hand-closed → pruned.
            crate::sandbox_registry::SessionEntry {
                session_id: "hand-closed".into(),
                cwd: "/work".into(),
                transcript: String::new(),
                started_at_ms: 2,
                name: None,
                pid: Some(4_000_000),
                owner_pid: Some(live_owner.pid),
                owner_started_at: Some(live_owner.started_at),
            },
            // Departed, owner gone (impossible pid) → died with its terminal → kept.
            crate::sandbox_registry::SessionEntry {
                session_id: "terminal-died".into(),
                cwd: "/work".into(),
                transcript: String::new(),
                started_at_ms: 3,
                name: None,
                pid: Some(4_000_001),
                owner_pid: Some(4_000_002),
                owner_started_at: Some("some-dead-terminal".into()),
            },
        ]);

        prune_closed_sessions_after(Duration::ZERO);

        let mut kept = registry_ids();
        kept.sort();
        assert_eq!(kept, ["live-session", "terminal-died"]);
    }

    #[test]
    fn prune_does_nothing_when_the_session_scan_fails() {
        // No `$HOME/.claude/sessions` directory → the scan errors (not "empty"),
        // and a failed scan must never be read as "everyone closed". P5.
        let _fixture =
            crate::sandbox_registry::tests::TempRegistry::with_home("reaper-prune-failclosed");
        // SAFETY: env access is serialized by the ENV_LOCK held by `_fixture`.
        unsafe {
            std::env::remove_var("LINERA_SANDBOX");
            std::env::remove_var("SANDBOX_NAME");
        }
        let live_owner = own_live_owner();
        // A hand-closed session that WOULD be pruned if the scan were trusted.
        seed(vec![crate::sandbox_registry::SessionEntry {
            session_id: "would-be-pruned".into(),
            cwd: "/work".into(),
            transcript: String::new(),
            started_at_ms: 1,
            name: None,
            pid: Some(4_000_000),
            owner_pid: Some(live_owner.pid),
            owner_started_at: Some(live_owner.started_at),
        }]);

        prune_closed_sessions_after(Duration::ZERO);

        assert_eq!(
            registry_ids(),
            ["would-be-pruned"],
            "a failed scan must leave the registry untouched"
        );
    }

    #[test]
    fn prune_is_a_noop_inside_a_sandbox() {
        let _fixture =
            crate::sandbox_registry::tests::TempRegistry::with_home("reaper-prune-sandbox");
        // SAFETY: env access is serialized by the ENV_LOCK held by `_fixture`.
        unsafe {
            std::env::set_var("LINERA_SANDBOX", "1");
            std::env::set_var("SANDBOX_NAME", "linera-agent");
        }
        // The seeded entry must be one a full prune pass WOULD drop (departed,
        // dead pid, owner alive), and the sessions dir must be readable — else
        // this test passes for the wrong reason (fail-closed abort, or an entry
        // every path keeps) and can't detect a broken in-sandbox guard.
        std::fs::create_dir_all(home_dir().unwrap().join(".claude").join("sessions")).unwrap();
        let live_owner = own_live_owner();
        seed(vec![crate::sandbox_registry::SessionEntry {
            session_id: "host-entry".into(),
            cwd: "/work".into(),
            transcript: String::new(),
            started_at_ms: 1,
            name: None,
            pid: Some(4_000_000),
            owner_pid: Some(live_owner.pid),
            owner_started_at: Some(live_owner.started_at),
        }]);

        prune_closed_sessions_after(Duration::ZERO);

        // SAFETY: still holding ENV_LOCK via `_fixture`.
        unsafe {
            std::env::remove_var("LINERA_SANDBOX");
            std::env::remove_var("SANDBOX_NAME");
        }
        assert_eq!(
            registry_ids(),
            ["host-entry"],
            "the reaper must not touch the local registry from inside a sandbox"
        );
    }

    #[test]
    fn prune_keeps_a_session_that_registered_after_both_scans() {
        // The post-scan window: a session starts after scan #2, its hook writes
        // the entry before the reaper's locked closure runs. Absent from both
        // scans, owner alive — only the in-closure now-check (its pid is
        // running) saves it. Modeled with this test's own pid and NO pointer
        // file, so the scans genuinely miss it.
        let _fixture =
            crate::sandbox_registry::tests::TempRegistry::with_home("reaper-postscan-race");
        // SAFETY: env access is serialized by the ENV_LOCK held by `_fixture`.
        unsafe {
            std::env::remove_var("LINERA_SANDBOX");
            std::env::remove_var("SANDBOX_NAME");
        }
        std::fs::create_dir_all(home_dir().unwrap().join(".claude").join("sessions")).unwrap();
        let live_owner = own_live_owner();
        seed(vec![crate::sandbox_registry::SessionEntry {
            session_id: "registered-after-scans".into(),
            cwd: "/work".into(),
            transcript: String::new(),
            started_at_ms: 1,
            name: None,
            pid: Some(std::process::id()),
            owner_pid: Some(live_owner.pid),
            owner_started_at: Some(live_owner.started_at),
        }]);

        prune_closed_sessions_after(Duration::ZERO);

        assert_eq!(
            registry_ids(),
            ["registered-after-scans"],
            "a live process's entry must survive even when both scans missed it"
        );
    }

    #[test]
    fn still_live_is_the_union_not_the_intersection() {
        // Torn-read forgiveness: present in either scan counts as live. An
        // intersection would judge a flickering session departed and prune it.
        let one = |id: &str| std::collections::HashSet::from([id.to_string()]);
        let live = still_live(one("only-in-first"), one("only-in-second"));
        assert!(live.contains("only-in-first"));
        assert!(live.contains("only-in-second"));
    }

    fn sc(pid: u32, host_tty: &str, alive: bool) -> SandboxSidecar {
        SandboxSidecar {
            pid,
            host_tty: host_tty.into(),
            alive,
            name: String::new(),
        }
    }

    fn ttys(values: &[&str]) -> HashSet<String> {
        values.iter().map(|s| (*s).to_string()).collect()
    }

    // ---- Tick-skip cache ----------------------------------------------

    #[test]
    fn state_string_is_deterministic_across_iteration_orders() {
        let a = ttys(&["/dev/ttys003", "/dev/ttys001", "/dev/ttys002"]);
        let b = ttys(&["/dev/ttys001", "/dev/ttys002", "/dev/ttys003"]);
        assert_eq!(state_string(&a), state_string(&b));
        assert_eq!(state_string(&a), "/dev/ttys001\n/dev/ttys002\n/dev/ttys003");
    }

    #[test]
    fn state_string_empty_set_is_empty_body() {
        assert_eq!(state_string(&HashSet::new()), "");
    }

    #[test]
    fn cache_skips_when_state_matches_and_within_max_age() {
        let action = decide_cache_action(
            Some("/dev/ttys001\n/dev/ttys002"),
            "/dev/ttys001\n/dev/ttys002",
            Some(Duration::from_secs(60)),
            MAX_CACHE_AGE,
        );
        assert_eq!(action, CacheAction::Skip);
    }

    #[test]
    fn cache_full_pass_when_state_changed() {
        let action = decide_cache_action(
            Some("/dev/ttys001"),
            "/dev/ttys001\n/dev/ttys002",
            Some(Duration::from_secs(60)),
            MAX_CACHE_AGE,
        );
        assert_eq!(action, CacheAction::FullPass);
    }

    #[test]
    fn cache_full_pass_when_age_exceeds_max() {
        let action = decide_cache_action(
            Some("/dev/ttys001"),
            "/dev/ttys001",
            Some(MAX_CACHE_AGE + Duration::from_secs(1)),
            MAX_CACHE_AGE,
        );
        assert_eq!(action, CacheAction::FullPass);
    }

    #[test]
    fn cache_full_pass_when_no_prev_state() {
        let action = decide_cache_action(None, "/dev/ttys001", None, MAX_CACHE_AGE);
        assert_eq!(action, CacheAction::FullPass);
    }

    #[test]
    fn cache_full_pass_when_age_unknown_even_if_state_matches() {
        // mtime-read failure (clock skew, FS oddity) → must NOT skip,
        // because we can't bound how stale the cache is.
        let action = decide_cache_action(Some("/dev/ttys001"), "/dev/ttys001", None, MAX_CACHE_AGE);
        assert_eq!(action, CacheAction::FullPass);
    }

    #[test]
    fn cache_skip_at_exact_max_age_boundary() {
        // age == max → still in window. The check is `<= max_age`.
        let action = decide_cache_action(
            Some("/dev/ttys001"),
            "/dev/ttys001",
            Some(MAX_CACHE_AGE),
            MAX_CACHE_AGE,
        );
        assert_eq!(action, CacheAction::Skip);
    }

    #[test]
    fn cache_full_pass_when_both_states_empty_but_no_prev() {
        // First-ever invocation with no host TTYs open: still need a
        // full pass to populate the cache.
        let action = decide_cache_action(None, "", None, MAX_CACHE_AGE);
        assert_eq!(action, CacheAction::FullPass);
    }

    #[test]
    fn cache_skips_when_both_states_empty_and_fresh() {
        let action =
            decide_cache_action(Some(""), "", Some(Duration::from_secs(10)), MAX_CACHE_AGE);
        assert_eq!(action, CacheAction::Skip);
    }

    #[test]
    fn alive_with_open_tty_is_current() {
        let plan = compute_orphans(&ttys(&["/dev/ttys001"]), &[sc(100, "/dev/ttys001", true)]);
        assert!(plan.kill.is_empty());
        assert!(plan.sweep.is_empty());
    }

    #[test]
    fn alive_with_closed_tty_is_kill() {
        let plan = compute_orphans(&ttys(&["/dev/ttys001"]), &[sc(200, "/dev/ttys999", true)]);
        assert_eq!(plan.kill, vec![sc(200, "/dev/ttys999", true)]);
        assert!(plan.sweep.is_empty());
    }

    #[test]
    fn dead_pid_goes_to_sweep_regardless_of_tty() {
        // dead AND tty closed
        let plan = compute_orphans(&ttys(&["/dev/ttys001"]), &[sc(300, "/dev/ttys999", false)]);
        assert!(plan.kill.is_empty());
        assert_eq!(plan.sweep, vec![sc(300, "/dev/ttys999", false)]);
        // dead AND tty still open (e.g. zombie sidecar after PID exit)
        let plan = compute_orphans(&ttys(&["/dev/ttys001"]), &[sc(301, "/dev/ttys001", false)]);
        assert!(plan.kill.is_empty());
        assert_eq!(plan.sweep, vec![sc(301, "/dev/ttys001", false)]);
    }

    #[test]
    fn tty_reuse_keeps_alive_in_open_set_kills_others() {
        // Two sidecars on /dev/ttys055: one alive AND in open_set (current),
        // one alive but on a different (closed) tty (orphan).
        let open = ttys(&["/dev/ttys055"]);
        let plan = compute_orphans(
            &open,
            &[
                sc(400, "/dev/ttys055", true),  // current
                sc(401, "/dev/ttys999", true),  // orphan (different tty, closed)
                sc(402, "/dev/ttys055", false), // disk-orphan (dead, same tty)
            ],
        );
        assert_eq!(plan.kill, vec![sc(401, "/dev/ttys999", true)]);
        assert_eq!(plan.sweep, vec![sc(402, "/dev/ttys055", false)]);
    }

    #[test]
    fn no_sidecars_means_no_orphans() {
        let plan = compute_orphans(&ttys(&["/dev/ttys001"]), &[]);
        assert!(plan.kill.is_empty());
        assert!(plan.sweep.is_empty());
    }

    #[test]
    fn no_open_ttys_means_every_alive_is_orphan() {
        let plan = compute_orphans(
            &ttys(&[]),
            &[sc(500, "/dev/ttys001", true), sc(501, "/dev/ttys002", true)],
        );
        assert_eq!(plan.kill.len(), 2);
    }

    // ---- Safety guard tests (decide_action) ----------------------------

    #[test]
    fn decide_skips_when_open_ttys_empty_and_alive_sidecars_exist() {
        // The footgun case: ps glitch returns no open TTYs but there are
        // alive sandbox claudes. compute_orphans would mark them all kill;
        // decide_action must refuse instead.
        let action = decide_action(
            &ttys(&[]),
            &[sc(500, "/dev/ttys001", true), sc(501, "/dev/ttys002", true)],
        );
        match action {
            Action::Skip(reason) => assert!(
                reason.contains("0 open host TTYs"),
                "unexpected skip reason: {reason}"
            ),
            Action::Execute(_) => panic!("must Skip when open_ttys empty and alive sidecars > 0"),
        }
    }

    #[test]
    fn decide_executes_when_open_ttys_empty_and_no_alive_sidecars() {
        // Pure dead-sweep case must still proceed even if open_ttys is
        // empty — there are no live sessions at risk.
        let action = decide_action(&ttys(&[]), &[sc(700, "/dev/ttys001", false)]);
        match action {
            Action::Execute(plan) => {
                assert_eq!(plan.kill.len(), 0);
                assert_eq!(plan.sweep.len(), 1);
            }
            Action::Skip(r) => panic!("must Execute when no alive sidecars; got Skip({r})"),
        }
    }

    #[test]
    fn decide_skips_when_kill_count_exceeds_cap() {
        // 11 alive orphans → over the cap of 10 → refuse. The user
        // intervenes manually with --dry-run.
        let mut sidecars = Vec::new();
        for i in 0..(MAX_KILLS_PER_PASS + 1) {
            sidecars.push(sc(1000 + i as u32, &format!("/dev/closed{i}"), true));
        }
        let action = decide_action(&ttys(&["/dev/ttys001"]), &sidecars);
        match action {
            Action::Skip(reason) => assert!(
                reason.contains("exceeds safety cap"),
                "unexpected skip reason: {reason}"
            ),
            Action::Execute(_) => panic!("must Skip when kill count exceeds cap"),
        }
    }

    #[test]
    fn decide_executes_at_exactly_the_cap() {
        // Boundary: exactly MAX_KILLS_PER_PASS is allowed.
        let mut sidecars = Vec::new();
        for i in 0..MAX_KILLS_PER_PASS {
            sidecars.push(sc(2000 + i as u32, &format!("/dev/closed{i}"), true));
        }
        let action = decide_action(&ttys(&["/dev/ttys001"]), &sidecars);
        match action {
            Action::Execute(plan) => assert_eq!(plan.kill.len(), MAX_KILLS_PER_PASS),
            Action::Skip(r) => panic!("must Execute at exactly cap; got Skip({r})"),
        }
    }

    // ---- Parser tests --------------------------------------------------

    #[test]
    fn extract_open_ttys_picks_sandbox_host_tty() {
        let ps = "\
  100 ?? Ss   0:00.01 /usr/sbin/sshd
  200 ?? S    0:00.05 sbx exec --env SANDBOX_HOST_TTY=/dev/ttys001 linera-agent bash
  201 ?? S    0:00.05 sbx exec --env SANDBOX_HOST_TTY=/dev/ttys055 linera-agent bash
  202 ?? S    0:00.05 sbx exec --env SANDBOX_HOST_TTY=/dev/ttys077 other-sandbox bash
  300 ?? S    0:00.05 ps -ax -o pid,command
";
        let set = extract_open_ttys(ps, "linera-agent");
        assert!(set.contains("/dev/ttys001"));
        assert!(set.contains("/dev/ttys055"));
        assert!(!set.contains("/dev/ttys077")); // wrong sandbox name
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn extract_open_ttys_honors_alternate_sandbox_name() {
        let ps = "\
  200 ?? S    0:00.05 sbx exec --env SANDBOX_HOST_TTY=/dev/ttys001 linera-agent bash
  201 ?? S    0:00.05 sbx exec --env SANDBOX_HOST_TTY=/dev/ttys055 my-team-sandbox bash
";
        let set = extract_open_ttys(ps, "my-team-sandbox");
        assert!(set.contains("/dev/ttys055"));
        assert!(!set.contains("/dev/ttys001"));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn sandbox_name_default_when_env_unset() {
        // Use a unique key so we don't disturb a real env value if a parallel
        // test happened to set CLAUDECTL_SANDBOX_NAME. Test the resolver
        // logic directly via a private helper that accepts the value.
        assert_eq!(resolve_or_default(None, "linera-agent"), "linera-agent");
        assert_eq!(resolve_or_default(Some(""), "linera-agent"), "linera-agent");
        assert_eq!(
            resolve_or_default(Some("my-team-sandbox"), "linera-agent"),
            "my-team-sandbox"
        );
    }

    #[test]
    fn resolve_sandbox_name_env_override_wins_over_auto_and_default() {
        assert_eq!(
            resolve_sandbox_name(Some("from-env"), Some("from-auto"), "default"),
            "from-env"
        );
    }

    #[test]
    fn resolve_sandbox_name_falls_back_to_auto_when_env_empty() {
        assert_eq!(
            resolve_sandbox_name(Some(""), Some("from-auto"), "default"),
            "from-auto"
        );
    }

    #[test]
    fn resolve_sandbox_name_falls_back_to_auto_when_env_none() {
        assert_eq!(
            resolve_sandbox_name(None, Some("from-auto"), "default"),
            "from-auto"
        );
    }

    #[test]
    fn resolve_sandbox_name_falls_back_to_default_when_no_signal() {
        assert_eq!(resolve_sandbox_name(None, None, "default"), "default");
    }

    #[test]
    fn resolve_sandbox_name_treats_empty_auto_as_no_signal() {
        // An empty auto-detect (parser couldn't pick a unique sandbox)
        // must fall through to the default, not to "".
        assert_eq!(resolve_sandbox_name(None, Some(""), "default"), "default");
    }

    // ---- sbx ls parser tests ------------------------------------------

    /// Verbatim `sbx ls --json` from Andre's box (v0.37.0, 2026-07-31), with
    /// the `workspaces` array shortened. Keeping the real `id`/`agent` fields
    /// is the point: they must be ignored without a serde error.
    const SBX_LS_JSON_ONE: &str = r#"{
  "sandboxes": [
    {
      "name": "linera-agent",
      "id": "2c6cf266-983b-4381-8090-e12c01fb3170",
      "agent": "claude",
      "status": "running",
      "workspaces": ["/Users/ndr/repos", "/Users/ndr/.claude"]
    }
  ]
}"#;

    #[test]
    fn parse_sbx_ls_returns_single_running_sandbox() {
        assert_eq!(
            parse_sbx_ls_for_single_running_sandbox(SBX_LS_JSON_ONE),
            Some("linera-agent".to_string())
        );
    }

    #[test]
    fn parse_sbx_ls_returns_none_when_no_sandboxes() {
        assert_eq!(
            parse_sbx_ls_for_single_running_sandbox(r#"{"sandboxes": []}"#),
            None
        );
    }

    #[test]
    fn parse_sbx_ls_returns_none_when_multiple_running() {
        let stdout = r#"{"sandboxes": [
            {"name": "sbx-a", "agent": "claude", "status": "running"},
            {"name": "sbx-b", "agent": "claude", "status": "running"}
        ]}"#;
        assert_eq!(parse_sbx_ls_for_single_running_sandbox(stdout), None);
    }

    #[test]
    fn parse_sbx_ls_returns_none_when_only_stopped() {
        let stdout = r#"{"sandboxes": [
            {"name": "linera-agent", "agent": "claude", "status": "stopped"}
        ]}"#;
        assert_eq!(parse_sbx_ls_for_single_running_sandbox(stdout), None);
    }

    #[test]
    fn parse_sbx_ls_returns_running_when_one_running_one_stopped() {
        // A stopped sandbox is not a candidate; the lone running one is.
        let stdout = r#"{"sandboxes": [
            {"name": "linera-agent", "agent": "claude", "status": "running"},
            {"name": "old-sbx", "agent": "claude", "status": "stopped"}
        ]}"#;
        assert_eq!(
            parse_sbx_ls_for_single_running_sandbox(stdout),
            Some("linera-agent".to_string())
        );
    }

    #[test]
    fn parse_sbx_ls_returns_none_for_empty_input() {
        assert_eq!(parse_sbx_ls_for_single_running_sandbox(""), None);
    }

    #[test]
    fn parse_sbx_ls_returns_none_for_garbage_input() {
        // Unparseable JSON → empty list → None, never a panic.
        assert_eq!(
            parse_sbx_ls_for_single_running_sandbox("totally bogus\noutput here\n"),
            None
        );
        // Well-formed JSON of the wrong shape must also degrade, not explode.
        assert_eq!(parse_sbx_ls_for_single_running_sandbox("[]"), None);
        assert_eq!(parse_sbx_ls_for_single_running_sandbox("{}"), None);
    }

    #[test]
    fn parse_sbx_ls_ignores_unknown_fields() {
        // A future `sbx` adding columns must not break the parse.
        let stdout = r#"{"sandboxes": [
            {"name": "linera-agent", "status": "running", "brand_new_field": 42}
        ], "some_new_top_level": true}"#;
        assert_eq!(
            parse_sbx_ls_for_single_running_sandbox(stdout),
            Some("linera-agent".to_string())
        );
    }

    /// Several running sandboxes is the whole point of the collector, and it is
    /// exactly the case the single-sandbox resolver deliberately refuses.
    const SBX_LS_THREE: &str = r#"{"sandboxes": [
        {"name": "linera-agent-a3f11b28", "agent": "claude", "status": "running"},
        {"name": "linera-agent-251d6f7c", "agent": "claude", "status": "running"},
        {"name": "old-task", "agent": "claude", "status": "stopped"},
        {"name": "linera-agent", "agent": "claude", "status": "running"}
    ]}"#;

    #[test]
    fn running_names_lists_every_running_sandbox_in_order() {
        assert_eq!(
            parse_sbx_ls_running_names(SBX_LS_THREE),
            vec![
                "linera-agent-a3f11b28".to_string(),
                "linera-agent-251d6f7c".to_string(),
                "linera-agent".to_string(),
            ]
        );
    }

    #[test]
    fn running_names_and_single_resolver_agree_on_the_one_sandbox_case() {
        // The two parsers share a body; assert they can't drift apart rather
        // than trusting that they won't.
        let stdout = SBX_LS_JSON_ONE;
        let names = parse_sbx_ls_running_names(stdout);
        assert_eq!(names.len(), 1);
        assert_eq!(
            parse_sbx_ls_for_single_running_sandbox(stdout),
            Some(names[0].clone())
        );
    }

    #[test]
    fn current_sandbox_prefers_a_pointer_that_is_actually_running() {
        let running = parse_sbx_ls_running_names(SBX_LS_THREE);
        assert_eq!(
            resolve_current_sandbox(Some("linera-agent-251d6f7c"), &running),
            Some("linera-agent-251d6f7c".to_string())
        );
    }

    #[test]
    fn current_sandbox_ignores_a_pointer_naming_a_dead_sandbox() {
        // A pointer left behind by a reaped sandbox must not win: marking a
        // corpse current would strand every live sandbox as "superseded".
        // Several are running, so there is no unambiguous fallback either.
        let running = parse_sbx_ls_running_names(SBX_LS_THREE);
        assert_eq!(resolve_current_sandbox(Some("long-gone"), &running), None);
    }

    #[test]
    fn current_sandbox_falls_back_to_the_only_running_sandbox() {
        // Today's world, before the wrapper writes any pointer.
        let running = vec!["linera-agent".to_string()];
        assert_eq!(
            resolve_current_sandbox(None, &running),
            Some("linera-agent".to_string())
        );
    }

    #[test]
    fn current_sandbox_refuses_to_guess_among_several() {
        let running = parse_sbx_ls_running_names(SBX_LS_THREE);
        assert_eq!(resolve_current_sandbox(None, &running), None);
    }

    #[test]
    fn current_sandbox_is_none_when_nothing_runs() {
        assert_eq!(resolve_current_sandbox(None, &[]), None);
        assert_eq!(resolve_current_sandbox(Some("linera-agent"), &[]), None);
    }

    fn registry_entry(pid: u32, id: &str, name: Option<&str>) -> sandbox_registry::SessionEntry {
        sandbox_registry::SessionEntry {
            session_id: id.to_string(),
            cwd: "/Users/ndr/repos/linera-infra".to_string(),
            transcript: String::new(),
            started_at_ms: 1_754_150_400_000,
            name: name.map(str::to_string),
            pid: Some(pid),
            owner_pid: None,
            owner_started_at: None,
        }
    }

    fn sample(pid: u32, cpu_percent: f32, mem_mb: f64) -> SandboxProc {
        SandboxProc {
            pid,
            cpu_percent,
            mem_mb,
        }
    }

    fn live_map(samples: Vec<SandboxProc>) -> HashMap<u32, SandboxProc> {
        samples.into_iter().map(|proc| (proc.pid, proc)).collect()
    }

    /// Verbatim `ps -o pid=,%cpu=,rss=,command=` output from inside a sandbox:
    /// right-aligned numeric columns, a full argv tail, and one non-claude
    /// process sharing the pid space.
    const PS_PROBE_OUTPUT: &str = "\
  11036   3.4 1048576 claude --dangerously-skip-permissions
  11200   0.0  524288 /usr/local/bin/claude --resume 9905252f-e0aa-43d0-b578-c3023b36b2fb
  11311  91.2  262144 node /opt/claudectl/bin/claudectl --json
";

    #[test]
    fn probe_parser_reads_cpu_and_memory_for_every_claude_row() {
        let procs = parse_sandbox_procs(PS_PROBE_OUTPUT);
        assert_eq!(procs.len(), 2, "parsed pids: {:?}", {
            let mut pids: Vec<u32> = procs.keys().copied().collect();
            pids.sort_unstable();
            pids
        });
        assert_eq!(procs[&11036].cpu_percent, 3.4);
        // rss is KiB; the renderer wants MB.
        assert_eq!(procs[&11036].mem_mb, 1024.0);
        assert_eq!(procs[&11200].cpu_percent, 0.0);
        assert_eq!(procs[&11200].mem_mb, 512.0);
    }

    #[test]
    fn probe_parser_drops_a_recycled_pid_running_something_else() {
        // 11311 is `claudectl`, not `claude`. Reporting it would attribute a
        // stranger's 91% CPU to a session that has actually exited — and, worse,
        // keep that dead session on screen forever.
        let procs = parse_sandbox_procs(PS_PROBE_OUTPUT);
        assert!(!procs.contains_key(&11311));
        // The same guard on the shapes that make a substring check wrong.
        for command in [
            "grep claude",
            "bash -lc 'exec sandbox-bootstrap claude --resume foo'",
            "claudectl --reap-orphans",
        ] {
            assert!(
                parse_sandbox_procs(&format!("  4242   1.0  1024 {command}")).is_empty(),
                "{command} must not count as a live claude session"
            );
        }
    }

    #[test]
    fn probe_parser_skips_malformed_rows_without_losing_the_good_ones() {
        let text = "\
notapid   1.0  1024 claude
7777 only-three fields
8888   2.5  2048 claude --resume x
   \n";
        let procs = parse_sandbox_procs(text);
        assert_eq!(procs.keys().copied().collect::<Vec<u32>>(), vec![8888]);
    }

    #[test]
    fn probe_parser_tolerates_empty_output() {
        // What `ps` prints when every recorded pid has exited.
        assert!(parse_sandbox_procs("").is_empty());
    }

    #[test]
    fn collected_row_carries_the_identity_the_registry_recorded() {
        let entries = vec![registry_entry(
            11036,
            "9905252f-e0aa-43d0-b578-c3023b36b2fb",
            Some("argo-validator-migration-strategy"),
        )];
        let rows = sessions_from_registry(&entries, &live_map(vec![sample(11036, 3.4, 1024.0)]));
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0]["session_id"], "9905252f-e0aa-43d0-b578-c3023b36b2fb",
            "without an id the renderer drops the row entirely"
        );
        assert_eq!(rows[0]["session_name"], "argo-validator-migration-strategy");
        assert_eq!(rows[0]["cwd"], "/Users/ndr/repos/linera-infra");
        assert_eq!(rows[0]["pid"], 11036);
        assert_eq!(rows[0]["started_at"], 1_754_150_400_000u64);
        assert_eq!(rows[0]["cpu"], 3.4);
        assert_eq!(rows[0]["mem_mb"], 1024.0);
    }

    #[test]
    fn collected_row_survives_from_snapshot_value() {
        // End to end: what the collector writes is what the renderer accepts.
        let entries = vec![registry_entry(11036, "abc-123", Some("my-title"))];
        let rows = sessions_from_registry(&entries, &live_map(vec![sample(11036, 7.5, 256.0)]));
        let session = crate::session::ClaudeSession::from_snapshot_value("linera-agent", &rows[0])
            .expect("a collected row must be renderable");
        assert_eq!(session.session_id, "abc-123");
        assert_eq!(session.display_name(), "my-title");
        assert_eq!(session.cwd, "/Users/ndr/repos/linera-infra");
        assert_eq!(session.pid, 11036);
        assert_eq!(session.cpu_percent, 7.5);
        assert_eq!(session.mem_mb, 256.0);
        assert_eq!(
            session.origin,
            crate::session::SessionOrigin::Sandbox("linera-agent".into())
        );
    }

    #[test]
    fn an_unnamed_session_still_renders_under_its_project() {
        let entries = vec![registry_entry(11036, "abc-123", None)];
        let rows = sessions_from_registry(&entries, &live_map(vec![sample(11036, 0.0, 1.0)]));
        let session = crate::session::ClaudeSession::from_snapshot_value("linera-agent", &rows[0])
            .expect("an unnamed session is still identifiable");
        assert_eq!(session.session_name, "");
        assert_eq!(session.display_name(), "linera-infra");
    }

    #[test]
    fn a_departed_session_is_left_out_of_the_snapshot() {
        // The slice is a mirror frozen at the last hook fire (and forever, once
        // `sbx rm` runs), so "recorded" never implies "still running".
        let entries = vec![
            registry_entry(11036, "alive", Some("still-here")),
            registry_entry(999, "departed", Some("long-gone")),
        ];
        let rows = sessions_from_registry(&entries, &live_map(vec![sample(11036, 1.0, 8.0)]));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["session_id"], "alive");
    }

    #[test]
    fn entries_without_identity_or_pid_are_skipped() {
        let mut no_pid = registry_entry(1, "no-pid", None);
        no_pid.pid = None;
        let entries = vec![registry_entry(11036, "", Some("no-id")), no_pid];
        let rows = sessions_from_registry(
            &entries,
            &live_map(vec![sample(11036, 1.0, 8.0), sample(1, 1.0, 8.0)]),
        );
        assert!(
            rows.is_empty(),
            "an unidentifiable row only inflates the file: {rows:?}"
        );
    }

    #[test]
    fn empty_registry_slice_collects_nothing() {
        assert!(sessions_from_registry(&[], &live_map(vec![sample(1, 1.0, 1.0)])).is_empty());
    }

    /// Spawn a `sleep`, kill it and reap it. Its pid is then definitively dead
    /// *and* in range — which an arbitrary large number is not. BSD `ps`
    /// rejects an out-of-range pid by discarding the WHOLE `-p` list, so
    /// spelling "dead pid" as `999999999` made this file's probe tests pass on
    /// procps and fail on macOS.
    fn reaped_pid() -> u32 {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn a process to then bury");
        let pid = child.id();
        child.kill().unwrap();
        child.wait().unwrap();
        pid
    }

    #[test]
    fn probe_script_prints_the_columns_the_parser_expects() {
        // Pins the WIRING, not just the parser: the script is executed for
        // real, so a reordered `-o` list or a header that stopped being
        // suppressed fails here rather than silently blanking CPU and MEM for
        // every foreign row. Runs on both CI runners, so it covers BSD ps and
        // procps alike.
        //
        // The dead pid alongside the live one is the production steady state,
        // not decoration: a registry slice routinely names sessions that have
        // since exited, and `ps` must still report the survivors.
        let dead = reaped_pid();
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn a probe target");
        let pid = child.id();
        let out = std::process::Command::new("bash")
            .args([
                "-c",
                PROC_PROBE_SCRIPT,
                "--",
                &pid.to_string(),
                &dead.to_string(),
            ])
            .output()
            .expect("run the probe script");
        let _ = child.kill();
        let _ = child.wait();

        assert!(
            out.status.success(),
            "probe script must exit 0 even with a dead pid in the list: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let rows: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(rows.len(), 1, "exactly the live pid, got {stdout:?}");
        let fields: Vec<&str> = rows[0].split_whitespace().collect();
        assert_eq!(fields[0].parse::<u32>().ok(), Some(pid), "column 1 is pid");
        assert!(fields[1].parse::<f32>().is_ok(), "column 2 is %cpu");
        assert!(fields[2].parse::<f64>().is_ok(), "column 3 is rss");
        assert!(
            fields[3..].join(" ").contains("sleep"),
            "column 4+ is the command line: {rows:?}"
        );
    }

    #[test]
    fn probe_script_exits_clean_with_no_pids_at_all() {
        let out = std::process::Command::new("bash")
            .args(["-c", PROC_PROBE_SCRIPT, "--"])
            .output()
            .expect("run the probe script");
        assert!(out.status.success());
        assert!(out.stdout.is_empty(), "no pids means no rows");
    }

    #[test]
    fn probe_script_exits_clean_when_every_recorded_pid_is_gone() {
        // `ps` exits non-zero when NONE of its pids exist, and that is the
        // ordinary steady state for a sandbox whose sessions have all ended.
        // Letting the status escape would make `probe_sandbox_procs` call the
        // whole sandbox a collection failure — and log one — on every tick,
        // for exactly the sandboxes where there was nothing to report.
        let (first, second) = (reaped_pid(), reaped_pid());
        let out = std::process::Command::new("bash")
            .args([
                "-c",
                PROC_PROBE_SCRIPT,
                "--",
                &first.to_string(),
                &second.to_string(),
            ])
            .output()
            .expect("run the probe script");
        assert!(
            out.status.success(),
            "an all-dead pid list is not an error: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(parse_sandbox_procs(&String::from_utf8_lossy(&out.stdout)).is_empty());
    }

    #[test]
    fn a_sandbox_with_no_recorded_sessions_is_never_probed() {
        // Each `sbx exec` is seconds of wrapper startup, paid per running
        // sandbox per tick, so the empty case must short-circuit before it.
        // The name is deliberately one no sandbox can have: reaching the exec
        // at all — whether `sbx` is absent (spawn error) or present (unknown
        // sandbox) — surfaces as `Err` and fails this assertion.
        let collected = collect_one_sandbox("claudectl-no-such-sandbox-4f2a91b7", &[]);
        assert_eq!(
            collected.expect("an empty registry slice must not shell out"),
            Vec::<serde_json::Value>::new()
        );
    }

    #[test]
    fn probe_script_and_parser_agree_on_a_real_claude_process() {
        // Full path, no fixtures: a process whose argv0 basename is `claude`,
        // probed by the real script and read by the real parser. This is the
        // only test that would catch the script and the parser drifting apart
        // in a way that still leaves both individually plausible.
        //
        // `exec -a` renames argv0 rather than staging a binary called `claude`:
        // copying `/bin/sleep` under that name works on a stock GNU/BSD box but
        // not where coreutils is the multi-call uutils build, which dispatches
        // on argv0 and exits with "unknown program 'claude'".
        let mut child = std::process::Command::new("bash")
            .args(["-c", r#"exec -a "$0" sleep 30"#, "/opt/sandbox/bin/claude"])
            .spawn()
            .expect("spawn a process posing as claude");
        let pid = child.id();

        // Poll rather than probe once. `spawn` returns as soon as *bash* is
        // running, and bash has yet to `exec`; during that window `ps` cannot
        // read the new `/proc/<pid>/cmdline` and falls back to the bracketed
        // `[sleep]` form, which the parser then drops as "not claude" —
        // correctly, since that is exactly the recycled-pid case it exists to
        // reject. The single immediate probe made this test a coin flip on a
        // loaded runner: it failed 1 run in 12 locally and took CI down twice
        // on a branch that never touched the probe.
        //
        // Waiting for the argv0 the test itself asked for is not a weakened
        // assertion — the assertion IS that the script and parser agree on a
        // process named `claude`, and until the exec lands there is no such
        // process to agree about.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let (procs, last_stdout) = loop {
            let out = std::process::Command::new("bash")
                .args(["-c", PROC_PROBE_SCRIPT, "--", &pid.to_string()])
                .output()
                .expect("run the probe script");
            let procs = parse_sandbox_procs(&String::from_utf8_lossy(&out.stdout));
            if procs.contains_key(&pid) || std::time::Instant::now() >= deadline {
                break (procs, out.stdout);
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let _ = child.kill();
        let _ = child.wait();

        let observed = procs.get(&pid).unwrap_or_else(|| {
            panic!(
                "pid {pid} never appeared as `claude` within 10s; last probe said {:?}",
                String::from_utf8_lossy(&last_stdout)
            )
        });
        assert!(
            observed.mem_mb > 0.0,
            "a running process has resident memory"
        );
    }

    #[test]
    fn regression_scan_script_drops_derived_placeholder_names() {
        // Executed against a fixture dir with plain bash. The reaper's grep
        // pipeline read the pointer's raw "name" and printed Claude Code's
        // auto-derived placeholder ("ndr-5e") in reap logs; it must honor
        // nameSource:"derived" the same way RawSession::title() does.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("101.terminal.json"),
            r#"{"host_tty":"/dev/ttys038"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("101.json"),
            r#"{"pid":101,"name":"ndr-5e","nameSource":"derived"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("202.terminal.json"),
            r#"{"host_tty":"/dev/ttys039"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("202.json"),
            r#"{"pid":202,"name":"real-title"}"#,
        )
        .unwrap();
        let script = sidecar_scan_script(dir.path().to_str().unwrap());
        let out = std::process::Command::new("bash")
            .args(["-c", &script])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "scan script failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let sidecars = parse_sandbox_sidecars(&String::from_utf8_lossy(&out.stdout));
        let name_of = |pid: u32| {
            sidecars
                .iter()
                .find(|s| s.pid == pid)
                .unwrap_or_else(|| panic!("pid {pid} missing from scan output"))
                .name
                .clone()
        };
        assert_eq!(name_of(101), "", "derived placeholder must not label logs");
        assert_eq!(name_of(202), "real-title");
    }

    #[test]
    fn parse_sandbox_sidecars_basic() {
        let text = "\
123\t/dev/ttys001\t1\tfix-validator-oom
456\t/dev/ttys999\t0\t
789\tnot-a-tty\t1\t
";
        let parsed = parse_sandbox_sidecars(text);
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].pid, 123);
        assert_eq!(parsed[0].host_tty, "/dev/ttys001");
        assert!(parsed[0].alive);
        assert_eq!(parsed[0].name, "fix-validator-oom");
        assert!(!parsed[1].alive);
        assert_eq!(parsed[1].name, "");
        assert!(parsed[2].alive);
    }

    #[test]
    fn parse_sandbox_sidecars_skips_malformed() {
        let text = "\
notapid\t/dev/ttys001\t1\t
123\tonly-two-fields
\t\t\t
";
        let parsed = parse_sandbox_sidecars(text);
        // first line: pid not numeric → skipped
        // second line: only 2 columns → skipped (alive_s missing)
        // third line: empty pid → not parseable → skipped
        assert_eq!(parsed.len(), 0);
    }

    // ---- Plist generator -----------------------------------------------

    /// Snapshot test pinned to the exact body of the hand-written plist
    /// driving Andre's auto-reaper since 2026-04-26. Drift here means a new
    /// install would not be byte-equivalent to the existing one — caller
    /// must update both intentionally.
    #[test]
    fn build_plist_matches_known_good_snapshot() {
        let exe = std::path::PathBuf::from("/Users/ndr/.cargo/bin/claudectl");
        let err = std::path::PathBuf::from("/Users/ndr/Library/Logs/claudectl-reaper.err.log");
        let home = std::path::PathBuf::from("/Users/ndr");
        let body = build_plist(&exe, 60, &err, &home);
        let expected = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>linera.claudectl-reaper</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/ndr/.cargo/bin/claudectl</string>
        <string>--reap-orphans</string>
    </array>

    <key>StartInterval</key>
    <integer>60</integer>

    <key>RunAtLoad</key>
    <false/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>/Users/ndr</string>
    </dict>

    <key>StandardOutPath</key>
    <string>/dev/null</string>

    <key>StandardErrorPath</key>
    <string>/Users/ndr/Library/Logs/claudectl-reaper.err.log</string>

    <key>ProcessType</key>
    <string>Background</string>

    <key>Nice</key>
    <integer>5</integer>
</dict>
</plist>
"#;
        assert_eq!(body, expected);
    }

    #[test]
    fn build_plist_substitutes_interval() {
        let exe = std::path::PathBuf::from("/x/y");
        let err = std::path::PathBuf::from("/e");
        let home = std::path::PathBuf::from("/h");
        let body = build_plist(&exe, 120, &err, &home);
        assert!(body.contains("<integer>120</integer>"));
        assert!(body.contains("<string>/x/y</string>"));
        assert!(body.contains("<string>/e</string>"));
        assert!(body.contains("<string>/h</string>"));
    }

    // ---- systemd unit generators --------------------------------------

    #[test]
    fn build_systemd_service_matches_known_good_snapshot() {
        let exe = std::path::PathBuf::from("/home/dev/.cargo/bin/claudectl");
        let body = build_systemd_service(&exe);
        let expected = "[Unit]\n\
Description=claudectl orphan reaper for in-sandbox claude processes\n\
\n\
[Service]\n\
Type=oneshot\n\
ExecStart=/home/dev/.cargo/bin/claudectl --reap-orphans\n\
Nice=5\n\
StandardOutput=null\n\
StandardError=append:%h/.local/state/claudectl-reaper.err.log\n";
        assert_eq!(body, expected);
    }

    #[test]
    fn build_systemd_timer_matches_known_good_snapshot() {
        let body = build_systemd_timer(60);
        let expected = "[Unit]\n\
Description=Periodic claudectl orphan reaper\n\
\n\
[Timer]\n\
Unit=claudectl-reaper.service\n\
OnUnitActiveSec=60s\n\
OnBootSec=60s\n\
Persistent=true\n\
\n\
[Install]\n\
WantedBy=timers.target\n";
        assert_eq!(body, expected);
    }

    #[test]
    fn build_systemd_timer_substitutes_interval() {
        let body = build_systemd_timer(300);
        assert!(body.contains("OnUnitActiveSec=300s"));
        assert!(body.contains("OnBootSec=300s"));
        assert!(body.contains("Persistent=true"));
        assert!(body.contains("WantedBy=timers.target"));
    }
}
