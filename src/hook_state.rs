//! Per-session deterministic state, populated by Claude Code hook callbacks.
//!
//! Claude Code does not write a permission-pending tool_use to the session JSONL
//! until the user approves it, so the JSONL alone cannot tell us whether a session
//! is sitting on a permission prompt. The `Notification` hook (matcher
//! `permission_prompt`) fires the moment that prompt opens, and `PreToolUse`,
//! `UserPromptSubmit`, and `Stop` fire when the prompt resolves. By recording
//! those events to a per-session JSON file, `infer_status` can return a
//! deterministic answer instead of guessing from CPU + JSONL tail.
//!
//! State files live at `~/.claudectl/state/<session_id>.json`. The file is
//! tiny (a few hundred bytes) and rewritten atomically on each hook event.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Per-session hook event timestamps and last-known prompt context.
///
/// All `last_*_ts_ms` fields are unix epoch milliseconds at the moment the
/// hook fired. A zero value means "never seen". `notification_kind` and
/// `current_tool_name` carry payload context from the most recent
/// `Notification` / `PreToolUse` events respectively.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookState {
    pub session_id: String,
    #[serde(default)]
    pub last_notification_ts_ms: u64,
    /// `notification_type` from the last Notification payload — e.g.
    /// `"permission_prompt"` (main agent), `"worker_permission_prompt"`
    /// (subagent), `"idle_prompt"`, `"auth_success"`.
    #[serde(default)]
    pub notification_kind: Option<String>,
    #[serde(default)]
    pub last_pretooluse_ts_ms: u64,
    #[serde(default)]
    pub last_posttooluse_ts_ms: u64,
    #[serde(default)]
    pub last_stop_ts_ms: u64,
    #[serde(default)]
    pub last_promptsubmit_ts_ms: u64,
    #[serde(default)]
    pub last_precompact_ts_ms: u64,
    /// `PostCompact` fires directly when auto-compact finishes — a more
    /// reliable "compaction done" signal than relying on `Stop`, which has
    /// been observed to never fire for sessions whose first turn triggers an
    /// auto-compact.
    #[serde(default)]
    pub last_postcompact_ts_ms: u64,
    #[serde(default)]
    pub last_subagentstop_ts_ms: u64,
    #[serde(default)]
    pub last_session_start_ts_ms: u64,
    #[serde(default)]
    pub last_session_end_ts_ms: u64,
    /// Tool name from the most recent `PreToolUse` payload (cleared by
    /// `PostToolUse`).
    #[serde(default)]
    pub current_tool_name: Option<String>,
}

/// Returns the directory holding per-session state files. Creates it if needed.
///
/// Honors the `CLAUDECTL_STATE_DIR` env var when set — used by tests to avoid
/// stomping on the real `~/.claudectl/state` directory.
pub fn state_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CLAUDECTL_STATE_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join(".claudectl/state")
}

/// Path to one session's state file.
fn state_path(session_id: &str) -> PathBuf {
    state_dir().join(format!("{session_id}.json"))
}

/// Process-wide monotonic millisecond timestamp.
///
/// Two `record_hook_event` calls back-to-back in the same process easily
/// land in the same wall-clock millisecond (system clock granularity ≈ ms;
/// hot-binary record-then-record is sub-ms). When that happens, `is_responding`
/// and `is_waiting_for_user` (both compare timestamps with strict `>`) become
/// order-blind: whichever check runs first wins, regardless of which event
/// was recorded later. That's how tests that record Stop then UserPromptSubmit
/// flake — both ts_ms are equal so neither comparison is strict-greater.
///
/// The fix is to enforce strictly-increasing per-process timestamps. The
/// atomic holds the most-recently-issued ms; the next call returns
/// `max(real_now_ms, last + 1)` and updates the atomic. Cross-process the
/// drift is bounded to a single process's burst of events (sub-millisecond
/// in practice), and Claude Code's hooks run in separate processes anyway,
/// so production sessions see real wall-clock timestamps with at most a few
/// ms of drift on rapid event bursts within one hook script.
fn now_ms() -> u64 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let real = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut last = LAST.load(Ordering::Relaxed);
    loop {
        let next = real.max(last + 1);
        match LAST.compare_exchange(last, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(observed) => last = observed,
        }
    }
}

impl HookState {
    /// Read state for a session. Returns `None` if no file exists yet.
    pub fn load(session_id: &str) -> Option<Self> {
        let path = state_path(session_id);
        let content = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Atomically write state to disk (write-temp + rename).
    fn save(&self) -> io::Result<()> {
        let dir = state_dir();
        fs::create_dir_all(&dir)?;
        let final_path = state_path(&self.session_id);
        let tmp_path = dir.join(format!(
            ".{}.json.tmp.{}",
            self.session_id,
            std::process::id()
        ));
        let json = serde_json::to_string(self).map_err(io::Error::other)?;
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Delete this session's state file (called on `SessionEnd`).
    fn remove(session_id: &str) -> io::Result<()> {
        let path = state_path(session_id);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }
}

/// Try to parse stdin as a Claude Code hook payload — without ever blocking.
///
/// Why this is harder than it looks: Claude Code hooks pipe a JSON payload on
/// stdin and close the writer side immediately, so EOF arrives within
/// microseconds. But many *non-hook* invocations also have a non-tty stdin —
/// e.g. `claudectl --json` run in a subshell, backgrounded, or invoked by
/// another script — and that stdin may stay open with no writer for the
/// entire run. A blind `read_to_string` would hang forever in those cases
/// (we hit this on the first install — `claudectl --json` from a backgrounded
/// shell never returned).
///
/// So: tty stdin ⇒ definitely not a hook. Otherwise poll(POLLIN, 50ms); if
/// no data is available in that window it's not a hook either. Hook payloads
/// are buffered and closed by the parent before we even start, so 50ms is
/// luxuriously generous.
pub fn try_read_hook_payload() -> io::Result<Option<serde_json::Value>> {
    use std::os::fd::AsRawFd;
    let fd = io::stdin().as_raw_fd();

    // SAFETY: isatty is always safe to call on a valid file descriptor.
    if unsafe { libc::isatty(fd) } == 1 {
        return Ok(None);
    }

    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll on a single valid fd with a finite timeout.
    let rc = unsafe { libc::poll(&mut pfd as *mut libc::pollfd, 1, 50) };
    if rc <= 0 || (pfd.revents & libc::POLLIN) == 0 {
        return Ok(None);
    }

    let mut buf = String::new();
    io::stdin().take(1024 * 1024).read_to_string(&mut buf)?;
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return Ok(None);
    };
    if value
        .get("hook_event_name")
        .and_then(|v| v.as_str())
        .is_none()
    {
        return Ok(None);
    }
    Ok(Some(value))
}

/// Record the live host (or sandbox) sessions into the restore registry.
///
/// This is the hook write, and it only ever adds and refreshes — it never
/// forgets. A hook fires from one session while a *different* terminal may be
/// mid-quit, so no hook is in a position to decide a session is gone for good;
/// that verdict belongs to the reaper (see [`crate::reaper`]), which samples
/// after the dust settles. Departed entries are therefore kept verbatim here.
///
/// Host side is a merge; inside a sandbox it stays a plain mirror of the live
/// set (`sbx rm` fires no hooks, so that slice freezes by itself, and container
/// pids say nothing about the host terminal anyway). Routing is on the sandbox
/// marker rather than a slice name, so a sandbox can never write the host's
/// file — even one named "host".
pub fn record_live_sessions(check: &crate::terminal_owner::OwnerCheck) -> io::Result<()> {
    let sandbox = crate::sandbox_registry::current_sandbox();
    // Read only to reuse owners already resolved; the authoritative copy is
    // re-read under the lock in update_local below.
    let known = match sandbox {
        Some(_) => Vec::new(),
        None => crate::sandbox_registry::load_local().sessions,
    };

    let live: Vec<_> = crate::discovery::live_sessions()
        .into_iter()
        .map(|session| {
            let owner = resolve_owner(&known, &session, sandbox.is_some(), check);
            let transcript = crate::discovery::transcript_path(&session.session_id, &session.cwd)
                .to_string_lossy()
                .into_owned();
            let name = (!session.session_name.is_empty()).then_some(session.session_name);
            crate::sandbox_registry::SessionEntry {
                session_id: session.session_id,
                cwd: session.cwd,
                transcript,
                started_at_ms: session.started_at,
                name,
                pid: Some(session.pid),
                owner_pid: owner.as_ref().map(|owner| owner.pid),
                owner_started_at: owner.map(|owner| owner.started_at),
            }
        })
        .collect();

    match sandbox {
        Some(sandbox) => crate::sandbox_registry::replace_sandbox_slice(&sandbox, live),
        None => crate::sandbox_registry::update_local(|previous| {
            crate::sandbox_registry::merge_live_keep_all(previous, live)
        }),
    }
}

/// The owner to record for a live session.
///
/// Inside a sandbox: none — a container pid says nothing about the host
/// terminal. On the host: reuse the owner already recorded for this exact
/// process (`matches_process`) — including a recorded `None`, so a session we
/// once failed to attribute isn't re-sampled on every event — and pay for a
/// `ps` only when meeting a genuinely new process.
///
/// A cached owner is reused only while its process still exists. A session can
/// outlive its terminal without departing — iTerm2's session restoration keeps
/// the whole `login → shell → claude` tree alive under `iTermServer` across an
/// app crash or update-relaunch — and an owner frozen at record time would then
/// stay dead forever, turning every later hand-close of that session into
/// "died with its terminal" restore material. Re-walking once the owner is gone
/// lands on whatever now anchors the survivor (for iTerm2, the server), whose
/// liveness gives correct verdicts again. The aliveness probe is a `kill -0`,
/// not a `ps`, so the zero-fork steady state stands.
fn resolve_owner(
    known: &[crate::sandbox_registry::SessionEntry],
    session: &crate::session::ClaudeSession,
    in_sandbox: bool,
    check: &crate::terminal_owner::OwnerCheck,
) -> Option<crate::terminal_owner::TerminalOwner> {
    if in_sandbox {
        return None;
    }
    match known.iter().find(|entry| {
        entry.session_id == session.session_id
            && entry.matches_process(session.pid, session.started_at)
    }) {
        Some(cached) => match cached.owner() {
            Some(owner) if !crate::discovery::pid_alive(owner.pid) => check.owner_of(session.pid),
            cached_owner => cached_owner,
        },
        None => check.owner_of(session.pid),
    }
}

/// Apply a Claude Code hook payload to the per-session state file.
///
/// Unknown event names are ignored (best-effort — Claude Code may add new
/// events that we haven't wired up yet, and that's fine). Payloads without a
/// `session_id` are also ignored, since we have nothing to key on.
pub fn record_hook_event(payload: &serde_json::Value) -> io::Result<()> {
    let Some(session_id) = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
    else {
        return Ok(());
    };
    let Some(event) = payload.get("hook_event_name").and_then(|v| v.as_str()) else {
        return Ok(());
    };

    // SessionEnd is the one event that removes state instead of updating it.
    // It also deliberately skips the registry write below: SessionEnd fires
    // while a session tears down — a terminal-app quit fires it for every
    // session at once, with pointer files vanishing as the live set collapses
    // to empty — and recording then erases exactly the entries
    // `--restore-sessions` needs seconds later. Forgetting a session is left to
    // the reaper, which runs on a timer and can tell one the user closed from
    // one its terminal took down with it.
    if event == "SessionEnd" {
        return HookState::remove(&session_id);
    }

    // Session registry: every other hook event records the live set, so
    // `--restore-sessions` (host) / `--restore-sbx-sessions` bring back the
    // right sessions after a Ghostty restart / `sbx rm`. Best-effort — registry
    // I/O never blocks the hook, and it only adds/refreshes (never forgets), so
    // the steady state costs no `ps` at all.
    let _ = record_live_sessions(&crate::terminal_owner::OwnerCheck::lazy());

    let mut state = HookState::load(&session_id).unwrap_or_default();
    state.session_id = session_id;
    let ts = now_ms();

    match event {
        "Notification" => {
            state.last_notification_ts_ms = ts;
            state.notification_kind = payload
                .get("notification_type")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
        }
        "PreToolUse" => {
            state.last_pretooluse_ts_ms = ts;
            state.current_tool_name = payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            // PreToolUse means the prompt resolved (approved). Clear the
            // permission_prompt notification so infer_status doesn't keep
            // reporting NeedsInput.
            if is_permission_prompt_kind(state.notification_kind.as_deref()) {
                state.notification_kind = None;
            }
        }
        "PostToolUse" => {
            state.last_posttooluse_ts_ms = ts;
            state.current_tool_name = None;
            // Defense-in-depth: when the user denies a permission prompt,
            // some flows route through PostToolUse (synthetic denial result)
            // without firing PreToolUse. Clear here too so a stale marker
            // doesn't get stuck.
            if is_permission_prompt_kind(state.notification_kind.as_deref()) {
                state.notification_kind = None;
            }
        }
        "Stop" => {
            state.last_stop_ts_ms = ts;
            state.notification_kind = None;
            state.current_tool_name = None;
        }
        "UserPromptSubmit" => {
            state.last_promptsubmit_ts_ms = ts;
            // A new prompt clears any pending notification (e.g. denial of
            // a permission prompt followed by typed input).
            state.notification_kind = None;
        }
        "PreCompact" => {
            state.last_precompact_ts_ms = ts;
        }
        "PostCompact" => {
            state.last_postcompact_ts_ms = ts;
        }
        "SubagentStop" => {
            state.last_subagentstop_ts_ms = ts;
        }
        "SessionStart" => {
            state.last_session_start_ts_ms = ts;
        }
        _ => return Ok(()),
    }

    state.save()
}

/// Whether a `notification_type` value from Claude Code represents an open
/// permission prompt. Covers both the main-agent dialog (`permission_prompt`)
/// and the subagent dialog (`worker_permission_prompt`) — both block the user
/// the same way, and claudectl classifies both as `NeedsInput`.
pub fn is_permission_prompt_kind(kind: Option<&str>) -> bool {
    matches!(kind, Some("permission_prompt" | "worker_permission_prompt"))
}

/// Maximum age for a permission prompt with no resolution events. If Stop,
/// PreToolUse, PostToolUse, and UserPromptSubmit have all been silent since
/// the notification fired, the prompt is almost certainly stale — the hook
/// configuration changed mid-session and resolution events stopped reaching
/// claudectl. Real permission prompts are resolved in seconds to minutes;
/// 30 minutes is a conservative upper bound.
const PERMISSION_PROMPT_MAX_SILENCE_MS: u64 = 30 * 60 * 1000;

/// Whether the session is currently sitting on a permission prompt.
///
/// Pure deterministic check: the `Notification (permission_prompt)` or
/// `Notification (worker_permission_prompt)` event must be the most recent
/// state-changing event for this session. Any later PreToolUse / PostToolUse
/// / Stop / UserPromptSubmit ⇒ the prompt was resolved (approved, denied, or
/// pivoted to a new prompt). No CPU or JSONL second-guessing — those
/// introduced the false-negatives we just had.
///
/// 750ms lower bound: auto-approved prompts (acceptEdits, allowlisted)
/// fire Notification + near-instant PreToolUse; the dialog never opens
/// visibly. Suppressing the marker for the first 750ms filters those out.
/// Real prompts sit far longer, so this costs them nothing.
///
/// 30 min upper bound: if no resolution event has reached claudectl in
/// 30 minutes, the hook configuration is almost certainly incomplete (e.g.
/// Stop lost its claudectl call mid-session) and the marker is stale. Without
/// this bound, a single missed Stop permanently locks the status to NeedsInput.
pub fn is_at_permission_prompt(state: &HookState) -> bool {
    if !is_permission_prompt_kind(state.notification_kind.as_deref()) {
        return false;
    }
    let notif = state.last_notification_ts_ms;
    if notif == 0 {
        return false;
    }
    let still_latest = notif > state.last_pretooluse_ts_ms
        && notif > state.last_posttooluse_ts_ms
        && notif > state.last_stop_ts_ms
        && notif > state.last_promptsubmit_ts_ms;
    if !still_latest {
        return false;
    }
    let age = now_ms().saturating_sub(notif);
    age > 750 && age < PERMISSION_PROMPT_MAX_SILENCE_MS
}

/// How long a session is allowed to sit in "compacting" before we give up and
/// stop reporting the status, even without a clear end-of-compact signal.
/// Auto-compact is a single model call over the transcript summary — it
/// should complete in seconds to a couple of minutes at the outside. If we're
/// still "compacting" five minutes later, something ate the resolution event
/// (we've seen `Stop` never fire for sessions whose first turn is an
/// auto-compact) and we're better off falling through to the real status.
const COMPACTING_MAX_AGE_MS: u64 = 5 * 60 * 1000;

/// Whether the session is currently auto-compacting. PreCompact has fired
/// and no resolution signal (`PostCompact` — the direct signal — or `Stop` —
/// the fallback signal for the post-compact assistant turn) has come in
/// since, AND the PreCompact is recent enough that compaction could
/// plausibly still be running.
pub fn is_compacting(state: &HookState) -> bool {
    let pre = state.last_precompact_ts_ms;
    if pre == 0 {
        return false;
    }
    let ended = state.last_postcompact_ts_ms.max(state.last_stop_ts_ms);
    if ended >= pre {
        return false;
    }
    now_ms().saturating_sub(pre) < COMPACTING_MAX_AGE_MS
}

/// Whether Claude is currently responding to a prompt.
///
/// True when *any* mid-turn event is more recent than the last `Stop`. Tools
/// coming and going inside a single response don't flip this — claude only
/// stops "responding" when `Stop` fires at the end of the turn. This is
/// what makes the status stable instead of flickering with each tool call.
pub fn is_responding(state: &HookState) -> bool {
    let stop = state.last_stop_ts_ms;
    state.last_promptsubmit_ts_ms > stop
        || state.last_pretooluse_ts_ms > stop
        || state.last_posttooluse_ts_ms > stop
}

/// Whether Claude has cleanly finished its turn — `Stop` is the latest
/// non-Notification event. Stable between turns; doesn't flicker.
pub fn is_waiting_for_user(state: &HookState) -> bool {
    let stop = state.last_stop_ts_ms;
    if stop == 0 {
        return false;
    }
    stop >= state.last_promptsubmit_ts_ms
        && stop >= state.last_pretooluse_ts_ms
        && stop >= state.last_posttooluse_ts_ms
}

/// Garbage-collect state files that are older than `max_age_secs` and have
/// no `last_session_*_ts` activity. Best-effort; errors are swallowed.
pub fn cleanup_stale(max_age_secs: u64) {
    let Ok(entries) = fs::read_dir(state_dir()) else {
        return;
    };
    let cutoff = SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(max_age_secs))
        .unwrap_or(UNIX_EPOCH);
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            if let Ok(modified) = meta.modified() {
                if modified < cutoff {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fresh_state(session_id: &str) -> HookState {
        HookState {
            session_id: session_id.into(),
            ..Default::default()
        }
    }

    #[test]
    fn permission_prompt_then_pretooluse_clears() {
        let mut s = fresh_state("sid");
        s.notification_kind = Some("permission_prompt".into());
        s.last_notification_ts_ms = now_ms().saturating_sub(2_000);
        assert!(is_at_permission_prompt(&s));

        // Simulate PreToolUse arriving (manually, mirroring record_hook_event)
        s.last_pretooluse_ts_ms = now_ms();
        s.notification_kind = None;
        assert!(!is_at_permission_prompt(&s));
    }

    #[test]
    fn permission_prompt_ages_out_after_30_minutes() {
        // Repro for the "stuck NeedsInput" failure mode: Stop hook loses its
        // claudectl call mid-session, so the notification fires and sets
        // notification_kind but no resolution event ever clears it. Without
        // the upper-bound timeout, the session stays NeedsInput forever.
        let mut s = fresh_state("sid");
        s.notification_kind = Some("permission_prompt".into());
        s.last_notification_ts_ms =
            now_ms().saturating_sub(PERMISSION_PROMPT_MAX_SILENCE_MS + 1_000);
        // All resolution timestamps are zero — no hook event ever fired.
        assert!(!is_at_permission_prompt(&s));
    }

    #[test]
    fn worker_permission_prompt_also_counts_as_needs_input() {
        let mut s = fresh_state("sid");
        // Backdate past the 750ms grace so the helper considers it open.
        s.last_notification_ts_ms = now_ms().saturating_sub(2_000);
        s.notification_kind = Some("worker_permission_prompt".into());
        assert!(is_at_permission_prompt(&s));

        // PreToolUse from a sibling or the approved tool clears the marker
        // via record_hook_event — verify the helper treats both kinds
        // uniformly.
        assert!(is_permission_prompt_kind(Some("permission_prompt")));
        assert!(is_permission_prompt_kind(Some("worker_permission_prompt")));
        assert!(!is_permission_prompt_kind(Some("idle_prompt")));
        assert!(!is_permission_prompt_kind(None));
    }

    #[test]
    fn compacting_lasts_until_stop_or_postcompact() {
        let mut s = fresh_state("sid");
        // Use recent timestamps so the age-out check doesn't short-circuit.
        let now = now_ms();
        s.last_precompact_ts_ms = now.saturating_sub(1_000);
        assert!(is_compacting(&s));

        // `Stop` clears it (legacy signal).
        s.last_stop_ts_ms = now;
        assert!(!is_compacting(&s));

        // Reset Stop, confirm `PostCompact` ALSO clears it (direct signal,
        // the reliable one — doesn't depend on Stop firing).
        s.last_stop_ts_ms = 0;
        assert!(is_compacting(&s));
        s.last_postcompact_ts_ms = now;
        assert!(!is_compacting(&s));
    }

    #[test]
    fn compacting_ages_out_without_resolution_signal() {
        // Defense against the observed case where `Stop` never fires for
        // sessions whose first turn is an auto-compact. Without the age-out
        // such sessions would stay `Compacting` forever and mask every real
        // `NeedsInput` that follows.
        let mut s = fresh_state("sid");
        s.last_precompact_ts_ms = now_ms().saturating_sub(COMPACTING_MAX_AGE_MS + 1_000);
        assert!(!is_compacting(&s));
    }

    #[test]
    fn responding_until_stop_fires() {
        let mut s = fresh_state("sid");
        s.last_promptsubmit_ts_ms = 1000;
        assert!(is_responding(&s));

        s.last_stop_ts_ms = 2000;
        assert!(!is_responding(&s));

        // Tools coming and going within a turn don't change responding state
        s.last_pretooluse_ts_ms = 3000;
        assert!(is_responding(&s));
        s.last_posttooluse_ts_ms = 3500;
        assert!(is_responding(&s));
        // Until Stop fires again
        s.last_stop_ts_ms = 4000;
        assert!(!is_responding(&s));
    }

    #[test]
    fn now_ms_is_strictly_monotonic_within_a_process() {
        // record_hook_event reads now_ms() per call; two back-to-back calls
        // must produce distinct ts so order-sensitive checks
        // (is_responding / is_waiting_for_user, both strict `>`) stay
        // deterministic when tests record many events in a hot binary.
        let a = now_ms();
        let b = now_ms();
        let c = now_ms();
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn waiting_for_user_after_stop() {
        let mut s = fresh_state("sid");
        s.last_stop_ts_ms = 1000;
        assert!(is_waiting_for_user(&s));

        // A new prompt invalidates "waiting".
        s.last_promptsubmit_ts_ms = 2000;
        assert!(!is_waiting_for_user(&s));
    }

    #[test]
    fn record_event_routes_correctly() {
        // Every event but SessionEnd reconciles the registry before the event
        // match, so this needs the fixture even though it only asserts on
        // parsing — without it the test rewrites the developer's real
        // `local-sessions.json` (and races the other fixture-holding tests).
        let _fixture =
            crate::sandbox_registry::tests::TempRegistry::with_home("record-event-routes");
        let no_sid = json!({"hook_event_name": "Stop"});
        assert!(record_hook_event(&no_sid).is_ok());

        let unknown = json!({"hook_event_name": "Mystery", "session_id": "x"});
        // Unknown events return Ok(()) without writing.
        assert!(record_hook_event(&unknown).is_ok());
    }

    #[test]
    fn resolve_owner_rewalks_when_the_cached_owner_died() {
        // The iTerm2 crash-relaunch shape: the session process survives its
        // recorded terminal. The cache hit must notice the owner is gone and
        // re-resolve instead of freezing the dead owner into the entry forever.
        let session = crate::session::ClaudeSession::from_raw(crate::session::RawSession {
            pid: std::process::id(),
            session_id: "aaa".to_string(),
            cwd: "/work".to_string(),
            started_at: 42,
            name: None,
        });
        let cached_with_dead_owner = crate::sandbox_registry::SessionEntry {
            session_id: "aaa".to_string(),
            cwd: "/work".to_string(),
            transcript: String::new(),
            started_at_ms: 42,
            name: None,
            pid: Some(std::process::id()),
            owner_pid: Some(2_000_000_000),
            owner_started_at: Some("long-gone".to_string()),
        };
        let check = crate::terminal_owner::OwnerCheck::lazy();
        let resolved = resolve_owner(
            std::slice::from_ref(&cached_with_dead_owner),
            &session,
            false,
            &check,
        );
        // Re-walked from this live process: lands on a real, alive ancestor —
        // not the recorded corpse.
        assert_ne!(
            resolved.as_ref().map(|owner| owner.pid),
            Some(2_000_000_000),
            "a dead cached owner must not be reused"
        );

        // And a cached owner whose process is alive IS reused verbatim (the
        // zero-fork steady state): our own real owner, exact lstart string.
        let live_owner = crate::terminal_owner::ProcessTable::snapshot()
            .unwrap()
            .owner_of(std::process::id())
            .expect("test process has an owner");
        let cached_alive = crate::sandbox_registry::SessionEntry {
            owner_pid: Some(live_owner.pid),
            owner_started_at: Some(live_owner.started_at.clone()),
            ..cached_with_dead_owner
        };
        let check = crate::terminal_owner::OwnerCheck::lazy();
        let resolved = resolve_owner(std::slice::from_ref(&cached_alive), &session, false, &check);
        assert_eq!(resolved, Some(live_owner));
    }

    #[test]
    fn try_read_payload_rejects_non_hook_json() {
        // Direct unit of the parsing branch: feed JSON without hook_event_name.
        let value: serde_json::Value = serde_json::from_str(r#"{"foo": 1}"#).unwrap();
        // Mimic the inner check in try_read_hook_payload.
        assert!(
            value
                .get("hook_event_name")
                .and_then(|v| v.as_str())
                .is_none()
        );
    }
}
