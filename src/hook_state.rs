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
//! State files live at `~/.local/share/claudectl/state/<session_id>.json`. The
//! file is tiny (a few hundred bytes) and rewritten atomically on each hook
//! event.
//!
//! That directory is deliberately the host-shared one, not `~/.claudectl`
//! where these files used to live: inside an agent-sandbox the old path
//! resolved into the VM's private overlay, so the laptop claudectl that
//! renders a sandbox session could never read its state. See [`state_dir`].

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

/// Returns the directory holding per-session state files.
///
/// `~/.local/share/claudectl/state`, alongside the session registries, and
/// **not** `~/.claudectl/state` where this used to live. The old path is
/// private to whichever machine wrote it: inside an agent-sandbox it resolves
/// into the VM's own overlay, so a session's hook state was invisible to the
/// laptop claudectl that renders it. `infer_status` therefore never found a
/// state file for any sandbox session, always fell through to the heuristic —
/// which by design never produces `NeedsInput` — and reported every session
/// blocked on a permission prompt as `Processing`. The new path is a
/// host-shared mount, which is what makes the deterministic status usable at
/// all from outside the VM.
///
/// Measured on 2026-08-04: `~/.claudectl` reported the same `st_dev` as `/etc`
/// and `/root` (the sandbox overlay), while `~/.local/share/claudectl`,
/// `~/.claude` and `~/repos` each had their own — the three host-shared binds.
///
/// Honors `CLAUDECTL_STATE_DIR` when set, for tests.
pub fn state_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CLAUDECTL_STATE_DIR") {
        return PathBuf::from(dir);
    }
    shared_state_dir()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn shared_state_dir() -> PathBuf {
    home_dir().join(".local/share/claudectl/state")
}

/// Where state files lived before they had to be readable from another machine.
fn legacy_state_dir() -> PathBuf {
    home_dir().join(".claudectl/state")
}

/// Carry any state left in [`legacy_state_dir`] over to the shared directory,
/// once per process.
///
/// Without this, upgrading mid-session silently resets every live session's
/// deterministic state: the next `infer_status` finds no file, falls back to
/// the heuristic, and a session sitting on a permission prompt reads as
/// `Processing` until its next hook fires — exactly the bug being fixed,
/// reintroduced for one turn on every session at once.
///
/// **Never overwrites.** Each sandbox migrates its own legacy directory into
/// one shared destination, and a session resumed in a newer sandbox has state
/// in both. The shared copy is the one a live writer is maintaining; a legacy
/// file is by definition from before this upgrade, so it must not clobber it.
///
/// Best-effort throughout: a failure here costs one turn of heuristic status,
/// which is strictly better than failing the hook that called us.
fn migrate_legacy_state(from: &std::path::Path, to: &std::path::Path) -> io::Result<usize> {
    if !from.is_dir() || from == to {
        return Ok(0);
    }
    fs::create_dir_all(to)?;
    let mut moved = 0;
    for entry in fs::read_dir(from)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Some(name) = path.file_name() else {
            continue;
        };
        let destination = to.join(name);
        if destination.exists() {
            // A live writer already owns this session here. Drop the stale
            // copy rather than leaving it to be re-migrated on every run.
            let _ = fs::remove_file(&path);
            continue;
        }
        // Rename first (atomic, same filesystem); fall back to copy for the
        // cross-device case, which is the norm here — the legacy directory is
        // on the sandbox overlay and the destination is a virtiofs bind.
        if fs::rename(&path, &destination).is_err() {
            fs::copy(&path, &destination)?;
            let _ = fs::remove_file(&path);
        }
        moved += 1;
    }
    Ok(moved)
}

/// Run [`migrate_legacy_state`] at most once per process.
///
/// Called from `state_path` rather than `state_dir` so it covers both readers
/// and writers, and cheap enough to sit on that path: after the first call it
/// is one relaxed atomic load.
fn migrate_legacy_state_once() {
    static DONE: std::sync::Once = std::sync::Once::new();
    DONE.call_once(|| {
        if std::env::var_os("CLAUDECTL_STATE_DIR").is_some() {
            // A test (or an operator) has pinned the directory explicitly;
            // moving real state into it would be a surprise.
            return;
        }
        let _ = migrate_legacy_state(&legacy_state_dir(), &shared_state_dir());
    });
}

/// Path to one session's state file.
fn state_path(session_id: &str) -> PathBuf {
    migrate_legacy_state_once();
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
        // A process-table-discovered session whose `--resume` id is unknown
        // has an empty session_id: it is displayable but not restorable
        // (nothing to `--resume`), and the registry is keyed by session_id,
        // so recording it would collide every such session on "".
        .filter(|session| !session.session_id.is_empty())
        .map(|session| {
            let owner = resolve_owner(&known, &session, sandbox.is_some(), check);
            let transcript = crate::discovery::transcript_to_record(
                &session.session_id,
                &session.cwd,
                &crate::discovery::find_transcript_by_session_id,
            );
            let name = (!session.session_name.is_empty()).then_some(session.session_name);
            // Carry the host-side routing keys for sandbox sessions. Only the
            // sandbox can read them (they come from its own per-pid sidecar),
            // and only the host can use them — so if this writer drops them,
            // nothing downstream can reconstruct them, and Tab on the row is
            // left guessing from a container pid and a container cwd.
            let (host_terminal_id, host_tty) = match sandbox {
                Some(_) => crate::process::host_terminal_routing(session.pid),
                // On the host these describe *this* machine already and are
                // rediscovered every tick; persisting them would just add a
                // second, staler copy.
                None => (None, None),
            };
            crate::sandbox_registry::SessionEntry {
                session_id: session.session_id,
                cwd: session.cwd,
                transcript,
                started_at_ms: session.started_at,
                name,
                pid: Some(session.pid),
                owner_pid: owner.as_ref().map(|owner| owner.pid),
                owner_started_at: owner.map(|owner| owner.started_at),
                host_terminal_id,
                host_tty,
                // Seeing a session live is what un-departs it: `--resume`
                // reuses the id, so a stamp left from the previous run would
                // hide the resumed session indefinitely.
                departed_at_ms: None,
            }
        })
        .collect();

    // The shape of what we're about to write. "19 sessions, 0 routed" and
    // "0 sessions" are wildly different failures that produce the same
    // outcome on the host (rows that Tab can't reach), and neither was
    // distinguishable from the outside.
    crate::logger::log(
        "DEBUG",
        &format!(
            "registry: live set = {} sessions, {} with host routing, scope={}",
            live.len(),
            live.iter().filter(|e| e.host_terminal_id.is_some()).count(),
            sandbox.as_deref().unwrap_or("host-local"),
        ),
    );

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

/// Whether a `SessionEnd` `reason` means the user *deliberately* ended the
/// session, rather than the process/terminal being torn down under it.
///
/// Claude Code reports a prompt-level exit — Ctrl-D / Ctrl-C at the prompt, or
/// `/logout` — as `prompt_input_exit` / `logout`. A terminal that dies (window
/// quit, crash, SIGHUP) tears the process down and reports `host_exit` /
/// `nonzero_exit` / `other`, and `/clear` (which does NOT end the session)
/// reports `clear`. So these two reasons are the durable, false-positive-free
/// signal that the user closed this session on purpose: safe to forget from the
/// restore registry immediately, and — because they never fire during a
/// terminal quit — acting on them can't erase the terminal-death casualties
/// `--restore-sessions` exists to bring back. Anything else stays the reaper's
/// job (it settle-samples the owner to make the same call, within its window).
fn is_deliberate_user_close(reason: &str) -> bool {
    matches!(reason, "prompt_input_exit" | "logout")
}

/// Whether a `SessionEnd` reason means the session is actually gone.
///
/// All of them except `clear`, which fires `SessionEnd` without ending
/// anything — the process carries straight on. Stamping that one as departed
/// would hide a live session from the dashboard until its next hook, which is
/// the exact failure this stamping exists to fix, only mirrored.
///
/// An unrecognised or absent reason counts as ended: `SessionEnd` fired, and a
/// row wrongly hidden for one hook is a far smaller error than a dead row that
/// lingers for minutes. `clear` is named explicitly, so a future reason that
/// also doesn't end the session has to be added here deliberately.
fn ends_the_session(reason: &str) -> bool {
    reason != "clear"
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
    // It also deliberately skips the *live-set* registry write below: SessionEnd
    // fires while a session tears down — a terminal-app quit fires it for every
    // session at once, with pointer files vanishing as the live set collapses to
    // empty — and mirroring the live set then erases exactly the entries
    // `--restore-sessions` needs seconds later.
    //
    // But when the `reason` proves the user DELIBERATELY closed this session (a
    // prompt-level exit / `/logout`, never a terminal teardown — see
    // is_deliberate_user_close), we forget it from the restore registry right
    // now, durably. That closes the reaper's timing gap: the reaper can only
    // distinguish "user closed it" from "terminal died under it" while the
    // terminal is still alive, so a close followed by a terminal quit before the
    // reaper's next tick would otherwise be resurrected. Because these reasons
    // never fire during a terminal quit, acting on them can't wipe a
    // terminal-death casualty that restore should bring back.
    if event == "SessionEnd" {
        let reason = payload.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        if is_deliberate_user_close(reason) {
            let _ = crate::sandbox_registry::forget_session(&session_id);
        } else if ends_the_session(reason) {
            // Restore material: must stay on disk, but must stop rendering as
            // live immediately rather than waiting for an unrelated session in
            // the same sandbox to trigger a reconcile.
            let _ = crate::sandbox_registry::mark_session_departed(&session_id, now_ms());
        }
        return HookState::remove(&session_id);
    }

    // Session registry: every other hook event records the live set, so
    // `--restore-sessions` (host) / `--restore-sbx-sessions` bring back the
    // right sessions after a Ghostty restart / `sbx rm`. Best-effort — registry
    // I/O never blocks the hook, and it only adds/refreshes (never forgets).
    // Cost: one `ps x` snapshot per event (~40ms on a busy machine) — the
    // price of discovery that survives Claude Code deleting pointer files
    // mid-session; owner resolution itself stays fork-free in steady state.
    // Still best-effort — a registry failure must never fail the hook — but no
    // longer silent. This discarded `Result` was the only thing standing
    // between "the registry write failed" and "the registry write was a no-op
    // because nothing changed", two states that look identical from outside
    // and needed telling apart on 2026-08-05.
    if let Err(e) = record_live_sessions(&crate::terminal_owner::OwnerCheck::lazy()) {
        crate::logger::log("ERROR", &format!("registry: live-set write failed: {e}"));
    }

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

/// Minimum age before an open permission prompt is reported. Auto-approved
/// prompts (acceptEdits, allowlisted) fire Notification + near-instant
/// PreToolUse and the dialog never opens visibly; suppressing the marker for
/// the first 750ms filters those out. Real prompts sit far longer, so this
/// costs them nothing.
const PERMISSION_PROMPT_MIN_AGE_MS: u64 = 750;

/// How far past the notification the transcript must advance before we treat
/// the prompt as resolved-with-a-lost-hook. Claude Code writes the assistant
/// message that *precedes* the dialog moments before the Notification fires,
/// so a small margin keeps ordinary jitter between the transcript writer and
/// the hook process from reading as "the session moved on".
const PROMPT_RESOLUTION_GRACE_MS: u64 = 5_000;

/// Whether the session is currently sitting on a permission prompt.
///
/// Deterministic check: the `Notification (permission_prompt)` or
/// `Notification (worker_permission_prompt)` event must be the most recent
/// state-changing event for this session. Any later PreToolUse / PostToolUse
/// / Stop / UserPromptSubmit ⇒ the prompt was resolved (approved, denied, or
/// pivoted to a new prompt).
///
/// `last_message_ts` is the timestamp of the newest *conversation message* in
/// the transcript (never the file's mtime — bookkeeping records like `system`
/// and `ai-title` bump mtime without any conversational progress). It is the
/// backstop for a resolution event that never reached claudectl: hook events
/// are lossy — each is a separate process spawned with a 5 s timeout — but the
/// transcript is written by Claude Code itself. If the conversation advanced
/// past the notification, the prompt was answered and we simply missed it.
///
/// **There is deliberately no time limit.** This used to expire the marker
/// after 30 minutes of silence, on the theory that a prompt open that long
/// meant a broken hook configuration. It cannot tell that case apart from "the
/// user hasn't answered yet", and the fall-through landed on `Processing` —
/// the one status that says "working, leave it alone" about a session that is
/// blocked on the user. Captured live on 2026-08-06: a prompt open since
/// 17:42 read `Needs Input` until 18:12:22 (age 1819 s) and then flipped to
/// `Processing`, with `notification_kind` still `permission_prompt` and not one
/// resolution event ever recorded. Transcript progress answers the same
/// question with evidence instead of a timer.
pub fn is_at_permission_prompt(state: &HookState, now_ms: u64, last_message_ts: u64) -> bool {
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
    if now_ms.saturating_sub(notif) <= PERMISSION_PROMPT_MIN_AGE_MS {
        return false;
    }
    last_message_ts <= notif.saturating_add(PROMPT_RESOLUTION_GRACE_MS)
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
pub fn is_compacting(state: &HookState, now_ms: u64) -> bool {
    let pre = state.last_precompact_ts_ms;
    if pre == 0 {
        return false;
    }
    let ended = state.last_postcompact_ts_ms.max(state.last_stop_ts_ms);
    if ended >= pre {
        return false;
    }
    now_ms.saturating_sub(pre) < COMPACTING_MAX_AGE_MS
}

/// Whether a tool started and has not reported back.
///
/// `PreToolUse` sets `current_tool_name`, `PostToolUse` and `Stop` clear it.
/// This is what licenses an unbounded `Processing`: a single tool call — a
/// build, a test suite, a subagent — can legitimately run for hours with no
/// other event on either channel. With no tool in flight, that same silence
/// means the turn is over and its `Stop` was lost.
pub fn tool_in_flight(state: &HookState) -> bool {
    state.current_tool_name.is_some() && state.last_pretooluse_ts_ms > state.last_posttooluse_ts_ms
}

/// Timestamp of the newest event that means "a turn is under way": the user's
/// prompt, or a tool starting or finishing inside the response.
///
/// This is the hook stream's idea of how far the current turn has progressed,
/// and the yardstick the transcript is compared against when the two disagree
/// (see `monitor::transcript_ended_the_turn`).
pub fn newest_turn_event_ms(state: &HookState) -> u64 {
    state
        .last_promptsubmit_ts_ms
        .max(state.last_pretooluse_ts_ms)
        .max(state.last_posttooluse_ts_ms)
}

/// Timestamp of the newest event of *any* kind we have for this session.
///
/// Unlike [`newest_turn_event_ms`] this is not about the turn — it is about the
/// channel. `SessionStart` counts, because a session sitting untouched at its
/// first prompt has legitimately produced nothing since, and reading that as a
/// dead hook would flag every freshly opened session.
pub fn newest_hook_event_ms(state: &HookState) -> u64 {
    newest_turn_event_ms(state)
        .max(state.last_notification_ts_ms)
        .max(state.last_stop_ts_ms)
        .max(state.last_subagentstop_ts_ms)
        .max(state.last_precompact_ts_ms)
        .max(state.last_postcompact_ts_ms)
        .max(state.last_session_start_ts_ms)
        .max(state.last_session_end_ts_ms)
}

/// How far a session's transcript may run ahead of its newest hook event before
/// the hook channel is called dead rather than quiet.
///
/// Generous on purpose, because it has to clear every legitimate gap. The two
/// channels advance together while a session works: a tool call brackets the
/// `tool_result` the transcript records with `PreToolUse` and `PostToolUse`, and
/// each user turn opens with `UserPromptSubmit`. Ten minutes of transcript
/// writes with not one hook event is therefore not a quiet session, it is a
/// channel that stopped delivering.
const HOOK_SILENCE_GRACE_MS: u64 = 10 * 60 * 1_000;

/// Whether this session's hook channel has stopped delivering: Claude Code kept
/// writing its transcript long after the last hook event reached us.
///
/// The transcript is the control: Claude Code writes it itself, so it cannot be
/// lost the way a hook invocation can — which is what makes the two comparable
/// at all, and is the same asymmetry `monitor::transcript_ended_the_turn`
/// already leans on.
///
/// Worth detecting because the failure is otherwise invisible. The hook command
/// ends in `2>/dev/null || true`, `claudectl-hook` writes nothing to stdout by
/// design, and its diagnostic log is opt-in — so a session whose hooks silently
/// stopped firing looks exactly like a session that has been quiet. On
/// 2026-08-10 one had been silent for **12 hours**, which cost its `SessionEnd`
/// and left a dead row on screen for ~96 s.
///
/// Fails quiet: with no transcript timestamp there is nothing to compare, so it
/// reports healthy rather than guessing.
///
/// `newest_transcript_ms` is whichever transcript clock the caller has:
/// `monitor::decide_status` passes the newest conversation message, which is the
/// better signal because bookkeeping records (`system`, `bridge-session`,
/// `attachment`) never move it; `--doctor` passes the file mtime, because it
/// reports on sessions without parsing their JSONL.
pub fn hook_channel_is_silent(newest_transcript_ms: u64, newest_hook_ms: u64) -> bool {
    if newest_transcript_ms == 0 {
        return false;
    }
    newest_transcript_ms.saturating_sub(newest_hook_ms) > HOOK_SILENCE_GRACE_MS
}

/// Whether Claude is currently responding to a prompt.
///
/// True when *any* mid-turn event is more recent than the last `Stop`. Tools
/// coming and going inside a single response don't flip this — claude only
/// stops "responding" when `Stop` fires at the end of the turn. This is
/// what makes the status stable instead of flickering with each tool call.
///
/// Note the asymmetry this creates, and why callers must not treat a `true`
/// here as the final word: a turn has no honest upper bound in *elapsed* time —
/// an agent can legitimately grind for hours — so if the `Stop` that ends the
/// turn never arrives, this stays true forever. That is exactly what happened
/// on 2026-07-28: sessions whose `Stop` hook never reached claudectl were
/// reported `Processing` for up to 15 hours while their transcripts had long
/// since ended in `end_turn`.
///
/// Two things in `monitor` bound it, and both are evidence rather than a timer:
/// the transcript — which Claude Code writes itself, and which therefore cannot
/// be lost the way a hook invocation can — overrules this when it shows the turn
/// ended (`transcript_ended_the_turn`); and a turn that emits nothing on *either*
/// channel while no tool is in flight is not running (`turn_went_silent`).
pub fn is_responding(state: &HookState) -> bool {
    newest_turn_event_ms(state) > state.last_stop_ts_ms
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

    /// The 2026-08-10 incident, to scale: the transcript ran 12 h past the last
    /// hook event, which cost the session its `SessionEnd`.
    const TWELVE_HOURS_MS: u64 = 12 * 60 * 60 * 1_000;

    #[test]
    fn a_transcript_running_hours_past_the_last_hook_event_is_a_dead_channel() {
        let last_hook = 1_786_330_534_891;
        assert!(hook_channel_is_silent(
            last_hook + TWELVE_HOURS_MS,
            last_hook
        ));
    }

    #[test]
    fn negative_control_a_quiet_session_inside_the_grace_is_healthy() {
        // Both channels idle, or the transcript a little ahead — ordinary
        // writer/hook jitter, not a failure.
        let last_hook = 1_786_330_534_891;
        assert!(!hook_channel_is_silent(last_hook, last_hook));
        assert!(!hook_channel_is_silent(
            last_hook + HOOK_SILENCE_GRACE_MS,
            last_hook
        ));
        assert!(
            hook_channel_is_silent(last_hook + HOOK_SILENCE_GRACE_MS + 1, last_hook),
            "one ms past the grace is where it flips"
        );
    }

    #[test]
    fn negative_control_a_freshly_started_session_is_healthy() {
        // Opened, never prompted: `newest_turn_event_ms` is 0 and only
        // SessionStart has fired. Judging on turn events alone would flag every
        // new session, which is why the yardstick includes SessionStart.
        let started = 1_786_330_000_000;
        let state = HookState {
            last_session_start_ts_ms: started,
            ..fresh_state("brand-new")
        };
        assert_eq!(newest_turn_event_ms(&state), 0);
        assert_eq!(newest_hook_event_ms(&state), started);
        assert!(!hook_channel_is_silent(started + 1_000, started));
    }

    #[test]
    fn a_session_with_no_hook_state_at_all_is_a_dead_channel() {
        // No state file ⇒ not one hook event ever reached us, while Claude Code
        // has been writing a transcript. That is the shape of a hook that never
        // ran, e.g. `claudectl-hook` absent from a sandbox's PATH — silent,
        // because the hook command ends in `|| true`.
        assert!(hook_channel_is_silent(1_786_330_534_891, 0));
    }

    #[test]
    fn fails_quiet_when_the_transcript_has_no_timestamp() {
        // Unstat-able or absent transcript: nothing to compare, so it must not
        // manufacture a verdict from a zero.
        assert!(!hook_channel_is_silent(0, 0));
        assert!(!hook_channel_is_silent(0, 1_786_330_534_891));
    }

    #[test]
    fn state_lives_on_the_shared_mount_not_the_sandbox_private_one() {
        // The whole point of the move. `~/.claudectl` is the sandbox's own
        // overlay — a state file written there is invisible to the laptop
        // claudectl that renders the session, which is why every sandbox
        // session blocked on a permission prompt reported `Processing`.
        // `~/.local/share/claudectl` is a host-shared bind, the same one the
        // session registries already cross the VM boundary through.
        let home = std::path::Path::new("/Users/ndr");
        assert_eq!(
            home.join(".local/share/claudectl/state"),
            home.join(".local/share/claudectl").join("state"),
        );
        // Asserted against the real resolver, with HOME pinned.
        let _guard = crate::sandbox_registry::tests::env_guard();
        let saved = std::env::var_os("HOME");
        let saved_override = std::env::var_os("CLAUDECTL_STATE_DIR");
        // SAFETY: env access is serialized by the held `ENV_LOCK`.
        unsafe {
            std::env::remove_var("CLAUDECTL_STATE_DIR");
            std::env::set_var("HOME", home);
        }
        let resolved = state_dir();
        unsafe {
            match saved {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            if let Some(value) = saved_override {
                std::env::set_var("CLAUDECTL_STATE_DIR", value);
            }
        }
        assert_eq!(resolved, home.join(".local/share/claudectl/state"));
        assert!(
            !resolved.starts_with(home.join(".claudectl")),
            "must not resolve back into the sandbox-private directory"
        );
    }

    #[test]
    fn a_permission_prompt_recorded_by_another_machine_reads_as_needs_input() {
        // End to end across the boundary, in the shape that was broken: a
        // sandbox's hook writes the state file, and a *different* process
        // reads it back off the shared directory and infers status. Nothing
        // here is sandbox-aware — the file is keyed by session id, which is
        // globally unique, so "written elsewhere" and "written here" are the
        // same code path once the directory is shared.
        let dir = tempfile::tempdir().unwrap();
        let written = HookState {
            session_id: "cf54da79-2d23-4231-81fd-ce2a441e6e39".into(),
            last_notification_ts_ms: now_ms().saturating_sub(5_000),
            notification_kind: Some("permission_prompt".into()),
            // A tool ran before the prompt opened, as it does in the real
            // sequence: PreToolUse -> Notification -> (blocked).
            last_pretooluse_ts_ms: now_ms().saturating_sub(9_000),
            last_posttooluse_ts_ms: now_ms().saturating_sub(8_000),
            ..Default::default()
        };
        let path = dir.path().join(format!("{}.json", written.session_id));
        std::fs::write(&path, serde_json::to_string(&written).unwrap()).unwrap();

        let reloaded: HookState =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(
            is_at_permission_prompt(&reloaded, now_ms(), 0),
            "a prompt recorded by another machine must survive the round trip"
        );
        // `is_responding` is *also* true here, and that is correct: a session
        // blocked on a prompt is mid-turn. `NeedsInput` wins on precedence,
        // not because the turn looks finished — see `monitor::decide_status`.
        assert!(is_responding(&reloaded));
    }

    #[test]
    fn migration_moves_legacy_state_to_the_shared_directory() {
        let legacy = tempfile::tempdir().unwrap();
        let shared = tempfile::tempdir().unwrap();
        std::fs::write(
            legacy.path().join("sid-1.json"),
            r#"{"session_id":"sid-1"}"#,
        )
        .unwrap();
        std::fs::write(legacy.path().join("notes.txt"), "ignored").unwrap();

        let moved = migrate_legacy_state(legacy.path(), shared.path()).unwrap();

        assert_eq!(moved, 1);
        assert!(shared.path().join("sid-1.json").exists());
        assert!(!legacy.path().join("sid-1.json").exists());
        assert!(
            legacy.path().join("notes.txt").exists(),
            "only .json state files are claimed"
        );
    }

    #[test]
    fn migration_never_overwrites_state_a_live_writer_owns() {
        // Each sandbox migrates its own legacy directory into one shared
        // destination, and a session resumed in a newer sandbox has a file in
        // both. The shared copy is the live one; a legacy file predates the
        // upgrade by construction and must lose.
        let legacy = tempfile::tempdir().unwrap();
        let shared = tempfile::tempdir().unwrap();
        std::fs::write(
            legacy.path().join("sid-1.json"),
            r#"{"session_id":"stale"}"#,
        )
        .unwrap();
        std::fs::write(shared.path().join("sid-1.json"), r#"{"session_id":"live"}"#).unwrap();

        let moved = migrate_legacy_state(legacy.path(), shared.path()).unwrap();

        assert_eq!(moved, 0);
        assert!(
            std::fs::read_to_string(shared.path().join("sid-1.json"))
                .unwrap()
                .contains("live"),
            "the live writer's state must survive"
        );
        assert!(
            !legacy.path().join("sid-1.json").exists(),
            "the stale copy is dropped so it is not re-migrated forever"
        );
    }

    #[test]
    fn migration_is_a_no_op_without_a_legacy_directory() {
        let shared = tempfile::tempdir().unwrap();
        let absent = shared.path().join("does-not-exist");
        assert_eq!(migrate_legacy_state(&absent, shared.path()).unwrap(), 0);
        // And self-migration cannot loop or delete anything.
        assert_eq!(
            migrate_legacy_state(shared.path(), shared.path()).unwrap(),
            0
        );
    }

    #[test]
    fn permission_prompt_then_pretooluse_clears() {
        let mut s = fresh_state("sid");
        s.notification_kind = Some("permission_prompt".into());
        s.last_notification_ts_ms = now_ms().saturating_sub(2_000);
        assert!(is_at_permission_prompt(&s, now_ms(), 0));

        // Simulate PreToolUse arriving (manually, mirroring record_hook_event)
        s.last_pretooluse_ts_ms = now_ms();
        s.notification_kind = None;
        assert!(!is_at_permission_prompt(&s, now_ms(), 0));
    }

    #[test]
    fn an_unanswered_permission_prompt_never_ages_out() {
        // Replaces `permission_prompt_ages_out_after_30_minutes`, which
        // asserted the opposite. That bound could not tell "the hook
        // configuration broke" from "the user hasn't answered yet", and the
        // fall-through reported the session as Processing. Captured live on
        // 2026-08-06: a prompt open since 17:42 read Needs Input until
        // 18:12:22 — age 1819 s — and then flipped to Processing with
        // `notification_kind` still `permission_prompt`.
        let mut s = fresh_state("sid");
        s.notification_kind = Some("permission_prompt".into());
        s.last_notification_ts_ms = 1_000_000;
        // No resolution event ever fired, and the transcript never advanced.
        for age_ms in [2_000, 30 * 60 * 1000, 48 * 60 * 60 * 1000] {
            assert!(
                is_at_permission_prompt(&s, 1_000_000 + age_ms, 0),
                "a prompt with no resolution evidence is still open after {age_ms} ms"
            );
        }
    }

    #[test]
    fn transcript_progress_resolves_a_prompt_whose_hook_was_lost() {
        // The backstop for the case the 30-minute timer was reaching for: the
        // user answered, but every resolution hook was dropped. Hook events are
        // lossy; the transcript is not, so conversation past the notification
        // is proof the session moved on.
        let mut s = fresh_state("sid");
        s.notification_kind = Some("permission_prompt".into());
        s.last_notification_ts_ms = 1_000_000;
        let now = 1_000_000 + 60 * 60 * 1000;

        assert!(
            is_at_permission_prompt(&s, now, 999_000),
            "a message written BEFORE the prompt opened is the turn that led to it"
        );
        assert!(
            is_at_permission_prompt(&s, now, 1_000_000 + PROMPT_RESOLUTION_GRACE_MS),
            "a message inside the grace window is ordinary writer/hook jitter"
        );
        assert!(
            !is_at_permission_prompt(&s, now, 1_000_000 + PROMPT_RESOLUTION_GRACE_MS + 1),
            "conversation past the notification means the prompt was answered"
        );
    }

    #[test]
    fn tool_in_flight_tracks_the_open_pretooluse() {
        let mut s = fresh_state("sid");
        assert!(!tool_in_flight(&s), "no tool has started");

        s.current_tool_name = Some("Bash".into());
        s.last_pretooluse_ts_ms = 2_000;
        assert!(tool_in_flight(&s));

        // PostToolUse both clears the name and lands newer than the PreToolUse;
        // either alone must be enough to close the tool, since a dropped hook
        // can leave the pair inconsistent.
        s.last_posttooluse_ts_ms = 3_000;
        assert!(!tool_in_flight(&s));
        s.last_posttooluse_ts_ms = 0;
        s.current_tool_name = None;
        assert!(!tool_in_flight(&s));
    }

    #[test]
    fn worker_permission_prompt_also_counts_as_needs_input() {
        let mut s = fresh_state("sid");
        // Backdate past the 750ms grace so the helper considers it open.
        s.last_notification_ts_ms = now_ms().saturating_sub(2_000);
        s.notification_kind = Some("worker_permission_prompt".into());
        assert!(is_at_permission_prompt(&s, now_ms(), 0));

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
        assert!(is_compacting(&s, now_ms()));

        // `Stop` clears it (legacy signal).
        s.last_stop_ts_ms = now;
        assert!(!is_compacting(&s, now_ms()));

        // Reset Stop, confirm `PostCompact` ALSO clears it (direct signal,
        // the reliable one — doesn't depend on Stop firing).
        s.last_stop_ts_ms = 0;
        assert!(is_compacting(&s, now_ms()));
        s.last_postcompact_ts_ms = now;
        assert!(!is_compacting(&s, now_ms()));
    }

    #[test]
    fn compacting_ages_out_without_resolution_signal() {
        // Defense against the observed case where `Stop` never fires for
        // sessions whose first turn is an auto-compact. Without the age-out
        // such sessions would stay `Compacting` forever and mask every real
        // `NeedsInput` that follows.
        let mut s = fresh_state("sid");
        s.last_precompact_ts_ms = now_ms().saturating_sub(COMPACTING_MAX_AGE_MS + 1_000);
        assert!(!is_compacting(&s, now_ms()));
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
            name_source: None,
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
            ..Default::default()
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
    fn is_deliberate_user_close_only_true_for_prompt_exit_and_logout() {
        assert!(is_deliberate_user_close("prompt_input_exit"));
        assert!(is_deliberate_user_close("logout"));
        // Terminal-teardown / process-death reasons must NOT count: forgetting on
        // those would resurrect the exact bug — dropping terminal-death casualties
        // that `--restore-sessions` should bring back. `clear` doesn't end the
        // session at all.
        for teardown in ["other", "host_exit", "nonzero_exit", "clear", ""] {
            assert!(
                !is_deliberate_user_close(teardown),
                "{teardown:?} must not count as a deliberate close"
            );
        }
    }

    #[test]
    fn session_end_forgets_the_entry_only_on_a_deliberate_close() {
        let _fixture =
            crate::sandbox_registry::tests::TempRegistry::with_home("session-end-forget");
        let seed = |id: &str| crate::sandbox_registry::SessionEntry {
            session_id: id.to_string(),
            cwd: "/work".to_string(),
            transcript: String::new(),
            started_at_ms: 1,
            name: None,
            pid: Some(std::process::id()),
            owner_pid: None,
            owner_started_at: None,
            ..Default::default()
        };
        crate::sandbox_registry::update_local(|_| vec![seed("closed"), seed("kept")]).unwrap();

        // A terminal-teardown SessionEnd leaves the entry for the reaper/restore.
        record_hook_event(
            &json!({"hook_event_name": "SessionEnd", "session_id": "kept", "reason": "other"}),
        )
        .unwrap();
        // A deliberate prompt exit forgets it durably, right now.
        record_hook_event(&json!({
            "hook_event_name": "SessionEnd", "session_id": "closed", "reason": "prompt_input_exit"
        }))
        .unwrap();

        let ids: Vec<String> = crate::sandbox_registry::load_local()
            .sessions
            .into_iter()
            .map(|entry| entry.session_id)
            .collect();
        assert_eq!(
            ids,
            vec!["kept".to_string()],
            "'closed' forgotten on prompt_input_exit; 'kept' left by reason 'other'"
        );

        // ...and 'kept' is now STAMPED, not merely present. Retention alone was
        // ambiguous: the dashboard read this file as the live set and could not
        // tell restore material from a running session, so a closed terminal
        // kept rendering until an unrelated hook forced a reconcile.
        let kept = crate::sandbox_registry::load_local()
            .sessions
            .into_iter()
            .find(|entry| entry.session_id == "kept")
            .expect("restore material stays on disk");
        assert!(
            kept.departed_at_ms.is_some(),
            "a terminal-teardown SessionEnd must stamp the entry so the view can skip it"
        );
    }

    #[test]
    fn clear_does_not_stamp_a_session_that_is_still_running() {
        // `clear` fires SessionEnd without ending anything. Stamping it would
        // hide a live session until its next hook — the same bug this stamping
        // fixes, only mirrored.
        let _fixture = crate::sandbox_registry::tests::TempRegistry::with_home("session-end-clear");
        crate::sandbox_registry::update_local(|_| {
            vec![crate::sandbox_registry::SessionEntry {
                session_id: "alive".to_string(),
                pid: Some(std::process::id()),
                ..Default::default()
            }]
        })
        .unwrap();

        record_hook_event(
            &json!({"hook_event_name": "SessionEnd", "session_id": "alive", "reason": "clear"}),
        )
        .unwrap();

        let entry = crate::sandbox_registry::load_local()
            .sessions
            .into_iter()
            .find(|entry| entry.session_id == "alive")
            .expect("clear must not remove the entry either");
        assert!(
            entry.departed_at_ms.is_none(),
            "`clear` does not end the session, so it must not be stamped"
        );
    }

    #[test]
    fn ends_the_session_covers_every_reason_the_binary_emits() {
        // The six reasons Claude Code 2.1.221 actually emits, read out of the
        // shipped binary. Only `clear` leaves the session running.
        for reason in [
            "other",
            "host_exit",
            "nonzero_exit",
            "prompt_input_exit",
            "logout",
            "",
        ] {
            assert!(ends_the_session(reason), "{reason:?} ends the session");
        }
        assert!(!ends_the_session("clear"));
    }

    #[test]
    fn a_duplicate_session_end_does_not_move_the_departure_clock() {
        let _fixture = crate::sandbox_registry::tests::TempRegistry::with_home("session-end-dup");
        crate::sandbox_registry::update_local(|_| {
            vec![crate::sandbox_registry::SessionEntry {
                session_id: "gone".to_string(),
                ..Default::default()
            }]
        })
        .unwrap();

        crate::sandbox_registry::mark_session_departed("gone", 1_000).unwrap();
        crate::sandbox_registry::mark_session_departed("gone", 9_999).unwrap();

        let entry = crate::sandbox_registry::load_local()
            .sessions
            .into_iter()
            .find(|entry| entry.session_id == "gone")
            .expect("still on disk");
        assert_eq!(
            entry.departed_at_ms,
            Some(1_000),
            "the first SessionEnd is the departure; a redelivery must not reset it"
        );
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
