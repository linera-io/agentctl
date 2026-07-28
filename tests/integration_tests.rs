use std::io::Write;
use std::sync::Once;
use std::time::Duration;

use claudectl::discovery;
use claudectl::models;
use claudectl::monitor;
use claudectl::session::{ClaudeSession, RawSession, SessionStatus, TelemetryStatus};

/// Point hook_state at a per-process tempdir before any test reads it. Without
/// this, infer_status would pick up real `~/.claudectl/state/*.json` files
/// from a developer's machine and tests would be non-hermetic.
fn isolate_hook_state_dir() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let dir =
            std::env::temp_dir().join(format!("claudectl-itest-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: set_var is unsafe in 2024 edition; tests are single-process
        // and this is set once before any other thread reads it.
        unsafe { std::env::set_var("CLAUDECTL_STATE_DIR", &dir) };
    });
}

/// Helper: create a minimal session for testing status inference.
fn make_session(cpu: f32, last_message_age_secs: u64) -> ClaudeSession {
    isolate_hook_state_dir();
    let raw = RawSession {
        pid: 1,
        session_id: "test-session".into(),
        cwd: "/tmp/test-project".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    s.cpu_percent = cpu;
    s.telemetry_status = TelemetryStatus::Available;
    s.usage_metrics_available = true;

    // Set last_message_ts relative to now
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    s.last_message_ts = now_ms.saturating_sub(last_message_age_secs * 1000);
    s
}

// ────────────────────────────────────────────────────────────────────────────
// Status Inference Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn status_high_cpu_always_processing() {
    let mut s = make_session(50.0, 0);
    monitor::infer_status(&mut s, "", "", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_high_cpu_overrides_waiting_for_task() {
    let mut s = make_session(10.0, 0);
    monitor::infer_status(&mut s, "assistant", "end_turn", true);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_high_cpu_overrides_end_turn() {
    let mut s = make_session(20.0, 60);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_waiting_for_task_no_longer_promotes_needs_input() {
    // The legacy `is_waiting_for_task` JSONL signal is no longer trusted as
    // a NeedsInput indicator — too many false positives. NeedsInput is
    // exclusively driven by the deterministic Notification hook now.
    // Heuristic still falls back to a sensible non-attention-grabbing state.
    let mut s = make_session(0.5, 10);
    monitor::infer_status(&mut s, "", "", true);
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn status_end_turn_recent_waiting_input() {
    // Assistant said end_turn, 2 minutes ago, low CPU
    let mut s = make_session(0.5, 120);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn status_end_turn_old_idle_in_heuristic_path() {
    // Heuristic-only path (no hook state). After 15 quiet minutes a stop_reason
    // of end_turn/stop_sequence is genuinely abandoned — show Idle so the user
    // can sort/filter past it. The deterministic Stop hook handles still-active
    // post-turn sessions before we reach this branch.
    let mut s = make_session(0.5, 15 * 60);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn status_end_turn_recent_waiting_input_still_works() {
    let mut s = make_session(0.5, 10 * 60);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn status_tool_use_low_cpu_no_longer_promotes_needs_input() {
    // assistant + tool_use + idle CPU used to be guessed as a permission
    // prompt — that was the central source of "Needs Input" false positives
    // (parked sessions, sessions with stale tool_use tail, etc.). NeedsInput
    // is now exclusively the Notification hook's call.
    let mut s = make_session(0.5, 30);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn status_tool_use_low_cpu_recent_processing() {
    // tool_use + low CPU + <5s ago = still processing (tool just fired)
    let mut s = make_session(0.5, 2);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_tool_use_high_cpu_processing() {
    // tool_use + high CPU = still crunching
    let mut s = make_session(15.0, 30);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_active_cpu_processing() {
    // CPU > 2.0 → Claude is actually thinking, regardless of age.
    let mut s = make_session(3.0, 30);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_recent_low_cpu_processing() {
    // Fresh user message + low CPU = still warming up; stay Processing.
    let mut s = make_session(0.5, 1);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_quiet_low_cpu_stays_processing() {
    // Heuristic fallback can't tell apart "permission prompt for an unflushed
    // tool_use" from "session was parked mid-conversation" — both look the
    // same. Stay Processing while still recent so we don't bury an actually-
    // active session, but age out to Idle eventually (covered separately).
    let mut s = make_session(0.5, 30);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_long_quiet_idle_in_heuristic_path() {
    // After 15 quiet minutes with no hook state, treat user-tail JSONL as
    // genuinely abandoned. The deterministic Notification hook would have
    // already flipped it to NeedsInput before reaching this branch if the
    // session was actually waiting on a permission prompt.
    let mut s = make_session(0.5, 15 * 60);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn status_no_signals_idle() {
    // No JSONL signals at all → Idle
    let mut s = make_session(0.0, 0);
    monitor::infer_status(&mut s, "", "", false);
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn status_no_telemetry_unknown() {
    isolate_hook_state_dir();
    let raw = RawSession {
        pid: 1,
        session_id: "test-session-no-telemetry".into(),
        cwd: "/tmp/test-project".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    monitor::infer_status(&mut s, "", "", false);
    assert_eq!(s.status, SessionStatus::Unknown);
}

// ────────────────────────────────────────────────────────────────────────────
// Deterministic hook-state path
// ────────────────────────────────────────────────────────────────────────────

/// Build a session with a unique `session_id` so each test owns its own
/// state file and can't be polluted by sibling tests.
fn session_with_id(id: &str, cpu: f32) -> ClaudeSession {
    isolate_hook_state_dir();
    let raw = RawSession {
        pid: 1,
        session_id: id.into(),
        cwd: "/tmp/test-project".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    s.cpu_percent = cpu;
    s.telemetry_status = TelemetryStatus::Available;
    s.usage_metrics_available = true;
    s.last_message_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    s
}

#[test]
fn hook_permission_prompt_marks_needs_input() {
    isolate_hook_state_dir();
    let sid = "hook-test-permission";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();

    // Backdate the notification past the 750ms grace period so the
    // suppression doesn't hide it during this test.
    let mut state = claudectl::hook_state::HookState::load(sid).unwrap();
    state.last_notification_ts_ms = state.last_notification_ts_ms.saturating_sub(2_000);
    let path = claudectl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    // Low CPU + permission_prompt marker (now older than grace) + JSONL has
    // NOT grown past the notification → NeedsInput (deterministic path).
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_notification_ts_ms.saturating_sub(1000);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_pretooluse_clears_permission_prompt() {
    isolate_hook_state_dir();
    let sid = "hook-test-approval";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    // Approval flipped the marker; we should now report Processing (a tool
    // is actively running, no PostToolUse yet).
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn hook_precompact_marks_compacting() {
    isolate_hook_state_dir();
    let sid = "hook-test-compacting";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreCompact",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Compacting);
}

#[test]
fn hook_stop_marks_waiting_input() {
    isolate_hook_state_dir();
    let sid = "hook-test-stop";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn hook_userpromptsubmit_after_stop_marks_responding() {
    isolate_hook_state_dir();
    let sid = "hook-test-followup";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();
    // User typed a follow-up — is_responding fires immediately, deterministic.
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn hook_waiting_input_ages_out_to_idle() {
    isolate_hook_state_dir();
    let sid = "hook-test-waiting-ages-out";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();

    // Backdate the Stop ts to >10 min ago so the age-out fires.
    let mut state = claudectl::hook_state::HookState::load(sid).unwrap();
    state.last_stop_ts_ms = state.last_stop_ts_ms.saturating_sub(11 * 60 * 1000);
    let path = claudectl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn hook_waiting_input_recent_stays_waiting() {
    isolate_hook_state_dir();
    let sid = "hook-test-waiting-recent";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn hook_responding_stable_across_tool_boundaries() {
    isolate_hook_state_dir();
    // The whole point of the is_responding check: tools coming and going
    // inside one turn don't flicker the status. UserPromptSubmit was the
    // most-recent-event when the turn started; PreToolUse/PostToolUse
    // happen during the response; status stays Processing the whole time.
    let sid = "hook-test-stable";
    for ev in [
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PreToolUse",
    ] {
        claudectl::hook_state::record_hook_event(&serde_json::json!({
            "hook_event_name": ev,
            "session_id": sid,
            "tool_name": "Bash",
        }))
        .unwrap();
    }

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_eq!(s.status, SessionStatus::Processing);

    // Stop fires → flips to WaitingInput, also stable.
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn hook_permission_prompt_cleared_by_subsequent_event() {
    isolate_hook_state_dir();
    // After Notification, ANY later state-changing event clears the prompt
    // regardless of which one. PreToolUse means approved; PostToolUse means
    // a tool finished (could be a denial result); UserPromptSubmit means
    // user typed past the dialog; Stop means turn ended. Whichever fires
    // first removes NeedsInput.
    let sid = "hook-test-cleared-by-pretooluse";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_worker_permission_prompt_marks_needs_input() {
    isolate_hook_state_dir();
    // Subagents fire `notification_type = "worker_permission_prompt"` instead
    // of `"permission_prompt"` (verified against Claude Code 2.1.117 binary).
    // Both must classify the session as NeedsInput.
    let sid = "hook-test-worker-permission";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "worker_permission_prompt",
    }))
    .unwrap();

    // Backdate past the 750ms grace period.
    let mut state = claudectl::hook_state::HookState::load(sid).unwrap();
    state.last_notification_ts_ms = state.last_notification_ts_ms.saturating_sub(2_000);
    let path = claudectl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_notification_ts_ms.saturating_sub(1000);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_worker_pretooluse_clears_permission_prompt() {
    isolate_hook_state_dir();
    // Approval of a subagent's prompt fires PreToolUse with the approved
    // tool — same semantic as the main-agent case.
    let sid = "hook-test-worker-approval";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "worker_permission_prompt",
    }))
    .unwrap();
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_permission_prompt_outranks_compacting() {
    isolate_hook_state_dir();
    // Edge case: both signals are set. NeedsInput wins because a pending
    // permission prompt is the most actionable state and because Compacting
    // has been observed to get stuck on sessions where Stop never fires —
    // without this precedence a real prompt would be silently masked.
    let sid = "hook-test-precedence";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreCompact",
        "session_id": sid,
    }))
    .unwrap();

    // Backdate the notification past the 750ms grace period.
    let mut state = claudectl::hook_state::HookState::load(sid).unwrap();
    state.last_notification_ts_ms = state.last_notification_ts_ms.saturating_sub(2_000);
    let path = claudectl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_postcompact_clears_compacting_without_stop() {
    isolate_hook_state_dir();
    // Auto-compact paths where Stop never fires: PostCompact is the direct
    // "compaction done" signal and must clear the Compacting status on its
    // own.
    let sid = "hook-test-postcompact";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreCompact",
        "session_id": sid,
    }))
    .unwrap();

    // Mid-compact: Compacting is the correct status.
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "", false);
    assert_eq!(s.status, SessionStatus::Compacting);

    // PostCompact arrives. Stop never fires.
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PostCompact",
        "session_id": sid,
    }))
    .unwrap();
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "", false);
    assert_ne!(s.status, SessionStatus::Compacting);
}

/// Replay a turn that ran and finished, but whose `Stop` hook never reached
/// claudectl — the 2026-07-28 incident. `mid_turn_age_secs` backdates every
/// recorded event so the transcript tail can be placed after them.
fn state_with_lost_stop(sid: &str, mid_turn_age_secs: u64) -> claudectl::hook_state::HookState {
    isolate_hook_state_dir();
    for (event, tool) in [
        ("UserPromptSubmit", None),
        ("PreToolUse", Some("Bash")),
        ("PostToolUse", Some("Bash")),
    ] {
        claudectl::hook_state::record_hook_event(&serde_json::json!({
            "hook_event_name": event,
            "session_id": sid,
            "tool_name": tool,
        }))
        .unwrap();
    }
    let mut state = claudectl::hook_state::HookState::load(sid).unwrap();
    let back = mid_turn_age_secs * 1000;
    state.last_promptsubmit_ts_ms = state.last_promptsubmit_ts_ms.saturating_sub(back);
    state.last_pretooluse_ts_ms = state.last_pretooluse_ts_ms.saturating_sub(back);
    state.last_posttooluse_ts_ms = state.last_posttooluse_ts_ms.saturating_sub(back);
    assert_eq!(state.last_stop_ts_ms, 0, "the lost Stop is the whole point");
    let path = claudectl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();
    state
}

#[test]
fn regression_lost_stop_does_not_pin_finished_turn_to_processing() {
    // 2026-07-28: three live sandbox sessions sat on `Processing` for hours
    // (one for 15h) while their transcripts ended in `assistant`/`end_turn`.
    // Every one had `last_stop_ts_ms == 0`: the Stop hook never reached
    // claudectl, `is_responding` had no staleness bound, and
    // `status_from_hook_state` returned early — so the JSONL heuristic that
    // would have said WaitingInput never ran.
    let sid = "regression-lost-stop-waiting";
    let state = state_with_lost_stop(sid, 120);

    // Transcript ended the turn AFTER the last recorded hook event.
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_posttooluse_ts_ms + 1_000;
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn regression_lost_stop_long_quiet_session_ages_to_idle() {
    // Same shape, but the finished turn is long past — the session belongs in
    // Idle, not Processing. (Live case: `d0ac5803`, last `end_turn` 15h old.)
    let sid = "regression-lost-stop-idle";
    let state = state_with_lost_stop(sid, 20 * 60);

    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_posttooluse_ts_ms + 1_000;
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn regression_lost_stop_veto_needs_a_finished_transcript_tail() {
    // The transcript only vetoes when it actually says the turn ended. A tail
    // still sitting on `tool_use` proves nothing, so the hook state stands and
    // the session stays Processing.
    let sid = "regression-lost-stop-tool-tail";
    let state = state_with_lost_stop(sid, 120);

    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_posttooluse_ts_ms + 1_000;
    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn regression_transcript_veto_never_masks_a_live_turn() {
    isolate_hook_state_dir();
    // The guard that keeps the fix honest: a turn that is genuinely under way
    // must stay Processing. The user's follow-up prompt is newer than the
    // transcript's last (previous-turn) `end_turn` message, so the stale tail
    // must not be read as "this turn is over".
    let sid = "regression-live-turn-not-masked";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": sid,
    }))
    .unwrap();

    let state = claudectl::hook_state::HookState::load(sid).unwrap();
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_promptsubmit_ts_ms.saturating_sub(1_000);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn regression_transcript_veto_never_masks_an_in_flight_tool() {
    isolate_hook_state_dir();
    // Same guard, tool edition: PreToolUse fired after the transcript's last
    // message (Claude Code writes the tool_use entry, then the tool runs).
    // The tail is older than the hook event, so no veto — still Processing.
    let sid = "regression-in-flight-tool-not-masked";
    claudectl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    let state = claudectl::hook_state::HookState::load(sid).unwrap();
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_pretooluse_ts_ms.saturating_sub(1_000);
    monitor::infer_status(&mut s, "assistant", "end_turn", false);
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_cpu_threshold_boundary() {
    // CPU exactly 5.0 — should NOT trigger Processing (threshold is >5.0)
    let mut s = make_session(5.0, 0);
    monitor::infer_status(&mut s, "", "", false);
    assert_eq!(s.status, SessionStatus::Idle);

    // CPU 5.1 — should trigger Processing
    let mut s2 = make_session(5.1, 0);
    monitor::infer_status(&mut s2, "", "", false);
    assert_eq!(s2.status, SessionStatus::Processing);
}

#[test]
fn status_persisted_tool_use_survives_empty_tick() {
    // Tool_use tail is no longer guessed as NeedsInput in the heuristic
    // path (Notification hook owns that signal). What we still want to
    // verify is that the persisted tool_use signal stays stable across
    // empty ticks — i.e., status doesn't drop to Idle the moment JSONL
    // stops growing.
    let mut s = make_session(0.5, 30);

    monitor::infer_status(&mut s, "assistant", "tool_use", false);
    let first_tick = s.status;
    assert_ne!(first_tick, SessionStatus::Idle);

    s.last_msg_type = "assistant".into();
    s.last_stop_reason = "tool_use".into();
    s.is_waiting_for_task = false;

    let msg_type = s.last_msg_type.clone();
    let stop_reason = s.last_stop_reason.clone();
    let waiting = s.is_waiting_for_task;
    monitor::infer_status(&mut s, &msg_type, &stop_reason, waiting);
    assert_eq!(s.status, first_tick);
}

#[test]
fn status_null_stop_reason_with_tool_use_inferred_from_content() {
    // Claude Code writes stop_reason: null for tool calls awaiting approval.
    // We still infer "tool_use" from content so the JSONL parser is correct.
    // The session no longer auto-promotes to NeedsInput from this signal —
    // that's the Notification hook's exclusive call.
    let jsonl = r#"{"type":"assistant","message":{"role":"assistant","model":"claude-opus-4-6","stop_reason":null,"content":[{"type":"tool_use","id":"toolu_01X","name":"Bash","input":{"command":"echo hi"}}],"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#;

    let (mut s, _file) = make_session_with_jsonl(jsonl);
    s.cpu_percent = 0.5;
    monitor::update_tokens(&mut s);

    assert_eq!(s.last_stop_reason, "tool_use");
    assert_eq!(s.pending_tool_name, Some("Bash".into()));
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

// ────────────────────────────────────────────────────────────────────────────
// Cost Estimation Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn cost_opus_tokens() {
    let mut s = make_session(0.0, 0);
    s.model = "opus-4.6".into();
    s.total_input_tokens = 1_000_000;
    s.total_output_tokens = 100_000;
    s.cache_read_tokens = 500_000;
    s.cache_write_tokens = 200_000;

    let cost = monitor::estimate_cost(&s);
    // plain_input = 1M - 500k - 200k = 300k
    // cost = 300k/1M * 15 + 100k/1M * 75 + 500k/1M * 1.875 + 200k/1M * 18.75
    //      = 0.3 * 15 + 0.1 * 75 + 0.5 * 1.875 + 0.2 * 18.75
    //      = 4.5 + 7.5 + 0.9375 + 3.75 = 16.6875
    let expected = 16.6875;
    assert!(
        (cost - expected).abs() < 0.001,
        "opus cost={cost}, expected={expected}"
    );
}

#[test]
fn cost_sonnet_tokens() {
    let mut s = make_session(0.0, 0);
    s.model = "sonnet-4.6".into();
    s.total_input_tokens = 100_000;
    s.total_output_tokens = 50_000;
    s.cache_read_tokens = 0;
    s.cache_write_tokens = 0;

    let cost = monitor::estimate_cost(&s);
    // plain_input = 100k
    // cost = 100k/1M * 3 + 50k/1M * 15 = 0.3 + 0.75 = 1.05
    let expected = 1.05;
    assert!(
        (cost - expected).abs() < 0.001,
        "sonnet cost={cost}, expected={expected}"
    );
}

#[test]
fn cost_haiku_tokens() {
    let mut s = make_session(0.0, 0);
    s.model = "haiku".into();
    s.total_input_tokens = 100_000;
    s.total_output_tokens = 50_000;
    s.cache_read_tokens = 0;
    s.cache_write_tokens = 0;

    let cost = monitor::estimate_cost(&s);
    // plain_input = 100k
    // cost = 100k/1M * 0.80 + 50k/1M * 4.0 = 0.08 + 0.2 = 0.28
    let expected = 0.28;
    assert!(
        (cost - expected).abs() < 0.001,
        "haiku cost={cost}, expected={expected}"
    );
}

#[test]
fn cost_unknown_model_defaults_to_opus() {
    let mut s = make_session(0.0, 0);
    s.model = "some-future-model".into();
    s.total_input_tokens = 1_000_000;
    s.total_output_tokens = 0;
    s.cache_read_tokens = 0;
    s.cache_write_tokens = 0;

    let cost = monitor::estimate_cost(&s);
    // Should use opus pricing: 1M/1M * 15 = 15.0
    let expected = 15.0;
    assert!(
        (cost - expected).abs() < 0.001,
        "unknown model cost={cost}, expected={expected}"
    );
}

#[test]
fn cost_zero_tokens() {
    let s = make_session(0.0, 0);
    let cost = monitor::estimate_cost(&s);
    assert_eq!(cost, 0.0);
}

// ────────────────────────────────────────────────────────────────────────────
// Model Context Max Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn context_max_opus() {
    assert_eq!(monitor::model_context_max("opus-4.6"), 1_000_000);
    assert_eq!(monitor::model_context_max("opus"), 1_000_000);
}

#[test]
fn context_max_sonnet() {
    assert_eq!(monitor::model_context_max("sonnet-4.6"), 200_000);
    assert_eq!(monitor::model_context_max("sonnet"), 200_000);
}

#[test]
fn context_max_haiku() {
    assert_eq!(monitor::model_context_max("haiku"), 200_000);
}

#[test]
fn context_max_unknown() {
    assert_eq!(monitor::model_context_max("unknown-model"), 200_000);
}

// ────────────────────────────────────────────────────────────────────────────
// Model Shortening Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn shorten_model_opus_46() {
    assert_eq!(
        monitor::shorten_model("claude-opus-4-6-20260401"),
        "opus-4.6"
    );
}

#[test]
fn shorten_model_opus_generic() {
    assert_eq!(monitor::shorten_model("claude-opus-20260101"), "opus");
}

#[test]
fn shorten_model_sonnet_46() {
    assert_eq!(
        monitor::shorten_model("claude-sonnet-4-6-20260401"),
        "sonnet-4.6"
    );
}

#[test]
fn shorten_model_sonnet_generic() {
    assert_eq!(monitor::shorten_model("claude-sonnet-20260101"), "sonnet");
}

#[test]
fn shorten_model_haiku() {
    assert_eq!(monitor::shorten_model("claude-haiku-4-5-20251001"), "haiku");
}

#[test]
fn shorten_model_unknown() {
    assert_eq!(monitor::shorten_model("gpt-4o"), "gpt-4o");
}

// ────────────────────────────────────────────────────────────────────────────
// JSONL Parsing Integration Tests (using temp files)
// ────────────────────────────────────────────────────────────────────────────

fn make_session_with_jsonl(content: &str) -> (ClaudeSession, tempfile::NamedTempFile) {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();

    let raw = RawSession {
        pid: 1,
        session_id: "test".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    s.jsonl_path = Some(file.path().to_path_buf());
    (s, file)
}

fn make_session_with_paths(
    cwd: String,
    session_id: String,
    jsonl_path: std::path::PathBuf,
) -> ClaudeSession {
    let raw = RawSession {
        pid: 1,
        session_id,
        cwd,
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    s.jsonl_path = Some(jsonl_path);
    s
}

fn write_jsonl(path: &std::path::Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

fn expected_cost(model: &str, input_tokens: u64, output_tokens: u64) -> f64 {
    let profile = models::resolve(model).profile;
    (input_tokens as f64 / 1_000_000.0) * profile.input_per_m
        + (output_tokens as f64 / 1_000_000.0) * profile.output_per_m
}

#[test]
fn jsonl_parse_token_usage() {
    let jsonl = r#"{"type":"assistant","message":{"model":"claude-opus-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":50000,"output_tokens":10000,"cache_read_input_tokens":20000,"cache_creation_input_tokens":5000}}}"#;

    let (mut s, _file) = make_session_with_jsonl(jsonl);
    monitor::update_tokens(&mut s);

    assert_eq!(s.total_input_tokens, 75000); // 50000 + 20000 + 5000
    assert_eq!(s.total_output_tokens, 10000);
    assert_eq!(s.cache_read_tokens, 20000);
    assert_eq!(s.cache_write_tokens, 5000);
    assert_eq!(s.model, "opus-4.6");
    assert_eq!(s.context_max, 1_000_000);
}

#[test]
fn jsonl_parse_multiple_entries() {
    let jsonl = concat!(
        r#"{"type":"user","message":{"type":"user"}}"#,
        "\n",
        r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6-20260401","stop_reason":"tool_use","usage":{"input_tokens":1000,"output_tokens":500,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
        "\n",
        r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":2000,"output_tokens":1000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );

    let (mut s, _file) = make_session_with_jsonl(jsonl);
    monitor::update_tokens(&mut s);

    assert_eq!(s.total_input_tokens, 3000); // 1000 + 2000
    assert_eq!(s.total_output_tokens, 1500); // 500 + 1000
    assert_eq!(s.model, "sonnet-4.6");
}

#[test]
fn jsonl_incremental_reads() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let line1 = r#"{"type":"assistant","message":{"model":"claude-opus-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":1000,"output_tokens":500,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#;
    writeln!(file, "{line1}").unwrap();
    file.flush().unwrap();

    let raw = RawSession {
        pid: 1,
        session_id: "test".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    s.jsonl_path = Some(file.path().to_path_buf());

    // First read
    monitor::update_tokens(&mut s);
    assert_eq!(s.total_input_tokens, 1000);
    assert_eq!(s.total_output_tokens, 500);

    // Second read with no new data — should not double-count
    monitor::update_tokens(&mut s);
    assert_eq!(s.total_input_tokens, 1000);
    assert_eq!(s.total_output_tokens, 500);

    // Append more data
    let line2 = r#"{"type":"assistant","message":{"model":"claude-opus-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":2000,"output_tokens":800,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#;
    writeln!(file, "{line2}").unwrap();
    file.flush().unwrap();

    // Third read — should pick up new data only
    monitor::update_tokens(&mut s);
    assert_eq!(s.total_input_tokens, 3000);
    assert_eq!(s.total_output_tokens, 1300);
}

#[test]
fn jsonl_empty_file() {
    let (mut s, _file) = make_session_with_jsonl("");
    monitor::update_tokens(&mut s);
    assert_eq!(s.total_input_tokens, 0);
    assert_eq!(s.total_output_tokens, 0);
}

#[test]
fn jsonl_corrupted_lines_skipped() {
    let jsonl = concat!(
        "not valid json at all\n",
        "{\"type\":\"something but no usage\"}\n",
        r#"{"type":"assistant","message":{"model":"claude-opus-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":5000,"output_tokens":1000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );

    let (mut s, _file) = make_session_with_jsonl(jsonl);
    monitor::update_tokens(&mut s);

    // Should still parse the valid line
    assert_eq!(s.total_input_tokens, 5000);
    assert_eq!(s.total_output_tokens, 1000);
}

#[test]
fn jsonl_waiting_for_task_no_longer_promotes_needs_input() {
    // The legacy `waiting_for_task` JSONL progress signal is parsed but no
    // longer promotes the session to NeedsInput — too unreliable. The
    // Notification hook owns NeedsInput now; this just confirms the heuristic
    // doesn't claim it.
    let jsonl = concat!(
        r#"{"type":"assistant","message":{"model":"claude-opus-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":1000,"output_tokens":500,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
        "\n",
        r#"{"type":"progress","data":"waiting_for_task"}"#,
    );

    let (mut s, _file) = make_session_with_jsonl(jsonl);
    s.cpu_percent = 0.5;
    monitor::update_tokens(&mut s);

    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn jsonl_missing_file() {
    let raw = RawSession {
        pid: 1,
        session_id: "test".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    s.jsonl_path = Some(std::path::PathBuf::from("/nonexistent/path.jsonl"));

    // Should not panic
    monitor::update_tokens(&mut s);
    assert_eq!(s.total_input_tokens, 0);
}

#[test]
fn jsonl_no_path() {
    let raw = RawSession {
        pid: 1,
        session_id: "test".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: None,
    };
    let mut s = ClaudeSession::from_raw(raw);
    // jsonl_path is None

    monitor::update_tokens(&mut s);
    assert_eq!(s.total_input_tokens, 0);
}

#[test]
fn jsonl_rolls_up_subagent_tokens_and_cost() {
    let temp = tempfile::tempdir().unwrap();
    let parent_jsonl = temp.path().join("parent.jsonl");
    write_jsonl(
        &parent_jsonl,
        r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":100000,"output_tokens":50000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );

    let session_id = format!("subagent-rollup-{}", std::process::id());
    let cwd = format!("/tmp/claudectl-rollup-{}", std::process::id());
    let slug = cwd.replace('/', "-");
    let uid = unsafe { libc::getuid() };
    let tasks_dir = std::path::PathBuf::from(format!("/tmp/claude-{uid}"))
        .join(&slug)
        .join(&session_id)
        .join("tasks");
    write_jsonl(
        &tasks_dir.join("agent-1.jsonl"),
        r#"{"type":"assistant","message":{"model":"claude-opus-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":200000,"output_tokens":50000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );
    write_jsonl(
        &tasks_dir.join("nested/agent-2.jsonl"),
        r#"{"type":"assistant","message":{"model":"claude-haiku-4-5-20260101","stop_reason":"end_turn","usage":{"input_tokens":50000,"output_tokens":10000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );

    let mut s = make_session_with_paths(cwd, session_id, parent_jsonl);
    discovery::scan_subagents(std::slice::from_mut(&mut s));
    monitor::update_tokens(&mut s);

    assert_eq!(s.active_subagent_count, 2);
    assert_eq!(s.subagent_count, 2);
    assert_eq!(s.total_input_tokens, 350_000);
    assert_eq!(s.total_output_tokens, 110_000);

    let expected = expected_cost("sonnet-4.6", 100_000, 50_000)
        + expected_cost("opus-4.6", 200_000, 50_000)
        + expected_cost("haiku", 50_000, 10_000);
    assert!((s.cost_usd - expected).abs() < 0.0001);
    assert!(!s.cost_estimate_unverified);

    let _ = std::fs::remove_dir_all(
        std::path::PathBuf::from(format!("/tmp/claude-{uid}"))
            .join(&slug)
            .join(&s.session_id),
    );
}

#[test]
fn subagent_rollup_persists_after_task_file_disappears() {
    let temp = tempfile::tempdir().unwrap();
    let parent_jsonl = temp.path().join("parent.jsonl");
    write_jsonl(
        &parent_jsonl,
        r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":100000,"output_tokens":10000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );

    let session_id = format!("subagent-persist-{}", std::process::id());
    let cwd = format!("/tmp/claudectl-persist-{}", std::process::id());
    let slug = cwd.replace('/', "-");
    let uid = unsafe { libc::getuid() };
    let subagent_root = std::path::PathBuf::from(format!("/tmp/claude-{uid}"))
        .join(&slug)
        .join(&session_id);
    let tasks_dir = subagent_root.join("tasks");
    write_jsonl(
        &tasks_dir.join("agent-1.jsonl"),
        r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6-20260401","stop_reason":"end_turn","usage":{"input_tokens":200000,"output_tokens":20000,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    );

    let mut s = make_session_with_paths(cwd, session_id, parent_jsonl);
    discovery::scan_subagents(std::slice::from_mut(&mut s));
    monitor::update_tokens(&mut s);

    assert_eq!(s.active_subagent_count, 1);
    assert_eq!(s.subagent_count, 1);
    assert_eq!(s.total_input_tokens, 300_000);
    assert_eq!(s.total_output_tokens, 30_000);

    std::fs::remove_dir_all(&subagent_root).unwrap();

    discovery::scan_subagents(std::slice::from_mut(&mut s));
    monitor::update_tokens(&mut s);

    assert_eq!(s.active_subagent_count, 0);
    assert_eq!(s.subagent_count, 1);
    assert_eq!(s.total_input_tokens, 300_000);
    assert_eq!(s.total_output_tokens, 30_000);
}

// ────────────────────────────────────────────────────────────────────────────
// Session Formatting Edge Cases
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn context_percent_zero_max() {
    let mut s = make_session(0.0, 0);
    s.context_max = 0;
    s.context_tokens = 1000;
    assert_eq!(s.context_percent(), 0.0);
}

#[test]
fn context_percent_zero_tokens() {
    let mut s = make_session(0.0, 0);
    s.context_max = 200_000;
    s.context_tokens = 0;
    assert_eq!(s.context_percent(), 0.0);
}

#[test]
fn context_percent_calculation() {
    let mut s = make_session(0.0, 0);
    s.context_max = 200_000;
    s.context_tokens = 100_000;
    assert!((s.context_percent() - 50.0).abs() < 0.01);
}

#[test]
fn sparkline_empty() {
    let s = make_session(0.0, 0);
    assert_eq!(s.format_sparkline(), "-");
}

#[test]
fn sparkline_records_and_renders() {
    let mut s = make_session(0.0, 0);
    s.status = SessionStatus::Processing;
    s.record_activity();
    s.status = SessionStatus::Idle;
    s.record_activity();

    let sparkline = s.format_sparkline();
    assert_eq!(sparkline.chars().count(), 2);
}

#[test]
fn sparkline_ring_buffer_limit() {
    let mut s = make_session(0.0, 0);
    for _ in 0..20 {
        s.status = SessionStatus::Processing;
        s.record_activity();
    }
    // Should be capped at 15
    assert_eq!(s.activity_history.len(), 15);
}

#[test]
fn json_export_format() {
    let mut s = make_session(0.0, 0);
    s.model = "opus-4.6".into();
    s.cost_usd = 1.234;
    s.total_input_tokens = 50000;
    s.total_output_tokens = 10000;
    s.elapsed = Duration::from_secs(300);

    let json = s.to_json_value();
    assert_eq!(json["pid"], 1);
    assert_eq!(json["status"], "Idle");
    assert_eq!(json["elapsed_secs"], 300);
    assert_eq!(json["tokens_in"], 50000);
    assert_eq!(json["tokens_out"], 10000);
    assert!(json["subagent_breakdown"].as_array().unwrap().is_empty());
}

#[test]
fn json_export_includes_subagent_breakdown() {
    let mut s = make_session(0.0, 0);
    s.active_subagent_jsonl_paths = vec![std::path::PathBuf::from(
        "/tmp/claude-1/-tmp-project/session-1/tasks/agent-2.jsonl",
    )];
    s.subagent_rollups.insert(
        std::path::PathBuf::from("/tmp/claude-1/-tmp-project/session-1/tasks/agent-1.jsonl"),
        claudectl::session::SubagentRollup {
            input_tokens: 20_000,
            output_tokens: 2_000,
            cost_usd: 0.4,
            usage_metrics_available: true,
            ..claudectl::session::SubagentRollup::default()
        },
    );
    s.subagent_rollups.insert(
        std::path::PathBuf::from("/tmp/claude-1/-tmp-project/session-1/tasks/agent-2.jsonl"),
        claudectl::session::SubagentRollup {
            input_tokens: 10_000,
            output_tokens: 1_000,
            cost_usd: 0.2,
            usage_metrics_available: true,
            ..claudectl::session::SubagentRollup::default()
        },
    );
    s.subagent_count = 2;
    s.active_subagent_count = 1;

    let json = s.to_json_value();
    let breakdown = json["subagent_breakdown"].as_array().unwrap();
    assert_eq!(breakdown.len(), 2);
    assert_eq!(breakdown[0]["label"], "completed");
    assert_eq!(breakdown[0]["state"], "Completed");
    assert_eq!(breakdown[0]["tokens_in"], 20000);
    assert_eq!(breakdown[1]["label"], "agent-2");
    assert_eq!(breakdown[1]["state"], "Active");
}

#[test]
fn burn_rate_formatting() {
    let mut s = make_session(0.0, 0);
    assert_eq!(s.format_burn_rate(), "-");

    s.burn_rate_per_hr = 0.50;
    assert_eq!(s.format_burn_rate(), "$0.50/h");

    s.burn_rate_per_hr = 3.5;
    assert_eq!(s.format_burn_rate(), "$3.5/h");
}

#[test]
fn mem_formatting() {
    let mut s = make_session(0.0, 0);
    assert_eq!(s.format_mem(), "-");

    s.mem_mb = 256.7;
    assert_eq!(s.format_mem(), "257M");
}

#[test]
fn context_bar_formatting() {
    let mut s = make_session(0.0, 0);
    assert_eq!(s.format_context_bar(10), "-");

    s.context_max = 200_000;
    s.context_tokens = 100_000; // 50%
    let bar = s.format_context_bar(10);
    assert!(bar.contains("50%"));
    assert!(bar.contains("█████"));
    assert!(bar.contains("░░░░░"));
}

// ────────────────────────────────────────────────────────────────────────────
// Session Recorder Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn session_recorder_produces_highlight_reel() {
    use claudectl::session_recorder::SessionRecorder;

    // Create empty JSONL first, then create recorder (which seeks to end),
    // then write events to simulate live session activity
    let mut jsonl_file = tempfile::NamedTempFile::new().unwrap();
    jsonl_file.flush().unwrap();

    let output_file = tempfile::NamedTempFile::new().unwrap();
    let output_path = output_file.path().to_str().unwrap().to_string() + ".cast";

    let mut rec = SessionRecorder::new(jsonl_file.path(), &output_path, "test-project", 120, 40)
        .expect("Failed to create session recorder");

    // Now write events AFTER recorder was created (simulates live recording)
    writeln!(jsonl_file, r#"{{"message":{{"role":"assistant","type":"message","content":[{{"type":"text","text":"I'll fix the authentication bug by updating the middleware."}}],"stop_reason":"tool_use"}}}}"#).unwrap();
    writeln!(jsonl_file, r#"{{"message":{{"role":"assistant","type":"message","content":[{{"type":"tool_use","name":"Edit","input":{{"file_path":"/src/auth.rs","old_string":"fn check()","new_string":"fn check_auth(token: &str)"}}}}],"stop_reason":"tool_use"}}}}"#).unwrap();
    writeln!(jsonl_file, r#"{{"message":{{"role":"assistant","type":"message","content":[{{"type":"tool_use","name":"Bash","input":{{"command":"cargo test"}}}}],"stop_reason":"tool_use"}}}}"#).unwrap();
    writeln!(jsonl_file, r#"{{"message":{{"role":"user","type":"message","content":[{{"type":"tool_result","content":"test result: ok. 12 passed","is_error":false}}]}}}}"#).unwrap();
    writeln!(jsonl_file, r#"{{"message":{{"role":"assistant","type":"message","content":[{{"type":"tool_use","name":"Read","input":{{"file_path":"/src/main.rs"}}}}],"stop_reason":"tool_use"}}}}"#).unwrap();
    jsonl_file.flush().unwrap();

    let had_events = rec.poll().expect("Failed to poll");
    assert!(had_events, "Should have found events in the JSONL");

    rec.finish().expect("Failed to finish recording");

    let content = std::fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // First line is the asciicast header
    assert!(
        lines[0].contains("\"version\":2"),
        "Should have asciicast v2 header"
    );
    assert!(
        lines[0].contains("test-project"),
        "Header should contain session name"
    );

    // Should have multiple frames (header + title card + events + finish)
    assert!(
        lines.len() >= 4,
        "Should have at least 4 lines (header + title + events + finish), got {}",
        lines.len()
    );

    // Should contain the Edit tool rendered as Claude Code style "Update(file)"
    let full = content.to_string();
    assert!(
        full.contains("Update"),
        "Should contain Update event for Edit tool"
    );
    assert!(full.contains("auth.rs"), "Should contain edited file name");

    // Should contain the Bash command rendered Claude Code style
    assert!(
        full.contains("bash command"),
        "Should contain bash command indicator"
    );
    assert!(full.contains("cargo test"), "Should contain bash command");

    // Read events should appear as brief gray context lines (not full highlight frames)
    assert!(
        full.contains("Read"),
        "Read tool should appear as context line"
    );

    // Should contain final summary
    assert!(
        full.contains("complete"),
        "Should contain completion message"
    );

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn session_recorder_empty_jsonl() {
    use claudectl::session_recorder::SessionRecorder;

    let jsonl_file = tempfile::NamedTempFile::new().unwrap();
    let output_file = tempfile::NamedTempFile::new().unwrap();
    let output_path = output_file.path().to_str().unwrap().to_string() + ".cast";

    let mut rec = SessionRecorder::new(jsonl_file.path(), &output_path, "empty-session", 80, 24)
        .expect("Failed to create recorder");

    let had_events = rec.poll().expect("Failed to poll");
    assert!(!had_events, "Empty JSONL should produce no events");

    rec.finish().expect("Failed to finish");

    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(
        content.contains("\"version\":2"),
        "Should still have header"
    );

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn recorder_cast_file_creation() {
    use claudectl::recorder::Recorder;

    let output_file = tempfile::NamedTempFile::new().unwrap();
    let output_path = output_file.path().to_str().unwrap().to_string() + ".cast";

    let mut rec = Recorder::new(&output_path, 120, 40).expect("Failed to create recorder");
    rec.capture(b"hello world");
    rec.flush_frame().expect("Failed to flush");
    rec.capture(b"second frame");
    rec.flush_frame().expect("Failed to flush");
    rec.finish().expect("Failed to finish");

    let content = std::fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    assert!(lines[0].contains("\"version\":2"));
    assert!(lines[0].contains("\"width\":120"));
    assert!(lines[0].contains("\"height\":40"));
    assert!(
        lines.len() == 3,
        "Should have header + 2 frames, got {}",
        lines.len()
    );
    assert!(lines[1].contains("hello world"));
    assert!(lines[2].contains("second frame"));

    let _ = std::fs::remove_file(&output_path);
}

// ────────────────────────────────────────────────────────────────────────────
// Transcript Discovery Tests (Issue #161)
//
// These tests mutate the HOME env var so projects_dir() resolves to a temp dir.
// A mutex serializes them to prevent concurrent HOME changes across threads.
// ────────────────────────────────────────────────────────────────────────────

use std::sync::Mutex;
static HOME_LOCK: Mutex<()> = Mutex::new(());

/// Helper: build a fake ~/.claude layout in a temp dir and run resolve_jsonl_paths.
/// Holds HOME_LOCK for the duration.
fn resolve_with_layout(
    cwd: &str,
    session_id: &str,
    slug_on_disk: &str,
) -> (ClaudeSession, tempfile::TempDir) {
    let _guard = HOME_LOCK.lock().unwrap();

    let home = tempfile::tempdir().unwrap();
    let original_home = std::env::var("HOME").ok();
    unsafe { std::env::set_var("HOME", home.path()) };

    let project_dir = home.path().join(".claude/projects").join(slug_on_disk);
    std::fs::create_dir_all(&project_dir).unwrap();
    let jsonl_content = r#"{"type":"assistant","message":{"model":"claude-opus-4-6","stop_reason":"end_turn","usage":{"input_tokens":1,"cache_creation_input_tokens":523,"cache_read_input_tokens":79425,"output_tokens":937}}}"#;
    std::fs::write(
        project_dir.join(format!("{session_id}.jsonl")),
        jsonl_content,
    )
    .unwrap();

    let raw = RawSession {
        pid: 86131,
        session_id: session_id.to_string(),
        cwd: cwd.to_string(),
        started_at: 1776421121745,
        name: None,
    };
    let mut session = ClaudeSession::from_raw(raw);
    discovery::resolve_jsonl_paths(std::slice::from_mut(&mut session));

    // Restore HOME
    if let Some(h) = original_home {
        unsafe { std::env::set_var("HOME", h) };
    }

    (session, home)
}

#[test]
fn resolve_jsonl_standard_cwd() {
    let (s, _home) = resolve_with_layout(
        "/Users/testuser/Repos/data-platform-answers",
        "db55eb53-8ff0-45b7-9f8f-0d5dfa51e701",
        "-Users-testuser-Repos-data-platform-answers",
    );
    assert!(
        s.jsonl_path.is_some(),
        "should find JSONL for standard cwd (no trailing slash)"
    );
}

#[test]
fn resolve_jsonl_trailing_slash_cwd() {
    let (s, _home) = resolve_with_layout(
        "/Users/testuser/Repos/data-platform-answers/",
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "-Users-testuser-Repos-data-platform-answers",
    );
    assert!(
        s.jsonl_path.is_some(),
        "should find JSONL even when cwd has trailing slash"
    );
}

#[test]
fn resolve_jsonl_cwd_with_hyphens() {
    let (s, _home) = resolve_with_layout(
        "/Users/dev/my-cool-project",
        "11111111-2222-3333-4444-555555555555",
        "-Users-dev-my-cool-project",
    );
    assert!(
        s.jsonl_path.is_some(),
        "should find JSONL when cwd contains hyphens"
    );
}

#[test]
fn resolve_jsonl_encoding_mismatch_fallback() {
    let _guard = HOME_LOCK.lock().unwrap();

    let home = tempfile::tempdir().unwrap();
    let original_home = std::env::var("HOME").ok();
    unsafe { std::env::set_var("HOME", home.path()) };

    let session_id = "deadbeef-1234-5678-9abc-def012345678";
    let cwd = "/Users/testuser/projects/webapp";

    // JSONL under a slug that does NOT match cwd_to_slug(cwd)
    let wrong_slug = "-some-other-encoding-of-the-cwd";
    let project_dir = home.path().join(".claude/projects").join(wrong_slug);
    std::fs::create_dir_all(&project_dir).unwrap();
    std::fs::write(
        project_dir.join(format!("{session_id}.jsonl")),
        r#"{"type":"assistant","message":{"model":"claude-opus-4-6","stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
    ).unwrap();

    let raw = RawSession {
        pid: 99999,
        session_id: session_id.to_string(),
        cwd: cwd.to_string(),
        started_at: 0,
        name: None,
    };
    let mut session = ClaudeSession::from_raw(raw);
    discovery::resolve_jsonl_paths(std::slice::from_mut(&mut session));

    if let Some(h) = original_home {
        unsafe { std::env::set_var("HOME", h) };
    }

    assert!(
        session.jsonl_path.is_some(),
        "should find JSONL via fallback scan when slug encoding differs"
    );
}

#[test]
fn resolve_jsonl_telemetry_available_after_resolution() {
    let (mut s, _home) = resolve_with_layout(
        "/Users/testuser/myproject",
        "face0000-face-face-face-faceface0000",
        "-Users-testuser-myproject",
    );
    assert!(s.jsonl_path.is_some(), "precondition: jsonl_path found");

    monitor::update_tokens(&mut s);
    assert_eq!(
        s.telemetry_status,
        TelemetryStatus::Available,
        "telemetry should be Available after parsing JSONL, not {:?}",
        s.telemetry_status
    );
    assert!(s.usage_metrics_available);
    assert!(s.own_output_tokens > 0, "should have parsed output tokens");
}
