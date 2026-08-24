use std::io::Write;
use std::sync::Once;
use std::time::Duration;

use agentctl::discovery;
use agentctl::models;
use agentctl::monitor;
use agentctl::session::{AgentSession, RawSession, SessionStatus, TelemetryStatus};

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
fn make_session(cpu: f32, last_message_age_secs: u64) -> AgentSession {
    isolate_hook_state_dir();
    let raw = RawSession {
        pid: 1,
        session_id: "test-session".into(),
        cwd: "/tmp/test-project".into(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
    s.cpu_rate_percent = Some(cpu);
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
fn a_session_blocked_on_a_permission_prompt_reads_as_needs_input() {
    // The bug this file's `isolate_hook_state_dir` normally hides, asserted
    // deliberately: a session whose hook state was written by *another*
    // machine — a sandbox — must reach `NeedsInput` on the host that renders
    // it. Before hook state moved to the shared mount this was unreachable:
    // `HookState::load` looked in the reader's own private `~/.claudectl`,
    // always missed for a sandbox session, and fell through to the heuristic,
    // which by design never produces `NeedsInput`.
    //
    // The transcript tail here is the one both stuck sessions had on
    // 2026-08-04 — `user` / `tool_result`, because a pending permission prompt
    // writes nothing to the JSONL — which is exactly what made the heuristic
    // answer `Processing`.
    let mut session = make_session(0.6, 5);
    session.session_id = "cf54da79-2d23-4231-81fd-ce2a441e6e39".into();

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let state = serde_json::json!({
        "session_id": session.session_id,
        "last_notification_ts_ms": now_ms - 5_000,
        "notification_kind": "permission_prompt",
        "last_pretooluse_ts_ms": now_ms - 9_000,
        "last_posttooluse_ts_ms": now_ms - 8_000,
    });
    let dir = std::env::var("CLAUDECTL_STATE_DIR").expect("isolated by make_session");
    std::fs::write(
        std::path::Path::new(&dir).join(format!("{}.json", session.session_id)),
        serde_json::to_string(&state).unwrap(),
    )
    .unwrap();

    monitor::infer_status(&mut session, "user", "");

    assert_eq!(
        session.status,
        SessionStatus::NeedsInput,
        "a pending permission prompt must not render as Processing"
    );

    // Negative control: same session, same transcript tail, no state file.
    // This is precisely the pre-fix situation, and it must still answer
    // Processing — proving the assertion above is carried by the shared state
    // and not by something incidental to the session fixture.
    std::fs::remove_file(std::path::Path::new(&dir).join(format!("{}.json", session.session_id)))
        .unwrap();
    monitor::infer_status(&mut session, "user", "");
    assert_eq!(
        session.status,
        SessionStatus::Processing,
        "without the state file the heuristic still cannot know"
    );
}

#[test]
fn status_high_cpu_always_processing() {
    let mut s = make_session(50.0, 0);
    monitor::infer_status(&mut s, "", "");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_a_finished_turn_outranks_high_cpu() {
    // **This assertion is deliberately the reverse of what it used to be.**
    // `status_high_cpu_overrides_waiting_for_task` and
    // `status_high_cpu_overrides_end_turn` asserted `Processing` here, which
    // encoded defect D3 as intended behaviour: CPU was checked before any
    // transcript evidence, so a session whose turn had demonstrably ended
    // rendered as busy. Claude Code's node process burns CPU on renders and
    // watchers while sitting at an empty prompt, and the number being compared
    // was `ps %cpu` — a lifetime average on Linux — so this branch latched.
    //
    // A turn that ended is over. CPU cannot un-end it; it only speaks for
    // sessions with no transcript evidence at all.
    for cpu in [10.0, 20.0, 95.0] {
        let mut s = make_session(cpu, 0);
        monitor::infer_status(&mut s, "assistant", "end_turn");
        assert_eq!(
            s.status,
            SessionStatus::WaitingInput,
            "end_turn with cpu={cpu} is a finished turn, not a busy one"
        );
    }
}

#[test]
fn status_waiting_for_task_no_longer_promotes_needs_input() {
    // The legacy `is_waiting_for_task` JSONL signal is no longer trusted as
    // a NeedsInput indicator — too many false positives. NeedsInput is
    // exclusively driven by the deterministic Notification hook now.
    // Heuristic still falls back to a sensible non-attention-grabbing state.
    let mut s = make_session(0.5, 10);
    monitor::infer_status(&mut s, "", "");
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn status_end_turn_recent_waiting_input() {
    // Assistant said end_turn, 2 minutes ago, low CPU
    let mut s = make_session(0.5, 120);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn status_end_turn_old_idle_in_heuristic_path() {
    // Heuristic-only path (no hook state). After 15 quiet minutes a stop_reason
    // of end_turn/stop_sequence is genuinely abandoned — show Idle so the user
    // can sort/filter past it. The deterministic Stop hook handles still-active
    // post-turn sessions before we reach this branch.
    let mut s = make_session(0.5, 15 * 60);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn status_end_turn_recent_waiting_input_still_works() {
    let mut s = make_session(0.5, 10 * 60);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn status_tool_use_low_cpu_no_longer_promotes_needs_input() {
    // assistant + tool_use + idle CPU used to be guessed as a permission
    // prompt — that was the central source of "Needs Input" false positives
    // (parked sessions, sessions with stale tool_use tail, etc.). NeedsInput
    // is now exclusively the Notification hook's call.
    let mut s = make_session(0.5, 30);
    monitor::infer_status(&mut s, "assistant", "tool_use");
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn status_tool_use_low_cpu_recent_processing() {
    // tool_use + low CPU + <5s ago = still processing (tool just fired)
    let mut s = make_session(0.5, 2);
    monitor::infer_status(&mut s, "assistant", "tool_use");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_tool_use_high_cpu_processing() {
    // tool_use + high CPU = still crunching
    let mut s = make_session(15.0, 30);
    monitor::infer_status(&mut s, "assistant", "tool_use");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_active_cpu_processing() {
    // CPU > 2.0 → Claude is actually thinking, regardless of age.
    let mut s = make_session(3.0, 30);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_recent_low_cpu_processing() {
    // Fresh user message + low CPU = still warming up; stay Processing.
    let mut s = make_session(0.5, 1);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_quiet_low_cpu_stays_processing() {
    // Heuristic fallback can't tell apart "permission prompt for an unflushed
    // tool_use" from "session was parked mid-conversation" — both look the
    // same. Stay Processing while still recent so we don't bury an actually-
    // active session, but age out to Idle eventually (covered separately).
    let mut s = make_session(0.5, 30);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_user_message_long_quiet_idle_in_heuristic_path() {
    // After 15 quiet minutes with no hook state, treat user-tail JSONL as
    // genuinely abandoned. The deterministic Notification hook would have
    // already flipped it to NeedsInput before reaching this branch if the
    // session was actually waiting on a permission prompt.
    let mut s = make_session(0.5, 15 * 60);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn status_no_signals_idle() {
    // No JSONL signals at all → Idle
    let mut s = make_session(0.0, 0);
    monitor::infer_status(&mut s, "", "");
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
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
    monitor::infer_status(&mut s, "", "");
    assert_eq!(s.status, SessionStatus::Unknown);
}

// ────────────────────────────────────────────────────────────────────────────
// Deterministic hook-state path
// ────────────────────────────────────────────────────────────────────────────

/// Build a session with a unique `session_id` so each test owns its own
/// state file and can't be polluted by sibling tests.
fn session_with_id(id: &str, cpu: f32) -> AgentSession {
    isolate_hook_state_dir();
    let raw = RawSession {
        pid: 1,
        session_id: id.into(),
        cwd: "/tmp/test-project".into(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
    s.cpu_rate_percent = Some(cpu);
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
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();

    // Backdate the notification past the 750ms grace period so the
    // suppression doesn't hide it during this test.
    let mut state = agentctl::hook_state::HookState::load(sid).unwrap();
    state.last_notification_ts_ms = state.last_notification_ts_ms.saturating_sub(2_000);
    let path = agentctl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    // Low CPU + permission_prompt marker (now older than grace) + JSONL has
    // NOT grown past the notification → NeedsInput (deterministic path).
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_notification_ts_ms.saturating_sub(1000);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_pretooluse_clears_permission_prompt() {
    isolate_hook_state_dir();
    let sid = "hook-test-approval";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    // Approval flipped the marker; we should now report Processing (a tool
    // is actively running, no PostToolUse yet).
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn hook_precompact_marks_compacting() {
    isolate_hook_state_dir();
    let sid = "hook-test-compacting";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreCompact",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Compacting);
}

#[test]
fn hook_stop_marks_waiting_input() {
    isolate_hook_state_dir();
    let sid = "hook-test-stop";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::WaitingInput);
}

#[test]
fn hook_userpromptsubmit_after_stop_marks_responding() {
    isolate_hook_state_dir();
    let sid = "hook-test-followup";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();
    // User typed a follow-up — is_responding fires immediately, deterministic.
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn hook_waiting_input_ages_out_to_idle() {
    isolate_hook_state_dir();
    let sid = "hook-test-waiting-ages-out";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();

    // Backdate the Stop ts to >10 min ago so the age-out fires.
    let mut state = agentctl::hook_state::HookState::load(sid).unwrap();
    state.last_stop_ts_ms = state.last_stop_ts_ms.saturating_sub(11 * 60 * 1000);
    let path = agentctl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    let mut s = session_with_id(sid, 0.5);
    // Backdate the transcript with it. `session_with_id` stamps the newest
    // message at *now*, and a session cannot have both a message this second
    // and a Stop eleven minutes old unless its hook channel has died — in which
    // case the honest answer is what the transcript says (a turn that ended a
    // moment ago, so `WaitingInput`), not the age-out being tested here. Quiet
    // on both channels is the scenario this test is named for.
    s.last_message_ts = s.last_message_ts.saturating_sub(11 * 60 * 1000);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::Idle);
}

#[test]
fn hook_waiting_input_recent_stays_waiting() {
    isolate_hook_state_dir();
    let sid = "hook-test-waiting-recent";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn");
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
        agentctl::hook_state::record_hook_event(&serde_json::json!({
            "hook_event_name": ev,
            "session_id": sid,
            "tool_name": "Bash",
        }))
        .unwrap();
    }

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use");
    assert_eq!(s.status, SessionStatus::Processing);

    // Stop fires → flips to WaitingInput, also stable.
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "end_turn");
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
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use");
    assert_ne!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_worker_permission_prompt_marks_needs_input() {
    isolate_hook_state_dir();
    // Subagents fire `notification_type = "worker_permission_prompt"` instead
    // of `"permission_prompt"` (verified against Claude Code 2.1.117 binary).
    // Both must classify the session as NeedsInput.
    let sid = "hook-test-worker-permission";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "worker_permission_prompt",
    }))
    .unwrap();

    // Backdate past the 750ms grace period.
    let mut state = agentctl::hook_state::HookState::load(sid).unwrap();
    state.last_notification_ts_ms = state.last_notification_ts_ms.saturating_sub(2_000);
    let path = agentctl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_notification_ts_ms.saturating_sub(1000);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_worker_pretooluse_clears_permission_prompt() {
    isolate_hook_state_dir();
    // Approval of a subagent's prompt fires PreToolUse with the approved
    // tool — same semantic as the main-agent case.
    let sid = "hook-test-worker-approval";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "worker_permission_prompt",
    }))
    .unwrap();
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "assistant", "tool_use");
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
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Notification",
        "session_id": sid,
        "notification_type": "permission_prompt",
    }))
    .unwrap();
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreCompact",
        "session_id": sid,
    }))
    .unwrap();

    // Backdate the notification past the 750ms grace period.
    let mut state = agentctl::hook_state::HookState::load(sid).unwrap();
    state.last_notification_ts_ms = state.last_notification_ts_ms.saturating_sub(2_000);
    let path = agentctl::hook_state::state_dir().join(format!("{sid}.json"));
    std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();

    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::NeedsInput);
}

#[test]
fn hook_postcompact_clears_compacting_without_stop() {
    isolate_hook_state_dir();
    // Auto-compact paths where Stop never fires: PostCompact is the direct
    // "compaction done" signal and must clear the Compacting status on its
    // own.
    let sid = "hook-test-postcompact";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreCompact",
        "session_id": sid,
    }))
    .unwrap();

    // Mid-compact: Compacting is the correct status.
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "");
    assert_eq!(s.status, SessionStatus::Compacting);

    // PostCompact arrives. Stop never fires.
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PostCompact",
        "session_id": sid,
    }))
    .unwrap();
    let mut s = session_with_id(sid, 0.5);
    monitor::infer_status(&mut s, "user", "");
    assert_ne!(s.status, SessionStatus::Compacting);
}

/// Replay a turn that ran and finished, but whose `Stop` hook never reached
/// claudectl — the 2026-07-28 incident. `mid_turn_age_secs` backdates every
/// recorded event so the transcript tail can be placed after them.
fn state_with_lost_stop(sid: &str, mid_turn_age_secs: u64) -> agentctl::hook_state::HookState {
    isolate_hook_state_dir();
    for (event, tool) in [
        ("UserPromptSubmit", None),
        ("PreToolUse", Some("Bash")),
        ("PostToolUse", Some("Bash")),
    ] {
        agentctl::hook_state::record_hook_event(&serde_json::json!({
            "hook_event_name": event,
            "session_id": sid,
            "tool_name": tool,
        }))
        .unwrap();
    }
    let mut state = agentctl::hook_state::HookState::load(sid).unwrap();
    let back = mid_turn_age_secs * 1000;
    state.last_promptsubmit_ts_ms = state.last_promptsubmit_ts_ms.saturating_sub(back);
    state.last_pretooluse_ts_ms = state.last_pretooluse_ts_ms.saturating_sub(back);
    state.last_posttooluse_ts_ms = state.last_posttooluse_ts_ms.saturating_sub(back);
    assert_eq!(state.last_stop_ts_ms, 0, "the lost Stop is the whole point");
    let path = agentctl::hook_state::state_dir().join(format!("{sid}.json"));
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
    monitor::infer_status(&mut s, "assistant", "end_turn");
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
    monitor::infer_status(&mut s, "assistant", "end_turn");
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
    monitor::infer_status(&mut s, "assistant", "tool_use");
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
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "Stop",
        "session_id": sid,
    }))
    .unwrap();
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": sid,
    }))
    .unwrap();

    let state = agentctl::hook_state::HookState::load(sid).unwrap();
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_promptsubmit_ts_ms.saturating_sub(1_000);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn regression_transcript_veto_never_masks_an_in_flight_tool() {
    isolate_hook_state_dir();
    // Same guard, tool edition: PreToolUse fired after the transcript's last
    // message (Claude Code writes the tool_use entry, then the tool runs).
    // The tail is older than the hook event, so no veto — still Processing.
    let sid = "regression-in-flight-tool-not-masked";
    agentctl::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "PreToolUse",
        "session_id": sid,
        "tool_name": "Bash",
    }))
    .unwrap();

    let state = agentctl::hook_state::HookState::load(sid).unwrap();
    let mut s = session_with_id(sid, 0.5);
    s.last_message_ts = state.last_pretooluse_ts_ms.saturating_sub(1_000);
    monitor::infer_status(&mut s, "assistant", "end_turn");
    assert_eq!(s.status, SessionStatus::Processing);
}

#[test]
fn status_cpu_threshold_boundary() {
    // CPU exactly 5.0 — should NOT trigger Processing (threshold is >5.0)
    let mut s = make_session(5.0, 0);
    monitor::infer_status(&mut s, "", "");
    assert_eq!(s.status, SessionStatus::Idle);

    // CPU 5.1 — should trigger Processing
    let mut s2 = make_session(5.1, 0);
    monitor::infer_status(&mut s2, "", "");
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

    monitor::infer_status(&mut s, "assistant", "tool_use");
    let first_tick = s.status;
    assert_ne!(first_tick, SessionStatus::Idle);

    s.last_msg_type = "assistant".into();
    s.last_stop_reason = "tool_use".into();
    s.is_waiting_for_task = false;

    let msg_type = s.last_msg_type.clone();
    let stop_reason = s.last_stop_reason.clone();
    monitor::infer_status(&mut s, &msg_type, &stop_reason);
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
    s.cpu_rate_percent = Some(0.5);
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

fn make_session_with_jsonl(content: &str) -> (AgentSession, tempfile::NamedTempFile) {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();

    let raw = RawSession {
        pid: 1,
        session_id: "test".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
    s.jsonl_path = Some(file.path().to_path_buf());
    (s, file)
}

fn make_session_with_paths(
    cwd: String,
    session_id: String,
    jsonl_path: std::path::PathBuf,
) -> AgentSession {
    let raw = RawSession {
        pid: 1,
        session_id,
        cwd,
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
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
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
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
fn regression_rename_does_not_revert_to_stale_scan_name() {
    // 2026-07-28: a /rename displayed for one tick and then reverted to the
    // stale pre-rename name. The monitor recovered the transcript's
    // custom-title once (incremental parse = one-shot), but every later
    // tick's discovery re-supplied the stale registry-recorded name and the
    // cross-tick merge let any non-empty scan name overwrite the title.
    let jsonl = r#"{"type":"custom-title","customTitle":"invoice-restriction-ended-wording"}"#;
    let (mut s, _file) = make_session_with_jsonl(jsonl);
    s.session_name = "ndr-5e".into(); // what the scan supplied at assembly
    monitor::update_tokens(&mut s);
    assert_eq!(s.session_name, "invoice-restriction-ended-wording");
    assert!(s.name_is_explicit);

    // Tick 2: discovery re-supplies the stale name; no new transcript bytes,
    // so the monitor cannot re-recover — the merge must hold the title.
    let fresh = AgentSession::from_raw(RawSession {
        pid: 1,
        session_id: "test".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: Some("ndr-5e".into()),
        name_source: None,
    });
    let (mut merged, _) = agentctl::app::merge_discovered_sessions(vec![s], vec![fresh]);
    assert_eq!(merged.len(), 1);
    monitor::update_tokens(&mut merged[0]);
    assert_eq!(
        merged[0].session_name, "invoice-restriction-ended-wording",
        "a stale scan name must not revert an explicit /rename title"
    );
}

#[test]
fn second_rename_overwrites_the_first() {
    // Explicit beats explicit: the transcript is append-only, so a later
    // custom-title record is the fresher user choice and must win.
    let jsonl = concat!(
        r#"{"type":"custom-title","customTitle":"first-title"}"#,
        "\n",
        r#"{"type":"custom-title","customTitle":"second-title"}"#,
    );
    let (mut s, _file) = make_session_with_jsonl(jsonl);
    monitor::update_tokens(&mut s);
    assert_eq!(s.session_name, "second-title");
    assert!(s.name_is_explicit);
}

#[test]
fn rotation_reestablishes_a_carried_over_title_from_the_new_transcript() {
    // A sessionId rotation releases the old explicit title (see the app.rs
    // merge test); Claude Code writes the custom-title record near the head
    // of a rotated transcript, and the rotated row re-parses from offset 0,
    // so a carried-over explicit title re-establishes in the same pass.
    let mut existing = AgentSession::from_raw(RawSession {
        pid: 9,
        session_id: "session-a".into(),
        cwd: "/tmp/test".into(),
        started_at: 0,
        name: Some("old-title".into()),
        name_source: None,
    });
    existing.name_is_explicit = true;

    let (mut fresh, file) =
        make_session_with_jsonl(r#"{"type":"custom-title","customTitle":"carried-title"}"#);
    fresh.pid = 9;
    fresh.session_id = "session-b".into();
    fresh.session_name = "registry-name".into();

    let (mut merged, _) = agentctl::app::merge_discovered_sessions(vec![existing], vec![fresh]);
    assert_eq!(merged[0].session_name, "registry-name");
    assert!(
        !merged[0].name_is_explicit,
        "rotation must release the hold"
    );

    // do_refresh_io re-resolves the rotated row's transcript; simulate it.
    merged[0].jsonl_path = Some(file.path().to_path_buf());
    monitor::update_tokens(&mut merged[0]);
    assert_eq!(merged[0].session_name, "carried-title");
    assert!(merged[0].name_is_explicit);
}

#[test]
fn agent_name_never_downgrades_an_explicit_title() {
    // An auto-derived agent-name record arriving after a /rename must not
    // replace the explicit title (it only ever fills a blank).
    let jsonl = concat!(
        r#"{"type":"custom-title","customTitle":"my-title"}"#,
        "\n",
        r#"{"type":"agent-name","agentName":"auto-junk"}"#,
    );
    let (mut s, _file) = make_session_with_jsonl(jsonl);
    monitor::update_tokens(&mut s);
    assert_eq!(s.session_name, "my-title");
    assert!(s.name_is_explicit);
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
    s.cpu_rate_percent = Some(0.5);
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
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
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
        name_source: None,
    };
    let mut s = AgentSession::from_raw(raw);
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
        agentctl::session::SubagentRollup {
            input_tokens: 20_000,
            output_tokens: 2_000,
            cost_usd: 0.4,
            usage_metrics_available: true,
            ..agentctl::session::SubagentRollup::default()
        },
    );
    s.subagent_rollups.insert(
        std::path::PathBuf::from("/tmp/claude-1/-tmp-project/session-1/tasks/agent-2.jsonl"),
        agentctl::session::SubagentRollup {
            input_tokens: 10_000,
            output_tokens: 1_000,
            cost_usd: 0.2,
            usage_metrics_available: true,
            ..agentctl::session::SubagentRollup::default()
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
    use agentctl::session_recorder::SessionRecorder;

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
    use agentctl::session_recorder::SessionRecorder;

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
    use agentctl::recorder::Recorder;

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
) -> (AgentSession, tempfile::TempDir) {
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
        name_source: None,
    };
    let mut session = AgentSession::from_raw(raw);
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
        name_source: None,
    };
    let mut session = AgentSession::from_raw(raw);
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

/// A session collected from another sandbox must end up with exactly the same
/// transcript-derived values as one discovered locally.
///
/// This is the invariant three bugs violated in different ways: Last, Context
/// and Activity rendered blank on sandbox rows, and status went stale, because
/// those values were taken from whatever the *collecting sandbox* reported
/// instead of computed here. The transcript lives on a host-shared mount, so
/// origin must make no difference at all to anything derived from it.
#[test]
fn a_foreign_session_derives_the_same_transcript_values_as_a_local_one() {
    isolate_hook_state_dir();
    let jsonl = r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hello"}]},"timestamp":"2026-08-03T12:00:00.000Z"}
{"type":"assistant","message":{"role":"assistant","model":"claude-opus-4-6","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":1200,"output_tokens":340,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#;

    let (mut local, _f1) = make_session_with_jsonl(jsonl);
    let (mut foreign, _f2) = make_session_with_jsonl(jsonl);
    foreign.origin = agentctl::session::SessionOrigin::Sandbox("linera-agent-4759da86c2f4".into());

    monitor::update_tokens(&mut local);
    monitor::update_tokens(&mut foreign);

    // The fields whose blankness was reported, asserted on the RENDERED form
    // where there is one — that is what was visibly broken.
    assert_eq!(foreign.last_user_message_ts, local.last_user_message_ts);
    assert_ne!(
        foreign.last_user_message_ts, 0,
        "Last renders '—' when this is 0, which is the reported bug"
    );
    assert_eq!(foreign.context_tokens, local.context_tokens);
    assert_eq!(foreign.context_max, local.context_max);
    assert_eq!(foreign.format_context_bar(6), local.format_context_bar(6));
    assert_eq!(foreign.format_tokens(), local.format_tokens());
    assert_eq!(foreign.format_cost(), local.format_cost());
    assert_eq!(foreign.status, local.status);
    assert_eq!(foreign.telemetry_status, local.telemetry_status);
    assert_eq!(
        foreign.usage_metrics_available,
        local.usage_metrics_available
    );

    // And the origin itself must survive the pass — it is what gates whether
    // we may signal the pid.
    assert!(!foreign.origin.is_addressable());
    assert!(local.origin.is_addressable());
}

// ────────────────────────────────────────────────────────────────────────────
// Status degradation
//
// `decide_status` is a pure function of its inputs, which is what makes this
// section possible: every scenario below is replayed at an arbitrary age
// without sleeping, setting an env var, or touching a real session. The three
// defects these tests pin were all *reachable only by waiting in real time*
// under the previous shape, which is why they shipped.
//
// The rule under test, in one sentence: **when the evidence for a status
// expires or goes missing, the fall-through must land on a weaker claim, never
// a stronger one.** `Processing` is the strongest and most misleading default —
// it tells the user "working, leave it alone" about a session that may be
// blocked on them.
// ────────────────────────────────────────────────────────────────────────────

use agentctl::hook_state::HookState;
use agentctl::monitor::{StatusInputs, decide_status};

/// Fixed epoch for every fixture. Any value works — the point is that the
/// tests choose it rather than reading a clock.
const T0: u64 = 1_785_814_692_000;
const MIN: u64 = 60_000;
const HOUR: u64 = 60 * MIN;

/// How strong a claim each status makes about the session. Aging a fixed set
/// of evidence must never move *up* this scale — that is the invariant the
/// three defects violated, stated once so a property test can check it.
/// Note that `NeedsInput` and `Processing` share a rank: both assert a
/// specific live state, and neither is a weaker claim than the other. The move
/// *between* them is therefore invisible to this scale — which is why
/// `property_time_alone_never_silences_a_session_that_needs_you` exists as a
/// separate property. That transition, `NeedsInput` -> `Processing`, is the
/// exact bug André reported, and a strength ordering alone does not catch it
/// (verified by reintroducing the old 30-minute bound: this property still
/// passed, that one fails).
fn claim_strength(status: SessionStatus) -> u8 {
    match status {
        // "This session is definitely in this specific state right now."
        SessionStatus::NeedsInput | SessionStatus::Processing | SessionStatus::Compacting => 4,
        // "It finished and is waiting for you."
        SessionStatus::WaitingInput => 3,
        // "It is alive and we cannot tell." An admission, not a claim.
        SessionStatus::Unknown => 2,
        SessionStatus::Idle => 1,
        SessionStatus::Finished => 0,
    }
}

struct Scenario {
    name: &'static str,
    state: Option<HookState>,
    last_msg_type: &'static str,
    last_stop_reason: &'static str,
    last_message_ts: u64,
    /// Outstanding tool calls by name.
    pending_tools: &'static [String],
    /// Whether the process was observed parenting anything, and when.
    has_child_process: Option<bool>,
    child_observed_at_ms: u64,
}

fn hook_state() -> HookState {
    HookState {
        session_id: "test-session".into(),
        ..Default::default()
    }
}

fn status_at(scenario: &Scenario, now_ms: u64, cpu_rate_percent: Option<f32>) -> SessionStatus {
    decide_status(&StatusInputs {
        hook_state: scenario.state.as_ref(),
        now_ms,
        last_msg_type: scenario.last_msg_type,
        last_stop_reason: scenario.last_stop_reason,
        last_message_ts: scenario.last_message_ts,
        cpu_rate_percent,
        telemetry_available: true,
        pending_tools: scenario.pending_tools,
        has_child_process: scenario.has_child_process,
        child_observed_at_ms: scenario.child_observed_at_ms,
    })
    .status
}

/// Every distinct shape of evidence a live session can present, at T0.
///
/// Deliberately includes the shapes that only arise when a hook is *lost* —
/// they are not exotic. Hook events are separate processes spawned with a 5 s
/// timeout, and one was caught being dropped live on 2026-08-06 (a transcript
/// carried a `stop_hook_summary` record 27 minutes newer than the `Stop` the
/// state file had recorded).
fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "permission prompt open, nothing has resolved it",
            state: Some(HookState {
                notification_kind: Some("permission_prompt".into()),
                last_notification_ts_ms: T0,
                last_promptsubmit_ts_ms: T0 - MIN,
                last_pretooluse_ts_ms: T0 - MIN,
                ..hook_state()
            }),
            last_msg_type: "user",
            last_stop_reason: "",
            last_message_ts: T0 - MIN,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "tool in flight",
            state: Some(HookState {
                current_tool_name: Some("Bash".into()),
                last_promptsubmit_ts_ms: T0 - MIN,
                last_pretooluse_ts_ms: T0,
                ..hook_state()
            }),
            last_msg_type: "assistant",
            last_stop_reason: "tool_use",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "turn under way, no tool in flight",
            state: Some(HookState {
                last_promptsubmit_ts_ms: T0 - MIN,
                last_pretooluse_ts_ms: T0 - MIN,
                last_posttooluse_ts_ms: T0,
                ..hook_state()
            }),
            last_msg_type: "user",
            last_stop_reason: "",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "stop fired, waiting for the user",
            state: Some(HookState {
                last_promptsubmit_ts_ms: T0 - 2 * MIN,
                last_pretooluse_ts_ms: T0 - 2 * MIN,
                last_posttooluse_ts_ms: T0 - MIN,
                last_stop_ts_ms: T0,
                ..hook_state()
            }),
            last_msg_type: "assistant",
            last_stop_reason: "end_turn",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "compacting",
            state: Some(HookState {
                last_promptsubmit_ts_ms: T0 - MIN,
                last_precompact_ts_ms: T0,
                ..hook_state()
            }),
            last_msg_type: "assistant",
            last_stop_reason: "tool_use",
            last_message_ts: T0 - MIN,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "stop was dropped; user interrupted, so the tail is a user message",
            state: Some(HookState {
                last_promptsubmit_ts_ms: T0 - 2 * MIN,
                last_pretooluse_ts_ms: T0,
                last_posttooluse_ts_ms: T0 - MIN,
                ..hook_state()
            }),
            last_msg_type: "user",
            last_stop_reason: "",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "stop was dropped; transcript ended the turn cleanly",
            state: Some(HookState {
                last_promptsubmit_ts_ms: T0 - 2 * MIN,
                last_pretooluse_ts_ms: T0 - 2 * MIN,
                last_posttooluse_ts_ms: T0 - MIN,
                ..hook_state()
            }),
            last_msg_type: "assistant",
            last_stop_reason: "end_turn",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "no hooks have ever fired, transcript mid-turn",
            state: None,
            last_msg_type: "assistant",
            last_stop_reason: "tool_use",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "no hooks have ever fired, transcript ended the turn",
            state: None,
            last_msg_type: "assistant",
            last_stop_reason: "end_turn",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        // Appended, not inserted: the tests above index this list positionally.
        Scenario {
            name: "hook channel died after Stop; the session started a new turn anyway",
            state: Some(HookState {
                last_promptsubmit_ts_ms: T0 - 9 * HOUR,
                last_pretooluse_ts_ms: T0 - 9 * HOUR,
                last_posttooluse_ts_ms: T0 - 9 * HOUR,
                last_stop_ts_ms: T0 - 9 * HOUR,
                last_session_start_ts_ms: T0 - 14 * MIN,
                notification_kind: Some("idle_prompt".into()),
                last_notification_ts_ms: T0 - 9 * HOUR,
                ..hook_state()
            }),
            last_msg_type: "assistant",
            last_stop_reason: "tool_use",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
        Scenario {
            name: "hook channel died mid-turn with a tool marker left in flight",
            state: Some(HookState {
                current_tool_name: Some("Bash".into()),
                last_promptsubmit_ts_ms: T0 - 13 * HOUR,
                last_pretooluse_ts_ms: T0 - 13 * HOUR,
                last_session_start_ts_ms: T0 - 14 * MIN,
                ..hook_state()
            }),
            last_msg_type: "assistant",
            last_stop_reason: "tool_use",
            last_message_ts: T0,
            pending_tools: &[],
            has_child_process: None,
            child_observed_at_ms: 0,
        },
    ]
}

#[test]
fn regression_a_permission_prompt_stays_needs_input_however_long_it_is_open() {
    // Captured live on 2026-08-06 at 30 s resolution. Session 95b83ac9 sat on a
    // permission prompt from 17:42 with `notification_kind` still
    // `permission_prompt` and not one resolution event ever recorded:
    //
    //   17:54:04  prompt_age=721s   rendered=Needs Input
    //   18:12:22  prompt_age=1819s  rendered=Processing   <- crossed 1800 s
    //   18:24:22  prompt_age=2539s  rendered=Idle
    //   18:34:18  prompt_age=3135s  rendered=Waiting
    //
    // "Needs Input changes to Processing by itself", exactly at the old
    // 30-minute bound, and then flapping. The prompt was still open the whole
    // time; nothing about the session had changed but the clock.
    let scenario = &scenarios()[0];
    for age in [MIN, 29 * MIN, 31 * MIN, 12 * HOUR, 48 * HOUR] {
        assert_eq!(
            status_at(scenario, T0 + age, Some(0.1)),
            SessionStatus::NeedsInput,
            "a prompt open for {} minutes with nothing to resolve it is still open",
            age / MIN
        );
    }
}

#[test]
fn regression_a_finished_turn_is_not_processing_however_busy_the_process_looks() {
    // Session ed1014c3 (pid 243), 2026-08-06: its transcript's last assistant
    // message was `end_turn` at 17:48:21, and claudectl rendered `Processing`.
    // `ps %cpu` said 5.7 — just over the 5.0 threshold — while the process was
    // using 0.20% measured over 5 s. `%cpu` is a lifetime average on Linux
    // (cputime 141.12 s / elapsed 2470 s = 5.71%), so the number never falls
    // back below the threshold for as long as the session lives.
    //
    // Two things had to be true for that render, and both are asserted here:
    // the CPU branch must sit *below* the transcript evidence, and the CPU
    // number itself must be a rate.
    let ended = &scenarios()[6];
    for cpu in [None, Some(0.0), Some(5.7), Some(90.0)] {
        assert_eq!(
            status_at(ended, T0 + MIN, cpu),
            SessionStatus::WaitingInput,
            "the turn ended; cpu_rate={cpu:?} cannot make it Processing"
        );
    }
}

#[test]
fn regression_a_dropped_stop_does_not_latch_processing_forever() {
    // `is_responding` stays true until a `Stop` arrives, and its transcript
    // veto only fires for an `assistant` + `end_turn` tail. Interrupt a turn
    // with ESC and the tail is a *user* message, so the veto cannot fire and
    // the session was pinned to `Processing` for the rest of its life — state
    // files were found on 2026-08-06 with turn markers 21 to 28 hours newer
    // than their last `Stop`.
    let interrupted = &scenarios()[5];
    assert_eq!(
        status_at(interrupted, T0 + MIN, None),
        SessionStatus::Processing,
        "a minute in, a turn with no Stop yet is genuinely still plausible"
    );
    for age in [16 * MIN, 2 * HOUR, 28 * HOUR] {
        let status = status_at(interrupted, T0 + age, None);
        assert_ne!(
            status,
            SessionStatus::Processing,
            "after {} minutes of silence on both channels the turn is not running",
            age / MIN
        );
        assert_eq!(
            status,
            SessionStatus::Unknown,
            "and we do not know what it is"
        );
    }
}

#[test]
fn a_tool_may_run_for_hours_without_being_declared_dead() {
    // The other half of the rule above: silence with a tool in flight is a
    // build, a test suite, or a subagent — not a dropped hook. Bounding this
    // one would trade a false `Processing` for a false `Unknown` on every
    // long-running command.
    //
    // Doubles as the guard on `UNRESOLVED_TOOL_CALL_NEEDS_YOU_MS`: that bound
    // is reachable only once the channel is dead, so a live tool must stay
    // `Processing` well past it. A three-hour build is not a permission prompt.
    let tool = &scenarios()[1];
    for age in [MIN, 16 * MIN, 21 * MIN, 3 * HOUR, 72 * HOUR] {
        assert_eq!(
            status_at(tool, T0 + age, None),
            SessionStatus::Processing,
            "a tool in flight for {} minutes is still a tool in flight",
            age / MIN
        );
    }
}

#[test]
fn regression_a_dead_hook_channel_does_not_latch_idle_over_a_live_transcript() {
    // Session cf54da79 (pid 16539, `claude --resume`), captured live on
    // 2026-08-17 at 12:26-12:29Z. Its transcript was emitting
    // `assistant`/`tool_use` records every few seconds; its hook state file had
    // not moved since `Stop` at 03:26Z, nine hours earlier, because the restore
    // storm that morning left ~40 sessions with `SessionStart` delivered and
    // every hook after it dropped. `is_waiting_for_user` believed that `Stop`
    // and returned before the transcript was ever consulted, so claudectl
    // reported `Idle` for a session that was visibly working. `--doctor` called
    // the same channel dead in the same snapshot — 39 of 84 — using a predicate
    // `decide_status` never asked.
    let resurrected = &scenarios()[9];
    assert_eq!(
        status_at(resurrected, T0 + 4_000, None),
        SessionStatus::Processing,
        "the transcript is four seconds old and mid-turn; the nine-hour-old Stop is not evidence"
    );
    // Dropping the stale state must not install a *new* latch in its place —
    // but nor may the claim expire into `Idle` while the tail is an
    // outstanding tool call. See
    // `regression_an_outstanding_tool_call_is_not_a_quiet_session`.
    assert_eq!(
        status_at(resurrected, T0 + 11 * MIN, None),
        SessionStatus::Processing,
        "the tool call is still outstanding, so the turn is still open"
    );
    assert_eq!(
        status_at(resurrected, T0 + 21 * MIN, None),
        SessionStatus::NeedsInput,
        "and once it has been outstanding this long with no channel, it needs the user"
    );
}

#[test]
fn regression_a_dead_hook_channel_does_not_latch_processing_either() {
    // The mirror image, same snapshot: session f431c406 sat at `Processing`
    // with `current_tool_name` still set and its newest turn event 13 hours old.
    // `tool_in_flight` deliberately licenses unbounded silence (see
    // `a_tool_may_run_for_hours_without_being_declared_dead`), so `turn_went_silent`
    // could never release it. That license is only defensible while the channel
    // that would report the tool finishing is still delivering.
    let latched = &scenarios()[10];
    assert_eq!(
        status_at(latched, T0 + MIN, None),
        SessionStatus::Processing,
        "the transcript is a minute old and mid-turn, so the verdict itself is right"
    );
    assert_eq!(
        status_at(latched, T0 + 11 * MIN, None),
        SessionStatus::Processing,
        "and it now rests on the transcript, which still shows the call outstanding"
    );
}

#[test]
fn regression_an_outstanding_tool_call_is_not_a_quiet_session() {
    // Session f431c406 (`declarative-high-throughput-networks`, pid 18085),
    // captured live 2026-08-17 18:12Z. Its transcript's last record was an
    // `assistant`/`tool_use` calling Bash at 18:00:09.897Z with no
    // `tool_result` after it and not one further byte written; the command was
    // a `python3 - <<'PY'` heredoc, which the sandbox's PreToolUse gate asks
    // about. It was sitting on a permission prompt.
    //
    // Its hook channel had been silent since its own `SessionStart` at
    // 12:14:47Z — 25 of 42 sessions were in that state — so
    // `is_at_permission_prompt` never ran, and the transcript fallback called
    // an outstanding tool call "quiet" and reported first `Processing`, then
    // `Idle`. `Idle` says nothing is happening. Something was: a tool call had
    // been outstanding for twelve minutes.
    //
    // The bug is that the previous fix asserted exactly this decay was correct
    // ("expires like any other"), using this very session as its fixture. A
    // `user` tail does expire like any other. A `tool_use` tail does not: it is
    // the one tail that proves the turn is still open.
    let blocked = &scenarios()[10];
    for age in [12 * MIN, 21 * MIN, 3 * HOUR, 72 * HOUR] {
        assert_ne!(
            status_at(blocked, T0 + age, None),
            SessionStatus::Idle,
            "at {} min a tool call is still outstanding; Idle says the row can be skipped",
            age / MIN
        );
    }
    assert_eq!(
        status_at(blocked, T0 + 12 * MIN, None),
        SessionStatus::Processing,
        "inside the bound we still say it is working"
    );
    for age in [21 * MIN, 3 * HOUR, 72 * HOUR] {
        assert_eq!(
            status_at(blocked, T0 + age, None),
            SessionStatus::NeedsInput,
            "past it, a session that cannot report its own progress needs the user"
        );
    }
}

#[test]
fn a_question_only_the_user_can_answer_needs_no_hook_at_all() {
    // The guarantee: for the two tools that nothing but a person can retire,
    // NeedsInput does not depend on the hook channel, the clock, or CPU. It is
    // read straight off the transcript, which Claude Code writes itself.
    //
    // Swept with NO hook state, at every age, at every CPU reading. If any
    // combination here fails to say NeedsInput, a session is sitting on a
    // question with nobody being told.
    for tool in ["ExitPlanMode", "AskUserQuestion"] {
        let pending = vec![tool.to_string()];
        let scenario = Scenario {
            name: "blocked on a question, no hooks have ever fired",
            state: None,
            last_msg_type: "assistant",
            last_stop_reason: "tool_use",
            last_message_ts: T0,
            pending_tools: Box::leak(pending.into_boxed_slice()),
            has_child_process: None,
            child_observed_at_ms: 0,
        };
        for age in [0, 1_000, 30_000, 11 * MIN, 21 * MIN, HOUR, 72 * HOUR] {
            for cpu in [None, Some(0.0), Some(50.0)] {
                assert_eq!(
                    status_at(&scenario, T0 + age, cpu),
                    SessionStatus::NeedsInput,
                    "{tool} outstanding at {} min, cpu={cpu:?}",
                    age / MIN
                );
            }
        }
    }
}

#[test]
fn a_question_outranks_hook_state_that_says_otherwise() {
    // Not merely a fallback for missing hook state — it must win over hook
    // state that is present and says something else. Both bugs this month were
    // a live hook marker being believed over the transcript: a nine-hour-old
    // `Stop` reading as Idle, and a latched `tool_in_flight` reading as
    // Processing. Neither may bury an open question.
    let pending = vec!["AskUserQuestion".to_string()];
    let stale_stop = Scenario {
        name: "hook state says the turn ended hours ago",
        state: Some(HookState {
            last_promptsubmit_ts_ms: T0 - 9 * HOUR,
            last_pretooluse_ts_ms: T0 - 9 * HOUR,
            last_stop_ts_ms: T0 - 9 * HOUR,
            notification_kind: Some("idle_prompt".into()),
            last_notification_ts_ms: T0 - 9 * HOUR,
            ..hook_state()
        }),
        last_msg_type: "assistant",
        last_stop_reason: "tool_use",
        last_message_ts: T0,
        pending_tools: Box::leak(pending.clone().into_boxed_slice()),
        has_child_process: None,
        child_observed_at_ms: 0,
    };
    assert_eq!(
        status_at(&stale_stop, T0 + 11 * MIN, None),
        SessionStatus::NeedsInput,
        "a latched Stop must not bury an open question"
    );

    let live_tool = Scenario {
        name: "hook state says a tool is in flight",
        state: Some(HookState {
            current_tool_name: Some("AskUserQuestion".into()),
            last_promptsubmit_ts_ms: T0 - MIN,
            last_pretooluse_ts_ms: T0,
            ..hook_state()
        }),
        last_msg_type: "assistant",
        last_stop_reason: "tool_use",
        last_message_ts: T0,
        pending_tools: Box::leak(pending.into_boxed_slice()),
        has_child_process: None,
        child_observed_at_ms: 0,
    };
    assert_eq!(
        status_at(&live_tool, T0 + MIN, None),
        SessionStatus::NeedsInput,
        "a healthy channel calls this tool in flight; it is in flight *on the user*"
    );
}

#[test]
fn an_answered_question_stops_needing_the_user() {
    // The release condition, and the reason this reads from `pending_tool_uses`
    // rather than the transcript tail: the entry is retired by id the moment
    // its `tool_result` lands. Without this the status would latch forever —
    // the failure mode of every latch fixed in this file.
    let answered = Scenario {
        name: "the question was answered",
        state: None,
        last_msg_type: "assistant",
        last_stop_reason: "tool_use",
        last_message_ts: T0,
        pending_tools: &[],
        has_child_process: None,
        child_observed_at_ms: 0,
    };
    assert_ne!(
        status_at(&answered, T0 + MIN, None),
        SessionStatus::NeedsInput,
        "nothing is outstanding, so nothing is waiting on the user"
    );
}

fn bash_pending(has_child: Option<bool>, observed_at_ms: u64) -> Scenario {
    Scenario {
        name: "a Bash call is outstanding",
        state: None,
        last_msg_type: "assistant",
        last_stop_reason: "tool_use",
        last_message_ts: T0,
        pending_tools: Box::leak(vec!["Bash".to_string()].into_boxed_slice()),
        has_child_process: has_child,
        child_observed_at_ms: observed_at_ms,
    }
}

#[test]
fn a_pending_command_with_no_child_to_run_it_is_a_permission_prompt() {
    // The case NeedsInput could not reach without the hook channel, and the one
    // that actually happened: f431c406 sat on a Bash permission prompt for a
    // `python3 - <<'PY'` heredoc the sandbox gate asks about. The transcript
    // cannot tell that from a running command — both are an outstanding
    // `tool_use` over a near-idle process. The child can: Claude Code spawns one
    // to run the command, and a prompt never gets that far.
    let blocked = bash_pending(Some(false), T0 + MIN);
    for age in [MIN, 11 * MIN, 21 * MIN, 3 * HOUR] {
        assert_eq!(
            status_at(&blocked, T0 + age, None),
            SessionStatus::NeedsInput,
            "no child at {} min means the command never started",
            age / MIN
        );
    }
}

#[test]
fn a_command_that_is_actually_running_is_not_a_prompt() {
    // The blast radius, and the reason this replaces #57's timer rather than
    // stacking on it: a three-hour build has a child the whole way through and
    // must never be called a prompt.
    let running = bash_pending(Some(true), T0 + MIN);
    for age in [MIN, 21 * MIN, 3 * HOUR, 72 * HOUR] {
        assert_eq!(
            status_at(&running, T0 + age, None),
            SessionStatus::Processing,
            "a command with a child is a command running"
        );
    }
}

#[test]
fn an_unmeasured_child_signal_claims_nothing() {
    // `None` is "not measured" — an older probe, or a sandbox the collector
    // could not place. Reading it as "no children" would turn every unprobed
    // session into a permission prompt, which is this month's bug wearing a new
    // hat: absence of evidence as evidence of absence.
    let unmeasured = bash_pending(None, T0 + MIN);
    assert_ne!(
        status_at(&unmeasured, T0 + 2 * MIN, None),
        SessionStatus::NeedsInput,
        "not measured must claim nothing"
    );
}

#[test]
fn an_observation_older_than_the_tool_call_claims_nothing() {
    // Both races that would report a running command as a prompt. The collector
    // runs every 300 s, so its answer routinely predates the call it would be
    // used to judge; and even a fresh observation can land in the gap between
    // the tool_use record and the child being spawned.
    let stale = bash_pending(Some(false), T0 - MIN);
    assert_ne!(
        status_at(&stale, T0 + 2 * MIN, None),
        SessionStatus::NeedsInput,
        "an observation taken before the call proves nothing about it"
    );

    let inside_margin = bash_pending(Some(false), T0 + 5_000);
    assert_ne!(
        status_at(&inside_margin, T0 + 2 * MIN, None),
        SessionStatus::NeedsInput,
        "within the spawn margin the child may simply not exist yet"
    );

    // And once the observation clears the margin, it counts.
    let clear = bash_pending(Some(false), T0 + 31_000);
    assert_eq!(
        status_at(&clear, T0 + 2 * MIN, None),
        SessionStatus::NeedsInput
    );
}

#[test]
fn an_ordinary_tool_is_not_a_question() {
    // The blast radius. Bash, Edit and Read all complete on their own, so an
    // outstanding one proves nothing about the user; those keep the old
    // treatment (Processing, then the 20-minute bound from #57). Only the two
    // tools that cannot self-retire get the immediate claim.
    let pending = vec!["Bash".to_string(), "Edit".to_string(), "Read".to_string()];
    let working = Scenario {
        name: "ordinary tools outstanding",
        state: None,
        last_msg_type: "assistant",
        last_stop_reason: "tool_use",
        last_message_ts: T0,
        pending_tools: Box::leak(pending.into_boxed_slice()),
        has_child_process: None,
        child_observed_at_ms: 0,
    };
    assert_eq!(
        status_at(&working, T0 + MIN, None),
        SessionStatus::Processing,
        "a running command is not a question"
    );
}

#[test]
fn a_dead_hook_channel_can_never_cost_a_needs_input() {
    // Dropping stale hook state would be unacceptable if it could ever discard
    // an open permission prompt — that is the one status André is notified on.
    // It cannot, and the two predicates are arithmetically exclusive rather
    // than merely untested: `is_at_permission_prompt` requires the transcript to
    // have advanced no more than PROMPT_RESOLUTION_GRACE_MS (5 s) past the
    // Notification, while a channel is only called dead once the transcript is
    // HOOK_SILENCE_GRACE_MS (10 min) past every hook event, the Notification
    // included. Swept here so a future change to either constant fails loudly.
    let prompt = &scenarios()[0];
    for age in [MIN, 31 * MIN, 12 * HOUR, 72 * HOUR] {
        assert_eq!(
            status_at(prompt, T0 + age, Some(50.0)),
            SessionStatus::NeedsInput,
            "an open prompt at age {} min must survive the channel-liveness filter",
            age / MIN
        );
    }
}

#[test]
fn property_time_alone_never_promotes_a_session_to_a_stronger_claim() {
    // The property that would have caught the permission-prompt defect the day
    // it was written. Nothing here knows *which* statuses are right — only that
    // letting the clock run, with every other input frozen, can never make
    // claudectl more confident than it already was.
    //
    // The sweep starts 1 s in on purpose: `is_at_permission_prompt` suppresses
    // the marker for its first 750 ms so auto-approved prompts (which fire a
    // Notification and an instant PreToolUse) never flash on screen. That
    // debounce is the one legitimate upward step, and it is over by then.
    for scenario in &scenarios() {
        for cpu in [None, Some(0.0), Some(50.0)] {
            let mut previous = status_at(scenario, T0 + 1_000, cpu);
            for age in [2_000, 30_000, 5 * MIN, 16 * MIN, HOUR, 12 * HOUR, 72 * HOUR] {
                let current = status_at(scenario, T0 + age, cpu);
                assert!(
                    claim_strength(current) <= claim_strength(previous),
                    "{}: at cpu={cpu:?}, aging to {} min moved {previous} -> {current}, \
                     which is a STRONGER claim than before",
                    scenario.name,
                    age / MIN
                );
                previous = current;
            }
        }
    }
}

#[test]
fn property_time_alone_never_silences_a_session_that_needs_you() {
    // `NeedsInput` is the one status the user must act on, and the only thing
    // that can end it is *evidence*: a resolution hook, or the conversation
    // moving on in the transcript. The clock is not evidence. A session that
    // quietly stops asking for you is worse than one that never asked — you
    // stop looking at the bucket you were told is empty.
    //
    // This is the property that pins the reported bug. `claim_strength` cannot:
    // `NeedsInput` and `Processing` are equally strong claims, so the move
    // between them is invisible to a strength ordering.
    for scenario in &scenarios() {
        for cpu in [None, Some(0.0), Some(50.0)] {
            if status_at(scenario, T0 + 1_000, cpu) != SessionStatus::NeedsInput {
                continue;
            }
            for age in [
                2_000,
                5 * MIN,
                29 * MIN,
                31 * MIN,
                HOUR,
                12 * HOUR,
                72 * HOUR,
            ] {
                assert_eq!(
                    status_at(scenario, T0 + age, cpu),
                    SessionStatus::NeedsInput,
                    "{}: at cpu={cpu:?}, waiting {} minutes made the session stop asking \
                     for the user, with nothing having resolved it",
                    scenario.name,
                    age / MIN
                );
            }
        }
    }
}

#[test]
fn property_processing_is_eventually_released_without_corroboration() {
    // The mirror of the property above, and the one that catches a latched
    // `Processing`: with no tool in flight, no new events on either channel and
    // no CPU to show for it, the claim must expire on its own. Without this a
    // single dropped `Stop` is permanent.
    for scenario in &scenarios() {
        let tool_in_flight = scenario
            .state
            .as_ref()
            .is_some_and(agentctl::hook_state::tool_in_flight);
        if tool_in_flight {
            continue; // Licensed to run indefinitely — asserted above.
        }
        for cpu in [None, Some(0.0)] {
            assert_ne!(
                status_at(scenario, T0 + 24 * HOUR, cpu),
                SessionStatus::Processing,
                "{}: still Processing a day later with cpu={cpu:?} and nothing to show for it",
                scenario.name
            );
        }
    }
}

#[test]
fn degradation_matrix() {
    // The readable form: what each shape of evidence renders as, fresh and
    // stale. Read the two columns as "what it says now" and "what it says once
    // the evidence has gone quiet".
    let expected: &[(&str, SessionStatus, SessionStatus)] = &[
        (
            "permission prompt open, nothing has resolved it",
            SessionStatus::NeedsInput,
            SessionStatus::NeedsInput,
        ),
        (
            "tool in flight",
            SessionStatus::Processing,
            SessionStatus::Processing,
        ),
        (
            "turn under way, no tool in flight",
            SessionStatus::Processing,
            SessionStatus::Unknown,
        ),
        (
            "stop fired, waiting for the user",
            SessionStatus::WaitingInput,
            SessionStatus::Idle,
        ),
        (
            "compacting",
            SessionStatus::Compacting,
            SessionStatus::Unknown,
        ),
        (
            "stop was dropped; user interrupted, so the tail is a user message",
            SessionStatus::Processing,
            SessionStatus::Unknown,
        ),
        (
            "stop was dropped; transcript ended the turn cleanly",
            SessionStatus::WaitingInput,
            SessionStatus::Idle,
        ),
        // The three `tool_use`-tail rows below degrade to `NeedsInput`, not
        // `Idle`. A day-old outstanding tool call on a *live* process is not a
        // quiet session: nothing but the user is going to resolve it.
        (
            "no hooks have ever fired, transcript mid-turn",
            SessionStatus::Processing,
            SessionStatus::NeedsInput,
        ),
        (
            "no hooks have ever fired, transcript ended the turn",
            SessionStatus::WaitingInput,
            SessionStatus::Idle,
        ),
        // Both dead-channel rows read exactly like their "no hooks have ever
        // fired" counterparts, which is the point: state we cannot date is
        // state we do not have.
        (
            "hook channel died after Stop; the session started a new turn anyway",
            SessionStatus::Processing,
            SessionStatus::NeedsInput,
        ),
        (
            "hook channel died mid-turn with a tool marker left in flight",
            SessionStatus::Processing,
            SessionStatus::NeedsInput,
        ),
    ];

    let scenarios = scenarios();
    assert_eq!(
        scenarios.len(),
        expected.len(),
        "every scenario needs a row here — a new one must not silently skip the matrix"
    );
    for (scenario, (name, fresh, stale)) in scenarios.iter().zip(expected) {
        assert_eq!(&scenario.name, name, "matrix rows must stay aligned");
        assert_eq!(
            status_at(scenario, T0 + MIN, Some(0.1)),
            *fresh,
            "{name}: one minute in"
        );
        assert_eq!(
            status_at(scenario, T0 + 24 * HOUR, Some(0.1)),
            *stale,
            "{name}: a day later"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Session identity survival
//
// A session that loses its cwd loses everything downstream: `transcript` is
// derived from it, `cwd_to_slug("")` is `-`, and the row then points at
// `projects/-/<id>.jsonl` — a path that does not exist. The row renders with no
// title, no project, and `Unreadable`, showing a bare pid.
//
// This is not hypothetical. Claude Code deletes pointer files mid-session, and
// discovery then falls back to the process table, which knows only the pid and
// the `--resume` uuid. On 2026-08-06 eighteen sessions rendered that way at
// once. Two independent defects had to hold for that:
//
//   1. the registry writer let a blank overwrite stored identity, and
//   2. nothing re-resolved a transcript path that was present but wrong,
//      so the damage was permanent rather than repaired on the next tick.
//
// Both are covered here. The second matters most: preventing the next clobber
// does nothing for rows already broken on disk.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn a_transcript_is_findable_by_session_id_alone() {
    // The primitive the repair path needs: every other lookup derives the
    // project directory from the cwd, which is exactly the field that is gone.
    let home = tempfile::tempdir().unwrap();
    let project = home.path().join(".claude/projects/-Users-ndr-work");
    std::fs::create_dir_all(&project).unwrap();
    let id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    std::fs::write(
        project.join(format!("{id}.jsonl")),
        r#"{"type":"user","cwd":"/Users/ndr/work","message":{"role":"user","content":"hi"}}"#,
    )
    .unwrap();

    let _guard = EnvGuard::set("HOME", home.path());
    let found = discovery::find_transcript_by_session_id(id).expect("must find it without a cwd");
    assert!(found.ends_with(format!("-Users-ndr-work/{id}.jsonl")));
    assert_eq!(
        discovery::recover_cwd_from_transcript(&found).as_deref(),
        Some("/Users/ndr/work"),
        "the transcript is the durable copy of the cwd the process table cannot supply"
    );
    assert_eq!(discovery::find_transcript_by_session_id("no-such-id"), None);
    assert_eq!(discovery::find_transcript_by_session_id(""), None);
}

#[test]
fn regression_a_blank_cwd_recovers_its_project_and_transcript() {
    // The whole repair, through the real resolver: a session with no cwd and a
    // recorded transcript path that does not exist must come back with both.
    let home = tempfile::tempdir().unwrap();
    let project = home.path().join(".claude/projects/-Users-ndr-work");
    std::fs::create_dir_all(&project).unwrap();
    let id = "11111111-2222-3333-4444-555555555555";
    std::fs::write(
        project.join(format!("{id}.jsonl")),
        r#"{"type":"user","cwd":"/Users/ndr/work","message":{"role":"user","content":"hi"}}"#,
    )
    .unwrap();

    let _guard = EnvGuard::set("HOME", home.path());
    let mut session = AgentSession::from_raw(RawSession {
        pid: 4242,
        session_id: id.into(),
        cwd: String::new(), // blanked by the process-table fallback
        started_at: 0,
        name: None,
        name_source: None,
    });
    // Present, wrong, and pointing at nothing — what the registry stores once
    // cwd is blank. Treating this as "already resolved" is what made the
    // breakage permanent.
    session.jsonl_path = Some(home.path().join(format!(".claude/projects/-/{id}.jsonl")));

    discovery::resolve_jsonl_paths(std::slice::from_mut(&mut session));

    assert_eq!(session.cwd, "/Users/ndr/work", "cwd recovered");
    assert_eq!(session.project_name, "work", "project column recovered");
    assert_eq!(
        session.jsonl_path.as_deref(),
        Some(project.join(format!("{id}.jsonl")).as_path()),
        "and it points at the transcript that actually exists"
    );
}

/// Scoped `HOME` override. Restores the previous value on drop so a panicking
/// assertion cannot leak a temp path into the rest of the suite — the failure
/// mode that let a test write to the real `~/.claude` in the first place.
struct EnvGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &std::path::Path) -> Self {
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}
