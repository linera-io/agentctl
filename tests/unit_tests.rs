use std::time::Duration;

// Test session status display and sorting
#[test]
fn test_session_status_sort_order() {
    use agentctl::session::SessionStatus;
    assert!(SessionStatus::NeedsInput.sort_key() < SessionStatus::Processing.sort_key());
    assert!(SessionStatus::Processing.sort_key() < SessionStatus::WaitingInput.sort_key());
    assert!(SessionStatus::WaitingInput.sort_key() < SessionStatus::Unknown.sort_key());
    assert!(SessionStatus::Unknown.sort_key() < SessionStatus::Idle.sort_key());
    assert!(SessionStatus::Idle.sort_key() < SessionStatus::Finished.sort_key());
}

#[test]
fn test_session_status_display() {
    use agentctl::session::SessionStatus;
    assert_eq!(SessionStatus::NeedsInput.to_string(), "Needs Input");
    assert_eq!(SessionStatus::Processing.to_string(), "Processing");
    assert_eq!(SessionStatus::WaitingInput.to_string(), "Waiting");
    assert_eq!(SessionStatus::Unknown.to_string(), "Unknown");
    assert_eq!(SessionStatus::Idle.to_string(), "Idle");
    assert_eq!(SessionStatus::Finished.to_string(), "Finished");
}

#[test]
fn test_session_from_raw() {
    use agentctl::session::{AgentSession, RawSession};
    let raw = RawSession {
        pid: 12345,
        session_id: "abc-123".to_string(),
        cwd: "/Users/test/projects/my-app".to_string(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let session = AgentSession::from_raw(raw);
    assert_eq!(session.pid, 12345);
    assert_eq!(session.project_name, "my-app");
    assert_eq!(session.display_name(), "my-app");
}

#[test]
fn test_session_display_name_prefers_session_name() {
    use agentctl::session::{AgentSession, RawSession};
    let raw = RawSession {
        pid: 1,
        session_id: "x".to_string(),
        cwd: "/tmp/foo".to_string(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut session = AgentSession::from_raw(raw);
    session.session_name = "my-custom-name".to_string();
    assert_eq!(session.display_name(), "my-custom-name");
}

#[test]
fn test_format_elapsed() {
    use agentctl::session::{AgentSession, RawSession};
    let raw = RawSession {
        pid: 1,
        session_id: "x".to_string(),
        cwd: "/tmp".to_string(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut session = AgentSession::from_raw(raw);
    session.elapsed = Duration::from_secs(3661);
    assert_eq!(session.format_elapsed(), "01:01:01");

    session.elapsed = Duration::from_secs(125);
    assert_eq!(session.format_elapsed(), "02:05");
}

#[test]
fn test_format_tokens() {
    use agentctl::session::{AgentSession, RawSession, TelemetryStatus};
    let raw = RawSession {
        pid: 1,
        session_id: "x".to_string(),
        cwd: "/tmp".to_string(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut session = AgentSession::from_raw(raw);

    assert_eq!(session.format_tokens(), "n/a");

    session.telemetry_status = TelemetryStatus::Available;
    session.usage_metrics_available = true;
    session.total_input_tokens = 1_500_000;
    session.total_output_tokens = 42_000;
    assert_eq!(session.format_tokens(), "1.5M/42.0k");
}

#[test]
fn test_format_cost() {
    use agentctl::session::{AgentSession, RawSession, TelemetryStatus};
    let raw = RawSession {
        pid: 1,
        session_id: "x".to_string(),
        cwd: "/tmp".to_string(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let mut session = AgentSession::from_raw(raw);

    assert_eq!(session.format_cost(), "n/a");

    session.telemetry_status = TelemetryStatus::Available;
    session.usage_metrics_available = true;
    session.cost_usd = 0.42;
    assert_eq!(session.format_cost(), "$0.42");

    session.cost_usd = 12.5;
    assert_eq!(session.format_cost(), "$12.5");
}

#[test]
fn test_cwd_to_project_name() {
    use agentctl::session::{AgentSession, RawSession};
    let cases = vec![
        ("/Users/foo/bar/my-project", "my-project"),
        ("/tmp", "tmp"),
        ("/a/b/c/deeply-nested", "deeply-nested"),
    ];
    for (cwd, expected) in cases {
        let raw = RawSession {
            pid: 1,
            session_id: "x".to_string(),
            cwd: cwd.to_string(),
            started_at: 0,
            name: None,
            name_source: None,
        };
        let session = AgentSession::from_raw(raw);
        assert_eq!(session.project_name, expected, "cwd={cwd}");
    }
}

#[test]
fn test_session_name_from_raw_name_field() {
    use agentctl::session::{AgentSession, RawSession};
    // When the session JSON has a `name` field (e.g. from Claude Code's
    // `/rename` slash command or topic auto-naming), it should become the
    // session's display name.
    let raw = RawSession {
        pid: 1,
        session_id: "abcd-efgh".into(),
        cwd: "/Users/jane/linera-protocol".into(),
        started_at: 0,
        name: Some("fix-validator-oom".into()),
        name_source: None,
    };
    let s = AgentSession::from_raw(raw);
    assert_eq!(s.session_name, "fix-validator-oom");
    assert_eq!(s.display_name(), "fix-validator-oom");
    // Cwd-derived project_name still computed as a fallback.
    assert_eq!(s.project_name, "linera-protocol");
}

#[test]
fn test_display_name_falls_back_to_cwd_when_name_absent() {
    use agentctl::session::{AgentSession, RawSession};
    let raw = RawSession {
        pid: 1,
        session_id: "abcd-efgh".into(),
        cwd: "/Users/jane/linera-protocol".into(),
        started_at: 0,
        name: None,
        name_source: None,
    };
    let s = AgentSession::from_raw(raw);
    assert_eq!(s.session_name, "");
    assert_eq!(s.display_name(), "linera-protocol");
}
