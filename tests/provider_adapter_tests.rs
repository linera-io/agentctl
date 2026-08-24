//! Contract tests for the provider adapter.
//!
//! These are parity tests first and design tests second: every assertion states
//! what Claude does *today*, so the extraction cannot quietly change behaviour.
//! A second product satisfying the same contract is what Task 6 adds.

use std::path::Path;

use agentctl::provider::AgentProvider;
use agentctl::providers::for_provider;

/// The registry never hands back an adapter for a different product.
///
/// Stated as an invariant over the whole enum rather than per product, so it
/// keeps holding as products are added.
#[test]
fn the_registry_never_returns_the_wrong_products_adapter() {
    for provider in AgentProvider::all() {
        if let Some(adapter) = for_provider(*provider) {
            assert_eq!(adapter.provider(), *provider);
        }
    }
}

/// Codex has its own adapter, and does not borrow Claude's.
///
/// This replaces the not-yet-wired tripwire from the adapter extraction; the
/// property it guarded — no silent fallback to Claude — is now asserted
/// positively, and again in `codex_rollout_tests`.
#[test]
fn codex_has_its_own_adapter_rather_than_claudes() {
    let codex = for_provider(AgentProvider::Codex).expect("Codex adapter");
    assert_eq!(codex.provider(), AgentProvider::Codex);
    assert_ne!(
        codex.executable(),
        for_provider(AgentProvider::Claude)
            .expect("Claude adapter")
            .executable(),
        "a Codex session must never be launched with Claude's binary"
    );
}

#[test]
fn the_claude_adapter_reports_claudes_identity() {
    let adapter = for_provider(AgentProvider::Claude).expect("Claude adapter");
    assert_eq!(adapter.executable(), "claude");
    assert_eq!(
        adapter.transcript_root(Path::new("/home/u")),
        Path::new("/home/u/.claude/projects"),
        "Claude transcripts live under ~/.claude/projects"
    );
}

/// Process matching must stay exactly as strict as it is today.
///
/// `is_claude_process` deliberately requires argv0's basename to be exactly
/// `claude`, which is what keeps our own binaries, `grep claude` and
/// `/usr/bin/env claude` from being mistaken for sessions. Loosening it during
/// the extraction would make the dashboard claim unrelated processes.
#[test]
fn claude_process_matching_keeps_its_current_strictness() {
    let adapter = for_provider(AgentProvider::Claude).expect("Claude adapter");

    for accepted in [
        "claude",
        "claude --resume abc",
        "/usr/local/bin/claude -p hi",
    ] {
        assert!(
            adapter.matches_process(accepted),
            "should match a real session: {accepted:?}"
        );
    }
    for rejected in [
        "claudectl --list",
        "agentctl --list",
        "grep claude",
        "/usr/bin/env claude",
        "codex",
        "",
    ] {
        assert!(
            !adapter.matches_process(rejected),
            "must not claim: {rejected:?}"
        );
    }
}

/// Launch and resume argv, in the exact order the terminals already emit.
#[test]
fn claude_launch_args_match_todays_invocation() {
    let adapter = for_provider(AgentProvider::Claude).expect("Claude adapter");

    assert!(adapter.launch_args(None, None).is_empty(), "bare launch");
    assert_eq!(
        adapter.launch_args(None, Some("abc-123")),
        vec!["--resume", "abc-123"]
    );
    assert_eq!(
        adapter.launch_args(Some("ship it"), None),
        vec!["-p", "ship it"]
    );
    assert_eq!(
        adapter.launch_args(Some("ship it"), Some("abc-123")),
        vec!["--resume", "abc-123", "-p", "ship it"],
        "resume precedes prompt, as the terminals emit it today"
    );
}

#[test]
fn the_claude_adapter_supports_rename() {
    assert!(
        for_provider(AgentProvider::Claude)
            .expect("Claude adapter")
            .supports_rename()
    );
}

/// The adapter must parse a real Claude transcript line the same way the
/// existing parser does — same function, reached through the trait.
#[test]
fn claude_transcript_parsing_is_unchanged() {
    let adapter = for_provider(AgentProvider::Claude).expect("Claude adapter");
    let line = r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-08-24T10:00:00.000Z"}"#;

    // Asserted positively first: comparing two `is_some()` calls would hold
    // just as well if the sample stopped parsing and both sides went None.
    assert!(
        adapter.parse_transcript_line(line).is_some(),
        "the sample must parse, or this test proves nothing"
    );
    assert_eq!(
        adapter.parse_transcript_line(line).is_some(),
        agentctl::transcript::parse_line(line).is_some(),
        "the adapter must not diverge from the parser it wraps"
    );
    assert!(
        adapter.parse_transcript_line("not json").is_none(),
        "garbage stays unparsed"
    );
}

/// The extraction must not move where Claude sessions are scanned from.
///
/// `discovery::projects_dir` now goes through the adapter; this pins it to the
/// literal it returned before, so a wrong `transcript_root` would show up as an
/// empty dashboard rather than a passing test.
#[test]
fn discovery_still_scans_the_same_claude_directory() {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let Some(home) = home else {
        return;
    };
    assert_eq!(
        agentctl::discovery::projects_dir(),
        home.join(".claude").join("projects")
    );
}
