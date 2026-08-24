//! Codex rollout parsing and process matching.
//!
//! The fixture is a real-shaped rollout built from `openai/codex` at tag
//! `rust-v0.148.0`, the release matching the `codex-cli` this was developed
//! against. Field placement there is not obvious and two choices are easy to get
//! backwards, so both are asserted explicitly below.

use agentctl::provider::AgentProvider;
use agentctl::providers::codex_rollout::summarize;
use agentctl::providers::for_provider;

fn fixture() -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/codex/rollout-basic.jsonl"),
    )
    .expect("fixture")
}

#[test]
fn a_rollout_yields_the_fields_the_dashboard_renders() {
    let s = summarize(&fixture());

    assert_eq!(s.cwd.as_deref(), Some("/home/u/repos/example"));
    assert_eq!(s.started_at.as_deref(), Some("2026-08-24T10:00:00.000Z"));
    assert_eq!(s.total_tokens, Some(12800));
    assert_eq!(s.context_window, Some(272000));
    assert_eq!(
        s.last_activity.as_deref(),
        Some("2026-08-24T10:00:20.000Z"),
        "last activity is the newest line, not the newest session_meta"
    );
}

/// `id` is the thread; `session_id` is a different field on the same object.
///
/// Reading the wrong one groups sessions incorrectly, and both are opaque
/// ULIDs, so nothing downstream would look wrong.
#[test]
fn the_thread_id_comes_from_id_not_session_id() {
    let s = summarize(&fixture());
    assert_eq!(s.thread_id.as_deref(), Some("01J0THREAD00000000000000000"));
    assert_eq!(s.session_id.as_deref(), Some("01J0SESSION0000000000000000"));
    assert_ne!(s.thread_id, s.session_id, "these are distinct fields");
}

/// The model lives in `turn_context`, never in `session_meta`.
#[test]
fn the_model_comes_from_turn_context() {
    assert_eq!(summarize(&fixture()).model.as_deref(), Some("gpt-5-codex"));

    let meta_only = r#"{"timestamp":"t","type":"session_meta","payload":{"id":"x","cwd":"/c","model":"wrong"}}"#;
    assert_eq!(
        summarize(meta_only).model,
        None,
        "a model key on session_meta must not be believed"
    );
}

/// A live session's last line is often half-written.
#[test]
fn a_truncated_final_line_does_not_discard_the_session() {
    let mut text = fixture();
    text.push_str("{\"timestamp\":\"2026-08-24T10:00:30.000Z\",\"type\":\"event_ms");

    let s = summarize(&text);
    assert_eq!(
        s.cwd.as_deref(),
        Some("/home/u/repos/example"),
        "the complete lines must still parse"
    );
    assert_eq!(
        s.last_activity.as_deref(),
        Some("2026-08-24T10:00:20.000Z"),
        "the torn line contributes nothing"
    );
}

/// The `type` tag is top level because `RolloutLine` flattens its item.
#[test]
fn a_nested_item_shape_is_not_what_codex_writes() {
    let nested =
        r#"{"timestamp":"t","item":{"type":"session_meta","payload":{"id":"x","cwd":"/c"}}}"#;
    assert_eq!(
        summarize(nested),
        agentctl::providers::codex_rollout::RolloutSummary {
            last_activity: Some("t".to_string()),
            ..Default::default()
        },
        "reading a nested item would mean the real format yields nothing"
    );
}

#[test]
fn codex_now_has_an_adapter() {
    let adapter = for_provider(AgentProvider::Codex).expect("Codex adapter");
    assert_eq!(adapter.provider(), AgentProvider::Codex);
    assert_eq!(adapter.executable(), "codex");
    assert!(!adapter.supports_rename(), "Codex has no in-place rename");
    assert_eq!(
        adapter.launch_args(None, Some("abc")),
        vec!["resume", "abc"],
        "codex resume is a subcommand, not Claude's --resume flag"
    );
}

/// Codex process matching must be as strict as Claude's.
#[test]
fn codex_process_matching_rejects_lookalikes() {
    let adapter = for_provider(AgentProvider::Codex).expect("Codex adapter");

    for accepted in [
        "codex",
        "codex resume 01J0",
        "codex --remote https://example.invalid",
        "/usr/local/bin/codex",
    ] {
        assert!(
            adapter.matches_process(accepted),
            "should match {accepted:?}"
        );
    }
    // `codex-code-mode-host` is a real sibling binary the npm package ships.
    for rejected in [
        "codex-code-mode-host",
        "grep codex",
        "/usr/bin/env codex",
        "claude",
        "agentctl --list",
        "",
    ] {
        assert!(
            !adapter.matches_process(rejected),
            "must not claim {rejected:?}"
        );
    }
}

/// Neither adapter may claim the other's processes.
#[test]
fn the_two_products_never_claim_each_others_processes() {
    let claude = for_provider(AgentProvider::Claude).expect("Claude adapter");
    let codex = for_provider(AgentProvider::Codex).expect("Codex adapter");

    assert!(claude.matches_process("claude") && !codex.matches_process("claude"));
    assert!(codex.matches_process("codex") && !claude.matches_process("codex"));
}
