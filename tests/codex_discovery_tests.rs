//! Discovery of Codex rollout files on disk.
//!
//! Codex writes `sessions/YYYY/MM/DD/<file>.jsonl`, so discovery has to recurse
//! — but boundedly, and without following symlinks back up the tree.

use std::path::Path;

use agentctl::providers::codex_rollout::{
    discover_rollout_files, discover_sessions_modified_since, summarize_file,
};

fn write(path: &Path, body: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

fn rollout(thread: &str, cwd: &str) -> String {
    format!(
        "{{\"timestamp\":\"2026-08-24T10:00:00.000Z\",\"type\":\"session_meta\",\
         \"payload\":{{\"id\":\"{thread}\",\"session_id\":\"s-{thread}\",\"cwd\":\"{cwd}\",\
         \"timestamp\":\"2026-08-24T10:00:00.000Z\"}}}}\n"
    )
}

#[test]
fn rollouts_are_found_through_the_dated_directory_layout() {
    let root = tempfile::tempdir().unwrap();
    write(
        &root.path().join("2026/08/24/rollout-a.jsonl"),
        &rollout("thread-a", "/w/a"),
    );
    write(
        &root.path().join("2026/08/23/rollout-b.jsonl"),
        &rollout("thread-b", "/w/b"),
    );

    let found = discover_rollout_files(root.path());
    assert_eq!(found.len(), 2, "both dated days must be reached: {found:?}");

    let sessions = discover_sessions_modified_since(root.path(), 0);
    let mut ids: Vec<&str> = sessions
        .iter()
        .filter_map(|s| s.thread_id.as_deref())
        .collect();
    ids.sort();
    assert_eq!(ids, ["thread-a", "thread-b"]);
}

/// A missing root is normal — most machines have no Codex install.
#[test]
fn a_missing_sessions_root_is_empty_not_an_error() {
    let root = tempfile::tempdir().unwrap();
    let absent = root.path().join("nope");
    assert!(discover_rollout_files(&absent).is_empty());
    assert!(discover_sessions_modified_since(&absent, 0).is_empty());
}

/// Only `.jsonl` files are rollouts.
#[test]
fn non_rollout_files_are_ignored() {
    let root = tempfile::tempdir().unwrap();
    write(&root.path().join("2026/08/24/notes.txt"), "not a rollout");
    write(&root.path().join("2026/08/24/x.jsonl"), &rollout("t", "/w"));

    let found = discover_rollout_files(root.path());
    assert_eq!(found.len(), 1);
    assert!(found[0].ends_with("x.jsonl"));
}

/// The walk is depth-bounded, so a deep or looping tree cannot hang a refresh.
#[test]
fn the_walk_stops_below_a_bounded_depth() {
    let root = tempfile::tempdir().unwrap();
    // sessions/YYYY/MM/DD/file is the real layout — depth 4 including the file.
    write(
        &root.path().join("2026/08/24/ok.jsonl"),
        &rollout("t", "/w"),
    );
    // One level deeper than Codex ever writes.
    write(
        &root.path().join("2026/08/24/extra/too-deep.jsonl"),
        &rollout("deep", "/w"),
    );

    let found = discover_rollout_files(root.path());
    assert!(
        found.iter().any(|p| p.ends_with("ok.jsonl")),
        "the real layout must be reached"
    );
    assert!(
        !found.iter().any(|p| p.ends_with("too-deep.jsonl")),
        "the walk must stop below the bound: {found:?}"
    );
}

/// A rollout with no `session_meta` has no identity to key a row on.
#[test]
fn a_rollout_without_session_meta_is_not_a_session() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("2026/08/24/headless.jsonl");
    write(
        &path,
        "{\"timestamp\":\"t\",\"type\":\"turn_context\",\"payload\":{\"model\":\"m\"}}\n",
    );

    assert_eq!(
        discover_rollout_files(root.path()).len(),
        1,
        "file is found"
    );
    assert!(
        summarize_file(&path).is_none(),
        "but it yields no session, since it has no thread id"
    );
    assert!(discover_sessions_modified_since(root.path(), 0).is_empty());
}

/// `codex -C/--cd <DIR>` moves the agent's root but not the process's cwd.
///
/// Correlating on `/proc/<pid>/cwd` alone misses every session started that
/// way, and misses it silently — an absent row, not a wrong one, which is the
/// hardest kind to notice.
#[test]
fn the_working_root_override_is_read_from_the_command_line() {
    use agentctl::providers::codex::working_root_override;

    // All three spellings reach the same clap argument.
    assert_eq!(
        working_root_override("-C /work/repo"),
        Some("/work/repo".to_string())
    );
    assert_eq!(
        working_root_override("--cd /work/repo"),
        Some("/work/repo".to_string())
    );
    assert_eq!(
        working_root_override("--cd=/work/repo"),
        Some("/work/repo".to_string())
    );
    assert_eq!(
        working_root_override("-C/work/repo"),
        Some("/work/repo".to_string())
    );

    // Present among other arguments.
    assert_eq!(
        working_root_override("resume abc --cd /work/repo \"do the thing\""),
        Some("/work/repo".to_string())
    );

    // Absent, or malformed, means fall back to the process cwd.
    assert_eq!(working_root_override(""), None);
    assert_eq!(working_root_override("resume abc"), None);
    assert_eq!(working_root_override("--cd"), None, "flag with no value");
    // clap accepts the `=` form for a short flag too.
    assert_eq!(
        working_root_override("-C=/work/repo"),
        Some("/work/repo".to_string())
    );
    assert_eq!(
        working_root_override("--config model=o3"),
        None,
        "an unrelated flag is not a working root"
    );
}

/// A registry round-trip must not turn a Codex session into a Claude one.
///
/// `AgentSession::from_raw` stamps Claude, so before the entry carried a
/// provider a restored Codex row would have been resumed with
/// `claude --resume <ULID>` — the wrong binary and the wrong id format.
#[test]
fn a_registry_entry_keeps_its_provider_across_a_round_trip() {
    use agentctl::provider::AgentProvider;
    use agentctl::sandbox_registry::SessionEntry;
    use agentctl::session::AgentSession;

    let entry = SessionEntry {
        provider: AgentProvider::Codex,
        session_id: "01J0THREAD".to_string(),
        cwd: "/work/repo".to_string(),
        pid: Some(4242),
        ..Default::default()
    };

    let session = AgentSession::from_registry_entry("sbx-1", &entry).expect("session");
    assert_eq!(session.provider, AgentProvider::Codex);

    // And the default stays Claude, so every entry written before the field
    // existed still means what it meant.
    let json = r#"{"session_id":"legacy","cwd":"/w","transcript":"","started_at_ms":0}"#;
    let legacy: SessionEntry = serde_json::from_str(json).expect("legacy entry parses");
    assert_eq!(legacy.provider, AgentProvider::Claude);
}

/// The per-tick walk must not re-read history that predates every live session.
///
/// A rollout is never pruned, so without an age floor the cost of a refresh
/// grows with total Codex history forever — measured at 263 ms per tick over
/// 200 files, on a 2-second timer.
#[test]
fn rollouts_older_than_every_live_session_are_skipped() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("2026/08/24/old.jsonl");
    write(&path, &rollout("ancient", "/w/old"));

    // Everything is newer than the epoch.
    assert_eq!(discover_sessions_modified_since(root.path(), 0).len(), 1);

    // Nothing is newer than the far future, so the file is never opened.
    let far_future_ms = u64::MAX / 2;
    assert!(
        discover_sessions_modified_since(root.path(), far_future_ms).is_empty(),
        "a rollout older than every live session must be skipped"
    );
}
