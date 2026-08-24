//! Discovery of Codex rollout files on disk.
//!
//! Codex writes `sessions/YYYY/MM/DD/<file>.jsonl`, so discovery has to recurse
//! — but boundedly, and without following symlinks back up the tree.

use std::path::Path;

use agentctl::providers::codex_rollout::{
    discover_rollout_files, discover_sessions, summarize_file,
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

    let sessions = discover_sessions(root.path());
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
    assert!(discover_sessions(&absent).is_empty());
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
    assert!(discover_sessions(root.path()).is_empty());
}
