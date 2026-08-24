//! Reading Codex rollout files.
//!
//! Codex writes one JSON object per line under `$CODEX_HOME/sessions/`. The
//! schema is derived from `serde` attributes in `openai/codex`, so it moves with
//! the release: everything here is read against tag `rust-v0.148.0`, matching
//! the `codex-cli` this was developed against.
//!
//! The outer `RolloutLine` flattens its item, so `type` and `payload` sit at the
//! TOP level of each line — there is no `item` key. Reading a nested one finds
//! nothing and reports an empty session rather than failing, which is why the
//! fixture is checked in beside the parser.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// What a rollout file tells us about a session.
///
/// Every field is optional because a rollout is appended to over a session's
/// life: a file captured one line in has a `session_meta` and nothing else, and
/// that is a valid state to render, not an error.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RolloutSummary {
    pub thread_id: Option<String>,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub model: Option<String>,
    pub started_at: Option<String>,
    /// Timestamp of the newest line, whatever kind it is.
    pub last_activity: Option<String>,
    pub total_tokens: Option<u64>,
    pub context_window: Option<u64>,
}

/// One rollout line, reading only the fields we render.
///
/// `#[serde(flatten)]` on the payload is deliberately NOT used: the payload
/// shape differs per `type`, and a flattened catch-all would silently accept a
/// line whose type we misread.
#[derive(Deserialize)]
struct Line {
    timestamp: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    payload: Option<serde_json::Value>,
}

/// Fold a rollout's lines into a summary.
///
/// Later lines win for `model` and `cwd`: `turn_context` is re-emitted per turn,
/// so the newest one is the session's current state rather than its first.
/// Unparseable lines are skipped — a partially written last line is normal for a
/// live session, and discarding the whole file for it would drop a running
/// session from the dashboard.
pub fn summarize(contents: &str) -> RolloutSummary {
    let mut out = RolloutSummary::default();

    for raw in contents.lines() {
        let Ok(line) = serde_json::from_str::<Line>(raw) else {
            continue;
        };
        if let Some(ts) = &line.timestamp {
            out.last_activity = Some(ts.clone());
        }
        let (Some(kind), Some(payload)) = (line.kind.as_deref(), line.payload.as_ref()) else {
            continue;
        };
        let str_field = |key: &str| {
            payload
                .get(key)
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };

        match kind {
            "session_meta" => {
                // `id` is the thread; `session_id` is a separate field on the
                // same object, and keying on it would group sessions wrongly.
                out.thread_id = str_field("id");
                out.session_id = str_field("session_id");
                out.cwd = str_field("cwd");
                out.started_at = str_field("timestamp");
            }
            "turn_context" => {
                // The model lives here, never in `session_meta`.
                if let Some(model) = str_field("model") {
                    out.model = Some(model);
                }
                if let Some(cwd) = str_field("cwd") {
                    out.cwd = Some(cwd);
                }
            }
            "event_msg" if payload.get("type").and_then(|v| v.as_str()) == Some("token_count") => {
                let info = payload.get("info");
                out.total_tokens = info
                    .and_then(|i| i.get("total_token_usage"))
                    .and_then(|u| u.get("total_tokens"))
                    .and_then(serde_json::Value::as_u64)
                    .or(out.total_tokens);
                out.context_window = info
                    .and_then(|i| i.get("model_context_window"))
                    .and_then(serde_json::Value::as_u64)
                    .or(out.context_window);
            }
            _ => {}
        }
    }

    out
}

/// Depth of the dated layout Codex writes: `sessions/YYYY/MM/DD/<file>.jsonl`.
///
/// Bounded rather than unlimited so a symlink loop or a stray deep tree under
/// the sessions root cannot turn a dashboard refresh into an unbounded walk.
const MAX_SCAN_DEPTH: usize = 4;

/// Rollout files under a sessions root.
///
/// Only `sessions/` is scanned. `archived_sessions/` is a sibling directory, so
/// archived threads are excluded by construction rather than by filtering — an
/// archived session is not a live one and must not appear on the dashboard.
///
/// A missing root is an empty list, not an error: not having Codex installed is
/// the common case, not a failure.
pub fn discover_rollout_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    collect(root, 1, &mut found);
    found.sort();
    found
}

fn collect(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_SCAN_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // `file_type` does not follow symlinks, so a link pointing back up the
        // tree is never descended into.
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect(&path, depth + 1, out);
        } else if kind.is_file() && path.extension().is_some_and(|e| e == "jsonl") {
            out.push(path);
        }
    }
}

/// Summarise one rollout file, or `None` if it cannot be read.
pub fn summarize_file(path: &Path) -> Option<RolloutSummary> {
    let contents = std::fs::read_to_string(path).ok()?;
    let summary = summarize(&contents);
    // A file with no `session_meta` has nothing to key a session on; rendering
    // it would produce a row with no identity.
    summary.thread_id.as_ref()?;
    Some(summary)
}

/// Every readable Codex session under a sessions root, newer than `since_ms`.
///
/// A rollout is never pruned, so the tree grows without bound while the live
/// set stays tiny. Re-reading and re-parsing all of it on every refresh tick
/// costs time proportional to total history rather than to live sessions — at
/// 200 files it was measured at 263 ms per tick, on a 2-second timer.
///
/// `since_ms` is the oldest live process's start time: a rollout not written
/// since before every live session began cannot belong to one of them, so it is
/// skipped on the `mtime` alone, without opening the file. Files whose metadata
/// cannot be read are kept rather than dropped — an unreadable stat must not
/// silently hide a live session.
pub fn discover_sessions_modified_since(root: &Path, since_ms: u64) -> Vec<RolloutSummary> {
    discover_rollout_files(root)
        .iter()
        .filter(|path| modified_since(path, since_ms))
        .filter_map(|path| summarize_file(path))
        .collect()
}

fn modified_since(path: &Path, since_ms: u64) -> bool {
    let Ok(modified) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return true;
    };
    let Ok(age) = modified.duration_since(std::time::UNIX_EPOCH) else {
        return true;
    };
    age.as_millis() as u64 >= since_ms
}
