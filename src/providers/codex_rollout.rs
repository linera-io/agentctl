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
    /// Position of the line within the rollout. Stable across re-reads, so it
    /// is what makes a usage row identifiable rather than merely re-appended.
    ordinal: Option<u64>,
}

/// One turn's token usage, as a delta.
///
/// Codex reports `total_token_usage` as a running total for the session, so a
/// row per event would re-bill the whole session every turn — the same
/// over-counting that made the Claude ledger read 11x high. These are the
/// differences between successive snapshots, which sum to the session total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageEvent {
    /// `<session or file stem>:<ordinal>` is unique and stable, so re-reading
    /// a rollout produces the same rows rather than duplicates.
    pub ordinal: u64,
    pub timestamp: Option<String>,
    /// The model in force at this point — `turn_context` is re-emitted per
    /// turn, and a session can change model mid-flight.
    pub model: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Per-turn token deltas from a rollout.
///
/// `last_token_usage` looks like the same thing, but it is only present on some
/// events and nothing guarantees the series partitions the total; differencing
/// the cumulative figure is exact by construction and needs no such assumption.
pub fn usage_events(contents: &str) -> Vec<UsageEvent> {
    let mut out = Vec::new();
    let mut model: Option<String> = None;
    let mut prev_input = 0u64;
    let mut prev_output = 0u64;

    for (idx, raw) in contents.lines().enumerate() {
        let Ok(line) = serde_json::from_str::<Line>(raw) else {
            continue;
        };
        let (Some(kind), Some(payload)) = (line.kind.as_deref(), line.payload.as_ref()) else {
            continue;
        };
        if kind == "turn_context" {
            if let Some(m) = payload.get("model").and_then(|v| v.as_str()) {
                model = Some(m.to_string());
            }
            continue;
        }
        if kind != "event_msg"
            || payload.get("type").and_then(|v| v.as_str()) != Some("token_count")
        {
            continue;
        }
        let Some(total) = payload.get("info").and_then(|i| i.get("total_token_usage")) else {
            continue;
        };
        let field = |k: &str| {
            total
                .get(k)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        };
        let (input, output) = (field("input_tokens"), field("output_tokens"));

        // Saturating: a counter that goes backwards is not a negative cost.
        let event = UsageEvent {
            ordinal: line.ordinal.unwrap_or(idx as u64),
            timestamp: line.timestamp.clone(),
            model: model.clone(),
            input_tokens: input.saturating_sub(prev_input),
            output_tokens: output.saturating_sub(prev_output),
        };
        prev_input = input.max(prev_input);
        prev_output = output.max(prev_output);
        if event.input_tokens > 0 || event.output_tokens > 0 {
            out.push(event);
        }
    }
    out
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

#[cfg(test)]
mod usage_tests {
    use super::*;

    const MULTI: &str = include_str!("../../tests/fixtures/codex/rollout-multi-turn.jsonl");
    const BASIC: &str = include_str!("../../tests/fixtures/codex/rollout-basic.jsonl");

    /// Codex reports a running total, so the events must be differenced. The
    /// deltas must also sum back to the session total — that is what makes the
    /// ledger's per-turn rows add up to what Codex says was spent.
    #[test]
    fn cumulative_totals_become_per_turn_deltas() {
        let events = usage_events(MULTI);
        assert_eq!(events.len(), 3, "one row per token_count event");

        assert_eq!(
            (events[0].input_tokens, events[0].output_tokens),
            (12_000, 800)
        );
        assert_eq!(
            (events[1].input_tokens, events[1].output_tokens),
            (13_000, 700)
        );
        assert_eq!(
            (events[2].input_tokens, events[2].output_tokens),
            (15_000, 900)
        );

        let summed: (u64, u64) = events.iter().fold((0, 0), |acc, e| {
            (acc.0 + e.input_tokens, acc.1 + e.output_tokens)
        });
        assert_eq!(
            summed,
            (40_000, 2_400),
            "deltas must sum to the session total"
        );
    }

    /// A session can change model mid-flight; each turn bills at the model in
    /// force when it ran, not at whatever the session ended on.
    #[test]
    fn each_turn_carries_the_model_in_force_at_the_time() {
        let events = usage_events(MULTI);
        assert_eq!(events[0].model.as_deref(), Some("gpt-5-codex"));
        assert_eq!(events[1].model.as_deref(), Some("gpt-5-codex"));
        assert_eq!(events[2].model.as_deref(), Some("gpt-5.4"));
    }

    /// The ordinal is what keys the row, so re-reading a rollout must produce
    /// the same identities rather than duplicates.
    #[test]
    fn ordinals_are_stable_across_reads() {
        let a = usage_events(MULTI);
        let b = usage_events(MULTI);
        assert_eq!(a, b);
        assert_eq!(
            a.iter().map(|e| e.ordinal).collect::<Vec<_>>(),
            vec![3, 4, 6]
        );
    }

    /// A rollout that is still being written ends mid-line; that must yield the
    /// turns so far, not nothing.
    #[test]
    fn a_truncated_rollout_yields_the_complete_turns() {
        let cut = &MULTI[..MULTI.len() - 40];
        let events = usage_events(cut);
        assert!(
            !events.is_empty(),
            "complete turns must survive a partial tail"
        );
    }

    #[test]
    fn a_single_turn_rollout_bills_its_whole_total() {
        let events = usage_events(BASIC);
        assert_eq!(events.len(), 1);
        assert_eq!(
            (events[0].input_tokens, events[0].output_tokens),
            (12_000, 800)
        );
    }
}
