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
