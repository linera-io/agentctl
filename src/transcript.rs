use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptRole {
    Assistant,
    User,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptUsage {
    pub input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone)]
pub enum TranscriptBlock {
    Text(String),
    ToolUse {
        id: Option<String>,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: Option<String>,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone)]
pub struct TranscriptMessage {
    pub role: TranscriptRole,
    pub model: Option<String>,
    pub stop_reason: Option<String>,
    pub usage: Option<TranscriptUsage>,
    pub content: Vec<TranscriptBlock>,
    /// Entry timestamp in unix epoch milliseconds, if the JSONL line carried
    /// a recognizable RFC-3339 `timestamp` field. Used to track when a user
    /// last interacted with the session.
    pub timestamp_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    WaitingForTask,
    Message(TranscriptMessage),
    /// The session's display name, recorded in the transcript itself:
    /// `{"type":"custom-title","customTitle":…}` (an explicit `/rename`) or
    /// `{"type":"agent-name","agentName":…}` (auto-derived). This is the only
    /// name source that survives Claude Code deleting the session pointer
    /// file mid-session and the restore registry losing its entry.
    SessionName {
        name: String,
        explicit: bool,
    },
}

pub fn parse_line(line: &str) -> Option<TranscriptEvent> {
    let entry: Value = serde_json::from_str(line).ok()?;

    if is_waiting_for_task(&entry) {
        return Some(TranscriptEvent::WaitingForTask);
    }

    match entry.get("type").and_then(|v| v.as_str()) {
        Some("custom-title") => {
            let name = entry.get("customTitle")?.as_str()?.trim().to_string();
            if !name.is_empty() {
                return Some(TranscriptEvent::SessionName {
                    name,
                    explicit: true,
                });
            }
        }
        Some("agent-name") => {
            let name = entry.get("agentName")?.as_str()?.trim().to_string();
            if !name.is_empty() {
                return Some(TranscriptEvent::SessionName {
                    name,
                    explicit: false,
                });
            }
        }
        _ => {}
    }

    let msg = entry.get("message")?;
    let role = message_role(&entry, msg)?;

    // Claude Code writes user prompts with `content` as a raw string, while
    // tool_result / text-block messages use an array of typed blocks. Handle
    // both shapes so a plain-string prompt is still visible as a Text block.
    let content: Vec<TranscriptBlock> = match msg.get("content") {
        Some(Value::String(s)) => vec![TranscriptBlock::Text(s.clone())],
        Some(Value::Array(blocks)) => blocks.iter().filter_map(parse_block).collect(),
        _ => Vec::new(),
    };

    Some(TranscriptEvent::Message(TranscriptMessage {
        role,
        model: msg
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        stop_reason: msg
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        usage: msg.get("usage").and_then(parse_usage),
        content,
        timestamp_ms: entry
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_rfc3339_utc_ms),
    }))
}

/// Parse an RFC-3339 UTC timestamp like "2026-04-19T22:57:04.552Z" into unix
/// epoch milliseconds. Accepts optional fractional seconds (0-9 digits) and
/// requires the trailing `Z` (Claude Code writes UTC). Returns `None` for any
/// format deviation — callers treat that as "no timestamp available".
pub fn parse_rfc3339_utc_ms(s: &str) -> Option<u64> {
    let s = s.strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    if date_parts.next().is_some() {
        return None;
    }
    let (hms, frac) = time.split_once('.').unwrap_or((time, ""));
    let mut hms_parts = hms.split(':');
    let hour: u32 = hms_parts.next()?.parse().ok()?;
    let minute: u32 = hms_parts.next()?.parse().ok()?;
    let second: u32 = hms_parts.next()?.parse().ok()?;
    if hms_parts.next().is_some() {
        return None;
    }
    let millis: u64 = if frac.is_empty() {
        0
    } else {
        // Pad/truncate the fractional part to exactly 3 digits (milliseconds).
        let mut buf = [b'0'; 3];
        for (i, b) in frac.bytes().take(3).enumerate() {
            buf[i] = b;
        }
        std::str::from_utf8(&buf).ok()?.parse().ok()?
    };
    if month == 0 || month > 12 || day == 0 || day > 31 || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let secs = days.checked_mul(86_400)?.checked_add(
        (hour as i64)
            .checked_mul(3600)?
            .checked_add((minute as i64).checked_mul(60)?)?
            .checked_add(second as i64)?,
    )?;
    if secs < 0 {
        return None;
    }
    (secs as u64).checked_mul(1000)?.checked_add(millis)
}

/// Days from 1970-01-01 for a proleptic Gregorian date. Howard Hinnant's
/// days_from_civil algorithm (https://howardhinnant.github.io/date_algorithms.html).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // 0..=399
    let m = m as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn is_waiting_for_task(entry: &Value) -> bool {
    if entry.get("type").and_then(|v| v.as_str()) != Some("progress") {
        return false;
    }

    match entry.get("data") {
        Some(Value::String(s)) => s.contains("waiting_for_task"),
        Some(Value::Object(map)) => map.values().any(|v| {
            v.as_str()
                .map(|s| s.contains("waiting_for_task"))
                .unwrap_or(false)
        }),
        _ => false,
    }
}

fn message_role(entry: &Value, msg: &Value) -> Option<TranscriptRole> {
    let role = msg
        .get("role")
        .and_then(|v| v.as_str())
        .or_else(|| entry.get("type").and_then(|v| v.as_str()))?;

    match role {
        "assistant" => Some(TranscriptRole::Assistant),
        "user" => Some(TranscriptRole::User),
        _ => None,
    }
}

fn parse_usage(value: &Value) -> Option<TranscriptUsage> {
    Some(TranscriptUsage {
        input_tokens: value.get("input_tokens")?.as_u64().unwrap_or(0),
        cache_read_input_tokens: value
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_creation_input_tokens: value
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: value
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    })
}

fn parse_block(block: &Value) -> Option<TranscriptBlock> {
    match block.get("type").and_then(|v| v.as_str())? {
        "text" => block
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| TranscriptBlock::Text(s.to_string())),
        "tool_use" => Some(TranscriptBlock::ToolUse {
            id: block.get("id").and_then(|v| v.as_str()).map(str::to_string),
            name: block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            input: block.get("input").cloned().unwrap_or(Value::Null),
        }),
        "tool_result" => Some(TranscriptBlock::ToolResult {
            tool_use_id: block
                .get("tool_use_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            content: block
                .get("content")
                .and_then(extract_text_content)
                .unwrap_or_default(),
            is_error: block
                .get("is_error")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        }),
        _ => None,
    }
}

fn extract_text_content(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }

    let blocks = value.as_array()?;
    let mut parts = Vec::new();
    for block in blocks {
        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
            parts.push(text);
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_real_fixture_line() {
        let line = include_str!("../tests/fixtures/real-transcript-line.json");
        let Some(TranscriptEvent::Message(msg)) = parse_line(line.trim()) else {
            panic!("expected message event");
        };
        assert_eq!(msg.role, TranscriptRole::Assistant);
        assert_eq!(msg.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(msg.model.as_deref(), Some("claude-sonnet-4-6-20260401"));
        assert_eq!(msg.content.len(), 2);
    }

    #[test]
    fn parse_legacy_fixture_line() {
        let line = include_str!("../tests/fixtures/legacy-transcript-line.json");
        let Some(TranscriptEvent::Message(msg)) = parse_line(line.trim()) else {
            panic!("expected message event");
        };
        assert_eq!(msg.role, TranscriptRole::Assistant);
        assert_eq!(msg.stop_reason.as_deref(), Some("end_turn"));
        assert!(msg.usage.is_some());
    }

    #[test]
    fn parse_waiting_for_task_progress() {
        let line = r#"{"type":"progress","data":"waiting_for_task"}"#;
        assert!(matches!(
            parse_line(line),
            Some(TranscriptEvent::WaitingForTask)
        ));
    }

    #[test]
    fn parse_rfc3339_with_millis() {
        // 2026-04-19T22:57:04.552Z — computed offline with:
        //   python -c 'import datetime as d; print(int(d.datetime(2026,4,19,22,57,4,552000,d.timezone.utc).timestamp()*1000))'
        assert_eq!(
            parse_rfc3339_utc_ms("2026-04-19T22:57:04.552Z"),
            Some(1_776_639_424_552)
        );
    }

    #[test]
    fn parse_rfc3339_epoch() {
        assert_eq!(parse_rfc3339_utc_ms("1970-01-01T00:00:00.000Z"), Some(0));
    }

    #[test]
    fn parse_rfc3339_without_fraction() {
        assert_eq!(
            parse_rfc3339_utc_ms("2026-04-19T22:57:04Z"),
            Some(1_776_639_424_000)
        );
    }

    #[test]
    fn parse_rfc3339_rejects_non_utc() {
        assert!(parse_rfc3339_utc_ms("2026-04-19T22:57:04+02:00").is_none());
    }

    #[test]
    fn parse_rfc3339_rejects_garbage() {
        assert!(parse_rfc3339_utc_ms("").is_none());
        assert!(parse_rfc3339_utc_ms("not-a-date").is_none());
        assert!(parse_rfc3339_utc_ms("2026-13-01T00:00:00Z").is_none());
    }

    #[test]
    fn parse_line_recovers_custom_title_and_agent_name() {
        let title = r#"{"type":"custom-title","customTitle":"resume-old-sessions-audit","sessionId":"3c20ad09-d564-429d-868c-d3a67dcace79"}"#;
        let Some(TranscriptEvent::SessionName { name, explicit }) = parse_line(title) else {
            panic!("expected SessionName from custom-title");
        };
        assert_eq!(name, "resume-old-sessions-audit");
        assert!(explicit, "custom-title is an explicit /rename");

        let agent = r#"{"type":"agent-name","agentName":"auto-topic","sessionId":"x"}"#;
        let Some(TranscriptEvent::SessionName { name, explicit }) = parse_line(agent) else {
            panic!("expected SessionName from agent-name");
        };
        assert_eq!(name, "auto-topic");
        assert!(!explicit, "agent-name is auto-derived");

        // Blank or absent names are not events.
        assert!(parse_line(r#"{"type":"custom-title","customTitle":"  "}"#).is_none());
        assert!(parse_line(r#"{"type":"agent-name"}"#).is_none());
    }

    #[test]
    fn parse_line_includes_timestamp_ms() {
        let line = r#"{"type":"user","timestamp":"2026-04-19T22:57:04.552Z","message":{"role":"user","content":"hello"}}"#;
        let Some(TranscriptEvent::Message(msg)) = parse_line(line) else {
            panic!("expected message event");
        };
        assert_eq!(msg.role, TranscriptRole::User);
        assert_eq!(msg.timestamp_ms, Some(1_776_639_424_552));
    }

    #[test]
    fn parse_line_user_with_string_content_is_text() {
        // Typical user prompts: `content` is a raw string, not a block array.
        let line = r#"{"type":"user","timestamp":"2026-04-19T22:57:04.000Z","message":{"role":"user","content":"hello there"}}"#;
        let Some(TranscriptEvent::Message(msg)) = parse_line(line) else {
            panic!("expected message event");
        };
        assert_eq!(msg.role, TranscriptRole::User);
        let texts: Vec<&str> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                TranscriptBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["hello there"]);
    }

    #[test]
    fn parse_line_distinguishes_user_text_from_tool_result() {
        // Real user prompt: content is an array with a Text block.
        let prompt = r#"{"type":"user","timestamp":"2026-04-19T22:57:04.000Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#;
        let Some(TranscriptEvent::Message(prompt_msg)) = parse_line(prompt) else {
            panic!("expected message event");
        };
        assert_eq!(prompt_msg.role, TranscriptRole::User);
        assert!(
            prompt_msg
                .content
                .iter()
                .any(|b| matches!(b, TranscriptBlock::Text(_))),
            "prompt should contain a Text block"
        );

        // Tool result: content is an array with a ToolResult block and no text.
        let tool_result = r#"{"type":"user","timestamp":"2026-04-19T22:58:00.000Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}"#;
        let Some(TranscriptEvent::Message(tr_msg)) = parse_line(tool_result) else {
            panic!("expected message event");
        };
        assert_eq!(tr_msg.role, TranscriptRole::User);
        assert!(
            !tr_msg
                .content
                .iter()
                .any(|b| matches!(b, TranscriptBlock::Text(_))),
            "tool_result should not contain a Text block"
        );
    }
}
