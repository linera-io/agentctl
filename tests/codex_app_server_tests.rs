//! Codex App Server protocol.
//!
//! Method and notification names come from the schema the installed binary
//! emits itself (`codex app-server generate-json-schema --experimental`,
//! codex-cli 0.148.0). The protocol is marked experimental upstream, so the
//! generator is the only source that cannot drift from the binary being driven.
//!
//! The handshake test runs a real child process over real pipes rather than a
//! mocked parser, so the framing and the blocking read loop are both exercised.

use std::io::{BufReader, Write};
use std::process::{Command, Stdio};

use agentctl::providers::codex_app_server::{
    Incoming, Method, Notification, RequestBuilder, RequestId, decode_line, handshake,
};

#[test]
fn request_ids_are_unique_and_correlate_back_to_their_method() {
    let mut b = RequestBuilder::default();
    let (id1, line1) = b.initialize("1.2.3");
    let (id2, _) = b.request(Method::ThreadList, serde_json::json!({}));

    assert_ne!(id1, id2, "two in-flight requests must not share an id");
    assert!(line1.ends_with('\n'), "requests are newline framed");
    assert!(line1.contains("\"method\":\"initialize\""));
    assert!(
        line1.contains("\"name\":\"agentctl\"") && line1.contains("\"version\":\"1.2.3\""),
        "clientInfo is required by the schema: {line1}"
    );

    assert_eq!(b.pending_count(), 2);
    assert_eq!(b.resolve(id2), Some(Method::ThreadList));
    assert_eq!(b.resolve(id1), Some(Method::Initialize));
    assert_eq!(b.pending_count(), 0);

    // A duplicate response must not resolve twice, or its result would be
    // applied to a request that already completed.
    assert_eq!(b.resolve(id1), None);
    assert_eq!(b.resolve(9999), None, "an id we never sent");
}

/// An error response carries BOTH an `id` and an `error`.
///
/// Checking `method` first would read it as a notification and silently lose
/// the failure, leaving the request pending forever.
#[test]
fn an_error_response_is_not_mistaken_for_a_notification() {
    let line = r#"{"id":7,"error":{"code":-32601,"message":"method not found"}}"#;
    assert_eq!(
        decode_line(line),
        Some(Incoming::Error {
            id: 7,
            message: "method not found".to_string()
        })
    );
}

#[test]
fn responses_and_notifications_are_told_apart() {
    assert_eq!(
        decode_line(r#"{"id":1,"result":{"ok":true}}"#),
        Some(Incoming::Response {
            id: 1,
            result: serde_json::json!({"ok": true})
        })
    );
    // `turn/started` carries a whole `Turn`; `Turn.id` is what correlates it
    // with the matching `turn/completed`.
    assert_eq!(
        decode_line(
            r#"{"method":"turn/started","params":{"threadId":"t-1","turn":{"id":"turn-7","items":[],"status":"active"}}}"#
        ),
        Some(Incoming::Notification(Notification::TurnStarted {
            thread_id: "t-1".to_string(),
            turn_id: "turn-7".to_string()
        }))
    );
    // The wire field is `threadName`. This fixture was hand-written as `name`
    // — the shape the *request* uses — and the decoder was written to match it,
    // so the test passed against a payload the server never sends.
    assert_eq!(
        decode_line(
            r#"{"method":"thread/name/updated","params":{"threadId":"t-1","threadName":"fix parser"}}"#
        ),
        Some(Incoming::Notification(Notification::ThreadNameUpdated {
            thread_id: "t-1".to_string(),
            name: Some("fix parser".to_string())
        }))
    );
    // Nullable and not required: a cleared name must stay distinguishable from
    // an absent one.
    assert_eq!(
        decode_line(r#"{"method":"thread/name/updated","params":{"threadId":"t-1"}}"#),
        Some(Incoming::Notification(Notification::ThreadNameUpdated {
            thread_id: "t-1".to_string(),
            name: None
        }))
    );
    // And the request-side key must NOT be accepted, or the bug comes back.
    assert_eq!(
        decode_line(
            r#"{"method":"thread/name/updated","params":{"threadId":"t-1","name":"wrong"}}"#
        ),
        Some(Incoming::Notification(Notification::ThreadNameUpdated {
            thread_id: "t-1".to_string(),
            name: None
        }))
    );
}

/// The server emits 72 notification methods; we model 5.
///
/// An unmodelled one must not be an error, or every protocol addition upstream
/// becomes a breaking change here.
#[test]
fn an_unmodelled_notification_is_carried_not_rejected() {
    assert_eq!(
        decode_line(r#"{"method":"thread/realtime/sdp","params":{}}"#),
        Some(Incoming::Notification(Notification::Other {
            method: "thread/realtime/sdp".to_string()
        }))
    );
}

/// A live stream can end mid-line; that must not kill the connection.
#[test]
fn undecodable_lines_are_skipped_rather_than_fatal() {
    assert_eq!(decode_line(""), None);
    assert_eq!(decode_line("{\"method\":\"turn/star"), None, "torn line");
    assert_eq!(decode_line("{}"), None, "neither id nor method");
    assert_eq!(decode_line("not json at all"), None);
}

/// Handshake against a real child process over real pipes.
///
/// The child replies to `initialize` only AFTER pushing a notification, which
/// is what the server does when it already has state. Discarding those would
/// leave the first render stale until something else changed.
#[test]
fn the_handshake_keeps_notifications_that_arrive_before_the_response() {
    let script = r#"
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
# Push state first, exactly as a server with live threads does.
print(json.dumps({"method": "thread/status/changed", "params": {"threadId": "t-9", "status": {"type": "active"}}}), flush=True)
print(json.dumps({"method": "turn/started", "params": {"threadId": "t-9", "turn": {"id": "turn-1", "items": [], "status": "active"}}}), flush=True)
# A response to an id we never sent must be ignored, not matched.
print(json.dumps({"id": 999, "result": {"stray": True}}), flush=True)
print(json.dumps({"id": req["id"], "result": {"userAgent": "codex"}}), flush=True)
"#;

    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn fake app server");

    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

    let (result, early) = handshake(&mut reader, &mut stdin, "0.33.0").expect("handshake");

    assert_eq!(result, serde_json::json!({"userAgent": "codex"}));
    assert_eq!(
        early,
        vec![
            Notification::ThreadStatusChanged {
                thread_id: "t-9".to_string(),
                status: "active".to_string()
            },
            Notification::TurnStarted {
                thread_id: "t-9".to_string(),
                turn_id: "turn-1".to_string()
            },
        ],
        "notifications preceding the response are kept, in order"
    );

    let _ = stdin.flush();
    drop(stdin);
    let _ = child.wait();
}

/// A server that dies before answering must be an error, not a hang.
#[test]
fn a_server_that_closes_early_fails_rather_than_blocking() {
    let mut child = Command::new("python3")
        .arg("-c")
        .arg("import sys; sys.stdin.readline()")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");

    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

    let err = handshake(&mut reader, &mut stdin, "0.33.0").expect_err("must fail");
    assert!(err.to_string().contains("closed the stream"), "got: {err}");
    let _ = child.wait();
}

/// The server sends REQUESTS to us, not just responses and notifications.
///
/// `ServerRequest.json` lists eleven, including every approval flow
/// (`item/commandExecution/requestApproval`, `applyPatchApproval`, …). Such a
/// message carries BOTH an `id` and a `method`, so classifying on `id` alone
/// reads it as a response to a request we never sent — it gets dropped, and the
/// server blocks forever waiting for an answer. That is the approval path this
/// task exists to enable, so getting it wrong disables approvals silently.
#[test]
fn a_request_from_the_server_is_not_mistaken_for_a_response() {
    let line =
        r#"{"id":12,"method":"item/commandExecution/requestApproval","params":{"threadId":"t-1"}}"#;
    match decode_line(line).expect("decodes") {
        Incoming::ServerRequest { id, method, params } => {
            assert_eq!(id, RequestId::Int(12));
            assert_eq!(method, "item/commandExecution/requestApproval");
            assert_eq!(params.get("threadId").and_then(|v| v.as_str()), Some("t-1"));
        }
        other => panic!("must decode as a server request, got {other:?}"),
    }
}

/// `RequestId` is `string | int64` in the schema, and the reply must echo it
/// back in the same form — answering a string id with a number is unmatched.
#[test]
fn a_string_request_id_survives_a_round_trip() {
    let line = r#"{"id":"abc-1","method":"applyPatchApproval","params":{}}"#;
    match decode_line(line).expect("decodes") {
        Incoming::ServerRequest { id, .. } => {
            assert_eq!(id, RequestId::Str("abc-1".to_string()));
            assert_eq!(id.to_value(), serde_json::json!("abc-1"));
        }
        other => panic!("expected a server request, got {other:?}"),
    }
}

/// A server request mid-handshake must surface, not be swallowed.
#[test]
fn a_server_request_during_the_handshake_is_reported() {
    let script = r#"
import sys, json
req = json.loads(sys.stdin.readline())
print(json.dumps({"id": "srv-1", "method": "applyPatchApproval", "params": {}}), flush=True)
print(json.dumps({"id": req["id"], "result": {}}), flush=True)
"#;
    let mut child = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));

    let err = handshake(&mut reader, &mut stdin, "0.33.0").expect_err("must not silently proceed");
    assert!(err.to_string().contains("applyPatchApproval"), "got: {err}");
    let _ = child.wait();
}

/// Four of the five modelled notifications carry a required payload, and the
/// first cut of this client threw all of it away.
///
/// Keeping only `threadId` meant the dashboard was told "status changed" and
/// never what it changed to, "tokens updated" and never how many. Field names
/// come from the generated schema: `ThreadStatus` is a tagged object, and the
/// counts are camelCase here (`totalTokens`, `modelContextWindow`) where the
/// rollout file spells the same idea snake_case — the two must not be shared.
#[test]
fn the_notifications_carry_their_payloads_not_just_a_thread_id() {
    assert_eq!(
        decode_line(
            r#"{"method":"thread/status/changed","params":{"threadId":"t-1","status":{"type":"idle"}}}"#
        ),
        Some(Incoming::Notification(Notification::ThreadStatusChanged {
            thread_id: "t-1".to_string(),
            status: "idle".to_string()
        })),
        "status is the whole point of the notification"
    );

    assert_eq!(
        decode_line(
            r#"{"method":"thread/tokenUsage/updated","params":{"threadId":"t-1","turnId":"turn-3","tokenUsage":{"total":{"totalTokens":12800,"inputTokens":12000,"outputTokens":800,"cachedInputTokens":0,"reasoningOutputTokens":0},"last":{"totalTokens":1280,"inputTokens":1200,"outputTokens":80,"cachedInputTokens":0,"reasoningOutputTokens":0},"modelContextWindow":272000}}}"#
        ),
        Some(Incoming::Notification(
            Notification::ThreadTokenUsageUpdated {
                thread_id: "t-1".to_string(),
                turn_id: "turn-3".to_string(),
                total_tokens: Some(12800),
                context_window: Some(272000)
            }
        ))
    );

    // `modelContextWindow` is nullable, and the counts may be absent.
    assert_eq!(
        decode_line(
            r#"{"method":"thread/tokenUsage/updated","params":{"threadId":"t-1","turnId":"turn-3","tokenUsage":{"modelContextWindow":null}}}"#
        ),
        Some(Incoming::Notification(
            Notification::ThreadTokenUsageUpdated {
                thread_id: "t-1".to_string(),
                turn_id: "turn-3".to_string(),
                total_tokens: None,
                context_window: None
            }
        ))
    );

    assert_eq!(
        decode_line(
            r#"{"method":"turn/completed","params":{"threadId":"t-1","turn":{"id":"turn-9","items":[],"status":"completed"}}}"#
        ),
        Some(Incoming::Notification(Notification::TurnCompleted {
            thread_id: "t-1".to_string(),
            turn_id: "turn-9".to_string()
        }))
    );
}
