//! Speaking Codex's App Server protocol.
//!
//! One JSON object per line over the child's stdin/stdout. Requests carry an
//! `id` and get exactly one response with the same `id`; notifications carry a
//! `method` and no `id`, and arrive interleaved with responses at any time.
//!
//! Every name here was taken from the schema the installed binary emits itself
//! (`codex app-server generate-json-schema --out <dir> --experimental`, 47
//! files, codex-cli 0.148.0) rather than from documentation or guesswork — the
//! protocol is `experimental` and moves, so the generator is the only source
//! that cannot be out of date with the binary being driven.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

/// The client name sent as `clientInfo.name` during `initialize`. The version
/// alongside it is supplied per call, not fixed here.
pub const CLIENT_NAME: &str = "agentctl";

/// Requests we send. Names verified against the generated `ClientRequest`
/// schema, whose `oneOf` lists 141 methods; these are the ones a dashboard
/// needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Initialize,
    ThreadList,
    ThreadRead,
    ThreadResume,
    ThreadNameSet,
    TurnStart,
}

impl Method {
    pub fn wire(&self) -> &'static str {
        match self {
            Self::Initialize => "initialize",
            Self::ThreadList => "thread/list",
            Self::ThreadRead => "thread/read",
            Self::ThreadResume => "thread/resume",
            Self::ThreadNameSet => "thread/name/set",
            Self::TurnStart => "turn/start",
        }
    }
}

/// A notification the server pushes without being asked.
///
/// Only the ones that change what a dashboard row shows are modelled; the
/// server emits 72 in total, and treating an unmodelled one as an error would
/// make every protocol addition a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Notification {
    ThreadStatusChanged {
        thread_id: String,
    },
    ThreadTokenUsageUpdated {
        thread_id: String,
    },
    ThreadNameUpdated {
        thread_id: String,
        /// `None` when the server cleared the name, which the schema models as
        /// a nullable, non-required field. Collapsing that to an empty string
        /// would make "cleared" and "not sent" the same value.
        name: Option<String>,
    },
    TurnStarted {
        thread_id: String,
    },
    TurnCompleted {
        thread_id: String,
    },
    /// A method we do not model. Carried rather than dropped so a caller can
    /// log it, and so "unknown" is distinguishable from "not a notification".
    Other {
        method: String,
    },
}

impl Notification {
    fn from_parts(method: &str, params: &serde_json::Value) -> Self {
        let thread_id = || {
            params
                .get("threadId")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        match method {
            "thread/status/changed" => Self::ThreadStatusChanged {
                thread_id: thread_id(),
            },
            "thread/tokenUsage/updated" => Self::ThreadTokenUsageUpdated {
                thread_id: thread_id(),
            },
            // The wire field is `threadName`, NOT `name`. The *request* side
            // (`ThreadSetNameParams`) does use `name`, which is the trap: the
            // notification does not, and reading `name` here yields an empty
            // string for every rename.
            "thread/name/updated" => Self::ThreadNameUpdated {
                thread_id: thread_id(),
                name: params
                    .get("threadName")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            },
            "turn/started" => Self::TurnStarted {
                thread_id: thread_id(),
            },
            "turn/completed" => Self::TurnCompleted {
                thread_id: thread_id(),
            },
            other => Self::Other {
                method: other.to_string(),
            },
        }
    }
}

/// A request id. The schema types it `string | int64`, and a server-initiated
/// request may use either, so it is echoed back verbatim rather than parsed
/// into a number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestId {
    Int(i64),
    Str(String),
}

impl RequestId {
    fn from_value(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::Number(n) => n.as_i64().map(Self::Int),
            serde_json::Value::String(s) => Some(Self::Str(s.clone())),
            _ => None,
        }
    }

    pub fn to_value(&self) -> serde_json::Value {
        match self {
            Self::Int(i) => serde_json::json!(i),
            Self::Str(s) => serde_json::json!(s),
        }
    }
}

/// One decoded line from the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Incoming {
    Response {
        id: i64,
        result: serde_json::Value,
    },
    Error {
        id: i64,
        message: String,
    },
    Notification(Notification),
    /// A request FROM the server, which expects a reply carrying the same id.
    ///
    /// The approval flows live here (`item/commandExecution/requestApproval`,
    /// `applyPatchApproval`, and nine others in `ServerRequest.json`). It has
    /// both an `id` and a `method`, so classifying on `id` alone reads it as a
    /// response to a request we never sent — and the server then blocks
    /// forever waiting for an answer that never comes.
    ServerRequest {
        id: RequestId,
        method: String,
        params: serde_json::Value,
    },
}

/// Decode one line.
///
/// `None` for a line that is not valid JSON or carries neither an id nor a
/// method. A live server's stream can contain a partially written final line,
/// and killing the connection over it would drop every session it reports.
pub fn decode_line(line: &str) -> Option<Incoming> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;

    let method = value.get("method").and_then(|v| v.as_str());

    // id + method is a request FROM the server; id alone is a response to ours;
    // method alone is a notification. Order matters: an error response carries
    // an id and an `error` but no method, while a server request carries an id
    // AND a method, so neither can be identified by the presence of `id` alone.
    if let (Some(raw_id), Some(method)) = (value.get("id"), method) {
        if let Some(id) = RequestId::from_value(raw_id) {
            return Some(Incoming::ServerRequest {
                id,
                method: method.to_string(),
                params: value
                    .get("params")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            });
        }
    }

    if let Some(id) = value.get("id").and_then(serde_json::Value::as_i64) {
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error")
                .to_string();
            return Some(Incoming::Error { id, message });
        }
        return Some(Incoming::Response {
            id,
            result: value
                .get("result")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        });
    }

    let method = method?;
    let params = value
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    Some(Incoming::Notification(Notification::from_parts(
        method, &params,
    )))
}

/// Outgoing request framing plus id allocation.
///
/// Ids are allocated here rather than by the caller so two in-flight requests
/// cannot collide — correlating a response to the wrong request is silent, and
/// would attribute one thread's state to another.
#[derive(Debug, Default)]
pub struct RequestBuilder {
    next_id: i64,
    pending: HashMap<i64, Method>,
}

#[derive(Serialize, Deserialize)]
struct WireRequest<'a> {
    id: i64,
    method: &'a str,
    params: serde_json::Value,
}

impl RequestBuilder {
    /// The `initialize` handshake. `clientInfo` is required by the schema;
    /// `capabilities` is optional and omitted.
    pub fn initialize(&mut self, version: &str) -> (i64, String) {
        self.request(
            Method::Initialize,
            serde_json::json!({
                "clientInfo": { "name": CLIENT_NAME, "version": version }
            }),
        )
    }

    /// Frame a request, returning its id and the line to write.
    pub fn request(&mut self, method: Method, params: serde_json::Value) -> (i64, String) {
        self.next_id += 1;
        let id = self.next_id;
        self.pending.insert(id, method);
        let wire = WireRequest {
            id,
            method: method.wire(),
            params,
        };
        let line = serde_json::to_string(&wire).unwrap_or_default();
        (id, format!("{line}\n"))
    }

    /// Which request an incoming id answers, consuming the correlation.
    ///
    /// `None` for an id we never sent, or one already answered — a duplicate
    /// response must not be applied twice.
    pub fn resolve(&mut self, id: i64) -> Option<Method> {
        self.pending.remove(&id)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Drive a handshake and read until the response to `initialize` arrives.
///
/// Notifications seen before the response are returned rather than discarded:
/// the server may push state as soon as it starts, and dropping it would leave
/// the first render stale until something else changed.
pub fn handshake<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    version: &str,
) -> std::io::Result<(serde_json::Value, Vec<Notification>)> {
    let mut builder = RequestBuilder::default();
    let (id, line) = builder.initialize(version);
    writer.write_all(line.as_bytes())?;
    writer.flush()?;

    let mut early = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            return Err(std::io::Error::other(
                "app server closed the stream before answering initialize",
            ));
        }
        match decode_line(buf.trim_end()) {
            Some(Incoming::Response { id: got, result }) if got == id => {
                builder.resolve(got);
                return Ok((result, early));
            }
            Some(Incoming::Error { id: got, message }) if got == id => {
                return Err(std::io::Error::other(format!(
                    "initialize failed: {message}"
                )));
            }
            Some(Incoming::Notification(n)) => early.push(n),
            // A server request arriving mid-handshake is surfaced, not
            // dropped: the server blocks until it is answered, and swallowing
            // it here would deadlock the connection before it is even up.
            Some(Incoming::ServerRequest { method, .. }) => {
                return Err(std::io::Error::other(format!(
                    "app server sent request {method} before initialize completed; \
                     answering server requests is not implemented yet"
                )));
            }
            // A response to an id we did not send, or an undecodable line.
            _ => continue,
        }
    }
}
