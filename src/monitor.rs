use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

use serde_json::Value;

use crate::hook_state::{self, HookState};
use crate::models;
use crate::session::{ClaudeSession, SessionStatus, SubagentRollup, TelemetryStatus};
use crate::transcript::{TranscriptBlock, TranscriptEvent, TranscriptRole, parse_line};

#[derive(Default)]
struct UsageRollup {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    cost_usd: f64,
    usage_metrics_available: bool,
    cost_estimate_unverified: bool,
}

impl UsageRollup {
    fn total_input_tokens(&self) -> u64 {
        self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
    }
}

/// Read new JSONL entries since last offset, accumulate token stats.
pub fn update_tokens(session: &mut ClaudeSession) {
    // Seed from persisted state so status inference works on ticks with no new JSONL.
    let mut last_type = session.last_msg_type.clone();
    let mut last_stop_reason = session.last_stop_reason.clone();
    let mut is_waiting_for_task = session.is_waiting_for_task;
    let mut saw_non_empty_line = false;
    let mut recognized_events = 0usize;
    let mut saw_parent_usage = false;
    let mut newest_message_ts = 0u64;
    let jsonl_path = session.jsonl_path.clone();

    match jsonl_path.as_ref() {
        Some(path) => {
            let mut file = match File::open(path) {
                Ok(f) => f,
                Err(_) => {
                    session.telemetry_status = TelemetryStatus::UnreadableTranscript;
                    finalize_usage(
                        session,
                        &last_type,
                        &last_stop_reason,
                        is_waiting_for_task,
                        false,
                    );
                    return;
                }
            };

            let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

            if file_len == 0 {
                session.telemetry_status = TelemetryStatus::Pending;
            } else {
                if session.jsonl_offset > file_len {
                    session.jsonl_offset = 0;
                    session.own_input_tokens = 0;
                    session.own_output_tokens = 0;
                    session.own_cache_read_tokens = 0;
                    session.own_cache_write_tokens = 0;
                    // Reset persisted inference state on file truncation
                    last_type.clear();
                    last_stop_reason.clear();
                    is_waiting_for_task = false;
                }

                if session.jsonl_offset < file_len {
                    if session.jsonl_offset > 0
                        && file.seek(SeekFrom::Start(session.jsonl_offset)).is_err()
                    {
                        finalize_usage(
                            session,
                            &last_type,
                            &last_stop_reason,
                            is_waiting_for_task,
                            false,
                        );
                        return;
                    }

                    let reader = BufReader::new(&file);

                    for line in reader.lines() {
                        let line = match line {
                            Ok(l) => l,
                            Err(_) => break,
                        };

                        if line.trim().is_empty() {
                            continue;
                        }
                        saw_non_empty_line = true;

                        let Some(event) = parse_line(&line) else {
                            continue;
                        };
                        recognized_events += 1;

                        match event {
                            TranscriptEvent::WaitingForTask => {
                                is_waiting_for_task = true;
                            }
                            TranscriptEvent::SessionName { name, explicit } => {
                                // Recover the display name from the transcript
                                // when every other source is gone (pointer file
                                // deleted mid-session, registry entry lost). An
                                // explicit `/rename` title always wins; an
                                // auto-derived agent-name only fills a blank.
                                //
                                // `name_is_explicit` makes the win durable:
                                // this event fires once per record (incremental
                                // parse), while scan-supplied names (a registry
                                // entry recorded before the rename) arrive
                                // every tick — without the flag the merge
                                // re-clobbers the title on the next tick and a
                                // rename "reverts" seconds after it is typed.
                                if explicit {
                                    session.session_name = name;
                                    session.name_is_explicit = true;
                                } else if session.session_name.is_empty() {
                                    session.session_name = name;
                                }
                            }
                            TranscriptEvent::Message(message) => {
                                is_waiting_for_task = false;
                                last_type = match message.role {
                                    TranscriptRole::Assistant => "assistant".to_string(),
                                    TranscriptRole::User => "user".to_string(),
                                };
                                if let Some(ts) = message.timestamp_ms {
                                    newest_message_ts = newest_message_ts.max(ts);
                                }

                                // Track the most recent user-originated text
                                // message. Tool results also arrive with the
                                // `user` role, so we require at least one Text
                                // block to distinguish a genuine prompt from
                                // Claude's tool-result injection.
                                if matches!(message.role, TranscriptRole::User) {
                                    let has_text = message
                                        .content
                                        .iter()
                                        .any(|b| matches!(b, TranscriptBlock::Text(_)));
                                    if has_text {
                                        if let Some(ts) = message.timestamp_ms {
                                            session.last_user_message_ts =
                                                session.last_user_message_ts.max(ts);
                                        }
                                    }
                                }

                                if let Some(reason) = message.stop_reason {
                                    last_stop_reason = reason;
                                } else {
                                    // Claude Code sometimes writes assistant messages
                                    // with stop_reason: null when a tool_use block is
                                    // awaiting user approval.  Infer from content.
                                    let has_tool_use = message
                                        .content
                                        .iter()
                                        .any(|b| matches!(b, TranscriptBlock::ToolUse { .. }));
                                    if has_tool_use {
                                        last_stop_reason = "tool_use".to_string();
                                    } else {
                                        last_stop_reason.clear();
                                    }
                                }

                                if let Some(usage) = message.usage {
                                    let input = usage.input_tokens;
                                    let cache_read = usage.cache_read_input_tokens;
                                    let cache_create = usage.cache_creation_input_tokens;
                                    let output = usage.output_tokens;

                                    session.own_input_tokens += input + cache_read + cache_create;
                                    session.own_output_tokens += output;
                                    session.own_cache_read_tokens += cache_read;
                                    session.own_cache_write_tokens += cache_create;
                                    saw_parent_usage = true;

                                    // Track context window: the input_tokens of the LAST API call
                                    // represents the current prompt/context size
                                    let context_size = input + cache_read + cache_create;
                                    if context_size > 0 {
                                        session.context_tokens = context_size;
                                    }
                                }

                                if let Some(model) = message.model {
                                    session.model = shorten_model(&model);
                                }

                                // Resume-safety: filter pending_tool_uses tracking so we
                                // don't flag `NeedsInput` based on tool_uses that were
                                // written to the JSONL by a PREVIOUS session process.
                                // A `claude --resume` reopens an old JSONL but doesn't
                                // re-display those historical permission prompts.
                                let is_current_session_msg = message
                                    .timestamp_ms
                                    .map(|ts| ts >= session.started_at)
                                    .unwrap_or(true);

                                for block in message.content {
                                    match &block {
                                        TranscriptBlock::ToolUse { id, name, input } => {
                                            record_tool_usage(name, input, session);
                                            // Track pending tool for rule-based auto-actions
                                            session.pending_tool_name = Some(name.clone());
                                            session.pending_tool_input = input
                                                .get("command")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string());
                                            // Track pending file path for conflict detection
                                            session.pending_file_path = if matches!(
                                                name.as_str(),
                                                "Edit" | "Write" | "NotebookEdit"
                                            ) {
                                                input
                                                    .get("file_path")
                                                    .and_then(|v| v.as_str())
                                                    .map(|s| s.to_string())
                                            } else {
                                                None
                                            };
                                            // Parallel-tool tracking: record every in-flight
                                            // tool_use by id so we can retire them individually
                                            // when their tool_results arrive — but only for
                                            // events from the CURRENT session process.
                                            if let (Some(id), true) = (id, is_current_session_msg) {
                                                session
                                                    .pending_tool_uses
                                                    .insert(id.clone(), name.clone());
                                            }
                                        }
                                        TranscriptBlock::ToolResult {
                                            tool_use_id,
                                            is_error,
                                            content,
                                        } => {
                                            session.last_tool_error = *is_error;
                                            if *is_error {
                                                session.total_error_count += 1;
                                                session.current_window_errors += 1;
                                                let truncated = if content.len() > 256 {
                                                    format!(
                                                        "{}...",
                                                        crate::session::truncate_str(content, 256)
                                                    )
                                                } else {
                                                    content.clone()
                                                };
                                                let tool_name = session
                                                    .pending_tool_name
                                                    .clone()
                                                    .unwrap_or_else(|| "?".into());
                                                session.last_error_message =
                                                    Some(truncated.clone());
                                                session.recent_errors.push(
                                                    crate::session::ErrorEntry {
                                                        tool_name,
                                                        message: truncated,
                                                    },
                                                );
                                                if session.recent_errors.len() > 5 {
                                                    session.recent_errors.remove(0);
                                                }
                                            } else {
                                                session.last_error_message = None;
                                            }
                                            // Retire this specific tool_use by id so
                                            // sibling parallel calls stay tracked.
                                            if let Some(id) = tool_use_id {
                                                session.pending_tool_uses.remove(id);
                                            }
                                            // Only clear the scalar "most-recent" trackers
                                            // once ALL tool_uses are retired; otherwise
                                            // rule-based actions lose context while a
                                            // parallel sibling is still in flight.
                                            if session.pending_tool_uses.is_empty() {
                                                session.pending_tool_name = None;
                                                session.pending_tool_input = None;
                                                session.pending_file_path = None;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }

                if recognized_events > 0 || session.telemetry_status.is_available() {
                    session.telemetry_status = TelemetryStatus::Available;
                } else if saw_non_empty_line {
                    session.telemetry_status = TelemetryStatus::UnsupportedTranscript;
                } else {
                    session.telemetry_status = TelemetryStatus::Pending;
                }

                session.jsonl_offset = file_len;
            }

            // `last_message_ts` is the newest *conversation* message, taken
            // from the record's own timestamp and accumulated across ticks
            // (this parse is incremental, so a tick with no new messages must
            // keep what earlier ticks learned).
            //
            // The file's mtime is only a fallback for transcripts that carry
            // no parsable timestamp at all. It is not the same quantity:
            // Claude Code appends bookkeeping records — `system`, `ai-title`,
            // `bridge-session`, `attachment` — after a turn ends, and each one
            // bumps mtime without any conversational progress. Reading mtime as
            // "last message" makes the quiet-session age-outs under-trigger and
            // would let those records masquerade as the user answering a
            // permission prompt.
            session.last_message_ts = session.last_message_ts.max(newest_message_ts);
            if session.last_message_ts == 0 {
                if let Ok(meta) = std::fs::metadata(path) {
                    if let Ok(modified) = meta.modified() {
                        session.last_message_ts = modified
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                    }
                }
            }
        }
        None => {
            session.telemetry_status = TelemetryStatus::MissingTranscript;
        }
    }

    finalize_usage(
        session,
        &last_type,
        &last_stop_reason,
        is_waiting_for_task,
        saw_parent_usage,
    );
}

fn finalize_usage(
    session: &mut ClaudeSession,
    last_type: &str,
    last_stop_reason: &str,
    is_waiting_for_task: bool,
    saw_parent_usage: bool,
) {
    let resolved_profile = models::resolve(&session.model);
    session.context_max = resolved_profile.profile.context_max;
    session.model_profile_source = resolved_profile.source.label().to_string();

    let subagent_rollup = refresh_subagent_rollups(session);
    session.subagent_input_tokens = subagent_rollup.total_input_tokens();
    session.subagent_output_tokens = subagent_rollup.output_tokens;
    session.subagent_cache_read_tokens = subagent_rollup.cache_read_tokens;
    session.subagent_cache_write_tokens = subagent_rollup.cache_write_tokens;
    session.subagent_count = session.subagent_rollups.len();

    session.total_input_tokens = session.own_input_tokens + session.subagent_input_tokens;
    session.total_output_tokens = session.own_output_tokens + session.subagent_output_tokens;
    session.cache_read_tokens = session.own_cache_read_tokens + session.subagent_cache_read_tokens;
    session.cache_write_tokens =
        session.own_cache_write_tokens + session.subagent_cache_write_tokens;

    let own_usage_metrics_available = saw_parent_usage
        || session.own_input_tokens > 0
        || session.own_output_tokens > 0
        || session.own_cache_read_tokens > 0
        || session.own_cache_write_tokens > 0;
    let (own_cost, own_cost_unverified) = estimate_cost_components(
        &session.model,
        session.own_input_tokens,
        session.own_output_tokens,
        session.own_cache_read_tokens,
        session.own_cache_write_tokens,
    );
    session.cost_usd = own_cost + subagent_rollup.cost_usd;
    session.usage_metrics_available =
        own_usage_metrics_available || subagent_rollup.usage_metrics_available;
    session.cost_estimate_unverified = (own_usage_metrics_available && own_cost_unverified)
        || subagent_rollup.cost_estimate_unverified;

    // Persist for next tick (so status inference works when no new JSONL arrives).
    session.last_msg_type = last_type.to_string();
    session.last_stop_reason = last_stop_reason.to_string();
    session.is_waiting_for_task = is_waiting_for_task;

    infer_status(session, last_type, last_stop_reason);
}

/// Load this session's hook state, stamp the clock, and record the verdict.
///
/// The thin IO shell around [`decide_status`]: every filesystem read and every
/// call to the system clock lives here, so the decision itself stays a pure
/// function of its inputs and can be replayed at any age from fixtures. That
/// separation is not cosmetic — status defects have repeatedly shipped because
/// the age-dependent branches were only reachable by waiting in real time.
pub fn infer_status(session: &mut ClaudeSession, last_msg_type: &str, last_stop_reason: &str) {
    let hook_state = if session.session_id.is_empty() {
        None
    } else {
        HookState::load(&session.session_id)
    };

    let verdict = decide_status(&StatusInputs {
        hook_state: hook_state.as_ref(),
        now_ms: now_ms(),
        last_msg_type,
        last_stop_reason,
        last_message_ts: session.last_message_ts,
        cpu_rate_percent: session.cpu_rate_percent,
        telemetry_available: session.telemetry_status.is_available(),
    });

    // Log the *evidence*, not just the ruling. Diagnosing a wrong status
    // otherwise means reconstructing it by hand from raw state files, `ps`,
    // and /proc sampling — which is exactly how the three defects this
    // function was rewritten for had to be found.
    if verdict.status != session.status {
        crate::logger::log(
            "DEBUG",
            &format!(
                "status: {} {} -> {} ({}) [msg={last_msg_type}/{last_stop_reason} cpu_rate={:?} hook={}]",
                session.pid,
                session.status,
                verdict.status,
                verdict.reason,
                session.cpu_rate_percent,
                hook_state.as_ref().map_or("none".into(), |s| format!(
                    "notif={:?} turn_age_ms={} stop_age_ms={} tool={:?}",
                    s.notification_kind,
                    now_ms().saturating_sub(hook_state::newest_turn_event_ms(s)),
                    now_ms().saturating_sub(s.last_stop_ts_ms),
                    s.current_tool_name,
                )),
            ),
        );
    }

    session.status = verdict.status;
}

/// Everything [`decide_status`] is allowed to look at. No clock, no filesystem.
pub struct StatusInputs<'a> {
    pub hook_state: Option<&'a HookState>,
    pub now_ms: u64,
    /// Role of the newest conversation message: `"assistant"`, `"user"`, or
    /// empty when the transcript yielded none.
    pub last_msg_type: &'a str,
    pub last_stop_reason: &'a str,
    /// Timestamp of the newest conversation message — never the transcript
    /// file's mtime, which bookkeeping records bump without any conversational
    /// progress.
    pub last_message_ts: u64,
    /// CPU used since the previous sample, as a percentage of one core.
    /// `None` means "not measured yet" and must never license a claim that the
    /// session is working.
    pub cpu_rate_percent: Option<f32>,
    pub telemetry_available: bool,
}

/// A status plus the branch that produced it, for the diagnostic log.
pub struct StatusVerdict {
    pub status: SessionStatus,
    pub reason: &'static str,
}

/// How long a turn may go with no hook event *and* no transcript growth before
/// we stop believing it is live.
///
/// Only applies when no tool is in flight — a single tool call can legitimately
/// run for hours in silence, and `hook_state::tool_in_flight` licenses exactly
/// that. What remains is a turn that is neither calling tools nor writing
/// messages, which is not a turn: its `Stop` was dropped. 15 minutes is well
/// past the longest tool-free stretch observed in practice (a session was seen
/// thinking for 15 minutes between its last tool result and `end_turn`), and
/// crossing it only ever moves a session to a *weaker* claim.
const TURN_SILENCE_MAX_MS: u64 = 15 * 60 * 1000;

/// A session waiting on the user this long reads as `Idle` instead — the user
/// has clearly walked away, and surfacing it as `Waiting` forever crowds the
/// bucket that means "ready for you right now".
const WAITING_TO_IDLE_MS: u64 = 10 * 60 * 1000;

/// CPU rate, as a percentage of one core, above which a session with no other
/// evidence is taken to be working. Compared against a *rate* measured between
/// two samples — never `ps`'s `%cpu`, which is a lifetime average on Linux and
/// a ~1-minute decaying average on macOS.
const BUSY_CPU_RATE_PERCENT: f32 = 5.0;

/// Decide a session's status from its evidence. Pure: same inputs, same answer,
/// on any machine at any time.
///
/// The branches are ordered by how strong a claim they make, and the ordering
/// is the whole point. A status is a claim: `NeedsInput` and `Processing` are
/// strong ("this session is definitely in this state"), `Waiting` is weaker,
/// `Unknown` is an admission of ignorance, `Idle` says only that nothing has
/// happened. **When the evidence for a claim expires or goes missing, the
/// fall-through must land on a weaker claim, never a stronger one.**
///
/// Every status defect fixed here was a violation of that rule: an expiring
/// permission-prompt marker fell through to `Processing` (a session blocked on
/// the user, reported as busy); a dropped `Stop` left `Processing` latched
/// forever; and a lifetime-average CPU number promoted finished sessions to
/// `Processing` ahead of a transcript that said the turn had ended. `Processing`
/// is the worst possible default — it means "it's working, leave it alone".
///
/// Two properties are asserted over this function in the test suite and are
/// the reason it is shaped this way:
///   * **time alone never promotes** — advancing the clock with all other
///     inputs fixed can never move a session *into* `Processing` or
///     `NeedsInput`;
///   * **eventual release** — with no tool in flight, no new events, and no CPU
///     to show for it, a `Processing` verdict must expire.
pub fn decide_status(inputs: &StatusInputs) -> StatusVerdict {
    let transcript_ended = transcript_ended_the_turn(inputs);

    if let Some(state) = inputs.hook_state {
        // A pending permission prompt outranks everything, including
        // Compacting and Processing: such a session is technically mid-turn,
        // but the state that matters to the user is "blocked on me".
        if hook_state::is_at_permission_prompt(state, inputs.now_ms, inputs.last_message_ts) {
            return verdict(SessionStatus::NeedsInput, "hook: permission prompt open");
        }
        if hook_state::is_compacting(state, inputs.now_ms) {
            return verdict(SessionStatus::Compacting, "hook: compacting");
        }
        if hook_state::is_responding(state) && !transcript_ended && !turn_went_silent(state, inputs)
        {
            return verdict(SessionStatus::Processing, "hook: turn in flight");
        }
        if hook_state::is_waiting_for_user(state) {
            return waiting_or_idle(state.last_stop_ts_ms, inputs, "hook: stop is newest event");
        }
    }

    // Transcript evidence outranks CPU. A turn that ended is over however busy
    // the process still looks — Claude Code's node process routinely burns CPU
    // on renders and watchers while sitting at an empty prompt.
    if transcript_ended {
        return waiting_or_idle(inputs.last_message_ts, inputs, "transcript: turn ended");
    }

    if inputs
        .cpu_rate_percent
        .is_some_and(|rate| rate > BUSY_CPU_RATE_PERCENT)
    {
        return verdict(SessionStatus::Processing, "cpu: burning a core");
    }

    if !inputs.telemetry_available && inputs.last_msg_type.is_empty() {
        return verdict(SessionStatus::Unknown, "no telemetry");
    }

    // Hook markers still say mid-turn, but the transcript did not corroborate
    // it and the session is not burning CPU. We genuinely do not know what
    // this session is doing — say so, rather than claiming it is working.
    if inputs.hook_state.is_some_and(hook_state::is_responding) {
        return verdict(
            SessionStatus::Unknown,
            "turn markers dangling, no corroboration",
        );
    }

    // Transcript-only fallbacks, for sessions whose hooks have never fired
    // (started before claudectl's auto-init ran). **Never produces
    // `NeedsInput`**: every attempt to guess "needs the user's attention" from
    // JSONL alone was wrong in some common case, and the deterministic
    // `Notification` hook owns that claim.
    if matches!(inputs.last_msg_type, "assistant" | "user") {
        // An `assistant` + `tool_use` tail or a `user` tail (prompt or tool
        // result) means work was under way as of that message. Recent ⇒
        // Processing; long quiet ⇒ Idle.
        return if inputs.now_ms.saturating_sub(inputs.last_message_ts) > WAITING_TO_IDLE_MS {
            verdict(SessionStatus::Idle, "transcript: mid-turn tail, long quiet")
        } else {
            verdict(
                SessionStatus::Processing,
                "transcript: mid-turn tail, recent",
            )
        };
    }

    verdict(SessionStatus::Idle, "no evidence of activity")
}

fn verdict(status: SessionStatus, reason: &'static str) -> StatusVerdict {
    StatusVerdict { status, reason }
}

/// `WaitingInput`, decaying to `Idle` once the user has been away long enough.
fn waiting_or_idle(since_ms: u64, inputs: &StatusInputs, reason: &'static str) -> StatusVerdict {
    if inputs.now_ms.saturating_sub(since_ms) > WAITING_TO_IDLE_MS {
        verdict(SessionStatus::Idle, reason)
    } else {
        verdict(SessionStatus::WaitingInput, reason)
    }
}

/// Whether a turn that the hook stream still calls live has gone silent on
/// *both* channels — no hook event and no new message — for longer than a real
/// turn ever does. See [`TURN_SILENCE_MAX_MS`].
fn turn_went_silent(state: &HookState, inputs: &StatusInputs) -> bool {
    if hook_state::tool_in_flight(state) {
        return false;
    }
    let newest = hook_state::newest_turn_event_ms(state).max(inputs.last_message_ts);
    inputs.now_ms.saturating_sub(newest) > TURN_SILENCE_MAX_MS
}

/// Whether the transcript proves the turn is over even though the hook stream
/// still looks mid-turn — the place the JSONL overrules a hook marker.
///
/// Hook events are lossy: each one is a separate `claudectl` process Claude
/// Code spawns with a 5 s timeout, and any invocation can be dropped. The
/// transcript is not — Claude Code writes it itself — so when it says the
/// assistant finished its turn we believe it.
///
/// Two conditions, both required:
///   1. the tail is an assistant message that *ended* (`end_turn` /
///      `stop_sequence`) — a `tool_use` tail proves nothing, the tool may still
///      be running;
///   2. that message is at least as new as the newest mid-turn hook event —
///      otherwise it describes the *previous* turn and vetoing on it would
///      report a live turn as finished. This is what keeps the fix from
///      trading one wrong status for its mirror image.
fn transcript_ended_the_turn(inputs: &StatusInputs) -> bool {
    if inputs.last_msg_type != "assistant"
        || !matches!(inputs.last_stop_reason, "end_turn" | "stop_sequence")
    {
        return false;
    }
    match inputs.hook_state {
        Some(state) => inputs.last_message_ts >= hook_state::newest_turn_event_ms(state),
        None => true,
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Estimate USD cost based on token usage and model.
#[allow(dead_code)]
pub fn estimate_cost(session: &ClaudeSession) -> f64 {
    estimate_cost_components(
        &session.model,
        session.total_input_tokens,
        session.total_output_tokens,
        session.cache_read_tokens,
        session.cache_write_tokens,
    )
    .0
}

/// Max context window tokens by model.
pub fn model_context_max(model: &str) -> u64 {
    models::resolve(model).profile.context_max
}

/// Extract tool usage stats and file paths from tool_use content blocks.
fn record_tool_usage(tool_name: &str, input: &Value, session: &mut ClaudeSession) {
    if tool_name.is_empty() {
        return;
    }

    session
        .tool_usage
        .entry(tool_name.to_string())
        .or_default()
        .calls += 1;

    if matches!(tool_name, "Edit" | "Write" | "NotebookEdit") {
        if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
            *session.files_modified.entry(path.to_string()).or_insert(0) += 1;
            // Reset file-read tracker for this path (it was just edited)
            session.file_reads_since_edit.remove(path);
        }
        // Track token efficiency: cumulative tokens at each edit event
        let total_tokens = session.total_input_tokens + session.total_output_tokens;
        session.total_tokens_at_edit_count += total_tokens;
        session.edit_event_count += 1;
        // Freeze baseline tokens-per-edit after first 5 edits
        if session.baseline_tokens_per_edit.is_none() && session.edit_event_count >= 5 {
            session.baseline_tokens_per_edit =
                Some(session.total_tokens_at_edit_count as f64 / session.edit_event_count as f64);
        }
    }

    // Track file reads for repetition detection
    if matches!(tool_name, "Read" | "Grep" | "Glob") {
        if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
            *session
                .file_reads_since_edit
                .entry(path.to_string())
                .or_insert(0) += 1;
        }
    }
}

pub fn shorten_model(model: &str) -> String {
    models::shorten_model(model)
}

fn refresh_subagent_rollups(session: &mut ClaudeSession) -> UsageRollup {
    for path in session.active_subagent_jsonl_paths.clone() {
        let rollup = session.subagent_rollups.entry(path.clone()).or_default();
        update_subagent_rollup(&path, rollup, &session.model);
    }

    let mut totals = UsageRollup::default();
    for rollup in session.subagent_rollups.values() {
        totals.input_tokens += rollup.input_tokens;
        totals.output_tokens += rollup.output_tokens;
        totals.cache_read_tokens += rollup.cache_read_tokens;
        totals.cache_write_tokens += rollup.cache_write_tokens;
        totals.cost_usd += rollup.cost_usd;
        totals.usage_metrics_available |= rollup.usage_metrics_available;
        totals.cost_estimate_unverified |= rollup.cost_estimate_unverified;
    }
    totals
}

fn update_subagent_rollup(
    path: &std::path::Path,
    rollup: &mut SubagentRollup,
    default_model: &str,
) {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return,
    };

    let file_len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    if rollup.jsonl_offset > file_len {
        *rollup = SubagentRollup::default();
    }

    if rollup.jsonl_offset >= file_len {
        rollup.jsonl_offset = file_len;
        return;
    }

    if rollup.jsonl_offset > 0 && file.seek(SeekFrom::Start(rollup.jsonl_offset)).is_err() {
        return;
    }

    let mut current_model = if rollup.model.is_empty() {
        default_model.to_string()
    } else {
        rollup.model.clone()
    };

    let reader = BufReader::new(&file);
    for line in reader.lines() {
        let Ok(line) = line else {
            break;
        };
        let Some(TranscriptEvent::Message(message)) = parse_line(&line) else {
            continue;
        };

        if let Some(model) = message.model {
            current_model = shorten_model(&model);
            rollup.model = current_model.clone();
        }

        let Some(usage) = message.usage else {
            continue;
        };

        rollup.input_tokens += usage.input_tokens;
        rollup.output_tokens += usage.output_tokens;
        rollup.cache_read_tokens += usage.cache_read_input_tokens;
        rollup.cache_write_tokens += usage.cache_creation_input_tokens;
        rollup.usage_metrics_available = true;

        let input_with_cache =
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens;
        let model_for_cost = if current_model.is_empty() {
            default_model
        } else {
            current_model.as_str()
        };
        let (delta_cost, unverified) = estimate_cost_components(
            model_for_cost,
            input_with_cache,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        );
        rollup.cost_usd += delta_cost;
        rollup.cost_estimate_unverified |= unverified;
    }

    rollup.jsonl_offset = file_len;
}

fn estimate_cost_components(
    model: &str,
    total_input_tokens: u64,
    total_output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
) -> (f64, bool) {
    let plain_input = total_input_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    let resolved = models::resolve(model);

    let cost = (plain_input as f64 / 1_000_000.0) * resolved.profile.input_per_m
        + (total_output_tokens as f64 / 1_000_000.0) * resolved.profile.output_per_m
        + (cache_read_tokens as f64 / 1_000_000.0) * resolved.profile.cache_read_per_m
        + (cache_write_tokens as f64 / 1_000_000.0) * resolved.profile.cache_write_per_m;

    (
        cost,
        resolved.source == models::ModelProfileSource::Fallback,
    )
}
