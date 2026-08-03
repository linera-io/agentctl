use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionStatus {
    NeedsInput,   // Blocked — waiting for user to approve/confirm (permission prompt)
    Compacting,   // Auto-compact in progress (PreCompact fired, no Stop yet)
    Processing,   // Actively generating or executing tools
    WaitingInput, // Done responding, waiting for user's next prompt
    Unknown,      // Process is alive, but transcript telemetry is unavailable
    Idle,         // No recent activity, stale session
    Finished,     // Process exited
}

impl fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeedsInput => write!(f, "Needs Input"),
            Self::Compacting => write!(f, "Compacting"),
            Self::Processing => write!(f, "Processing"),
            Self::WaitingInput => write!(f, "Waiting"),
            Self::Unknown => write!(f, "Unknown"),
            Self::Idle => write!(f, "Idle"),
            Self::Finished => write!(f, "Finished"),
        }
    }
}

impl SessionStatus {
    /// Inverse of [`fmt::Display`], for reading a status back out of collected
    /// `--json` output. Unrecognised input becomes `Unknown` rather than a
    /// guess — the variant that already means "the process is there but we
    /// can't say what it's doing".
    ///
    /// Kept adjacent to the `Display` impl on purpose; `status_labels_round_trip`
    /// asserts the two stay inverses instead of trusting that they will.
    pub fn from_label(label: &str) -> Self {
        match label {
            "Needs Input" => Self::NeedsInput,
            "Compacting" => Self::Compacting,
            "Processing" => Self::Processing,
            "Waiting" => Self::WaitingInput,
            "Idle" => Self::Idle,
            "Finished" => Self::Finished,
            _ => Self::Unknown,
        }
    }

    pub fn sort_key(&self) -> u8 {
        match self {
            Self::NeedsInput => 0,
            Self::Compacting => 1,
            Self::Processing => 2,
            Self::WaitingInput => 3,
            Self::Unknown => 4,
            Self::Idle => 5,
            Self::Finished => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryStatus {
    Pending,
    Available,
    MissingTranscript,
    UnreadableTranscript,
    UnsupportedTranscript,
}

impl TelemetryStatus {
    pub fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Available => "Available",
            Self::MissingTranscript => "No transcript",
            Self::UnreadableTranscript => "Unreadable transcript",
            Self::UnsupportedTranscript => "Unsupported transcript",
        }
    }

    /// Inverse of [`label`](Self::label), for reading a collected row back.
    /// Unrecognised input becomes `Pending` — "we don't know yet" — rather than
    /// a guess. `telemetry_labels_round_trip` keeps the two in step.
    pub fn from_label(label: &str) -> Self {
        match label {
            "Available" => Self::Available,
            "No transcript" => Self::MissingTranscript,
            "Unreadable transcript" => Self::UnreadableTranscript,
            "Unsupported transcript" => Self::UnsupportedTranscript,
            _ => Self::Pending,
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Available => "Available",
            Self::MissingTranscript => "No transcript",
            Self::UnreadableTranscript => "Unreadable",
            Self::UnsupportedTranscript => "Unsupported",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawSession {
    pub pid: u32,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub cwd: String,
    #[serde(rename = "startedAt")]
    pub started_at: u64,
    /// Session display name set by Claude Code (via the `/rename` slash
    /// command or topic-based auto-naming). When present, it is preferred
    /// over the cwd-derived `project_name` in `display_name()`.
    #[serde(default)]
    pub name: Option<String>,
    /// Where `name` came from. Claude Code ≥ 2.1.220 mints a placeholder
    /// name (`<cwd-basename>-<hex>`, e.g. "ndr-5e") at every session
    /// start/resume and marks it `"nameSource": "derived"`; a real `/rename`
    /// title carries no source marker. Absent in older pointer files.
    #[serde(rename = "nameSource", default)]
    pub name_source: Option<String>,
}

impl RawSession {
    /// The session's title, with Claude Code's auto-derived placeholder
    /// filtered out. A placeholder is minted fresh on every start/resume, so
    /// ingesting it as a title both displays noise and — because the registry
    /// merge lets an incoming `Some` name overwrite the stored one — clobbers
    /// a real `/rename` title on resume. Treat it as "not named" so the
    /// transcript/registry recovery paths supply the real title instead.
    fn title(&self) -> Option<String> {
        if self.name_source.as_deref() == Some("derived") {
            None
        } else {
            self.name.clone()
        }
    }
}

/// Connection target for a host-side terminal when claudectl runs inside the
/// agent-sandbox microVM. Filled from the per-PID terminal sidecar written
/// by the sandbox wrappers; mirrors the env vars each terminal exports.
///
/// On macOS-host sandboxes (Ghostty/iTerm2/Warp/Apple) the bridge speaks
/// AppleScript over `sandbox-osa-bridge`, so this field stays None and the
/// macOS arms in `terminals/mod.rs` pick the matcher by `terminal_id`/`tty`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTerminalTarget {
    Kitty {
        socket: String,
        window_id: String,
    },
    Tmux {
        socket: String,
        pane: String,
    },
    WezTerm {
        pane_id: u64,
        unix_socket: Option<String>,
    },
}

/// Where a session was observed — which machine or microVM its process runs in.
///
/// A property of the session, not of how we happened to learn about it: a
/// session's pid, `~/.claude/sessions` sidecar and `ps` row all live inside one
/// VM, and none of them mean anything outside it. Carrying this on the session
/// (rather than in a lookup table beside it) is what stops a foreign row being
/// silently treated as local — the distinction that decides whether we may
/// signal its pid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionOrigin {
    /// Discovered natively by this claudectl — its own sandbox, or the laptop
    /// when running on the host. Its pid is addressable by us.
    Local,
    /// Collected from another sandbox via the host-side snapshot. Its pid
    /// belongs to a different VM and must never be signalled from here.
    Sandbox(String),
}

impl SessionOrigin {
    /// Label for the ORIGIN column. `Local` needs context we don't hold here —
    /// the caller passes what "here" is called (the sandbox name, or `None` on
    /// the laptop).
    pub fn label(&self, local_name: Option<&str>) -> String {
        match self {
            Self::Local => local_name.unwrap_or("laptop").to_string(),
            Self::Sandbox(name) => name.clone(),
        }
    }

    /// Can this process be signalled from here? False for anything in another
    /// VM, where our pid namespace would hit an unrelated local process.
    pub fn is_addressable(&self) -> bool {
        matches!(self, Self::Local)
    }
}

#[derive(Debug, Clone)]
pub struct ClaudeSession {
    pub pid: u32,
    pub session_id: String,
    /// Which machine/VM this session runs in. See [`SessionOrigin`].
    pub origin: SessionOrigin,
    pub cwd: String,
    pub project_name: String,
    pub started_at: u64,
    pub elapsed: Duration,
    pub tty: String,
    /// Host-side terminal-application id (currently only populated for
    /// Ghostty via the agent-sandbox terminal sidecar). When set, terminal
    /// matchers should prefer it over CWD/title heuristics. None for
    /// host-native claude sessions and terminals that don't expose a stable
    /// id (iTerm2/Apple/Warp rely on `tty` instead).
    pub terminal_id: Option<String>,
    /// Host-side terminal connection target (kitty socket+window, tmux
    /// socket+pane, wezterm pane id+optional socket). Populated from the
    /// agent-sandbox terminal sidecar when claudectl runs inside the
    /// sandbox and the host runs a Linux desktop terminal. None for
    /// macOS-host sandboxes (which use osa-bridge) and for host-native
    /// claudectl runs (which talk to the local terminal CLIs directly).
    pub host_terminal_target: Option<HostTerminalTarget>,
    /// True once `process::fetch_and_enrich` has attempted to read the
    /// per-PID `terminal.json` sidecar at least once. The sidecar is
    /// written exactly once at session start by sandbox-bootstrap and
    /// never mutated, so re-reading every tick burned ~70 ms / tick at
    /// 40 sandboxed sessions for no information gain.
    pub sidecar_loaded: bool,
    /// Failed sidecar probes so far. A first-tick probe can lose a race
    /// (registry mid-write, transient /proc hiccup) and a session frozen
    /// without routing keeps Tab-switching broken for the TUI's lifetime,
    /// so absence is retried with a bounded budget instead of being cached
    /// as final on the first attempt.
    pub sidecar_attempts: u8,
    /// Which terminal application this session runs in, detected once from the
    /// session process's own environment (TERM_PROGRAM / GHOSTTY_RESOURCES_DIR /
    /// KITTY_WINDOW_ID / …). Lets claudectl focus/input/approve a session that
    /// lives in a *different* terminal than the one claudectl itself runs in
    /// (e.g. claudectl in iTerm2 switching to a session in Ghostty). None until
    /// resolved or when no terminal signal is present; callers then fall back to
    /// the terminal claudectl itself runs in.
    pub terminal: Option<crate::terminals::Terminal>,
    /// True once `process::fetch_and_enrich` has attempted to resolve `terminal`
    /// (one `ps eww` per pid); avoids repeating the probe every refresh tick.
    pub terminal_resolved: bool,
    pub status: SessionStatus,
    pub cpu_percent: f32,
    pub cpu_history: Vec<f32>, // Last N CPU readings for smoothing
    pub mem_mb: f64,
    pub own_input_tokens: u64,
    pub own_output_tokens: u64,
    pub own_cache_read_tokens: u64,
    pub own_cache_write_tokens: u64,
    pub subagent_input_tokens: u64,
    pub subagent_output_tokens: u64,
    pub subagent_cache_read_tokens: u64,
    pub subagent_cache_write_tokens: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub model: String,
    pub command_args: String,
    pub session_name: String,
    /// True once `session_name` holds an explicit `/rename` title recovered
    /// from the transcript's `custom-title` records (monitor). An explicit
    /// title is the user's own choice and the transcript is its durable
    /// source of truth — a scan-supplied name (registry entry recorded
    /// before a rename, recreated pointer) must not overwrite it, or the
    /// cross-tick merge re-clobbers the title seconds after every rename.
    /// A newer `custom-title` record still updates it (explicit beats
    /// explicit; the transcript is append-only, so later means fresher).
    pub name_is_explicit: bool,
    pub jsonl_path: Option<PathBuf>,
    pub jsonl_offset: u64,
    pub last_message_ts: u64,
    /// Unix epoch millis of the most recent user-originated text message in
    /// the transcript (fresh prompt or text reply; excludes tool results).
    /// 0 means never seen.
    pub last_user_message_ts: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: f64,
    pub context_tokens: u64,
    pub context_max: u64,
    pub prev_cost_usd: f64,
    pub burn_rate_per_hr: f64,
    pub subagent_count: usize,
    pub active_subagent_count: usize,
    pub active_subagent_jsonl_paths: Vec<PathBuf>,
    pub subagent_rollups: HashMap<PathBuf, SubagentRollup>,
    pub activity_history: Vec<u8>, // Ring buffer of status levels (0-7) for sparkline, one per tick
    pub files_modified: HashMap<String, u32>, // file path -> edit count
    pub tool_usage: HashMap<String, ToolStats>, // tool name -> call count & tokens
    pub worktree_id: Option<String>, // Resolved git toplevel + git-dir, for conflict detection
    pub telemetry_status: TelemetryStatus,
    pub usage_metrics_available: bool,
    pub cost_estimate_unverified: bool,
    pub model_profile_source: String,
    /// Persisted across ticks so status inference works when no new JSONL arrives.
    pub last_msg_type: String,
    pub last_stop_reason: String,
    pub is_waiting_for_task: bool,
    /// Pending tool call details for rule-based auto-actions.
    pub pending_tool_name: Option<String>,
    pub pending_tool_input: Option<String>, // Extracted command string (for Bash)
    pub pending_file_path: Option<String>,  // File path for pending Edit/Write/NotebookEdit
    pub has_file_conflict: bool,            // Pending file edit conflicts with another session
    /// All in-flight tool calls keyed by `tool_use_id` with their names. An entry
    /// is added when a ToolUse block is parsed and removed when the matching
    /// ToolResult arrives. Supports parallel tool calls — `pending_tool_name`
    /// above only tracks the most recent, so this map is the source of truth
    /// for "is any tool still pending".
    pub pending_tool_uses: HashMap<String, String>,
    pub last_tool_error: bool,
    pub last_error_message: Option<String>,
    pub recent_errors: Vec<ErrorEntry>, // Last 5 errors (ring buffer)
    // ── Cognitive health tracking ────────────────────────────────────
    /// Cumulative tokens at each Edit/Write event (for efficiency trending).
    pub total_tokens_at_edit_count: u64,
    /// Number of Edit/Write events (for averaging tokens-per-edit).
    pub edit_event_count: u32,
    /// Baseline tokens-per-edit, frozen after first 5 edits.
    pub baseline_tokens_per_edit: Option<f64>,
    /// Error count ring buffer: one entry per window (~10s each).
    pub error_counts_per_window: Vec<u32>, // max 10 entries
    /// Accumulator for current error window.
    pub current_window_errors: u32,
    /// Ticks since last window flush.
    pub window_tick_counter: u32,
    /// Baseline error rate (errors per window), frozen after 3 windows.
    pub baseline_error_rate: Option<f64>,
    /// File reads since last edit: path -> read count. Reset when file is edited.
    pub file_reads_since_edit: HashMap<String, u32>,
    /// All-time error count.
    pub total_error_count: u32,
    /// Cached composite decay score (0-100), recomputed each tick.
    pub decay_score: u32,
}

/// A captured tool error with context.
#[derive(Debug, Clone)]
pub struct ErrorEntry {
    pub tool_name: String,
    pub message: String,
}

/// Per-tool usage statistics.
#[derive(Debug, Clone, Default)]
pub struct ToolStats {
    pub calls: u32,
}

#[derive(Debug, Clone, Default)]
pub struct SubagentRollup {
    pub jsonl_offset: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: f64,
    pub model: String,
    pub cost_estimate_unverified: bool,
    pub usage_metrics_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentState {
    Active,
    Completed,
}

#[derive(Debug, Clone)]
pub struct SubagentBreakdown {
    pub label: String,
    pub state: SubagentState,
    pub count: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: f64,
    pub usage_metrics_available: bool,
    pub cost_estimate_unverified: bool,
}

impl SubagentBreakdown {
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    pub fn state_label(&self) -> String {
        match self.state {
            SubagentState::Active => "Active".to_string(),
            SubagentState::Completed if self.count > 1 => format!("Completed ({})", self.count),
            SubagentState::Completed => "Completed".to_string(),
        }
    }

    pub fn display_label(&self) -> String {
        if self.state == SubagentState::Completed && self.label == "completed" && self.count > 1 {
            format!("completed ({})", self.count)
        } else {
            self.label.clone()
        }
    }

    pub fn format_tokens(&self) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        let total = self.total_input_tokens() + self.output_tokens;
        if total == 0 {
            return "-".to_string();
        }
        format_count(self.total_input_tokens()) + "/" + &format_count(self.output_tokens)
    }

    pub fn format_cost(&self) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        if self.cost_usd < 0.01 {
            return "-".to_string();
        }
        if self.cost_usd < 1.0 {
            format!(
                "${:.2}{}",
                self.cost_usd,
                if self.cost_estimate_unverified {
                    "?"
                } else {
                    ""
                }
            )
        } else {
            format!(
                "${:.1}{}",
                self.cost_usd,
                if self.cost_estimate_unverified {
                    "?"
                } else {
                    ""
                }
            )
        }
    }
}

impl ClaudeSession {
    pub fn from_raw(raw: RawSession) -> Self {
        let project_name = raw.cwd.rsplit('/').next().unwrap_or("unknown").to_string();
        let session_name = raw.title().unwrap_or_default();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let elapsed_ms = now_ms.saturating_sub(raw.started_at);
        let elapsed = Duration::from_millis(elapsed_ms);

        Self {
            pid: raw.pid,
            session_id: raw.session_id,
            // Everything reaching `from_raw` came from this VM's own sidecars
            // or process table, so it is by construction local. Foreign
            // sessions are built by `from_snapshot_value` instead.
            origin: SessionOrigin::Local,
            cwd: raw.cwd,
            project_name,
            started_at: raw.started_at,
            elapsed,
            tty: String::new(),
            terminal_id: None,
            host_terminal_target: None,
            sidecar_loaded: false,
            sidecar_attempts: 0,
            terminal: None,
            terminal_resolved: false,
            status: SessionStatus::Idle,
            cpu_percent: 0.0,
            cpu_history: Vec::new(),
            mem_mb: 0.0,
            own_input_tokens: 0,
            own_output_tokens: 0,
            own_cache_read_tokens: 0,
            own_cache_write_tokens: 0,
            subagent_input_tokens: 0,
            subagent_output_tokens: 0,
            subagent_cache_read_tokens: 0,
            subagent_cache_write_tokens: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            model: String::new(),
            command_args: String::new(),
            session_name,
            name_is_explicit: false,
            jsonl_path: None,
            jsonl_offset: 0,
            last_message_ts: 0,
            last_user_message_ts: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cost_usd: 0.0,
            context_tokens: 0,
            context_max: 0,
            prev_cost_usd: 0.0,
            burn_rate_per_hr: 0.0,
            subagent_count: 0,
            active_subagent_count: 0,
            active_subagent_jsonl_paths: Vec::new(),
            subagent_rollups: HashMap::new(),
            activity_history: Vec::new(),
            files_modified: HashMap::new(),
            tool_usage: HashMap::new(),
            worktree_id: None,
            telemetry_status: TelemetryStatus::Pending,
            usage_metrics_available: false,
            cost_estimate_unverified: false,
            model_profile_source: "built-in".into(),
            last_msg_type: String::new(),
            last_stop_reason: String::new(),
            is_waiting_for_task: false,
            pending_tool_name: None,
            pending_tool_input: None,
            pending_file_path: None,
            has_file_conflict: false,
            pending_tool_uses: HashMap::new(),
            last_tool_error: false,
            last_error_message: None,
            recent_errors: Vec::new(),
            total_tokens_at_edit_count: 0,
            edit_event_count: 0,
            baseline_tokens_per_edit: None,
            error_counts_per_window: Vec::new(),
            current_window_errors: 0,
            window_tick_counter: 0,
            baseline_error_rate: None,
            file_reads_since_edit: HashMap::new(),
            total_error_count: 0,
            decay_score: 0,
        }
    }

    /// Record current status into the activity sparkline ring buffer.
    /// Max 15 entries (one per tick, at 2s default = 30s of history).
    pub fn record_activity(&mut self) {
        let level = match self.status {
            SessionStatus::Processing => 7,
            SessionStatus::Compacting => 5,
            SessionStatus::NeedsInput => 4,
            SessionStatus::WaitingInput => 2,
            SessionStatus::Unknown => 2,
            SessionStatus::Idle => 1,
            SessionStatus::Finished => 0,
        };
        self.activity_history.push(level);
        if self.activity_history.len() > 15 {
            self.activity_history.remove(0);
        }

        // Flush error window every 5 ticks (~10s at default 2s interval)
        self.window_tick_counter += 1;
        if self.window_tick_counter >= 5 {
            self.error_counts_per_window
                .push(self.current_window_errors);
            if self.error_counts_per_window.len() > 10 {
                self.error_counts_per_window.remove(0);
            }
            // Freeze baseline error rate after 3 windows
            if self.baseline_error_rate.is_none() && self.error_counts_per_window.len() >= 3 {
                let sum: u32 = self.error_counts_per_window.iter().sum();
                self.baseline_error_rate =
                    Some(sum as f64 / self.error_counts_per_window.len() as f64);
            }
            self.current_window_errors = 0;
            self.window_tick_counter = 0;
        }
    }

    /// Render the sparkline as unicode block characters.
    pub fn format_sparkline(&self) -> String {
        const BLOCKS: &[char] = &[
            ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}',
            '\u{2587}', '\u{2588}',
        ];
        if self.activity_history.is_empty() {
            return String::from("-");
        }
        self.activity_history
            .iter()
            .map(|&level| BLOCKS[level.min(8) as usize])
            .collect()
    }

    /// Rebuild a session from one row of the host-collected snapshot
    /// (`sandboxes.json`), i.e. a session belonging to another sandbox.
    ///
    /// The reaper's collector assembles those rows from each sandbox's
    /// hook-written registry plus an in-VM `ps` probe. Older snapshots on disk
    /// were assembled from that sandbox's own `claudectl --json` and carry more
    /// fields (cost, status, activity); they are still read here, but the host
    /// recomputes every transcript-derived one immediately afterwards in
    /// `app::do_refresh_io`, so what they say no longer decides anything.
    ///
    /// Returns `None` when the entry carries no `session_id` — an unidentified
    /// row can be neither resumed nor de-duplicated against local discovery,
    /// so showing it would be worse than omitting it.
    ///
    /// Only fields the collected payload actually carries are overlaid. The
    /// rest keep `from_raw`'s defaults rather than being invented: a fabricated
    /// zero would render as a real measurement, and "we didn't collect this"
    /// must not look like "this session is idle".
    pub fn from_snapshot_value(sandbox: &str, value: &serde_json::Value) -> Option<Self> {
        let session_id = value.get("session_id")?.as_str()?.to_string();
        let str_field = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let raw = RawSession {
            pid: value
                .get("pid")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
            session_id,
            cwd: str_field("cwd"),
            started_at: value
                .get("started_at")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            // Pass the recorded title through as-is. It has already survived
            // the placeholder filter on the producing side, so re-running
            // `title()` here would be a second filter over clean input.
            name: Some(str_field("session_name")).filter(|name| !name.is_empty()),
            name_source: None,
        };
        let mut session = Self::from_raw(raw);
        session.origin = SessionOrigin::Sandbox(sandbox.to_string());

        if let Some(cpu) = value.get("cpu").and_then(serde_json::Value::as_f64) {
            session.cpu_percent = cpu as f32;
        }
        if let Some(mem) = value.get("mem_mb").and_then(serde_json::Value::as_f64) {
            session.mem_mb = mem;
        }
        // The cost/token block is emitted as null when the producing side had
        // no usage metrics. Prefer the producer's own flag; fall back to "a
        // cost number is present" for rows collected from an older claudectl
        // that didn't emit the flag.
        let has_metrics = value
            .get("usage_metrics_available")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or_else(|| {
                value
                    .get("cost_usd")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
            });
        if has_metrics {
            session.usage_metrics_available = true;
            session.cost_usd = value
                .get("cost_usd")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            session.burn_rate_per_hr = value
                .get("burn_rate_per_hr")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            session.total_input_tokens = value
                .get("tokens_in")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            session.total_output_tokens = value
                .get("tokens_out")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            // Both sides of the ratio: `context_percent` returns 0 unless it
            // has each of them, which rendered the Context bar blank on every
            // collected row.
            session.context_tokens = value
                .get("context_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            session.context_max = value
                .get("context_max")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
        }
        if let Some(ts) = value
            .get("last_user_message_ts")
            .and_then(serde_json::Value::as_u64)
        {
            session.last_user_message_ts = ts;
        }
        if let Some(history) = value
            .get("activity_history")
            .and_then(serde_json::Value::as_array)
        {
            session.activity_history = history
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|level| level as u8)
                .collect();
        }
        if let Some(status) = value.get("status").and_then(serde_json::Value::as_str) {
            session.status = SessionStatus::from_label(status);
        }
        if let Some(state) = value
            .get("telemetry")
            .and_then(|t| t.get("state"))
            .and_then(serde_json::Value::as_str)
        {
            session.telemetry_status = TelemetryStatus::from_label(state);
        }
        Some(session)
    }

    pub fn display_name(&self) -> &str {
        if !self.session_name.is_empty() {
            &self.session_name
        } else {
            &self.project_name
        }
    }

    pub fn format_subagent_summary(&self) -> String {
        if self.subagent_count == 0 {
            return "0".to_string();
        }
        if self.active_subagent_count == 0 || self.active_subagent_count == self.subagent_count {
            return self.subagent_count.to_string();
        }
        format!(
            "{} total ({} active)",
            self.subagent_count, self.active_subagent_count
        )
    }

    pub fn subagent_breakdown(&self) -> Vec<SubagentBreakdown> {
        if self.subagent_rollups.is_empty() {
            return Vec::new();
        }

        let active_paths: HashSet<&PathBuf> = self.active_subagent_jsonl_paths.iter().collect();
        let mut active_rows = Vec::new();
        let mut completed_rows = Vec::new();

        for (path, rollup) in &self.subagent_rollups {
            let row = SubagentBreakdown {
                label: subagent_label(path),
                state: if active_paths.contains(path) {
                    SubagentState::Active
                } else {
                    SubagentState::Completed
                },
                count: 1,
                input_tokens: rollup.input_tokens,
                output_tokens: rollup.output_tokens,
                cache_read_tokens: rollup.cache_read_tokens,
                cache_write_tokens: rollup.cache_write_tokens,
                cost_usd: rollup.cost_usd,
                usage_metrics_available: rollup.usage_metrics_available,
                cost_estimate_unverified: rollup.cost_estimate_unverified,
            };

            if row.state == SubagentState::Active {
                active_rows.push(row);
            } else {
                completed_rows.push(row);
            }
        }

        active_rows.sort_by(|a, b| a.label.cmp(&b.label));

        let mut rows = Vec::new();
        if !completed_rows.is_empty() {
            let mut aggregate = SubagentBreakdown {
                label: "completed".to_string(),
                state: SubagentState::Completed,
                count: completed_rows.len(),
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                cost_usd: 0.0,
                usage_metrics_available: false,
                cost_estimate_unverified: false,
            };

            for row in completed_rows {
                aggregate.input_tokens += row.input_tokens;
                aggregate.output_tokens += row.output_tokens;
                aggregate.cache_read_tokens += row.cache_read_tokens;
                aggregate.cache_write_tokens += row.cache_write_tokens;
                aggregate.cost_usd += row.cost_usd;
                aggregate.usage_metrics_available |= row.usage_metrics_available;
                aggregate.cost_estimate_unverified |= row.cost_estimate_unverified;
            }

            rows.push(aggregate);
        }

        rows.extend(active_rows);
        rows
    }

    pub fn format_elapsed(&self) -> String {
        let secs = self.elapsed.as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        if h > 0 {
            format!("{h:02}:{m:02}:{s:02}")
        } else {
            format!("{m:02}:{s:02}")
        }
    }

    pub fn format_tokens(&self) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        let total = self.total_input_tokens + self.total_output_tokens;
        if total == 0 {
            return String::from("-");
        }
        format_count(self.total_input_tokens) + "/" + &format_count(self.total_output_tokens)
    }

    pub fn format_mem(&self) -> String {
        if self.mem_mb < 1.0 {
            return String::from("-");
        }
        format!("{:.0}M", self.mem_mb)
    }

    pub fn format_cost(&self) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        if self.cost_usd < 0.01 {
            return String::from("-");
        }
        if self.cost_usd < 1.0 {
            format!(
                "${:.2}{}",
                self.cost_usd,
                if self.cost_estimate_unverified {
                    "?"
                } else {
                    ""
                }
            )
        } else {
            format!(
                "${:.1}{}",
                self.cost_usd,
                if self.cost_estimate_unverified {
                    "?"
                } else {
                    ""
                }
            )
        }
    }

    pub fn context_percent(&self) -> f64 {
        if !self.usage_metrics_available {
            return 0.0;
        }
        if self.context_max == 0 || self.context_tokens == 0 {
            return 0.0;
        }
        (self.context_tokens as f64 / self.context_max as f64) * 100.0
    }

    /// Format context as "450k/1M 45%" or a visual bar
    pub fn format_context(&self) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        if self.context_tokens == 0 {
            return String::from("-");
        }
        let pct = self.context_percent();
        format!("{}%", pct as u32)
    }

    /// Visual bar for context usage: ████░░ 62%
    pub fn format_context_bar(&self, width: usize) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        let pct = self.context_percent();
        if pct == 0.0 {
            return String::from("-");
        }
        let filled = ((pct / 100.0) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!(
            "{}{} {}%",
            "█".repeat(filled),
            "░".repeat(empty),
            pct as u32
        )
    }

    /// Produce a JSON-serializable value for --json export.
    pub fn to_json_value(&self) -> serde_json::Value {
        let cost_usd = if self.usage_metrics_available {
            serde_json::json!((self.cost_usd * 100.0).round() / 100.0)
        } else {
            serde_json::Value::Null
        };
        let burn_rate = if self.usage_metrics_available {
            serde_json::json!((self.burn_rate_per_hr * 100.0).round() / 100.0)
        } else {
            serde_json::Value::Null
        };
        let context_pct = if self.usage_metrics_available {
            serde_json::json!((self.context_percent() * 100.0).round() / 100.0)
        } else {
            serde_json::Value::Null
        };
        let tokens_in = if self.usage_metrics_available {
            serde_json::json!(self.total_input_tokens)
        } else {
            serde_json::Value::Null
        };
        let tokens_out = if self.usage_metrics_available {
            serde_json::json!(self.total_output_tokens)
        } else {
            serde_json::Value::Null
        };

        serde_json::json!({
            // Identity and placement. Together these are what let a consumer
            // reconstruct and act on a row it did not observe itself:
            // `session_id` for `claude --resume <id>`, `cwd` for resuming in
            // the right directory, and `session_name` because `project` below
            // is `display_name()` — which collapses the title and the project
            // into one string and so can't fill both columns on the way back.
            // Without these a consumer can only address a session by pid,
            // which means nothing outside the VM that produced it.
            "session_id": self.session_id,
            "cwd": self.cwd,
            "session_name": self.session_name,
            "started_at": self.started_at,
            // Raw inputs behind the rendered columns, not just the rendered
            // values. `context_pct` below is a computed number, and a consumer
            // that only receives it cannot draw the Context bar, which needs
            // both sides of the ratio. Same for Last and Activity: they are
            // driven by fields that had no representation here at all, so
            // every collected row rendered them empty.
            "context_tokens": self.context_tokens,
            "context_max": self.context_max,
            "last_user_message_ts": self.last_user_message_ts,
            "activity_history": self.activity_history,
            "pid": self.pid,
            "project": self.display_name(),
            "status": self.status.to_string(),
            "telemetry": {
                "state": self.telemetry_status.label(),
                "usage_metrics_available": self.usage_metrics_available,
            },
            "estimate": {
                "verified": !self.cost_estimate_unverified,
                "profile_source": self.model_profile_source,
            },
            "context_pct": context_pct,
            "cost_usd": cost_usd,
            "burn_rate_per_hr": burn_rate,
            "elapsed_secs": self.elapsed.as_secs(),
            "cpu": self.cpu_percent,
            "mem_mb": (self.mem_mb * 100.0).round() / 100.0,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "subagents": self.subagent_count,
            "active_subagents": self.active_subagent_count,
            "subagent_breakdown": self.subagent_breakdown().into_iter().map(|row| {
                serde_json::json!({
                    "label": row.display_label(),
                    "state": row.state_label(),
                    "count": row.count,
                    "tokens_in": if row.usage_metrics_available {
                        serde_json::json!(row.total_input_tokens())
                    } else {
                        serde_json::Value::Null
                    },
                    "tokens_out": if row.usage_metrics_available {
                        serde_json::json!(row.output_tokens)
                    } else {
                        serde_json::Value::Null
                    },
                    "cost_usd": if row.usage_metrics_available {
                        serde_json::json!((row.cost_usd * 100.0).round() / 100.0)
                    } else {
                        serde_json::Value::Null
                    },
                })
            }).collect::<Vec<_>>(),
            "decay_score": if self.usage_metrics_available { serde_json::json!(self.decay_score) } else { serde_json::Value::Null },
            "last_error": self.last_error_message,
            "recent_errors": self.recent_errors.iter().map(|e| {
                serde_json::json!({
                    "tool": e.tool_name,
                    "message": e.message,
                })
            }).collect::<Vec<_>>(),
            "files_modified": self.files_modified,
            "tool_usage": self.tool_usage.iter().map(|(k, v)| {
                (k.clone(), serde_json::json!({"calls": v.calls}))
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
        })
    }

    pub fn format_burn_rate(&self) -> String {
        if !self.usage_metrics_available {
            return "n/a".to_string();
        }
        if self.burn_rate_per_hr < 0.01 {
            return String::from("-");
        }
        if self.burn_rate_per_hr < 1.0 {
            format!(
                "${:.2}/h{}",
                self.burn_rate_per_hr,
                if self.cost_estimate_unverified {
                    "?"
                } else {
                    ""
                }
            )
        } else {
            format!(
                "${:.1}/h{}",
                self.burn_rate_per_hr,
                if self.cost_estimate_unverified {
                    "?"
                } else {
                    ""
                }
            )
        }
    }

    pub fn telemetry_label(&self) -> &'static str {
        self.telemetry_status.label()
    }

    pub fn has_usage_metrics(&self) -> bool {
        self.usage_metrics_available
    }
}

/// Truncate a string to at most `max_bytes` bytes, landing on a valid
/// UTF-8 character boundary. Returns the original string if already short enough.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn subagent_label(path: &Path) -> String {
    let components: Vec<String> = path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect();

    if let Some(tasks_idx) = components.iter().position(|component| component == "tasks") {
        let relative = &components[tasks_idx + 1..];
        if !relative.is_empty() {
            let mut label = relative.join("/");
            if let Some(stripped) = label.strip_suffix(".jsonl") {
                label = stripped.to_string();
            }
            return label;
        }
    }

    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("subagent")
        .to_string()
}

#[cfg(test)]
mod origin_tests {
    use super::*;

    const ALL_STATUSES: [SessionStatus; 7] = [
        SessionStatus::NeedsInput,
        SessionStatus::Compacting,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::Unknown,
        SessionStatus::Idle,
        SessionStatus::Finished,
    ];

    #[test]
    fn status_labels_round_trip() {
        // `from_label` is a hand-written inverse of `Display`. Assert they are
        // actually inverses rather than trusting two match arms to stay in
        // step — a drifted pair would silently render every collected foreign
        // session as Unknown.
        for status in ALL_STATUSES {
            assert_eq!(
                SessionStatus::from_label(&status.to_string()),
                status,
                "round trip failed for {status}"
            );
        }
    }

    #[test]
    fn telemetry_labels_round_trip() {
        for state in [
            TelemetryStatus::Pending,
            TelemetryStatus::Available,
            TelemetryStatus::MissingTranscript,
            TelemetryStatus::UnreadableTranscript,
            TelemetryStatus::UnsupportedTranscript,
        ] {
            assert_eq!(
                TelemetryStatus::from_label(state.label()),
                state,
                "round trip failed for {}",
                state.label()
            );
        }
        assert_eq!(
            TelemetryStatus::from_label("something new"),
            TelemetryStatus::Pending
        );
    }

    #[test]
    fn every_rendered_column_survives_a_round_trip_through_the_snapshot() {
        // The Last / Context / Activity columns rendered blank on every
        // collected row because the fields behind them were never emitted.
        // Assert on the RENDERED output, not just the fields — that is what
        // was actually broken.
        let mut original = ClaudeSession::from_raw(RawSession {
            pid: 4242,
            session_id: "abc-123".into(),
            cwd: "/Users/ndr/repos/linera-infra".into(),
            started_at: 1_700_000_000_000,
            name: Some("fix-scylla-disk".into()),
            name_source: None,
        });
        original.usage_metrics_available = true;
        original.cost_usd = 12.5;
        original.burn_rate_per_hr = 3.25;
        original.total_input_tokens = 1000;
        original.total_output_tokens = 250;
        original.context_tokens = 450_000;
        original.context_max = 1_000_000;
        original.last_user_message_ts = 1_700_000_500_000;
        original.activity_history = vec![0, 3, 7, 2];
        original.telemetry_status = TelemetryStatus::Available;
        original.status = SessionStatus::Processing;

        let collected = original.to_json_value();
        let restored = ClaudeSession::from_snapshot_value("linera-agent-2a14db7ea350", &collected)
            .expect("a session we just serialised must be restorable");

        assert_eq!(restored.last_user_message_ts, original.last_user_message_ts);
        assert_eq!(restored.activity_history, original.activity_history);
        assert_eq!(restored.context_tokens, original.context_tokens);
        assert_eq!(restored.context_max, original.context_max);
        assert_eq!(restored.telemetry_status, original.telemetry_status);
        // Rendered forms must match, not merely the backing numbers.
        assert_eq!(
            restored.format_context_bar(6),
            original.format_context_bar(6)
        );
        assert_eq!(restored.format_sparkline(), original.format_sparkline());
        assert_eq!(restored.format_tokens(), original.format_tokens());
        assert_eq!(restored.format_cost(), original.format_cost());
        assert!(
            restored.context_percent() > 0.0,
            "Context rendered as empty, which is the bug"
        );
    }

    #[test]
    fn negative_control_stripping_the_new_fields_reproduces_the_blank_columns() {
        // Proves the test above discriminates rather than passing vacuously:
        // remove exactly the keys this PR added and the three columns go blank
        // again, which is what the laptop was showing for every sandbox row.
        let mut original = ClaudeSession::from_raw(RawSession {
            pid: 1,
            session_id: "abc-123".into(),
            cwd: "/tmp".into(),
            started_at: 0,
            name: None,
            name_source: None,
        });
        original.usage_metrics_available = true;
        original.cost_usd = 1.0;
        original.context_tokens = 450_000;
        original.context_max = 1_000_000;
        original.last_user_message_ts = 1_700_000_500_000;
        original.activity_history = vec![1, 2, 3];

        let mut collected = original.to_json_value();
        let map = collected.as_object_mut().unwrap();
        for key in [
            "context_tokens",
            "context_max",
            "last_user_message_ts",
            "activity_history",
        ] {
            map.remove(key);
        }

        let restored = ClaudeSession::from_snapshot_value("sbx", &collected).expect("restorable");
        assert_eq!(
            restored.context_percent(),
            0.0,
            "Context would render blank"
        );
        assert_eq!(
            restored.format_sparkline(),
            "-",
            "Activity would render '-'"
        );
        assert_eq!(restored.last_user_message_ts, 0, "Last would render '—'");
    }

    #[test]
    fn a_row_without_metrics_does_not_fabricate_a_context_bar() {
        let bare = serde_json::json!({
            "session_id": "abc-123",
            "pid": 1,
            "status": "Idle",
            "usage_metrics_available": false,
        });
        let restored = ClaudeSession::from_snapshot_value("sbx", &bare).expect("restorable");
        assert!(!restored.usage_metrics_available);
        assert_eq!(restored.context_percent(), 0.0);
        assert_eq!(restored.activity_history.len(), 0);
        assert_eq!(restored.last_user_message_ts, 0);
    }

    #[test]
    fn unknown_status_label_degrades_to_unknown() {
        assert_eq!(
            SessionStatus::from_label("Something From The Future"),
            SessionStatus::Unknown
        );
        assert_eq!(SessionStatus::from_label(""), SessionStatus::Unknown);
    }

    #[test]
    fn origin_label_names_the_local_context() {
        // Inside a sandbox the local origin is that sandbox; on the laptop it
        // has no name of its own.
        assert_eq!(
            SessionOrigin::Local.label(Some("linera-agent-a3f1")),
            "linera-agent-a3f1"
        );
        assert_eq!(SessionOrigin::Local.label(None), "laptop");
        assert_eq!(
            SessionOrigin::Sandbox("linera-agent-251d".into()).label(Some("linera-agent-a3f1")),
            "linera-agent-251d"
        );
    }

    #[test]
    fn only_local_sessions_are_addressable() {
        // The guard that stops us signalling a pid in someone else's namespace.
        assert!(SessionOrigin::Local.is_addressable());
        assert!(!SessionOrigin::Sandbox("elsewhere".into()).is_addressable());
    }

    fn collected(extra: serde_json::Value) -> serde_json::Value {
        let mut base = serde_json::json!({
            "session_id": "abc-123",
            "cwd": "/Users/ndr/repos/linera-infra",
            "session_name": "fix-scylla-disk",
            "started_at": 1_700_000_000_000u64,
            "pid": 4242,
            "project": "fix-scylla-disk",
            "status": "Processing",
        });
        let (base_map, extra_map) = (base.as_object_mut().unwrap(), extra.as_object().unwrap());
        for (key, value) in extra_map {
            base_map.insert(key.clone(), value.clone());
        }
        base
    }

    #[test]
    fn snapshot_value_rebuilds_identity_and_origin() {
        let session = ClaudeSession::from_snapshot_value(
            "linera-agent-251d",
            &collected(serde_json::json!({})),
        )
        .expect("well-formed entry");
        assert_eq!(session.session_id, "abc-123");
        assert_eq!(session.pid, 4242);
        assert_eq!(session.cwd, "/Users/ndr/repos/linera-infra");
        assert_eq!(session.status, SessionStatus::Processing);
        assert_eq!(
            session.origin,
            SessionOrigin::Sandbox("linera-agent-251d".into())
        );
        assert!(!session.origin.is_addressable());
        // `session_name` must survive separately from `project`, or the Name
        // and Project columns collapse into one on the way back.
        assert_eq!(session.display_name(), "fix-scylla-disk");
    }

    #[test]
    fn snapshot_value_without_an_id_is_dropped() {
        let mut value = collected(serde_json::json!({}));
        value.as_object_mut().unwrap().remove("session_id");
        assert!(ClaudeSession::from_snapshot_value("sbx", &value).is_none());
    }

    #[test]
    fn snapshot_value_does_not_invent_uncollected_metrics() {
        // No cost block in the payload means the producing side had no usage
        // metrics. A fabricated 0.0 would render as a real measurement.
        let session = ClaudeSession::from_snapshot_value("sbx", &collected(serde_json::json!({})))
            .expect("well-formed entry");
        assert!(!session.usage_metrics_available);
        assert_eq!(session.cost_usd, 0.0);
        assert_eq!(session.total_input_tokens, 0);
    }

    #[test]
    fn snapshot_value_overlays_collected_metrics() {
        let session = ClaudeSession::from_snapshot_value(
            "sbx",
            &collected(serde_json::json!({
                "cost_usd": 12.5,
                "burn_rate_per_hr": 3.25,
                "tokens_in": 1000,
                "tokens_out": 250,
                "cpu": 7.5,
                "mem_mb": 512.0,
            })),
        )
        .expect("well-formed entry");
        assert!(session.usage_metrics_available);
        assert_eq!(session.cost_usd, 12.5);
        assert_eq!(session.burn_rate_per_hr, 3.25);
        assert_eq!(session.total_input_tokens, 1000);
        assert_eq!(session.total_output_tokens, 250);
        assert_eq!(session.mem_mb, 512.0);
    }

    #[test]
    fn locally_discovered_sessions_are_local() {
        let session = ClaudeSession::from_raw(RawSession {
            pid: 1,
            session_id: "local-1".into(),
            cwd: "/tmp".into(),
            started_at: 0,
            name: None,
            name_source: None,
        });
        assert_eq!(session.origin, SessionOrigin::Local);
        assert!(session.origin.is_addressable());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session() -> ClaudeSession {
        ClaudeSession::from_raw(RawSession {
            pid: 1,
            session_id: "session-1".into(),
            cwd: "/tmp/project".into(),
            started_at: 0,
            name: None,
            name_source: None,
        })
    }

    #[test]
    fn regression_derived_pointer_name_is_not_a_title() {
        // Verbatim pointer shape written by Claude Code 2.1.220 (2026-07-28):
        // the auto-derived placeholder ("<cwd-basename>-<hex>") must not
        // become the session title — displayed it is noise, and recorded it
        // overwrites a real stored /rename title on resume.
        let raw: RawSession = serde_json::from_str(
            r#"{"pid":2647641,"sessionId":"d74ca77e-09ba-42cc-a148-290b6ed2ac98",
                "cwd":"/Users/ndr","startedAt":1785261352874,
                "name":"ndr-5e","nameSource":"derived"}"#,
        )
        .unwrap();
        assert_eq!(raw.name_source.as_deref(), Some("derived"));
        assert!(ClaudeSession::from_raw(raw).session_name.is_empty());
    }

    #[test]
    fn renamed_pointer_title_is_kept() {
        // A /rename title carries no nameSource marker.
        let raw: RawSession = serde_json::from_str(
            r#"{"pid":1018766,"sessionId":"944e1a21-95f9-4630-b422-ef2e9a90876e",
                "cwd":"/Users/ndr","startedAt":1785153171968,
                "name":"detect-bad-slow-validators"}"#,
        )
        .unwrap();
        assert_eq!(
            ClaudeSession::from_raw(raw).session_name,
            "detect-bad-slow-validators"
        );
    }

    #[test]
    fn unfamiliar_name_source_is_trusted() {
        // Only "derived" marks a placeholder; any other (future) source is a
        // real title and must survive.
        let raw: RawSession = serde_json::from_str(
            r#"{"pid":1,"sessionId":"s","cwd":"/x","startedAt":0,
                "name":"my-title","nameSource":"custom"}"#,
        )
        .unwrap();
        assert_eq!(ClaudeSession::from_raw(raw).session_name, "my-title");
    }

    #[test]
    fn subagent_breakdown_groups_completed_and_lists_active_rows() {
        let mut session = make_session();
        let completed = PathBuf::from("/tmp/claude-1/-tmp-project/session-1/tasks/agent-1.jsonl");
        let active =
            PathBuf::from("/tmp/claude-1/-tmp-project/session-1/tasks/nested/agent-2.jsonl");

        session.active_subagent_jsonl_paths = vec![active.clone()];
        session.subagent_rollups.insert(
            completed,
            SubagentRollup {
                input_tokens: 10_000,
                output_tokens: 2_000,
                cost_usd: 0.25,
                usage_metrics_available: true,
                ..SubagentRollup::default()
            },
        );
        session.subagent_rollups.insert(
            active,
            SubagentRollup {
                input_tokens: 40_000,
                output_tokens: 8_000,
                cost_usd: 1.5,
                usage_metrics_available: true,
                ..SubagentRollup::default()
            },
        );

        let rows = session.subagent_breakdown();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].display_label(), "completed");
        assert_eq!(rows[0].state, SubagentState::Completed);
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[0].format_tokens(), "10.0k/2.0k");
        assert_eq!(rows[1].display_label(), "nested/agent-2");
        assert_eq!(rows[1].state, SubagentState::Active);
        assert_eq!(rows[1].format_cost(), "$1.5");
    }

    #[test]
    fn subagent_breakdown_collapses_multiple_completed_rows() {
        let mut session = make_session();

        for name in ["agent-1.jsonl", "agent-2.jsonl"] {
            let path = PathBuf::from(format!("/tmp/claude-1/-tmp-project/session-1/tasks/{name}"));
            session.subagent_rollups.insert(
                path,
                SubagentRollup {
                    input_tokens: 10_000,
                    output_tokens: 1_000,
                    cost_usd: 0.2,
                    usage_metrics_available: true,
                    ..SubagentRollup::default()
                },
            );
        }

        let rows = session.subagent_breakdown();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].display_label(), "completed (2)");
        assert_eq!(rows[0].count, 2);
        assert_eq!(rows[0].format_tokens(), "20.0k/2.0k");
    }

    // ── Cognitive health tracking tests ──────────────────────────────

    #[test]
    fn error_window_flush() {
        let mut s = make_session();
        s.current_window_errors = 3;
        // Call record_activity 5 times to trigger one window flush
        for _ in 0..5 {
            s.record_activity();
        }
        assert_eq!(s.error_counts_per_window.len(), 1);
        assert_eq!(s.error_counts_per_window[0], 3);
        assert_eq!(s.current_window_errors, 0);
        assert_eq!(s.window_tick_counter, 0);
    }

    #[test]
    fn baseline_error_rate_freezes() {
        let mut s = make_session();
        // Simulate 3 windows of errors
        for errors in [2, 3, 4] {
            s.current_window_errors = errors;
            for _ in 0..5 {
                s.record_activity();
            }
        }
        assert_eq!(s.error_counts_per_window.len(), 3);
        let baseline = s.baseline_error_rate.expect("baseline should be set");
        // baseline = (2+3+4)/3 = 3.0
        assert!((baseline - 3.0).abs() < 0.01);

        // Add another window — baseline should NOT change
        s.current_window_errors = 10;
        for _ in 0..5 {
            s.record_activity();
        }
        assert_eq!(s.baseline_error_rate.unwrap(), baseline);
    }
}
