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

/// What the ORIGIN column shows for a session running outside any sandbox.
pub const HOST_LABEL: &str = "host";

impl SessionOrigin {
    /// Label for the ORIGIN column. `Local` needs context we don't hold here —
    /// the caller passes what "here" is called (the sandbox name, or `None`
    /// when running outside one).
    ///
    /// "host" rather than "laptop", which assumed hardware nobody promised, or
    /// "local", which is not a distinction: a sandbox runs on this same machine
    /// and is just as local. Host is what the sandbox runs *on*, and it is the
    /// word the rest of this codebase already uses for that machine —
    /// `SANDBOX_HOST_TTY`, host-shared, host-side, host-native.
    pub fn label(&self, local_name: Option<&str>) -> String {
        match self {
            Self::Local => local_name.unwrap_or(HOST_LABEL).to_string(),
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
pub struct AgentSession {
    pub pid: u32,
    pub session_id: String,
    /// Which agent product this session belongs to.
    pub provider: crate::provider::AgentProvider,
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
    /// CPU used since the previous sample, as a percentage of one core.
    /// `None` until two samples exist — see [`crate::cpu`] for why `ps`'s
    /// `%cpu` column cannot answer this question. Renamed from `cpu_percent`
    /// deliberately: the quantity changed, and every reader had to be revisited.
    pub cpu_rate_percent: Option<f32>,
    /// Whether this session's process was last seen parenting another process.
    /// `None` means not measured, which is not the same as "no children".
    pub has_child_process: Option<bool>,
    /// When [`Self::has_child_process`] was observed.
    pub child_observed_at_ms: u64,
    /// Previous cumulative CPU-time sample, carried across refresh ticks by
    /// `app::merge_discovered_sessions`. Without that hand-off there is never a
    /// pair to difference and the rate stays permanently unknown.
    pub cpu_sample: Option<crate::cpu::CpuSample>,
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

impl AgentSession {
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
            // Discovery only finds Claude today; Codex process discovery sets
            // this from the `ps` row rather than assuming.
            provider: crate::provider::AgentProvider::Claude,
            // Everything reaching `from_raw` came from this VM's own sidecars
            // or process table, so it is by construction local. Foreign
            // sessions are built by `from_registry_entry` instead, which
            // overwrites this.
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
            cpu_rate_percent: None,
            has_child_process: None,
            child_observed_at_ms: 0,
            cpu_sample: None,
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

    /// Build a foreign session row from one hook-written registry entry.
    ///
    /// This is the *membership* constructor: the registry is written by hooks
    /// inside each sandbox onto the host-shared `~/.local/share/claudectl`
    /// mount, so a session appears here the moment it starts and leaves the
    /// moment it deliberately closes — no collector tick in between. The
    /// snapshot is only consulted afterwards, to overlay the two facts
    /// (`cpu`, `mem_mb`) the host genuinely cannot measure for another VM.
    ///
    /// `transcript` is taken as the JSONL path when the entry carries one. It
    /// names a file on the shared `~/.claude` mount, so the host can read it
    /// directly, and doing so here is what lets cost, context and status be
    /// recomputed live for a session the collector has never seen.
    ///
    /// Returns `None` for an entry with no `session_id`: an unidentifiable row
    /// can be neither resumed nor de-duplicated against local discovery, so
    /// showing it would be worse than omitting it.
    pub fn from_registry_entry(
        sandbox: &str,
        entry: &crate::sandbox_registry::SessionEntry,
    ) -> Option<Self> {
        if entry.session_id.is_empty() {
            return None;
        }
        let raw = RawSession {
            pid: entry.pid.unwrap_or(0),
            session_id: entry.session_id.clone(),
            cwd: entry.cwd.clone(),
            started_at: entry.started_at_ms,
            // The registry's `name` has already been through the placeholder
            // filter on the writing side, so pass it through with no source
            // marker rather than letting `title()` filter it a second time.
            name: entry.name.clone(),
            name_source: None,
        };
        let mut session = Self::from_raw(raw);
        // `from_raw` stamps Claude; a registry entry knows better. Without this
        // a restored Codex row would be resumed with `claude --resume <ULID>`.
        session.provider = entry.provider;
        session.origin = SessionOrigin::Sandbox(sandbox.to_string());
        if !entry.transcript.is_empty() {
            session.jsonl_path = Some(PathBuf::from(&entry.transcript));
        }
        // The only host-meaningful routing this entry carries. `pid` is the
        // sandbox's own namespace and `cwd` is a path inside it, so without
        // these the terminal matchers have nothing to key on and fall through
        // to matching every surface sitting in `$HOME`.
        session.terminal_id = entry.host_terminal_id.clone();
        if let Some(tty) = &entry.host_tty {
            session.tty = tty.clone();
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

    /// CPU used since the previous sample, as a percentage of one core.
    ///
    /// `-` when no rate has been measured yet — one refresh tick after a
    /// session appears, and for a sandbox session until two collector passes
    /// have run. Rendering an unmeasured session as `0.0` would be a claim
    /// nothing supports.
    pub fn format_cpu(&self) -> String {
        match self.cpu_rate_percent {
            Some(rate) => format!("{rate:.1}"),
            None => String::from("-"),
        }
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
            // Which product the row came from: a consumer resuming it needs to
            // know whether to invoke claude or codex.
            "provider": self.provider.label(),
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
            // null rather than 0 when the rate has not been measured yet:
            // consumers must be able to tell "idle" from "not known".
            "cpu": self.cpu_rate_percent,
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

    fn registry_entry(
        session_id: &str,
        name: Option<&str>,
    ) -> crate::sandbox_registry::SessionEntry {
        crate::sandbox_registry::SessionEntry {
            session_id: session_id.to_string(),
            cwd: "/Users/ndr/repos/linera-infra".to_string(),
            transcript: String::new(),
            started_at_ms: 1_700_000_000_000,
            name: name.map(str::to_string),
            pid: Some(4242),
            owner_pid: None,
            owner_started_at: None,
            ..Default::default()
        }
    }

    #[test]
    fn registry_entry_rebuilds_identity_and_marks_the_origin_foreign() {
        let session = AgentSession::from_registry_entry(
            "linera-agent-251d",
            &registry_entry("abc-123", Some("fix-scylla-disk")),
        )
        .expect("well-formed entry");
        assert_eq!(session.session_id, "abc-123");
        assert_eq!(session.pid, 4242);
        assert_eq!(session.cwd, "/Users/ndr/repos/linera-infra");
        assert_eq!(
            session.origin,
            SessionOrigin::Sandbox("linera-agent-251d".into())
        );
        // The pid belongs to another VM; signalling it here would hit an
        // unrelated local process.
        assert!(!session.origin.is_addressable());
        assert_eq!(session.display_name(), "fix-scylla-disk");
    }

    #[test]
    fn registry_entry_without_an_id_is_dropped() {
        assert!(
            AgentSession::from_registry_entry("sbx", &registry_entry("", Some("nameless")))
                .is_none()
        );
    }

    #[test]
    fn an_unnamed_entry_still_renders_under_its_project() {
        let session = AgentSession::from_registry_entry("sbx", &registry_entry("abc-123", None))
            .expect("an unnamed session is still identifiable");
        assert_eq!(session.session_name, "");
        assert_eq!(session.display_name(), "linera-infra");
    }

    #[test]
    fn the_recorded_transcript_becomes_the_jsonl_path() {
        // This is what lets the host recompute cost, context and status for a
        // session it has never collected: the transcript lives on the shared
        // `~/.claude` mount, so naming it here is the whole handoff to
        // `monitor::update_tokens`.
        let mut entry = registry_entry("abc-123", Some("named"));
        entry.transcript = "/Users/ndr/.claude/projects/-Users-ndr/abc-123.jsonl".to_string();
        let session = AgentSession::from_registry_entry("sbx", &entry).expect("well-formed entry");
        assert_eq!(
            session.jsonl_path.as_deref(),
            Some(Path::new(
                "/Users/ndr/.claude/projects/-Users-ndr/abc-123.jsonl"
            ))
        );
    }

    #[test]
    fn an_entry_without_a_transcript_leaves_the_path_for_discovery_to_resolve() {
        // `do_refresh_io` falls back to `resolve_jsonl_paths` when this is
        // None; inventing a path here would defeat that.
        let session = AgentSession::from_registry_entry("sbx", &registry_entry("abc-123", None))
            .expect("well-formed entry");
        assert!(session.jsonl_path.is_none());
    }

    /// The host routing keys are the one thing in the entry that means anything
    /// outside the sandbox that wrote it — `pid` is a container pid and `cwd` a
    /// container path. Dropping them here left the Ghostty matcher with nothing
    /// to key on, so Tab fell through to "every surface sitting in $HOME".
    #[test]
    fn registry_entry_carries_host_terminal_routing_onto_the_session() {
        let mut entry = registry_entry("abc-123", None);
        entry.host_terminal_id = Some("8161D3F2-17C1-40CA-814F-4D714DB8F7BC".to_string());
        entry.host_tty = Some("/dev/ttys031".to_string());

        let session = AgentSession::from_registry_entry("sbx", &entry).expect("well-formed entry");
        assert_eq!(
            session.terminal_id.as_deref(),
            Some("8161D3F2-17C1-40CA-814F-4D714DB8F7BC")
        );
        assert_eq!(session.tty, "/dev/ttys031");
    }

    /// An entry written by an older sandbox has neither field. It must not
    /// invent a tty — an empty one falls through to the cwd chain, while a
    /// fabricated one would match the wrong surface (or none).
    #[test]
    fn registry_entry_without_routing_leaves_the_session_unrouted() {
        let session = AgentSession::from_registry_entry("sbx", &registry_entry("abc-123", None))
            .expect("well-formed entry");
        assert!(session.terminal_id.is_none());
        assert!(session.tty.is_empty());
    }

    #[test]
    fn registry_entry_does_not_fabricate_metrics() {
        // The registry carries identity only. A zero cost rendered as a real
        // measurement would be worse than a blank cell.
        let session = AgentSession::from_registry_entry("sbx", &registry_entry("abc-123", None))
            .expect("well-formed entry");
        assert!(!session.usage_metrics_available);
        assert_eq!(session.cost_usd, 0.0);
        assert_eq!(session.total_input_tokens, 0);
        assert_eq!(session.context_percent(), 0.0);
    }

    #[test]
    fn origin_label_names_the_local_context() {
        // Inside a sandbox the local origin is that sandbox; on the laptop it
        // has no name of its own.
        assert_eq!(
            SessionOrigin::Local.label(Some("linera-agent-a3f1")),
            "linera-agent-a3f1"
        );
        assert_eq!(
            SessionOrigin::Local.label(None),
            "host",
            "not 'laptop' (assumes hardware) and not 'local' (a sandbox is local too)"
        );
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

    #[test]
    fn locally_discovered_sessions_are_local() {
        let session = AgentSession::from_raw(RawSession {
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

    fn make_session() -> AgentSession {
        AgentSession::from_raw(RawSession {
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
        assert!(AgentSession::from_raw(raw).session_name.is_empty());
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
            AgentSession::from_raw(raw).session_name,
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
        assert_eq!(AgentSession::from_raw(raw).session_name, "my-title");
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
