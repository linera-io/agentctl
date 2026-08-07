use std::collections::HashMap;
use std::time::Duration;

use crate::session::{ClaudeSession, RawSession, SessionStatus, TelemetryStatus, ToolStats};

/// Fake project definitions for demo mode.
const PROJECTS: &[(&str, &str, &str)] = &[
    ("acme-api", "/Users/dev/projects/acme-api", "opus-4.6"),
    ("acme-api", "/Users/dev/projects/acme-api", "opus-4.6"),
    (
        "web-frontend",
        "/Users/dev/projects/web-frontend",
        "sonnet-4.6",
    ),
    ("ml-pipeline", "/Users/dev/projects/ml-pipeline", "opus-4.6"),
    (
        "ml-pipeline",
        "/Users/dev/worktrees/ml-pipeline-feat",
        "sonnet-4.6",
    ),
    (
        "infra-terraform",
        "/Users/dev/projects/infra-terraform",
        "haiku",
    ),
    ("docs-site", "/Users/dev/projects/docs-site", "sonnet-4.6"),
    ("mobile-app", "/Users/dev/projects/mobile-app", "opus-4.6"),
];

/// Deterministic status progression per session (cycles through these).
const STATUS_SEQUENCES: &[&[SessionStatus]] = &[
    &[
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::NeedsInput,
        SessionStatus::NeedsInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
    ],
    &[
        SessionStatus::Processing,
        SessionStatus::NeedsInput,
        SessionStatus::NeedsInput,
        SessionStatus::NeedsInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
    ],
    &[
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::NeedsInput,
        SessionStatus::Processing,
    ],
    &[
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
    ],
    &[
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::NeedsInput,
    ],
    &[
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::NeedsInput,
        SessionStatus::Processing,
    ],
    &[
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::WaitingInput,
    ],
    &[
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::NeedsInput,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::Processing,
        SessionStatus::WaitingInput,
        SessionStatus::Processing,
    ],
];

/// Pending tool calls assigned to NeedsInput sessions by index.
/// (tool_name, command/input_summary)
const PENDING_TOOLS: &[(&str, &str)] = &[
    ("Bash", "cargo test --workspace"),
    ("Bash", "cargo clippy -- -D warnings"),
    ("Bash", "npm run build && npm test"),
    ("Bash", "python train.py --epochs 50"),
    ("Bash", "rm -rf /tmp/cache && rm -rf node_modules"),
    ("Bash", "terraform apply -auto-approve"),
    ("Bash", "npm run deploy -- --prod"),
    ("Bash", "git push --force origin main"),
];

/// Generate deterministic fake sessions for demo mode.
pub fn generate_sessions(tick: u32) -> Vec<ClaudeSession> {
    let base_pid = 10000u32;

    PROJECTS
        .iter()
        .enumerate()
        .map(|(i, (name, cwd, model))| {
            let pid = base_pid + (i as u32 * 1111);
            let raw = RawSession {
                pid,
                session_id: format!("demo-{:04x}-{:04x}-{:04x}", i, i * 7, i * 13),
                cwd: cwd.to_string(),
                started_at: 0, // Will be overridden
                name: None,
                name_source: None,
            };
            let mut s = ClaudeSession::from_raw(raw);
            s.project_name = name.to_string();
            s.model = model.to_string();
            s.telemetry_status = TelemetryStatus::Available;
            s.usage_metrics_available = true;

            // Deterministic status from sequence
            let seq = STATUS_SEQUENCES[i % STATUS_SEQUENCES.len()];
            s.status = seq[(tick as usize) % seq.len()];

            // Deterministic metrics that grow over time
            let base_tokens = (i as u64 + 1) * 50_000 + (tick as u64) * 2_000;
            s.total_input_tokens = base_tokens;
            s.total_output_tokens = base_tokens / 5;
            s.cache_read_tokens = base_tokens / 3;
            s.cache_write_tokens = base_tokens / 10;

            // Context grows over time, different rates per session
            let ctx_rate = 0.3 + (i as f64 * 0.08);
            let ctx_pct = ((tick as f64 * ctx_rate) % 95.0) + 5.0;
            s.context_max = crate::monitor::model_context_max(model);
            s.context_tokens = (s.context_max as f64 * ctx_pct / 100.0) as u64;

            // Cost grows over time
            s.cost_usd = (i as f64 + 1.0) * 0.15 + (tick as f64) * 0.03 * (i as f64 + 1.0);
            s.burn_rate_per_hr = if matches!(s.status, SessionStatus::Processing) {
                2.0 + (i as f64 * 0.8)
            } else {
                0.0
            };

            // Elapsed time
            let base_elapsed = (i as u64 + 1) * 300 + tick as u64 * 2;
            s.elapsed = Duration::from_secs(base_elapsed);

            // CPU/MEM
            s.cpu_rate_percent = Some(match s.status {
                SessionStatus::Processing => 15.0 + (i as f32 * 3.0),
                SessionStatus::NeedsInput => 0.3,
                _ => 0.8,
            });
            s.mem_mb = 200.0 + (i as f64 * 50.0);

            // Subagents for some sessions
            if i == 0 || i == 3 {
                s.subagent_count = 2 + (tick as usize % 3);
            }

            // Activity sparkline history
            for t in 0..15 {
                let past_tick = if tick > 15 { tick - 15 + t } else { t };
                let past_status = seq[(past_tick as usize) % seq.len()];
                let level = match past_status {
                    SessionStatus::Processing => 7,
                    SessionStatus::Compacting => 5,
                    SessionStatus::NeedsInput => 4,
                    SessionStatus::WaitingInput => 2,
                    SessionStatus::Unknown => 2,
                    SessionStatus::Idle => 1,
                    SessionStatus::Finished => 0,
                };
                s.activity_history.push(level);
            }

            // Worktree IDs — sessions 0 and 1 share same worktree (conflict!)
            // Session 4 is a worktree of project 3's repo (no conflict)
            s.worktree_id = Some(cwd.to_string());

            // Tool usage for detail panel
            if tick > 3 {
                let mut tools = HashMap::new();
                tools.insert(
                    "Bash".to_string(),
                    ToolStats {
                        calls: 12 + (i as u32 * 3),
                    },
                );
                tools.insert(
                    "Read".to_string(),
                    ToolStats {
                        calls: 25 + (i as u32 * 5),
                    },
                );
                tools.insert(
                    "Edit".to_string(),
                    ToolStats {
                        calls: 8 + (i as u32 * 2),
                    },
                );
                tools.insert(
                    "Grep".to_string(),
                    ToolStats {
                        calls: 6 + (i as u32),
                    },
                );
                s.tool_usage = tools;
            }

            // File changes for detail panel
            if tick > 5 {
                let mut files = HashMap::new();
                files.insert(format!("/Users/dev/projects/{name}/src/main.rs"), 3);
                files.insert(format!("/Users/dev/projects/{name}/src/lib.rs"), 1);
                if i % 2 == 0 {
                    files.insert(format!("/Users/dev/projects/{name}/Cargo.toml"), 1);
                }
                s.files_modified = files;
            }

            // ── Health-triggering overrides ──────────────────────────────

            // Session 2 (web-frontend): Low cache hit ratio → 🔥 critical
            if i == 2 {
                s.total_input_tokens = 120_000 + (tick as u64 * 3_000);
                s.cache_read_tokens = 5_000; // ~4% — well under 10% critical threshold
                s.cache_write_tokens = 2_000;
            }

            // Session 3 (ml-pipeline): Context saturation → 🧠 critical
            if i == 3 && tick > 6 {
                let saturation = 0.91 + ((tick as f64 - 6.0) * 0.005).min(0.07);
                s.context_tokens = (s.context_max as f64 * saturation) as u64;
            }

            // Session 5 (infra-terraform): Stalled → 🐌 warning
            // High cost, long elapsed, but NO file edits
            if i == 5 {
                s.cost_usd = 7.50 + (tick as f64) * 0.12;
                s.elapsed = Duration::from_secs(900 + tick as u64 * 5);
                s.files_modified.clear(); // Zero file edits despite spending
            }

            // Session 7 (mobile-app): Cost spike → 💸
            if i == 7 && tick > 4 {
                // Base cost accumulates slowly, then burn rate spikes
                s.cost_usd = 3.0 + (tick as f64) * 0.05;
                let elapsed_hrs = s.elapsed.as_secs_f64() / 3600.0;
                let avg_rate = if elapsed_hrs > 0.01 {
                    s.cost_usd / elapsed_hrs
                } else {
                    1.0
                };
                // Spike burn rate to 6x average
                s.burn_rate_per_hr = avg_rate * 6.0;
            }

            // Session 3 (ml-pipeline): Severe cognitive decay (⊘) — high context + all signals
            if i == 3 && tick > 4 {
                s.baseline_tokens_per_edit = Some(4000.0);
                s.edit_event_count = 15;
                s.total_tokens_at_edit_count = 15 * 12_000; // 12k/edit vs 4k baseline = 3x
                s.error_counts_per_window = vec![0, 1, 1, 3, 5, 6, 8];
                s.baseline_error_rate = Some(0.7);
                s.file_reads_since_edit.insert("src/pipeline.rs".into(), 5);
                s.file_reads_since_edit
                    .insert("src/data_loader.rs".into(), 3);
            }

            // Session 0 (acme-api): Early cognitive decay (◐) — moderate context + some signals
            if i == 0 && tick > 8 {
                s.baseline_tokens_per_edit = Some(5000.0);
                s.edit_event_count = 10;
                s.total_tokens_at_edit_count = 10 * 7_500; // 7.5k/edit vs 5k baseline = 1.5x
                s.error_counts_per_window = vec![1, 1, 2, 2, 3];
                s.baseline_error_rate = Some(1.3);
                s.file_reads_since_edit.insert("src/main.rs".into(), 3);
                // Push context to ~65% for moderate decay
                let ctx_pct = 0.60 + ((tick as f64 - 8.0) * 0.01).min(0.15);
                s.context_tokens = (s.context_max as f64 * ctx_pct) as u64;
            }

            // Session 4 (ml-pipeline worktree): Loop detection → 🔄
            if i == 4 && tick > 5 {
                s.last_tool_error = true;
                s.tool_usage
                    .entry("Bash".to_string())
                    .and_modify(|t| t.calls = 15 + (tick % 5))
                    .or_insert(ToolStats {
                        calls: 15 + (tick % 5),
                    });
            }

            // ── Pending tool info for rules/brain ────────────────────────

            if s.status == SessionStatus::NeedsInput {
                let (tool, cmd) = PENDING_TOOLS[i % PENDING_TOOLS.len()];
                s.pending_tool_name = Some(tool.to_string());
                s.pending_tool_input = Some(cmd.to_string());
            }

            s
        })
        .collect()
}

/// Scripted demo events that simulate rules, brain, and routing actions.
/// Returns a status message for specific ticks, cycling every CYCLE_LEN ticks.
pub fn demo_event(tick: u32) -> Option<DemoEvent> {
    const CYCLE_LEN: u32 = 24;
    let phase = tick % CYCLE_LEN;

    match phase {
        // Brain auto-approve (show the brain working)
        2 => Some(DemoEvent {
            message: "Brain: auto-approved Bash(cargo test --workspace) for acme-api [92%]"
                .into(),
            kind: EventKind::BrainSuggestion,
        }),

        // Rule firing
        4 => Some(DemoEvent {
            message:
                "Rule 'deny-rm-rf': denied ml-pipeline (Bash: rm -rf /tmp/cache)"
                    .into(),
            kind: EventKind::RuleAction,
        }),

        // Brain deny
        6 => Some(DemoEvent {
            message: "Brain: denied Bash(terraform apply -auto-approve) for infra-terraform — destructive without plan review [87%]".into(),
            kind: EventKind::BrainSuggestion,
        }),

        // Cognitive decay alert
        8 => Some(DemoEvent {
            message: "Health: ml-pipeline cognitive decay at 82/100 — session degrading, consider restart".into(),
            kind: EventKind::HealthAlert,
        }),

        // Brain auto-approve
        10 => Some(DemoEvent {
            message: "Brain: auto-approved Bash(npm run build && npm test) for web-frontend [95%]"
                .into(),
            kind: EventKind::BrainSuggestion,
        }),

        // Brain override by deny rule
        12 => Some(DemoEvent {
            message:
                "Brain suggested approve, but deny rule 'deny-force-push' overrides (git push --force)"
                    .into(),
            kind: EventKind::BrainOverride,
        }),

        // Brain learning signal
        14 => Some(DemoEvent {
            message: "Brain: auto-approved Edit(src/auth.rs) for acme-api — learned from 8 prior approvals [88%]"
                .into(),
            kind: EventKind::BrainSuggestion,
        }),

        // Inter-session routing
        16 => Some(DemoEvent {
            message: "Routed summary from ml-pipeline → docs-site: \"Added training pipeline with checkpoint support\"".into(),
            kind: EventKind::Route,
        }),

        // Stall alert
        18 => Some(DemoEvent {
            message: "Health: infra-terraform stalled — $8.40 spent, 16 min, no file edits".into(),
            kind: EventKind::HealthAlert,
        }),

        // Brain approve with context
        20 => Some(DemoEvent {
            message: "Brain: auto-approved Bash(cargo clippy -- -D warnings) for acme-api [94%]"
                .into(),
            kind: EventKind::BrainSuggestion,
        }),

        // Context saturation alert
        22 => Some(DemoEvent {
            message: "Health: ml-pipeline context at 94% — auto-restart checkpoint saved".into(),
            kind: EventKind::HealthAlert,
        }),

        _ => None,
    }
}

/// A scripted event in the demo timeline.
pub struct DemoEvent {
    pub message: String,
    pub kind: EventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    RuleAction,
    BrainSuggestion,
    BrainOverride,
    Route,
    HealthAlert,
}

/// Generate demo rules for display.
pub fn demo_rules() -> Vec<crate::rules::AutoRule> {
    use crate::rules::{AutoRule, RuleAction};

    vec![
        {
            let mut r = AutoRule::new("approve-cargo".into(), RuleAction::Approve);
            r.match_tool = vec!["Bash".into()];
            r.match_command = vec!["cargo".into()];
            r
        },
        {
            let mut r = AutoRule::new("deny-rm-rf".into(), RuleAction::Deny);
            r.match_command = vec!["rm -rf".into()];
            r
        },
        {
            let mut r = AutoRule::new("deny-force-push".into(), RuleAction::Deny);
            r.match_command = vec!["--force".into()];
            r
        },
        {
            let mut r = AutoRule::new("kill-runaway".into(), RuleAction::Terminate);
            r.match_cost_above = Some(20.0);
            r
        },
    ]
}
