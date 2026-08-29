#![allow(dead_code)]

use std::collections::HashMap;

use super::decisions::{DecisionRecord, DistilledPreferences};
use super::insights::{Insight, InsightCategory, InsightSeverity, epoch_now};
use super::preferences::{PreferenceCondition, PreferencePattern};
use crate::rules::RuleAction;

// ────────────────────────────────────────────────────────────────────────────
// Detection algorithms
// ────────────────────────────────────────────────────────────────────────────

/// Extract a command keyword for grouping (first two tokens).
/// Duplicated from decisions.rs because that function is private.
pub(crate) fn extract_command_keyword(command: Option<&str>) -> Option<String> {
    let cmd = command?.trim();
    if cmd.is_empty() {
        return None;
    }
    let tokens: Vec<&str> = cmd.split_whitespace().take(2).collect();
    Some(tokens.join(" "))
}

/// Detect tools/commands that are repeatedly rejected by the user.
pub(crate) fn detect_friction_patterns(decisions: &[DecisionRecord]) -> Vec<Insight> {
    let mut groups: HashMap<(String, Option<String>), (u32, u32)> = HashMap::new();

    for d in decisions {
        let tool = d.tool.clone().unwrap_or_else(|| "*".to_string());
        let cmd = extract_command_keyword(d.command.as_deref());
        let key = (tool, cmd);
        let entry = groups.entry(key).or_insert((0, 0));
        entry.0 += 1; // total
        if d.is_negative() {
            entry.1 += 1; // rejected
        }
    }

    let now = epoch_now();
    let mut insights = Vec::new();

    for ((tool, cmd), (total, rejected)) in &groups {
        if *rejected < 3 || *total < 3 {
            continue;
        }
        let rejection_rate = *rejected as f64 / *total as f64;
        if rejection_rate < 0.6 {
            continue;
        }

        let cmd_part = cmd
            .as_ref()
            .map(|c| format!(" \"{c}\""))
            .unwrap_or_default();

        let severity = if rejection_rate >= 0.9 {
            InsightSeverity::Warning
        } else {
            InsightSeverity::Suggestion
        };

        insights.push(Insight {
            fingerprint: format!("friction:{}:{}", tool, cmd.as_deref().unwrap_or("*")),
            generated_at: now,
            category: InsightCategory::FrictionPattern,
            severity,
            summary: format!(
                "[{tool}]{cmd_part} rejected {rejected}/{total} times ({:.0}%)",
                rejection_rate * 100.0
            ),
            suggestion: Some(format!("consider adding deny rule for [{tool}]{cmd_part}")),
            evidence_count: *total,
        });
    }

    insights
}

/// Detect repeated errors from the same tool across sessions.
pub(crate) fn detect_error_loops(decisions: &[DecisionRecord]) -> Vec<Insight> {
    // Group by PID, then find consecutive errors for same tool
    let mut pid_groups: HashMap<u32, Vec<&DecisionRecord>> = HashMap::new();
    for d in decisions {
        pid_groups.entry(d.pid).or_default().push(d);
    }

    // Count how many sessions had error loops for each (tool, cmd) combo
    let mut loop_counts: HashMap<(String, Option<String>), u32> = HashMap::new();

    for session_decisions in pid_groups.values() {
        let mut streak_tool: Option<String> = None;
        let mut streak_cmd: Option<String> = None;
        let mut streak_count: u32 = 0;

        for d in session_decisions {
            let has_error = d
                .context
                .as_ref()
                .map(|c| c.last_tool_error)
                .unwrap_or(false);
            let tool = d.tool.clone().unwrap_or_default();
            let cmd = extract_command_keyword(d.command.as_deref());

            if has_error && Some(&tool) == streak_tool.as_ref() {
                streak_count += 1;
            } else if has_error {
                // New error streak
                streak_tool = Some(tool.clone());
                streak_cmd = cmd.clone();
                streak_count = 1;
            } else {
                // No error — check if previous streak was long enough
                if streak_count >= 3 {
                    if let Some(ref t) = streak_tool {
                        *loop_counts
                            .entry((t.clone(), streak_cmd.clone()))
                            .or_insert(0) += 1;
                    }
                }
                streak_tool = None;
                streak_cmd = None;
                streak_count = 0;
            }
        }
        // Check trailing streak
        if streak_count >= 3 {
            if let Some(ref t) = streak_tool {
                *loop_counts
                    .entry((t.clone(), streak_cmd.clone()))
                    .or_insert(0) += 1;
            }
        }
    }

    let now = epoch_now();
    loop_counts
        .into_iter()
        .filter(|(_, count)| *count >= 1)
        .map(|((tool, cmd), count)| {
            let cmd_part = cmd
                .as_ref()
                .map(|c| format!(" \"{c}\""))
                .unwrap_or_default();
            Insight {
                fingerprint: format!("error_loop:{}:{}", tool, cmd.as_deref().unwrap_or("*")),
                generated_at: now,
                category: InsightCategory::ErrorLoop,
                severity: if count >= 3 {
                    InsightSeverity::Warning
                } else {
                    InsightSeverity::Suggestion
                },
                summary: format!(
                    "[{tool}]{cmd_part} hit 3+ consecutive errors in {count} session(s)"
                ),
                suggestion: Some(format!("investigate why [{tool}]{cmd_part} keeps failing")),
                evidence_count: count,
            }
        })
        .collect()
}

/// Detect sessions frequently hitting high context usage.
pub(crate) fn detect_context_blowouts(decisions: &[DecisionRecord]) -> Vec<Insight> {
    // Group by PID, check if any decision in session had context > 80%
    let mut pid_max_context: HashMap<u32, u8> = HashMap::new();
    for d in decisions {
        if let Some(ref ctx) = d.context {
            let entry = pid_max_context.entry(d.pid).or_insert(0);
            if ctx.context_pct > *entry {
                *entry = ctx.context_pct;
            }
        }
    }

    if pid_max_context.is_empty() {
        return Vec::new();
    }

    // Only look at recent sessions (last 20 PIDs by insertion order)
    let recent: Vec<u8> = pid_max_context
        .values()
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(20)
        .collect();

    let blowout_count = recent.iter().filter(|&&pct| pct > 80).count();
    let total = recent.len();
    let blowout_rate = blowout_count as f64 / total as f64;

    if blowout_rate < 0.4 || blowout_count < 2 {
        return Vec::new();
    }

    vec![Insight {
        fingerprint: "context_blowout:global".to_string(),
        generated_at: epoch_now(),
        category: InsightCategory::ContextBlowout,
        severity: if blowout_rate >= 0.7 {
            InsightSeverity::Warning
        } else {
            InsightSeverity::Suggestion
        },
        summary: format!(
            "Context >80% in {blowout_count}/{total} recent sessions ({:.0}%)",
            blowout_rate * 100.0
        ),
        suggestion: Some("consider earlier /compact when context approaches 70%".to_string()),
        evidence_count: blowout_count as u32,
    }]
}

/// A table name unique to the (tool, command) pair a suggestion describes.
///
/// Keying on the tool alone collided: one insight is emitted per pattern, so
/// three separate Bash denials all suggested `[rules.deny-bash]`. Pasting them
/// leaves ONE rule matching only the last, because `ensure_rule` reuses a rule
/// by name and every matcher is last-write-wins — so two of the three denials
/// the user believes they added silently do not exist.
/// Slugifying alone is not enough: it is lossy, so it collides. `rm -rf` and
/// `rm/rf` both reduce to `bash-rm-rf`, and any all-punctuation command —
/// realistic, since `command_pattern` is only the first two tokens — reduces to
/// bare `bash`. A short digest of the untouched command restores uniqueness
/// while the readable part keeps the name meaningful.
fn rule_slug(tool: &str, command: Option<&str>) -> String {
    let readable = |text: &str| {
        let mut out = String::with_capacity(text.len());
        let mut pending_dash = false;
        for ch in text.to_lowercase().chars() {
            if ch.is_ascii_alphanumeric() {
                out.push(ch);
                pending_dash = false;
            } else if !pending_dash {
                out.push('-');
                pending_dash = true;
            }
        }
        out.trim_matches('-').to_string()
    };

    let mut slug = readable(tool);
    if slug.is_empty() {
        slug.push_str("rule");
    }
    let Some(command) = command else {
        return slug;
    };

    let stem = readable(command);
    if !stem.is_empty() {
        slug.push('-');
        // Cap the readable part: `command_pattern` is two tokens, but nothing
        // bounds their length, and a table name is read by a human.
        slug.push_str(stem.get(..32).unwrap_or(&stem).trim_matches('-'));
    }
    slug.push('-');
    slug.push_str(&short_digest(command));
    slug
}

/// Characters the config reader cannot round-trip: `,` splits an array, `#`
/// truncates the line, and quotes are stripped.
const UNQUOTABLE: [char; 4] = [',', '#', '"', '\''];

/// The rule matcher expressing this condition, or `None` if the rule language
/// has no equivalent (context level and time-of-day have no matcher).
fn condition_matcher(condition: &PreferenceCondition) -> Option<String> {
    match condition {
        PreferenceCondition::CostAbove(usd) => Some(format!("match_cost_above = {usd}")),
        PreferenceCondition::NoErrors => Some("match_last_error = false".to_string()),
        PreferenceCondition::HasErrors => Some("match_last_error = true".to_string()),
        PreferenceCondition::NoFileConflict => Some("match_file_conflict = false".to_string()),
        PreferenceCondition::HasFileConflict => Some("match_file_conflict = true".to_string()),
        PreferenceCondition::CostBelow(_)
        | PreferenceCondition::ContextBelow(_)
        | PreferenceCondition::ContextAbove(_)
        | PreferenceCondition::HourRange(_, _) => None,
    }
}

/// Paste-ready TOML for one detected pattern, or prose where no exact rule can
/// be produced — the user pastes this verbatim, so a near-miss is worse than none.
fn rule_suggestion(action: &str, pattern: &PreferencePattern) -> String {
    let target = crate::product::project_config(std::path::Path::new("."));
    let command = pattern.command_pattern.as_deref();

    // Five of the nine conditions have an exact matcher; a pattern carrying any
    // of the rest cannot be stated as a rule, and dropping it silently would
    // widen the rule to cases the user never agreed to.
    let Some(condition_lines) = pattern
        .conditions
        .iter()
        .map(condition_matcher)
        .collect::<Option<Vec<_>>>()
    else {
        return format!(
            "observed only under a condition auto-rules cannot express — review \
             before adding a rule for [{}]",
            pattern.tool
        );
    };

    // `distill_preferences` uses "*" when a decision carries no tool, but
    // `match_tool` is an exact-equality test: nothing is named "*", so the rule
    // could never fire, and omitting the matcher would match every tool.
    if pattern.tool == "*" {
        return format!(
            "seen across tools rather than one — an auto-rule has to name a \
             single tool, so add this{} by hand",
            command.map(|c| format!(" for {c:?}")).unwrap_or_default()
        );
    }

    if pattern.tool.contains(UNQUOTABLE) || command.is_some_and(|c| c.contains(UNQUOTABLE)) {
        return format!(
            "add a rule for [{}]{} by hand — it contains a character the config \
             reader cannot round-trip",
            pattern.tool,
            command
                .map(|c| format!(" matching {c:?}"))
                .unwrap_or_default()
        );
    }

    // The engine's wildcard is an OMITTED matcher; `["*"]` is a live substring
    // test for an asterisk, so it never fires and a deny would fail open.
    let command_line = command
        .map(|c| format!("\nmatch_command = [\"{c}\"]"))
        .unwrap_or_default();
    let condition_block = condition_lines
        .iter()
        .map(|l| format!("\n{l}"))
        .collect::<String>();

    // `match_command` is a substring test and the pattern is only the command's
    // first two tokens, so an approve covers more than was ever observed. A
    // full-line comment survives the parser, so the warning travels with paste.
    let caveat = match (action, command) {
        ("approve", Some(c)) => format!("# also matches any command containing {c:?}\n"),
        _ => String::new(),
    };

    format!(
        "add to {}:\n{caveat}[rules.{}-{}]\nmatch_tool = [\"{}\"]{command_line}{condition_block}\naction = \"{action}\"",
        target.display(),
        action,
        rule_slug(&pattern.tool, command),
        pattern.tool,
    )
}

/// FNV-1a, four hex chars. Deterministic across runs so a suggestion the user
/// already pasted keeps the same name; not a security hash.
fn short_digest(text: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in text.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{:04x}", hash & 0xffff)
}

/// Detect high-confidence patterns that could become AutoRules.
pub(crate) fn detect_missing_rules(
    _decisions: &[DecisionRecord],
    prefs: &DistilledPreferences,
) -> Vec<Insight> {
    let now = epoch_now();
    let mut insights = Vec::new();

    for p in &prefs.patterns {
        if p.sample_count < 5 || p.confidence < 0.8 {
            continue;
        }

        // `accept_rate` measures agreement with the BRAIN, not a wish for the
        // call to proceed — both extremes are decisive, and `preferred_action`
        // is what they already resolved to.
        let consistency = p.accept_rate.max(1.0 - p.accept_rate);
        if consistency < 0.9 {
            continue;
        }

        // Only approve and deny are safe to hand over: `send` would type the
        // engine's "continue" default, `terminate`/`kill` destroy sessions, and
        // route/spawn/delegate have no rule form.
        let action = match RuleAction::parse(&p.preferred_action) {
            Some(a @ (RuleAction::Approve | RuleAction::Deny)) => a.label(),
            _ => continue,
        };

        let cmd_part = p
            .command_pattern
            .as_ref()
            .map(|c| format!(" \"{c}\""))
            .unwrap_or_default();

        insights.push(Insight {
            fingerprint: format!(
                "missing_rule:{action}:{}:{}",
                p.tool,
                p.command_pattern.as_deref().unwrap_or("*")
            ),
            generated_at: now,
            category: InsightCategory::MissingRule,
            severity: InsightSeverity::Suggestion,
            summary: format!(
                "{action} [{}]{cmd_part} (consistent in {:.0}% of {} decisions)",
                p.tool,
                consistency * 100.0,
                p.sample_count,
            ),
            suggestion: Some(rule_suggestion(action, p)),
            evidence_count: p.sample_count,
        });
    }

    insights
}

/// Detect tools where brain accuracy is low.
pub(crate) fn detect_accuracy_gaps(prefs: &DistilledPreferences) -> Vec<Insight> {
    let now = epoch_now();
    prefs
        .tool_accuracy
        .iter()
        .filter(|ta| ta.total >= 5 && ta.confidence_threshold > 0.7)
        .map(|ta| {
            let accuracy = if ta.total > 0 {
                (ta.correct as f64 / ta.total as f64) * 100.0
            } else {
                0.0
            };
            Insight {
                fingerprint: format!("accuracy_gap:{}", ta.tool),
                generated_at: now,
                category: InsightCategory::AccuracyGap,
                severity: if accuracy < 50.0 {
                    InsightSeverity::Warning
                } else {
                    InsightSeverity::Suggestion
                },
                summary: format!(
                    "Brain accuracy for [{}] is {:.0}% (threshold raised to {:.2})",
                    ta.tool, accuracy, ta.confidence_threshold,
                ),
                suggestion: Some(format!(
                    "more training data needed for [{}] — brain defers these to manual review",
                    ta.tool,
                )),
                evidence_count: ta.total,
            }
        })
        .collect()
}

/// Convert temporal patterns from distillation into insights.
pub(crate) fn detect_temporal_friction(prefs: &DistilledPreferences) -> Vec<Insight> {
    let now = epoch_now();
    prefs
        .temporal
        .iter()
        .filter(|tp| tp.strength > 0.3 && tp.sample_count >= 3)
        .map(|tp| {
            // Use first 40 chars of description as fingerprint suffix
            let fp_suffix: String = tp
                .description
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ')
                .take(40)
                .collect::<String>()
                .replace(' ', "_");
            Insight {
                fingerprint: format!("temporal:{fp_suffix}"),
                generated_at: now,
                category: InsightCategory::TemporalFriction,
                severity: InsightSeverity::Info,
                summary: tp.description.clone(),
                suggestion: None,
                evidence_count: tp.sample_count,
            }
        })
        .collect()
}

/// Detect increasing burn rate trends and cost outliers.
pub(crate) fn detect_cost_patterns(decisions: &[DecisionRecord]) -> Vec<Insight> {
    let burn_rates: Vec<f64> = decisions
        .iter()
        .filter_map(|d| d.context.as_ref().map(|c| c.burn_rate_per_hr))
        .filter(|r| *r > 0.0)
        .collect();

    if burn_rates.len() < 10 {
        return Vec::new();
    }

    let mut insights = Vec::new();
    let now = epoch_now();

    // Compare first half vs second half burn rates
    let mid = burn_rates.len() / 2;
    let first_avg: f64 = burn_rates[..mid].iter().sum::<f64>() / mid as f64;
    let second_avg: f64 = burn_rates[mid..].iter().sum::<f64>() / (burn_rates.len() - mid) as f64;

    if first_avg > 0.0 {
        let increase = (second_avg - first_avg) / first_avg;
        if increase > 0.5 {
            insights.push(Insight {
                fingerprint: "cost_trend:increasing".to_string(),
                generated_at: now,
                category: InsightCategory::CostPattern,
                severity: if increase > 1.0 {
                    InsightSeverity::Warning
                } else {
                    InsightSeverity::Suggestion
                },
                summary: format!(
                    "Burn rate trending up: ${:.2}/hr -> ${:.2}/hr ({:+.0}%)",
                    first_avg,
                    second_avg,
                    increase * 100.0,
                ),
                suggestion: Some(
                    "consider setting a budget with --budget or reviewing costly operations"
                        .to_string(),
                ),
                evidence_count: burn_rates.len() as u32,
            });
        }
    }

    // Detect cost outlier sessions
    let mut per_session_cost: HashMap<u32, f64> = HashMap::new();
    for d in decisions {
        if let Some(ref ctx) = d.context {
            let entry = per_session_cost.entry(d.pid).or_insert(0.0);
            if ctx.cost_usd > *entry {
                *entry = ctx.cost_usd;
            }
        }
    }

    if per_session_cost.len() >= 3 {
        let costs: Vec<f64> = per_session_cost.values().copied().collect();
        let avg: f64 = costs.iter().sum::<f64>() / costs.len() as f64;
        let outlier_count = costs.iter().filter(|&&c| c > avg * 2.0 && c > 1.0).count();

        if outlier_count >= 2 {
            insights.push(Insight {
                fingerprint: "cost_trend:outliers".to_string(),
                generated_at: now,
                category: InsightCategory::CostPattern,
                severity: InsightSeverity::Info,
                summary: format!(
                    "{outlier_count} sessions cost >2x average (avg ${avg:.2})"
                ),
                suggestion: Some(
                    "review high-cost sessions — consider budget limits or earlier session restarts"
                        .to_string(),
                ),
                evidence_count: outlier_count as u32,
            });
        }
    }

    insights
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::decisions::{
        DecisionContext, DecisionType, DistilledPreferences, PreferencePattern, TemporalPattern,
        ToolAccuracy,
    };

    fn make_decision(tool: &str, command: &str, user_action: &str, pid: u32) -> DecisionRecord {
        DecisionRecord {
            timestamp: "0".to_string(),
            pid,
            project: "test".to_string(),
            tool: Some(tool.to_string()),
            command: Some(command.to_string()),
            brain_action: "approve".to_string(),
            brain_confidence: 0.8,
            brain_reasoning: String::new(),
            user_action: user_action.to_string(),
            context: None,
            outcome: None,
            decision_type: DecisionType::Session,
            suggested_at: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_decision_with_context(
        tool: &str,
        command: &str,
        user_action: &str,
        pid: u32,
        context_pct: u8,
        last_error: bool,
        burn_rate: f64,
        cost: f64,
    ) -> DecisionRecord {
        let mut d = make_decision(tool, command, user_action, pid);
        d.context = Some(DecisionContext {
            cost_usd: cost,
            context_pct,
            last_tool_error: last_error,
            error_message: None,
            model: "test".to_string(),
            elapsed_secs: 100,
            files_modified_count: 0,
            total_tool_calls: 10,
            has_file_conflict: false,
            status: "Processing".to_string(),
            burn_rate_per_hr: burn_rate,
            recent_error_count: 0,
            subagent_count: 0,
            hour: Some(10),
        });
        d
    }

    #[test]
    fn test_friction_patterns_detected() {
        let decisions: Vec<DecisionRecord> = (0..10)
            .map(|i| make_decision("Bash", "npm install", "reject", i))
            .collect();

        let insights = detect_friction_patterns(&decisions);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].category, InsightCategory::FrictionPattern);
        assert!(insights[0].summary.contains("npm install"));
        assert!(insights[0].summary.contains("10/10"));
    }

    #[test]
    fn test_friction_below_threshold_not_detected() {
        // 2 rejections out of 5 = 40%, below 60% threshold
        let mut decisions = Vec::new();
        for i in 0..3 {
            decisions.push(make_decision("Bash", "cargo test", "accept", i));
        }
        for i in 3..5 {
            decisions.push(make_decision("Bash", "cargo test", "reject", i));
        }

        let insights = detect_friction_patterns(&decisions);
        assert!(insights.is_empty());
    }

    #[test]
    fn test_error_loops_detected() {
        // 4 consecutive errors for same tool in one session
        let decisions: Vec<DecisionRecord> = (0..4)
            .map(|_| {
                make_decision_with_context(
                    "Write",
                    "src/main.rs",
                    "accept",
                    100,
                    50,
                    true,
                    1.0,
                    0.5,
                )
            })
            .collect();

        let insights = detect_error_loops(&decisions);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].category, InsightCategory::ErrorLoop);
    }

    #[test]
    fn test_context_blowouts_detected() {
        // 5 sessions all hitting >80% context
        let decisions: Vec<DecisionRecord> = (0..5)
            .map(|pid| {
                make_decision_with_context("Read", "file.rs", "accept", pid, 85, false, 1.0, 0.5)
            })
            .collect();

        let insights = detect_context_blowouts(&decisions);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].category, InsightCategory::ContextBlowout);
    }

    #[test]
    fn test_missing_rules_detected() {
        let prefs = DistilledPreferences {
            patterns: vec![
                PreferencePattern {
                    tool: "Bash".to_string(),
                    command_pattern: Some("cargo test".to_string()),
                    preferred_action: "approve".to_string(),
                    sample_count: 15,
                    accept_rate: 1.0,
                    conditions: Vec::new(),
                    confidence: 1.0,
                },
                PreferencePattern {
                    tool: "Bash".to_string(),
                    command_pattern: Some("rm -rf".to_string()),
                    preferred_action: "deny".to_string(),
                    sample_count: 6,
                    accept_rate: 0.0,
                    conditions: Vec::new(),
                    confidence: 1.0,
                },
            ],
            tool_accuracy: Vec::new(),
            total_decisions: 21,
            overall_accuracy: 0.8,
            temporal: Vec::new(),
        };

        let insights = detect_missing_rules(&[], &prefs);
        assert_eq!(insights.len(), 2);
        assert!(insights.iter().any(|i| i.summary.contains("cargo test")));
        assert!(insights.iter().any(|i| i.summary.contains("rm -rf")));
    }

    /// A suggested rule must be in the syntax the config parser actually reads.
    #[test]
    fn a_suggested_rule_uses_syntax_the_parser_accepts() {
        let prefs = DistilledPreferences {
            patterns: vec![PreferencePattern {
                tool: "Bash".to_string(),
                command_pattern: Some("rm -rf".to_string()),
                preferred_action: "deny".to_string(),
                sample_count: 6,
                accept_rate: 0.0,
                conditions: Vec::new(),
                confidence: 1.0,
            }],
            tool_accuracy: Vec::new(),
            total_decisions: 6,
            overall_accuracy: 0.8,
            temporal: Vec::new(),
        };

        let insights = detect_missing_rules(&[], &prefs);
        let suggestion = insights[0].suggestion.clone().expect("a suggestion");

        // The path is whatever `product::project_config` resolves to — asserting
        // it is never `.claudectl.toml` would contradict legacy resolution, which
        // returns exactly that when only the legacy file exists.
        assert!(
            suggestion.contains(
                &crate::product::project_config(std::path::Path::new("."))
                    .display()
                    .to_string()
            ),
            "the target must be the resolved config path: {suggestion}"
        );

        // Substring-matching the header would pass even when the header is
        // glued to the prose prefix and so is not a header at all. Parse the
        // TOML the user would actually paste and assert a rule comes back.
        let toml: String = suggestion
            .lines()
            .skip_while(|line| !line.starts_with('['))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            toml.starts_with("[rules."),
            "the table header must own its own line: {suggestion:?}"
        );

        let parsed = crate::config::parse_config_file_for_test(&toml);
        assert_eq!(
            parsed.len(),
            1,
            "the suggestion must produce exactly 1 rule"
        );
        assert!(
            parsed[0].name.starts_with("deny-bash-rm-rf-"),
            "readable stem plus a digest: {}",
            parsed[0].name
        );
        assert_eq!(parsed[0].action, crate::rules::RuleAction::Deny);
        assert_eq!(parsed[0].match_tool, vec!["Bash".to_string()]);
        assert_eq!(parsed[0].match_command, vec!["rm -rf".to_string()]);
    }

    /// Two denials for the same tool must not collide on one table name.
    ///
    /// `ensure_rule` reuses a rule by name and every matcher is
    /// last-write-wins, so identical headers silently collapse into one rule
    /// matching only the last pattern — the earlier denials would not exist.
    #[test]
    fn two_suggestions_for_one_tool_get_distinct_rule_names() {
        let pattern = |cmd: &str| PreferencePattern {
            tool: "Bash".to_string(),
            command_pattern: Some(cmd.to_string()),
            preferred_action: "deny".to_string(),
            sample_count: 6,
            accept_rate: 0.0,
            conditions: Vec::new(),
            confidence: 1.0,
        };
        let prefs = DistilledPreferences {
            patterns: vec![pattern("rm -rf"), pattern("curl | sh")],
            tool_accuracy: Vec::new(),
            total_decisions: 12,
            overall_accuracy: 0.8,
            temporal: Vec::new(),
        };

        let names: Vec<String> = detect_missing_rules(&[], &prefs)
            .iter()
            .filter_map(|i| i.suggestion.clone())
            .filter_map(|s| {
                s.lines()
                    .find(|l| l.starts_with("[rules."))
                    .map(str::to_string)
            })
            .collect();

        assert_eq!(names.len(), 2);
        assert_ne!(
            names[0], names[1],
            "both denials suggested the same table: {names:?}"
        );
    }

    /// Slugifying is lossy, so the readable part alone is not a unique name.
    ///
    /// `rm -rf` and `rm/rf` both reduce to `bash-rm-rf`, and any command made
    /// only of punctuation reduces to nothing at all — so without a digest,
    /// distinct denials would still collide onto one table and silently
    /// overwrite each other.
    #[test]
    fn commands_that_slugify_identically_still_get_distinct_names() {
        let collide = ["rm -rf", "rm/rf"];
        let empty_stem = ["|", ">", "", "\u{65e5}\u{672c}"];

        let names: Vec<String> = collide
            .iter()
            .chain(empty_stem.iter())
            .map(|c| rule_slug("Bash", Some(c)))
            .collect();

        let unique: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "names collided: {names:?}");
        assert!(
            names.iter().all(|n| !n.is_empty() && !n.ends_with('-')),
            "no empty or dangling-dash table names: {names:?}"
        );

        // Stable across calls, so re-running does not rename a rule the user
        // has already pasted.
        assert_eq!(rule_slug("Bash", Some("rm -rf")), names[0]);
        // No command at all stays clean, with no trailing digest.
        assert_eq!(rule_slug("Bash", None), "bash");
    }

    #[test]
    fn test_accuracy_gaps_detected() {
        let prefs = DistilledPreferences {
            patterns: Vec::new(),
            tool_accuracy: vec![ToolAccuracy {
                tool: "Edit".to_string(),
                total: 10,
                correct: 4,
                confidence_threshold: 0.85,
            }],
            total_decisions: 10,
            overall_accuracy: 0.4,
            temporal: Vec::new(),
        };

        let insights = detect_accuracy_gaps(&prefs);
        assert_eq!(insights.len(), 1);
        assert!(insights[0].summary.contains("Edit"));
        assert!(insights[0].summary.contains("40%"));
    }

    #[test]
    fn test_cost_trend_detected() {
        let mut decisions = Vec::new();
        // First 10: low burn rate
        for i in 0..10 {
            decisions.push(make_decision_with_context(
                "Bash", "cmd", "accept", i, 50, false, 1.0, 0.5,
            ));
        }
        // Next 10: high burn rate (3x increase)
        for i in 10..20 {
            decisions.push(make_decision_with_context(
                "Bash", "cmd", "accept", i, 50, false, 3.0, 1.5,
            ));
        }

        let insights = detect_cost_patterns(&decisions);
        assert!(
            insights
                .iter()
                .any(|i| i.fingerprint == "cost_trend:increasing")
        );
    }

    #[test]
    fn test_temporal_friction_detected() {
        let prefs = DistilledPreferences {
            patterns: Vec::new(),
            tool_accuracy: Vec::new(),
            total_decisions: 50,
            overall_accuracy: 0.8,
            temporal: vec![TemporalPattern {
                description: "After 3+ errors: user usually denies (n=5)".to_string(),
                sample_count: 5,
                strength: 0.6,
            }],
        };

        let insights = detect_temporal_friction(&prefs);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].category, InsightCategory::TemporalFriction);
    }

    fn one(tool: &str, command: Option<&str>, conditions: Vec<PreferenceCondition>) -> String {
        rule_suggestion(
            "deny",
            &PreferencePattern {
                tool: tool.to_string(),
                command_pattern: command.map(str::to_string),
                preferred_action: "deny".to_string(),
                sample_count: 6,
                accept_rate: 0.0,
                conditions,
                confidence: 1.0,
            },
        )
    }

    fn toml_of(suggestion: &str) -> String {
        suggestion
            .lines()
            .skip_while(|l| !l.starts_with('['))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// No command pattern means an omitted matcher, never `["*"]`.
    #[test]
    fn a_tool_only_pattern_omits_the_command_matcher() {
        let suggestion = one("Read", None, Vec::new());
        assert!(
            !suggestion.contains("match_command"),
            "no command means no matcher: {suggestion}"
        );

        let parsed = crate::config::parse_config_file_for_test(&toml_of(&suggestion));
        assert_eq!(parsed.len(), 1);
        assert!(
            parsed[0].match_command.is_empty(),
            "an empty matcher is the wildcard; {:?} is not",
            parsed[0].match_command
        );
        assert_eq!(parsed[0].match_tool, vec!["Read".to_string()]);
    }

    /// `["sort -k1,1"]` parses back as two patterns, the stray `1` matching
    /// almost anything by substring — so such commands get prose, not TOML.
    #[test]
    fn a_command_the_parser_would_mangle_falls_back_to_prose() {
        for command in ["sort -k1,1", "awk -F,", "open f.html#top", "say \"hi\""] {
            let suggestion = one("Bash", Some(command), Vec::new());
            assert!(
                !suggestion.contains("[rules."),
                "{command:?} must not be printed as a rule: {suggestion}"
            );
            assert!(
                suggestion.contains("by hand"),
                "{command:?} should tell the user to write it themselves: {suggestion}"
            );
        }

        // A clean command still gets paste-ready TOML, and round-trips exactly.
        let suggestion = one("Bash", Some("rm -rf"), Vec::new());
        let parsed = crate::config::parse_config_file_for_test(&toml_of(&suggestion));
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].match_command, vec!["rm -rf".to_string()]);
    }

    /// A condition with an exact matcher must be carried into the rule, not
    /// dropped — dropping it widens the rule past what the user agreed to.
    #[test]
    fn an_expressible_condition_becomes_a_matcher() {
        let suggestion = one(
            "Bash",
            Some("cargo test"),
            vec![PreferenceCondition::CostAbove(1.0)],
        );
        let parsed = crate::config::parse_config_file_for_test(&toml_of(&suggestion));
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].match_cost_above, Some(1.0), "{suggestion}");

        let flags = one(
            "Bash",
            Some("cargo test"),
            vec![
                PreferenceCondition::HasErrors,
                PreferenceCondition::NoFileConflict,
            ],
        );
        let parsed = crate::config::parse_config_file_for_test(&toml_of(&flags));
        assert_eq!(parsed[0].match_last_error, Some(true), "{flags}");
        assert_eq!(parsed[0].match_file_conflict, Some(false), "{flags}");
    }

    /// A condition with no matcher must fall back to prose, and one
    /// inexpressible condition taints the whole set.
    #[test]
    fn an_inexpressible_condition_is_not_emitted_as_a_rule() {
        for condition in [
            PreferenceCondition::CostBelow(1.0),
            PreferenceCondition::ContextAbove(80),
            PreferenceCondition::ContextBelow(20),
            PreferenceCondition::HourRange(8, 18),
        ] {
            let suggestion = one("Bash", Some("cargo test"), vec![condition.clone()]);
            assert!(
                !suggestion.contains("[rules."),
                "{condition:?} has no matcher and must not become a rule: {suggestion}"
            );
        }

        let mixed = one(
            "Bash",
            Some("cargo test"),
            vec![
                PreferenceCondition::CostAbove(1.0),
                PreferenceCondition::HourRange(8, 18),
            ],
        );
        assert!(
            !mixed.contains("[rules."),
            "one inexpressible condition must taint the set: {mixed}"
        );
    }

    /// `distill_preferences` writes `"*"` for a decision with no tool, and
    /// `match_tool` is exact equality — `["*"]` matches nothing, so a pasted
    /// deny would never fire.
    #[test]
    fn a_toolless_pattern_is_not_emitted_as_a_rule() {
        for command in [None, Some("rm -rf")] {
            let suggestion = one("*", command, Vec::new());
            assert!(
                !suggestion.contains("[rules."),
                "a tool-less pattern must not become a rule: {suggestion}"
            );
            assert!(
                !suggestion.contains("match_tool"),
                "and must not print a matcher at all: {suggestion}"
            );
        }
    }

    fn pattern(preferred_action: &str, accept_rate: f64) -> DistilledPreferences {
        DistilledPreferences {
            patterns: vec![PreferencePattern {
                tool: "Bash".to_string(),
                command_pattern: Some("rm -rf".to_string()),
                preferred_action: preferred_action.to_string(),
                sample_count: 6,
                accept_rate,
                conditions: Vec::new(),
                confidence: 1.0,
            }],
            tool_accuracy: Vec::new(),
            total_decisions: 6,
            overall_accuracy: 0.8,
            temporal: Vec::new(),
        }
    }

    /// `accept_rate` is agreement with the BRAIN, not a wish to proceed, so a
    /// unanimous accept_rate over a brain that kept denying means deny.
    #[test]
    fn the_suggested_action_follows_the_preference_not_the_accept_rate() {
        let insights = detect_missing_rules(&[], &pattern("deny", 1.0));
        assert_eq!(insights.len(), 1);
        let suggestion = insights[0].suggestion.clone().expect("a suggestion");
        assert!(
            !suggestion.contains("action = \"approve\""),
            "agreeing with 6 denials must not suggest auto-approving it: {suggestion}"
        );

        let parsed = crate::config::parse_config_file_for_test(&toml_of(&suggestion));
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].action, crate::rules::RuleAction::Deny);
        assert!(
            insights[0].summary.starts_with("deny "),
            "{}",
            insights[0].summary
        );
    }

    /// Only approve and deny survive as a rule. `send` would type the engine's
    /// "continue" default, `terminate`/`kill` would destroy sessions, and
    /// route/spawn/delegate have no rule form at all.
    #[test]
    fn a_preference_the_rule_language_cannot_express_is_not_suggested() {
        for action in [
            "route",
            "spawn",
            "delegate",
            "send",
            "terminate",
            "kill",
            "",
        ] {
            assert!(
                detect_missing_rules(&[], &pattern(action, 1.0)).is_empty(),
                "{action:?} must not be handed over as a pasteable rule"
            );
        }
    }

    /// An approve rule matches a superset of what was observed, and the caveat
    /// saying so must survive being pasted.
    #[test]
    fn an_approve_suggestion_says_it_matches_more_than_it_saw() {
        let insights = detect_missing_rules(&[], &pattern("approve", 1.0));
        let suggestion = insights[0].suggestion.clone().expect("a suggestion");
        assert!(
            suggestion.contains("# also matches any command containing \"rm -rf\""),
            "{suggestion}"
        );

        // Keep the comment: the parser must skip it, not choke on it.
        let pasted = suggestion
            .lines()
            .skip_while(|l| !l.starts_with('#') && !l.starts_with('['))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            pasted.starts_with('#'),
            "the comment must be pasted: {pasted}"
        );
        let parsed = crate::config::parse_config_file_for_test(&pasted);
        assert_eq!(parsed.len(), 1, "the comment must not break parsing");
        assert_eq!(parsed[0].action, RuleAction::Approve);

        // A deny matches a superset too, but that direction is fail-safe.
        let deny = detect_missing_rules(&[], &pattern("deny", 0.0))[0]
            .suggestion
            .clone()
            .expect("a suggestion");
        assert!(!deny.contains('#'), "no caveat on a deny: {deny}");
    }
}
