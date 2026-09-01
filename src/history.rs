use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// A completed session record persisted to CSV.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub timestamp: String, // ISO 8601
    pub pid: u32,
    pub project: String,
    pub model: String,
    pub duration_secs: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    /// The agent label as recorded. A product this build does not know is
    /// kept verbatim rather than misattributed to a known one.
    pub provider: String,
}

fn history_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    crate::product::shared_state_root(&home)
}

fn history_path() -> PathBuf {
    history_dir().join("history.csv")
}

/// Append a session record to the history CSV.
pub fn record_session(session: &crate::session::AgentSession) {
    let dir = history_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }

    let path = history_path();
    let needs_header = !path.exists();

    let file = OpenOptions::new().create(true).append(true).open(&path);

    let Ok(mut file) = file else { return };

    if needs_header {
        let _ = writeln!(file, "{HEADER}");
    }

    let ts = crate::logger::timestamp_now();
    let project = session.display_name().replace(',', ";");
    let model = session.model.replace(',', ";");

    let _ = writeln!(
        file,
        "{}",
        format_row(
            &ts,
            session.pid,
            &project,
            &model,
            session.elapsed.as_secs(),
            session.total_input_tokens,
            session.total_output_tokens,
            session.cost_usd,
            session.provider,
        )
    );
}

/// One CSV row. Separate from the writer so a test can round-trip it through
/// [`parse_record`] — a field the writer and reader disagree about is the whole
/// hazard of widening this format.
#[allow(clippy::too_many_arguments)]
fn format_row(
    ts: &str,
    pid: u32,
    project: &str,
    model: &str,
    duration_secs: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    provider: crate::provider::AgentProvider,
) -> String {
    format!(
        "{ts},{pid},{project},{model},{duration_secs},{input_tokens},{output_tokens},{cost_usd:.4},{}",
        provider.label()
    )
}

/// The CSV header. Kept beside [`format_row`] so the two cannot drift.
const HEADER: &str =
    "timestamp,pid,project,model,duration_secs,input_tokens,output_tokens,cost_usd,provider";

/// Parse one CSV row. A missing or blank 9th field reads as Claude — rows
/// predating the column are Claude by construction — while a present but
/// unrecognised one is kept verbatim rather than folded into a known product.
fn parse_record(line: &str) -> Option<SessionRecord> {
    // `needs_header` is read before the open and unlocked, so two processes
    // ending a session at once both write one — a mid-file header parsed as
    // data becomes a phantom $0.00 session.
    if line.starts_with("timestamp,") {
        return None;
    }
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 8 {
        return None;
    }
    Some(SessionRecord {
        timestamp: fields[0].to_string(),
        pid: fields[1].parse().unwrap_or(0),
        project: fields[2].to_string(),
        model: fields[3].to_string(),
        duration_secs: fields[4].parse().unwrap_or(0),
        input_tokens: fields[5].parse().unwrap_or(0),
        output_tokens: fields[6].parse().unwrap_or(0),
        cost_usd: fields[7].parse().unwrap_or(0.0),
        // Absent means a pre-column row, which is Claude by construction. A
        // label this build does not recognise is kept as written: attributing
        // another product's spend to Claude would be a lie about money.
        provider: match fields.get(8).map(|l| l.trim()) {
            None | Some("") => crate::provider::AgentProvider::Claude.label().to_string(),
            Some(raw) => crate::provider::AgentProvider::from_label(raw)
                .map(|p| p.label().to_string())
                .unwrap_or_else(|| raw.to_string()),
        },
    })
}

/// Load all history records, optionally filtered by a time window.
pub fn load_history(since_secs: Option<u64>) -> Vec<SessionRecord> {
    let path = history_path();
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let Some(record) = parse_record(&line) else {
            continue;
        };

        // Filter by time window if specified
        if let Some(window) = since_secs {
            if let Some(record_secs) = parse_timestamp_epoch(&record.timestamp) {
                if now_secs.saturating_sub(record_secs) > window {
                    continue;
                }
            }
        }

        records.push(record);
    }

    records
}

/// Parse an ISO 8601 timestamp to epoch seconds (simplified).
fn parse_timestamp_epoch(ts: &str) -> Option<u64> {
    // Format: 2026-04-11T14:30:00Z
    if ts.len() < 19 {
        return None;
    }
    let year: u64 = ts[0..4].parse().ok()?;
    let month: u64 = ts[5..7].parse().ok()?;
    let day: u64 = ts[8..10].parse().ok()?;
    let hour: u64 = ts[11..13].parse().ok()?;
    let min: u64 = ts[14..16].parse().ok()?;
    let sec: u64 = ts[17..19].parse().ok()?;

    // Approximate days from epoch (good enough for filtering)
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap(y) { 366 } else { 365 };
    }
    let month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += month_days[m as usize];
        if m == 2 && is_leap(year) {
            days += 1;
        }
    }
    days += day - 1;

    Some(days * 86400 + hour * 3600 + min * 60 + sec)
}

fn is_leap(y: u64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Print a tabular history view.
pub fn print_history(since: &str) {
    let since_secs = parse_duration(since);
    let records = load_history(since_secs);

    if records.is_empty() {
        println!("No session history found.");
        if since_secs.is_some() {
            println!("  (filtered to last {since})");
        }
        return;
    }

    // The rule is measured from the header rather than hand-counted, so adding
    // a column cannot leave it short again.
    let header = format!(
        "{:<22} {:<7} {:<8} {:<20} {:<12} {:>10} {:>12} {:>12} {:>10}",
        "Timestamp", "PID", "Agent", "Project", "Model", "Duration", "Input", "Output", "Cost"
    );
    let rule = "-".repeat(header.chars().count());
    println!("{header}");
    println!("{rule}");

    let mut total_cost = 0.0;
    let mut total_duration = 0u64;
    let mut total_input = 0u64;
    let mut total_output = 0u64;

    for r in &records {
        let dur = format_duration(r.duration_secs);
        let cost = if r.cost_usd < 1.0 {
            format!("${:.2}", r.cost_usd)
        } else {
            format!("${:.1}", r.cost_usd)
        };

        println!(
            "{:<22} {:<7} {:<8} {:<20} {:<12} {:>10} {:>12} {:>12} {:>10}",
            &r.timestamp[..19.min(r.timestamp.len())],
            r.pid,
            truncate(&r.provider, 8),
            truncate(&r.project, 20),
            truncate(&r.model, 12),
            dur,
            format_count(r.input_tokens),
            format_count(r.output_tokens),
            cost,
        );

        total_cost += r.cost_usd;
        total_duration += r.duration_secs;
        total_input += r.input_tokens;
        total_output += r.output_tokens;
    }

    println!("{rule}");
    let total_cost_str = if total_cost < 1.0 {
        format!("${:.2}", total_cost)
    } else {
        format!("${:.1}", total_cost)
    };
    println!(
        "{:<22} {:<7} {:<8} {:<20} {:<12} {:>10} {:>12} {:>12} {:>10}",
        format!("{} sessions", records.len()),
        "",
        "",
        "",
        "",
        format_duration(total_duration),
        format_count(total_input),
        format_count(total_output),
        total_cost_str,
    );
}

/// Print aggregate statistics.
pub fn print_stats(since: &str) {
    let since_secs = parse_duration(since);
    let records = load_history(since_secs);

    if records.is_empty() {
        println!("No session history found.");
        return;
    }

    let total_cost: f64 = records.iter().map(|r| r.cost_usd).sum();
    let total_duration: u64 = records.iter().map(|r| r.duration_secs).sum();
    let total_input: u64 = records.iter().map(|r| r.input_tokens).sum();
    let total_output: u64 = records.iter().map(|r| r.output_tokens).sum();
    let avg_cost = total_cost / records.len() as f64;
    let avg_duration = total_duration / records.len() as u64;

    println!("Session Statistics (last {since})");
    println!("{}", "=".repeat(45));
    println!("  Sessions:         {}", records.len());
    println!("  Total cost:       ${:.2}", total_cost);
    println!("  Avg cost/session: ${:.2}", avg_cost);
    println!("  Total duration:   {}", format_duration(total_duration));
    println!("  Avg duration:     {}", format_duration(avg_duration));
    println!(
        "  Total tokens:     {} in / {} out",
        format_count(total_input),
        format_count(total_output)
    );
    println!();

    // Per-agent breakdown. Sorted largest-cost first, like the project table.
    let mut agents: std::collections::HashMap<&str, (f64, u64, usize)> =
        std::collections::HashMap::new();
    for r in &records {
        let entry = agents.entry(r.provider.as_str()).or_default();
        entry.0 += r.cost_usd;
        entry.1 += r.duration_secs;
        entry.2 += 1;
    }
    if agents.len() > 1 {
        let mut agent_list: Vec<_> = agents.into_iter().collect();
        agent_list.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let head = format!(
            "  {:<25} {:>8} {:>10} {:>10}",
            "Agent", "Sessions", "Duration", "Cost"
        );
        println!("  Per-agent breakdown:");
        println!("{head}");
        println!("  {}", "-".repeat(head.trim_start().chars().count()));
        for (name, (cost, dur, count)) in &agent_list {
            let cost_str = if *cost < 1.0 {
                format!("${:.2}", cost)
            } else {
                format!("${:.1}", cost)
            };
            println!(
                "  {:<25} {:>8} {:>10} {:>10}",
                truncate(name, 25),
                count,
                format_duration(*dur),
                cost_str,
            );
        }
        println!();
    }

    // Per-project breakdown
    let mut projects: std::collections::HashMap<String, (f64, u64, usize)> =
        std::collections::HashMap::new();
    for r in &records {
        let entry = projects.entry(r.project.clone()).or_default();
        entry.0 += r.cost_usd;
        entry.1 += r.duration_secs;
        entry.2 += 1;
    }

    let mut project_list: Vec<_> = projects.into_iter().collect();
    project_list.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

    let head = format!(
        "  {:<25} {:>8} {:>10} {:>10}",
        "Project", "Sessions", "Duration", "Cost"
    );
    println!("  Per-project breakdown:");
    println!("{head}");
    println!("  {}", "-".repeat(head.trim_start().chars().count()));
    for (name, (cost, dur, count)) in &project_list {
        let cost_str = if *cost < 1.0 {
            format!("${:.2}", cost)
        } else {
            format!("${:.1}", cost)
        };
        println!(
            "  {:<25} {:>8} {:>10} {:>10}",
            truncate(name, 25),
            count,
            format_duration(*dur),
            cost_str,
        );
    }

    // Per-model breakdown
    let mut models: std::collections::HashMap<String, (f64, usize)> =
        std::collections::HashMap::new();
    for r in &records {
        let model = if r.model.is_empty() {
            "unknown".to_string()
        } else {
            r.model.clone()
        };
        let entry = models.entry(model).or_default();
        entry.0 += r.cost_usd;
        entry.1 += 1;
    }

    let mut model_list: Vec<_> = models.into_iter().collect();
    model_list.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

    println!();
    println!("  Per-model breakdown:");
    let head = format!("  {:<20} {:>8} {:>10}", "Model", "Sessions", "Cost");
    println!("{head}");
    println!("  {}", "-".repeat(head.trim_start().chars().count()));
    for (name, (cost, count)) in &model_list {
        let cost_str = if *cost < 1.0 {
            format!("${:.2}", cost)
        } else {
            format!("${:.1}", cost)
        };
        println!("  {:<20} {:>8} {:>10}", truncate(name, 20), count, cost_str);
    }
}

/// Parse a duration string like "24h", "30m", "7d" into seconds.
pub fn parse_duration(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (num_str, unit) = s.split_at(s.len() - 1);
    let num: u64 = num_str.parse().ok()?;
    match unit {
        "s" => Some(num),
        "m" => Some(num * 60),
        "h" => Some(num * 3600),
        "d" => Some(num * 86400),
        "w" => Some(num * 604800),
        _ => None,
    }
}

pub(crate) fn format_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h{m:02}m")
    } else {
        format!("{m}m{s:02}s")
    }
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

/// Bound a field to `max` columns for display.
///
/// Counts characters, not bytes: the old byte slice panicked on any non-ASCII
/// project name (`&s[..17]` landing inside a multi-byte char). Control
/// characters are dropped — these values come off disk and go to a terminal.
fn truncate(s: &str, max: usize) -> String {
    let clean: String = s.chars().filter(|c| !c.is_control()).collect();
    if clean.chars().count() <= max {
        return clean;
    }
    clean
        .chars()
        .take(max.saturating_sub(3))
        .collect::<String>()
        + "..."
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file written before the provider column keeps working, and a file
    /// upgraded mid-life holds BOTH widths at once — the row a user's history
    /// actually looks like after they upgrade.
    #[test]
    fn rows_of_either_width_survive_in_the_same_file() {
        let old_row = "2026-08-31T10:00:00Z,111,proj,claude-opus-5,60,10,20,0.0500";
        let new_row = "2026-08-31T10:01:00Z,222,proj,gpt-5,30,5,7,0.0100,Codex";

        let old = parse_record(old_row).expect("an 8-field row must still parse");
        assert_eq!(old.pid, 111);
        assert_eq!(
            old.cost_usd, 0.05,
            "cost must not be corrupted by the new column"
        );
        assert_eq!(
            old.provider, "Claude",
            "rows predating the column are Claude by construction"
        );

        let new = parse_record(new_row).expect("a 9-field row must parse");
        assert_eq!(new.pid, 222);
        assert_eq!(new.cost_usd, 0.01);
        assert_eq!(new.provider, "Codex");
    }

    /// What the writer emits must be what the reader reads, for every product.
    #[test]
    fn a_written_row_round_trips_through_the_parser() {
        for provider in crate::provider::AgentProvider::all() {
            let row = format_row(
                "2026-08-31T10:00:00Z",
                4242,
                "proj",
                "some-model",
                1800,
                123,
                456,
                0.5,
                *provider,
            );
            let back = parse_record(&row).expect("a freshly written row must parse");
            assert_eq!(back.pid, 4242);
            assert_eq!(back.project, "proj");
            assert_eq!(back.model, "some-model");
            assert_eq!(back.duration_secs, 1800);
            assert_eq!(back.input_tokens, 123);
            assert_eq!(back.output_tokens, 456);
            assert_eq!(back.cost_usd, 0.5);
            assert_eq!(back.provider, provider.label());
        }
    }

    /// The header must name exactly as many columns as a row has fields.
    #[test]
    fn the_header_and_a_row_have_the_same_width() {
        let row = format_row(
            "2026-08-31T10:00:00Z",
            1,
            "p",
            "m",
            1,
            1,
            1,
            0.0,
            crate::provider::AgentProvider::Claude,
        );
        assert_eq!(
            HEADER.split(',').count(),
            row.split(',').count(),
            "header {HEADER:?} disagrees with row {row:?}"
        );
    }

    /// An unrecognised provider must not drop the row — the cost is still real.
    #[test]
    fn an_unknown_provider_label_keeps_the_row() {
        let row = "2026-08-31T10:00:00Z,333,proj,m,60,10,20,0.2500,Gemini";
        let r = parse_record(row).expect("an unknown provider must not drop the row");
        assert_eq!(r.cost_usd, 0.25);
        assert_eq!(
            r.provider, "Gemini",
            "an unknown product must not be booked against Claude"
        );
    }

    /// A header can land mid-file when two processes end a session at once.
    /// Parsed as data it becomes a phantom $0.00 session in every total.
    #[test]
    fn a_header_line_is_never_data_wherever_it_appears() {
        assert!(parse_record(HEADER).is_none());
        assert!(
            parse_record(
                "timestamp,pid,project,model,duration_secs,input_tokens,output_tokens,cost_usd"
            )
            .is_none(),
            "the pre-column header must be rejected too"
        );
    }

    /// A known label is normalised so case differences do not split a group in
    /// the per-agent breakdown.
    #[test]
    fn a_known_label_is_normalised_for_grouping() {
        for raw in ["codex", "CODEX", " Codex "] {
            let row = format!("T,1,p,m,60,10,20,0.1000,{raw}");
            let r = parse_record(&row).expect("row parses");
            assert_eq!(r.provider, "Codex", "{raw:?} must normalise");
        }
    }

    /// `truncate` byte-sliced, so any non-ASCII project name crashed
    /// `--history` outright: `&s[..17]` landing inside a multi-byte char.
    #[test]
    fn truncate_does_not_panic_on_a_multibyte_name() {
        // 24 bytes, 8 chars — the old slice at byte 17 split a 3-byte char.
        let name = "日本語日本語日本";
        assert_eq!(truncate(name, 20).chars().count(), 8, "fits, so unchanged");
        let long = "日本語日本語日本語日本語日本語";
        assert_eq!(
            truncate(long, 12).chars().count(),
            12,
            "must bound by CHARS and not panic"
        );
        assert!(truncate(long, 12).ends_with("..."));
    }

    /// These values come off disk and go to a terminal, so an escape sequence
    /// in a provider label must not reach it.
    #[test]
    fn truncate_strips_control_characters() {
        let hostile = "\u{1b}[31mRED\u{1b}[0m";
        let out = truncate(hostile, 30);
        assert!(!out.contains('\u{1b}'), "escape must not survive: {out:?}");
        assert_eq!(out, "[31mRED[0m");
        assert!(!truncate("a\rb\nc", 30).contains('\r'));
    }

    /// A truncated row is still rejected.
    #[test]
    fn a_short_row_is_rejected() {
        assert!(parse_record("2026-08-31T10:00:00Z,1,p,m,60,10,20").is_none());
    }

    /// `label` and `from_label` must stay inverse for every product.
    #[test]
    fn every_provider_label_round_trips() {
        for p in crate::provider::AgentProvider::all() {
            assert_eq!(
                crate::provider::AgentProvider::from_label(p.label()),
                Some(*p),
                "{} must round-trip",
                p.label()
            );
        }
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("24h"), Some(86400));
        assert_eq!(parse_duration("30m"), Some(1800));
        assert_eq!(parse_duration("7d"), Some(604800));
        assert_eq!(parse_duration("1w"), Some(604800));
        assert_eq!(parse_duration(""), None);
        assert_eq!(parse_duration("abc"), None);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(3661), "1h01m");
        assert_eq!(format_duration(125), "2m05s");
        assert_eq!(format_duration(0), "0m00s");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world!", 8), "hello...");
    }

    #[test]
    fn test_parse_timestamp_epoch() {
        // 2026-01-01T00:00:00Z
        let ts = parse_timestamp_epoch("2026-01-01T00:00:00Z").unwrap();
        // Should be reasonable (after 2025)
        assert!(ts > 1735689600); // 2025-01-01
        assert!(ts < 1798761600); // 2027-01-01
    }

    #[test]
    fn test_is_leap() {
        assert!(is_leap(2000));
        assert!(is_leap(2024));
        assert!(!is_leap(1900));
        assert!(!is_leap(2023));
    }

    /// The history file must live inside the host-shared, bind-mounted root.
    ///
    /// Exercised through the real resolver with `HOME` pinned, not by
    /// re-deriving the path in the test: the regression this guards against was
    /// a call site quietly routed to `~/.local/share/agentctl`, which no
    /// assertion about `product` alone would have caught.
    #[test]
    fn history_dir_resolves_inside_the_shared_state_root() {
        let _guard = crate::sandbox_registry::tests::env_guard();
        let home = tempfile::tempdir().unwrap();
        let saved = std::env::var_os("HOME");
        // SAFETY: env access is serialised by the held lock.
        unsafe { std::env::set_var("HOME", home.path()) };
        let resolved = history_dir();
        unsafe {
            match saved {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }
        assert_eq!(resolved, crate::product::shared_state_root(home.path()));
        assert!(
            resolved.ends_with("claudectl"),
            "the bind mount is named claudectl, got {}",
            resolved.display()
        );
    }
}
