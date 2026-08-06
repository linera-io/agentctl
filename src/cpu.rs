//! Measuring how hard a session is working *right now*.
//!
//! `ps`'s `%cpu` column looks like the answer and is not. It means two
//! different things depending on the platform, and neither is "current":
//!
//! * Linux (procps-ng) — "CPU time used divided by the time the process has
//!   been running (cputime/realtime ratio)". A **lifetime average that never
//!   decays**: a session that worked hard for ten minutes reads busy for hours
//!   afterwards. <https://man7.org/linux/man-pages/man1/ps.1.html>
//! * macOS (BSD) — "a decaying average over up to a minute of previous (real)
//!   time". Closer, but still reads busy for up to a minute after a turn ends.
//!   <https://keith.github.io/xcode-man-pages/ps.1.html>
//!
//! Measured live on 2026-08-06 against an idle `claude` pid: `ps %cpu` said
//! **5.7**, sampling `/proc` over 5 s said **0.20**, and `cputime 141.12 s /
//! elapsed 2470 s` reproduced `ps`'s number exactly. claudectl's status
//! inference compared that 5.7 against a 5.0 threshold and reported a finished
//! session as `Processing`.
//!
//! The honest measurement is a rate: sample the process's *cumulative* CPU time
//! twice and divide the delta by the wall-clock delta. `cputime` is available
//! from `ps` on both platforms (on BSD it is an alias for `time`), and both
//! samples come from ticks claudectl already performs — no extra process spawn,
//! no sleeping.

/// One cumulative CPU-time sample: seconds of CPU consumed by a process since
/// it started, and the wall-clock instant the sample was taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuSample {
    pub cputime_secs: f64,
    pub sampled_at_ms: u64,
}

/// CPU used between two samples, as a percentage of one core.
///
/// `None` — never zero — whenever the question cannot be answered: no previous
/// sample, no wall-clock progress, or a counter/clock that went backwards
/// (`ps` rounds to whole seconds, and a pid can be recycled between ticks).
/// Callers must treat `None` as "unknown" and not as evidence of idleness *or*
/// of work; status inference will not claim a session is `Processing` on a
/// rate it never measured.
pub fn cpu_rate_percent(prev: CpuSample, cur: CpuSample) -> Option<f32> {
    let elapsed_ms = cur.sampled_at_ms.checked_sub(prev.sampled_at_ms)?;
    if elapsed_ms == 0 {
        return None;
    }
    let cpu_delta = cur.cputime_secs - prev.cputime_secs;
    if cpu_delta < 0.0 {
        return None;
    }
    Some((cpu_delta / (elapsed_ms as f64 / 1000.0) * 100.0) as f32)
}

/// Parse `ps`'s cumulative CPU-time column into seconds.
///
/// Accepts every rendering the two platforms produce, since the exact one is
/// documented on neither: `MM:SS`, `HH:MM:SS`, `DD-HH:MM:SS`, and any of those
/// with fractional seconds (`MM:SS.ss`, which BSD uses for short-lived
/// processes). Linux procps renders `00:01:31`.
///
/// Returns `None` for anything else, which propagates as an unknown rate — the
/// safe direction. `cpu_matches_this_platforms_ps` pins this against real `ps`
/// output on whatever machine the tests run on, so a format this does not
/// handle fails the suite rather than silently disabling the signal.
pub fn parse_cputime_secs(field: &str) -> Option<f64> {
    let field = field.trim();
    if field.is_empty() {
        return None;
    }

    let (days, hms) = match field.split_once('-') {
        Some((days, rest)) => (days.parse::<u64>().ok()?, rest),
        None => (0, field),
    };

    let parts: Vec<&str> = hms.split(':').collect();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [m, s] => (0u64, m.parse::<u64>().ok()?, s),
        [h, m, s] => (h.parse::<u64>().ok()?, m.parse::<u64>().ok()?, s),
        _ => return None,
    };
    let seconds: f64 = seconds.parse().ok()?;
    if seconds < 0.0 {
        return None;
    }

    Some((days * 86_400 + hours * 3_600 + minutes * 60) as f64 + seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(cputime_secs: f64, sampled_at_ms: u64) -> CpuSample {
        CpuSample {
            cputime_secs,
            sampled_at_ms,
        }
    }

    #[test]
    fn rate_is_cpu_delta_over_wall_delta() {
        // 1 s of CPU over 2 s of wall clock = half a core.
        let rate = cpu_rate_percent(sample(10.0, 1_000), sample(11.0, 3_000)).unwrap();
        assert!((rate - 50.0).abs() < 0.01, "got {rate}");

        // A fully saturated core.
        let rate = cpu_rate_percent(sample(10.0, 1_000), sample(12.0, 3_000)).unwrap();
        assert!((rate - 100.0).abs() < 0.01, "got {rate}");

        // Multi-core work reads above 100%, deliberately: the threshold is
        // expressed in cores, so a 4-way parallel build should look busier
        // than a single-threaded one.
        let rate = cpu_rate_percent(sample(10.0, 1_000), sample(18.0, 3_000)).unwrap();
        assert!((rate - 400.0).abs() < 0.01, "got {rate}");
    }

    #[test]
    fn an_idle_process_rates_zero_not_unknown() {
        // The distinction matters: 0.0 is evidence of idleness and suppresses
        // the Processing claim; None is absence of evidence and must not be
        // read as either.
        let rate = cpu_rate_percent(sample(141.12, 1_000), sample(141.12, 3_000)).unwrap();
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn unanswerable_comparisons_are_none_never_zero() {
        // No wall-clock progress: two samples inside the same millisecond.
        assert_eq!(
            cpu_rate_percent(sample(1.0, 1_000), sample(2.0, 1_000)),
            None
        );
        // Clock went backwards (sample order inverted).
        assert_eq!(
            cpu_rate_percent(sample(1.0, 3_000), sample(2.0, 1_000)),
            None
        );
        // Counter went backwards — a recycled pid, not a process that
        // un-consumed CPU. Reporting 0.0 here would claim the new occupant is
        // idle on the strength of the old one's numbers.
        assert_eq!(
            cpu_rate_percent(sample(9.0, 1_000), sample(2.0, 3_000)),
            None
        );
    }

    #[test]
    fn parses_every_ps_cputime_rendering() {
        assert_eq!(parse_cputime_secs("00:01:31"), Some(91.0)); // Linux procps
        assert_eq!(parse_cputime_secs("0:00.03"), Some(0.03)); // BSD, short-lived
        assert_eq!(parse_cputime_secs("12:34"), Some(754.0));
        assert_eq!(parse_cputime_secs("1:02:03"), Some(3_723.0));
        assert_eq!(parse_cputime_secs("2-03:04:05"), Some(183_845.0));
        assert_eq!(parse_cputime_secs("  00:00:00  "), Some(0.0));
    }

    #[test]
    fn unparsable_cputime_is_none_so_the_rate_stays_unknown() {
        for bad in ["", "   ", "-", "abc", "1:2:3:4", "x:01:31", "01:31:xy"] {
            assert_eq!(parse_cputime_secs(bad), None, "{bad:?} must not parse");
        }
    }

    /// The parser is written against two man pages, only one of which documents
    /// its own format. This runs the real `ps` on whatever platform the suite
    /// is executing on and asserts the parser accepts what it prints — so a
    /// rendering neither man page describes fails here instead of silently
    /// turning the CPU signal off in production.
    #[test]
    fn cpu_matches_this_platforms_ps() {
        let pid = std::process::id();
        let out = std::process::Command::new("ps")
            .args(["-o", "cputime=", "-p", &pid.to_string()])
            .output()
            .expect("ps must be available");
        assert!(out.status.success(), "ps failed for our own pid");
        let field = String::from_utf8_lossy(&out.stdout);
        let field = field.trim();
        assert!(
            parse_cputime_secs(field).is_some(),
            "this platform's `ps -o cputime=` prints {field:?}, which parse_cputime_secs rejects"
        );
    }
}
