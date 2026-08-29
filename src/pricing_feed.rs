//! Live model prices, refreshed from LiteLLM's public price table.
//!
//! The static table in `models.rs` is a point-in-time snapshot, and it goes
//! stale *silently*: Claude Fable 5 shipped, matched no entry, and was billed
//! at the unknown-model fallback for three months before anyone noticed. A
//! fresher hard-coded table would only postpone that. This fetches the prices
//! instead, and demotes the static table to the offline fallback.
//!
//! LiteLLM is the same source `ccusage` uses. Its numbers were checked against
//! <https://platform.claude.com/docs/en/about-claude/pricing> on 2026-08-29 and
//! matched exactly for every model in a real corpus, Claude and GPT alike.
//!
//! Prices are cached on disk and refreshed at most daily. Every failure mode —
//! no network, malformed JSON, a nonsense price — falls back to the static
//! table rather than guessing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use serde_json::Value;

use crate::models::ModelProfile;

const FEED_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
const CACHE_BASENAME: &str = "model_prices.json";
const REFRESH_AFTER: Duration = Duration::from_secs(24 * 60 * 60);
const FETCH_TIMEOUT_SECS: u64 = 20;

/// Reject a parsed price outside this range. A price of zero would silently
/// make a model free; the upper bound catches a units mix-up (the feed is
/// per-token, so anything past $1000/MTok means we misread the scale).
const MAX_SANE_PER_M: f64 = 1000.0;

static REFRESH_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

fn cache_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    crate::product::shared_state_root(&home).join(CACHE_BASENAME)
}

/// True when the cache is missing or older than `REFRESH_AFTER`.
fn is_stale(path: &std::path::Path, now: SystemTime) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return true;
    };
    let Ok(modified) = meta.modified() else {
        return true;
    };
    now.duration_since(modified)
        .map(|age| age >= REFRESH_AFTER)
        .unwrap_or(false)
}

/// Load cached prices into `models`, then refresh the cache in the background
/// if it has aged out. Call once at startup: the current process uses whatever
/// is already on disk, and the fetch benefits the next run rather than
/// blocking this one.
pub fn install_and_refresh() {
    load_cached();
    if is_stale(&cache_path(), SystemTime::now()) {
        refresh_in_background();
    }
}

fn load_cached() {
    let path = cache_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return;
    };
    match parse_feed(&raw) {
        Ok(prices) => {
            let n = prices.len();
            report_disagreements(&prices);
            crate::models::set_feed_prices(prices);
            crate::logger::log(
                "INFO",
                &format!("model prices: {n} loaded from {}", path.display()),
            );
        }
        Err(e) => crate::logger::log(
            "WARN",
            &format!(
                "model prices: {} is unusable ({e}); falling back to the built-in table",
                path.display()
            ),
        ),
    }
}

fn refresh_in_background() {
    if REFRESH_IN_FLIGHT
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let work = || {
        match fetch(FEED_URL) {
            Ok(body) => match parse_feed(&body) {
                // Only overwrite the cache with something we could parse, so a
                // truncated or reshaped feed can't cost us the last good copy.
                Ok(prices) => {
                    let path = cache_path();
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if std::fs::write(&path, &body).is_ok() {
                        let n = prices.len();
                        report_disagreements(&prices);
                        crate::models::set_feed_prices(prices);
                        crate::logger::log("INFO", &format!("model prices: refreshed, {n} models"));
                    }
                }
                Err(e) => crate::logger::log(
                    "WARN",
                    &format!(
                        "model prices: fetched feed is unusable ({e}); keeping the cached copy"
                    ),
                ),
            },
            Err(e) => crate::logger::log(
                "WARN",
                &format!("model prices: refresh failed ({e}); keeping the cached copy"),
            ),
        }
        REFRESH_IN_FLIGHT.store(false, Ordering::Release);
    };

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(work);
    } else {
        std::thread::Builder::new()
            .name("price-refresh".into())
            .spawn(work)
            .map_err(|_| REFRESH_IN_FLIGHT.store(false, Ordering::Release))
            .ok();
    }
}

/// Shelling out to curl, as the rest of the codebase does rather than carrying
/// an HTTP stack for a once-a-day GET.
fn fetch(url: &str) -> Result<String, String> {
    let output = Command::new("curl")
        .args([
            "-sS",
            "--fail",
            "--max-time",
            &FETCH_TIMEOUT_SECS.to_string(),
            url,
        ])
        .output()
        .map_err(|e| format!("curl failed to start: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "curl exit {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("response was not utf-8: {e}"))
}

/// LiteLLM quotes per-token USD; we bill per million.
fn per_m(entry: &Value, field: &str) -> Option<f64> {
    let v = entry.get(field)?.as_f64()? * 1_000_000.0;
    (v.is_finite() && v > 0.0 && v <= MAX_SANE_PER_M).then_some(v)
}

fn parse_feed(raw: &str) -> Result<HashMap<String, ModelProfile>, String> {
    let root: Value = serde_json::from_str(raw).map_err(|e| format!("bad json: {e}"))?;
    let obj = root.as_object().ok_or("top level is not an object")?;

    let mut out = HashMap::new();
    for (name, entry) in obj {
        // Input and output are the two we cannot infer; without both, the
        // entry is unusable and the static table serves the model better.
        let (Some(input), Some(output)) = (
            per_m(entry, "input_cost_per_token"),
            per_m(entry, "output_cost_per_token"),
        ) else {
            continue;
        };
        // Providers that don't bill cache writes separately omit the field;
        // those tokens bill as ordinary input.
        let cache_write = per_m(entry, "cache_creation_input_token_cost").unwrap_or(input);
        out.insert(
            name.to_lowercase(),
            ModelProfile {
                input_per_m: input,
                output_per_m: output,
                cache_read_per_m: per_m(entry, "cache_read_input_token_cost").unwrap_or(input),
                cache_write_per_m: cache_write,
                cache_write_1h_per_m: per_m(entry, "cache_creation_input_token_cost_above_1hr")
                    .unwrap_or(cache_write),
                context_max: entry
                    .get("max_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            },
        );
    }
    if out.is_empty() {
        return Err("no usable entries".into());
    }
    Ok(out)
}

/// Log where the feed and the built-in table materially disagree.
///
/// This is the staleness signal: a divergence means either the static table has
/// aged out (the common case, and exactly what went unnoticed for three months)
/// or the feed is wrong. Either way someone should look, and the numbers are
/// moving under us silently otherwise.
fn report_disagreements(prices: &HashMap<String, ModelProfile>) {
    const MATERIAL: f64 = 0.20;
    for (fragment, built_in) in crate::models::built_in_prices() {
        let Some(feed) = feed_entry_for(prices, &fragment) else {
            continue;
        };
        let diff = (feed.input_per_m - built_in.input_per_m).abs() / built_in.input_per_m;
        if diff > MATERIAL {
            crate::logger::log(
                "WARN",
                &format!(
                    "model prices: '{fragment}' built-in ${:.2}/MTok input vs feed ${:.2} — \
                     the built-in table has drifted; update MODELS in models.rs",
                    built_in.input_per_m, feed.input_per_m
                ),
            );
        }
    }
}

/// The built-in table is keyed by id *fragment* (`opus-5`) while the feed is
/// keyed by full model name (`claude-opus-5`). Prefer an exact hit, then the
/// shortest containing key so a dated snapshot doesn't stand in for the
/// canonical entry.
fn feed_entry_for<'a>(
    prices: &'a HashMap<String, ModelProfile>,
    fragment: &str,
) -> Option<&'a ModelProfile> {
    if let Some(hit) = prices.get(fragment) {
        return Some(hit);
    }
    prices
        .iter()
        .filter(|(name, _)| name.contains(fragment))
        .min_by_key(|(name, _)| name.len())
        .map(|(_, profile)| profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "claude-opus-5": {
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 2.5e-05,
            "cache_read_input_token_cost": 5e-07,
            "cache_creation_input_token_cost": 6.25e-06,
            "cache_creation_input_token_cost_above_1hr": 1e-05,
            "max_input_tokens": 1000000
        },
        "gpt-5-codex": {
            "input_cost_per_token": 1.25e-06,
            "output_cost_per_token": 1e-05,
            "cache_read_input_token_cost": 1.25e-07,
            "max_input_tokens": 272000
        },
        "no-prices": { "max_input_tokens": 1000 },
        "free-model": { "input_cost_per_token": 0, "output_cost_per_token": 0 },
        "absurd": { "input_cost_per_token": 1.0, "output_cost_per_token": 1.0 }
    }"#;

    #[test]
    fn parses_anthropic_entry_into_per_million_prices() {
        let p = parse_feed(SAMPLE).unwrap();
        let opus = p.get("claude-opus-5").expect("opus present");
        assert_eq!(opus.input_per_m, 5.0);
        assert_eq!(opus.output_per_m, 25.0);
        assert_eq!(opus.cache_read_per_m, 0.5);
        assert_eq!(opus.cache_write_per_m, 6.25);
        assert_eq!(opus.cache_write_1h_per_m, 10.0);
        assert_eq!(opus.context_max, 1_000_000);
    }

    /// OpenAI doesn't bill cache writes separately, so the field is absent and
    /// those tokens must bill as ordinary input rather than as free.
    #[test]
    fn missing_cache_write_falls_back_to_input() {
        let p = parse_feed(SAMPLE).unwrap();
        let codex = p.get("gpt-5-codex").expect("codex present");
        assert_eq!(codex.input_per_m, 1.25);
        assert_eq!(codex.output_per_m, 10.0);
        assert_eq!(codex.cache_read_per_m, 0.125);
        assert_eq!(codex.cache_write_per_m, 1.25);
        assert_eq!(codex.cache_write_1h_per_m, 1.25);
    }

    #[test]
    fn entries_without_both_base_prices_are_skipped() {
        let p = parse_feed(SAMPLE).unwrap();
        assert!(!p.contains_key("no-prices"));
        assert!(
            !p.contains_key("free-model"),
            "a zero price would bill as free"
        );
    }

    /// $1/token is $1,000,000/MTok — a units mix-up, not a price.
    #[test]
    fn absurd_prices_are_rejected() {
        let p = parse_feed(SAMPLE).unwrap();
        assert!(!p.contains_key("absurd"));
    }

    #[test]
    fn malformed_feeds_are_errors_not_empty_tables() {
        assert!(parse_feed("not json").is_err());
        assert!(parse_feed("[]").is_err());
        assert!(
            parse_feed("{}").is_err(),
            "an empty table must not look like success"
        );
    }

    #[test]
    fn staleness_is_measured_against_the_cache_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prices.json");
        assert!(is_stale(&path, SystemTime::now()), "missing cache is stale");

        std::fs::write(&path, "{}").unwrap();
        assert!(
            !is_stale(&path, SystemTime::now()),
            "just-written cache is fresh"
        );
        assert!(
            is_stale(
                &path,
                SystemTime::now() + REFRESH_AFTER + Duration::from_secs(1)
            ),
            "cache past the TTL is stale"
        );
    }
}

#[cfg(test)]
mod live_feed {
    use super::*;

    /// The static table must still agree with the feed. Run on demand (it hits
    /// the network) to answer "has models.rs gone stale?" — the question that
    /// went unasked for the three months Fable 5 was mispriced.
    #[test]
    #[ignore = "hits the network; run with --ignored to audit the static table"]
    fn built_in_table_still_agrees_with_the_feed() {
        let body = fetch(FEED_URL).expect("feed fetch");
        let prices = parse_feed(&body).expect("feed parse");

        let mut drifted = Vec::new();
        for (fragment, built_in) in crate::models::built_in_prices() {
            let Some(feed) = feed_entry_for(&prices, &fragment) else {
                continue;
            };
            for (field, ours, theirs) in [
                ("input", built_in.input_per_m, feed.input_per_m),
                ("output", built_in.output_per_m, feed.output_per_m),
                (
                    "cache_read",
                    built_in.cache_read_per_m,
                    feed.cache_read_per_m,
                ),
                (
                    "cache_write_5m",
                    built_in.cache_write_per_m,
                    feed.cache_write_per_m,
                ),
                (
                    "cache_write_1h",
                    built_in.cache_write_1h_per_m,
                    feed.cache_write_1h_per_m,
                ),
            ] {
                if (ours - theirs).abs() > 1e-9 {
                    drifted.push(format!(
                        "{fragment} {field}: built-in {ours} vs feed {theirs}"
                    ));
                }
            }
        }
        assert!(
            drifted.is_empty(),
            "static table has drifted:\n  {}",
            drifted.join("\n  ")
        );
    }
}
