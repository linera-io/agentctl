use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelProfile {
    pub input_per_m: f64,
    pub output_per_m: f64,
    pub cache_read_per_m: f64,
    /// 5-minute cache write (1.25x base input).
    pub cache_write_per_m: f64,
    /// 1-hour cache write (2x base input). Transcripts report the split via
    /// `usage.cache_creation.ephemeral_1h_input_tokens`; 64% of cache-write
    /// tokens on a real corpus are 1-hour, so charging them at the 5m rate
    /// undercounts materially.
    pub cache_write_1h_per_m: f64,
    pub context_max: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelOverride {
    pub name: String,
    pub profile: ModelProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelProfileSource {
    BuiltIn,
    Override,
    /// The refreshed price table — see `pricing_feed`.
    Feed,
    Fallback,
}

impl ModelProfileSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::BuiltIn => "built-in",
            Self::Override => "override",
            Self::Feed => "feed",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModelProfile {
    pub key: String,
    pub profile: ModelProfile,
    pub source: ModelProfileSource,
}

static MODEL_OVERRIDES: OnceLock<Mutex<HashMap<String, ModelProfile>>> = OnceLock::new();

/// Prices from the refreshed feed, keyed by lowercased model id. Empty until
/// `pricing_feed::install_and_refresh` runs, which is what makes the static
/// table below the offline fallback rather than the source of truth.
static FEED_PRICES: OnceLock<Mutex<HashMap<String, ModelProfile>>> = OnceLock::new();

fn feed_store() -> &'static Mutex<HashMap<String, ModelProfile>> {
    FEED_PRICES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn set_feed_prices(prices: HashMap<String, ModelProfile>) {
    if let Ok(mut guard) = feed_store().lock() {
        *guard = prices;
    }
}

/// The static table, for comparison against the feed.
pub fn built_in_prices() -> Vec<(String, ModelProfile)> {
    MODELS
        .iter()
        .map(|e| (e.fragment.to_string(), e.profile))
        .collect()
}

/// One published model version: the id fragment that identifies it, the label
/// shown in the UI, and its prices per million tokens.
struct ModelEntry {
    /// Matched against the model id with `.` normalised to `-`.
    fragment: &'static str,
    label: &'static str,
    profile: ModelProfile,
}

/// Cache rates are fixed multiples of base input for every published model,
/// so they are derived rather than transcribed per row.
/// <https://platform.claude.com/docs/en/about-claude/pricing#prompt-caching>
const CACHE_READ_MULTIPLIER: f64 = 0.1;
const CACHE_WRITE_5M_MULTIPLIER: f64 = 1.25;
const CACHE_WRITE_1H_MULTIPLIER: f64 = 2.0;

const fn entry(
    fragment: &'static str,
    label: &'static str,
    input: f64,
    output: f64,
    context_max: u64,
) -> ModelEntry {
    ModelEntry {
        fragment,
        label,
        profile: ModelProfile {
            input_per_m: input,
            output_per_m: output,
            cache_read_per_m: input * CACHE_READ_MULTIPLIER,
            cache_write_per_m: input * CACHE_WRITE_5M_MULTIPLIER,
            cache_write_1h_per_m: input * CACHE_WRITE_1H_MULTIPLIER,
            context_max,
        },
    }
}

/// Published prices, verified against
/// <https://platform.claude.com/docs/en/about-claude/pricing> on 2026-08-27.
///
/// Ordered most-specific-first: the first fragment the model id contains wins,
/// so `opus-4-1` must precede any shorter `opus` fragment. Versions are listed
/// explicitly because families no longer share a price — Sonnet 5 is $2/$10
/// while Sonnet 4.6 is $3/$15, and collapsing them overcharged every Sonnet 5
/// message.
const MODELS: &[ModelEntry] = &[
    entry("fable-5", "fable-5", 10.0, 50.0, 1_000_000),
    entry("mythos-5", "mythos-5", 10.0, 50.0, 1_000_000),
    entry("opus-4-1", "opus-4.1", 15.0, 75.0, 200_000),
    entry("opus-4-5", "opus-4.5", 5.0, 25.0, 200_000),
    entry("opus-4-6", "opus-4.6", 5.0, 25.0, 1_000_000),
    entry("opus-4-7", "opus-4.7", 5.0, 25.0, 1_000_000),
    entry("opus-4-8", "opus-4.8", 5.0, 25.0, 1_000_000),
    entry("opus-5", "opus-5", 5.0, 25.0, 1_000_000),
    entry("sonnet-4-5", "sonnet-4.5", 3.0, 15.0, 200_000),
    entry("sonnet-4-6", "sonnet-4.6", 3.0, 15.0, 1_000_000),
    entry("sonnet-5", "sonnet-5", 2.0, 10.0, 1_000_000),
    entry("haiku-3-5", "haiku-3.5", 0.8, 4.0, 200_000),
    entry("haiku-4-5", "haiku-4.5", 1.0, 5.0, 200_000),
];

/// Newest known version of each family, used when the id names a family but a
/// version we don't have a price for — a new release is far likelier than a
/// retired one. Always reported as `Fallback` so the guess is visible.
const FAMILY_DEFAULTS: &[(&str, &str)] = &[
    ("opus", "opus-5"),
    ("sonnet", "sonnet-5"),
    ("haiku", "haiku-4-5"),
    ("fable", "fable-5"),
    ("mythos", "mythos-5"),
];

fn normalise_id(model: &str) -> String {
    model.trim().to_lowercase().replace('.', "-")
}

fn lookup(fragment: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|e| e.fragment == fragment)
}

/// Exact version match on the model id, if we publish a price for it.
fn exact_entry(model: &str) -> Option<&'static ModelEntry> {
    let id = normalise_id(model);
    MODELS.iter().find(|e| id.contains(e.fragment))
}

/// Family match for an id whose version we don't recognise.
fn family_entry(model: &str) -> Option<&'static ModelEntry> {
    let id = normalise_id(model);
    FAMILY_DEFAULTS
        .iter()
        .find(|(family, _)| id.contains(family))
        .and_then(|(_, newest)| lookup(newest))
}

/// Short label for the UI. Falls back to the raw id so an unknown model is
/// visible as itself rather than silently bucketed into a priced family.
pub fn shorten_model(model: &str) -> String {
    if let Some(hit) = exact_entry(model) {
        return hit.label.into();
    }
    if let Some((family, _)) = FAMILY_DEFAULTS
        .iter()
        .find(|(family, _)| normalise_id(model).contains(family))
    {
        return (*family).into();
    }
    model.to_string()
}

pub fn set_overrides(overrides: Vec<ModelOverride>) {
    let store = MODEL_OVERRIDES.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut guard) = store.lock() else {
        return;
    };
    guard.clear();
    for mut override_ in overrides {
        // An override that omits the 1h rate would otherwise price 1h cache
        // writes at zero. Derive it from the 5m rate via the published
        // multipliers (2x base / 1.25x base).
        if override_.profile.cache_write_1h_per_m == 0.0 {
            override_.profile.cache_write_1h_per_m =
                override_.profile.cache_write_per_m / 1.25 * 2.0;
        }
        let raw = override_.name.trim().to_lowercase();
        let shortened = shorten_model(&override_.name).to_lowercase();
        guard.insert(raw, override_.profile);
        guard.insert(shortened, override_.profile);
    }
}

pub fn resolve(model: &str) -> ResolvedModelProfile {
    let empty = HashMap::new();
    let store = MODEL_OVERRIDES.get_or_init(|| Mutex::new(HashMap::new()));
    let guard = store.lock().ok();
    let overrides = guard.as_deref().unwrap_or(&empty);
    resolve_with_overrides(model, overrides)
}

pub(crate) fn resolve_with_overrides(
    model: &str,
    overrides: &HashMap<String, ModelProfile>,
) -> ResolvedModelProfile {
    let raw_key = model.trim().to_lowercase();
    let short_key = shorten_model(model).to_lowercase();

    if let Some(profile) = overrides
        .get(&raw_key)
        .or_else(|| overrides.get(&short_key))
        .copied()
    {
        return ResolvedModelProfile {
            key: if raw_key.is_empty() {
                short_key
            } else {
                raw_key
            },
            profile,
            source: ModelProfileSource::Override,
        };
    }

    // The feed outranks the static table: the table is a snapshot and the feed
    // is maintained, so a model released after our last release is priced
    // correctly instead of falling through to a guess. A user override still
    // wins over both.
    if let Some(profile) = feed_store()
        .lock()
        .ok()
        .and_then(|feed| feed.get(&raw_key).copied())
    {
        return ResolvedModelProfile {
            key: shorten_model(model),
            profile,
            source: ModelProfileSource::Feed,
        };
    }

    if let Some(hit) = exact_entry(model) {
        return ResolvedModelProfile {
            key: hit.label.into(),
            profile: hit.profile,
            source: ModelProfileSource::BuiltIn,
        };
    }

    // No price for this exact version. Guess the family's newest and say so —
    // Claude Fable 5 shipped priced at the unknown-model fallback and nobody
    // noticed for three months because this path was silent.
    if let Some(hit) = family_entry(model) {
        warn_unpriced(model, hit.label);
        return ResolvedModelProfile {
            key: short_key,
            profile: hit.profile,
            source: ModelProfileSource::Fallback,
        };
    }

    warn_unpriced(model, FALLBACK_LABEL);
    ResolvedModelProfile {
        key: if short_key.is_empty() {
            "unknown".into()
        } else {
            short_key
        },
        profile: fallback_profile(),
        source: ModelProfileSource::Fallback,
    }
}

/// Models already reported, so a per-message resolve doesn't spam the log.
static WARNED_MODELS: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();

/// Log an unpriced model once, naming the id and the profile we substituted.
fn warn_unpriced(model: &str, substituted: &str) {
    let store = WARNED_MODELS.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let Ok(mut seen) = store.lock() else {
        return;
    };
    let id = normalise_id(model);
    if !seen.insert(id.clone()) {
        return;
    }
    crate::logger::log(
        "WARN",
        &format!(
            "no published price for model '{id}' — billing it as '{substituted}'. \
             Add a [models.\"{id}\"] override to config, or update MODELS in models.rs."
        ),
    );
}

/// Reported for a model naming no known family. Priced at the current
/// flagship-Opus tier; any value here is a guess, which is why it is logged.
const FALLBACK_LABEL: &str = "opus-5 (unknown model)";

fn fallback_profile() -> ModelProfile {
    entry("", "", 5.0, 25.0, 200_000).profile
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cache rates are derived by multiplication, so exact float equality is
    /// the wrong comparison (3.0 * 0.1 != 0.3).
    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Serialize the tests that mutate the process-global override and feed
    /// tables — in parallel they clear each other's fixtures mid-assertion.
    /// Same convention as `usage_ledger`'s `cache_test_lock`.
    fn model_state_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn resolve_builtin_profile() {
        let resolved = resolve_with_overrides("claude-opus-4-6-20260401", &HashMap::new());
        assert_eq!(resolved.source, ModelProfileSource::BuiltIn);
        assert_eq!(resolved.profile.context_max, 1_000_000);
    }

    #[test]
    fn resolve_override_profile() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "gpt-4o".into(),
            ModelProfile {
                input_per_m: 1.0,
                output_per_m: 2.0,
                cache_read_per_m: 0.5,
                cache_write_per_m: 1.5,
                cache_write_1h_per_m: 2.4,
                context_max: 128_000,
            },
        );
        let resolved = resolve_with_overrides("gpt-4o", &overrides);
        assert_eq!(resolved.source, ModelProfileSource::Override);
        assert_eq!(resolved.profile.context_max, 128_000);
    }

    #[test]
    fn resolve_fallback_profile() {
        let resolved = resolve_with_overrides("mystery-model", &HashMap::new());
        assert_eq!(resolved.source, ModelProfileSource::Fallback);
        assert_eq!(resolved.profile.context_max, 200_000);
    }

    /// Every id observed in a real `~/.claude/projects` corpus, priced against
    /// the published table. Fable 5 matched no family for three months and was
    /// silently billed at the unknown-model fallback.
    #[test]
    fn published_prices_for_every_model_in_the_corpus() {
        let cases: &[(&str, f64, f64, f64, f64)] = &[
            // id, input, output, cache_read, cache_write_5m
            ("claude-fable-5", 10.0, 50.0, 1.00, 12.50),
            ("claude-opus-5", 5.0, 25.0, 0.50, 6.25),
            ("claude-opus-4-8", 5.0, 25.0, 0.50, 6.25),
            ("claude-opus-4-7", 5.0, 25.0, 0.50, 6.25),
            ("claude-opus-4-6", 5.0, 25.0, 0.50, 6.25),
            ("claude-sonnet-5", 2.0, 10.0, 0.20, 2.50),
            ("claude-sonnet-4-6", 3.0, 15.0, 0.30, 3.75),
            ("claude-haiku-4-5-20251001", 1.0, 5.0, 0.10, 1.25),
        ];
        for (id, input, output, read, write) in cases {
            let r = resolve_with_overrides(id, &HashMap::new());
            assert_eq!(r.source, ModelProfileSource::BuiltIn, "{id} is unpriced");
            assert_eq!(r.profile.input_per_m, *input, "{id} input");
            assert_eq!(r.profile.output_per_m, *output, "{id} output");
            assert!(near(r.profile.cache_read_per_m, *read), "{id} cache read");
            assert!(
                near(r.profile.cache_write_per_m, *write),
                "{id} cache write 5m"
            );
        }
    }

    /// Sonnet 5 ($2/$10) and Sonnet 4.6 ($3/$15) must not collapse together.
    #[test]
    fn same_family_different_versions_are_priced_apart() {
        let five = resolve_with_overrides("claude-sonnet-5", &HashMap::new());
        let four_six = resolve_with_overrides("claude-sonnet-4-6", &HashMap::new());
        assert_ne!(five.profile.input_per_m, four_six.profile.input_per_m);
        assert_eq!(five.key, "sonnet-5");
        assert_eq!(four_six.key, "sonnet-4.6");
    }

    /// A retired version must not be shadowed by its family prefix.
    #[test]
    fn specific_versions_win_over_family_prefixes() {
        let retired = resolve_with_overrides("claude-opus-4-1-20250805", &HashMap::new());
        assert_eq!(retired.profile.input_per_m, 15.0);
        assert_eq!(retired.key, "opus-4.1");
    }

    /// An unknown version of a known family is guessed, but reported as a guess.
    #[test]
    fn unknown_version_of_known_family_is_flagged() {
        let r = resolve_with_overrides("claude-opus-9-9", &HashMap::new());
        assert_eq!(r.source, ModelProfileSource::Fallback);
        assert_eq!(r.profile.input_per_m, 5.0);
    }

    /// Cache reads are 0.1x base input and 1h writes are 2x, per the published
    /// multipliers.
    #[test]
    fn cache_multipliers_match_published_ratios() {
        for id in ["claude-opus-5", "claude-sonnet-4-6", "claude-haiku-4-5"] {
            let p = resolve_with_overrides(id, &HashMap::new()).profile;
            assert!(
                (p.cache_read_per_m - p.input_per_m * 0.1).abs() < 1e-9,
                "{id} read"
            );
            assert!(
                (p.cache_write_per_m - p.input_per_m * 1.25).abs() < 1e-9,
                "{id} 5m"
            );
            assert!(
                (p.cache_write_1h_per_m - p.input_per_m * 2.0).abs() < 1e-9,
                "{id} 1h"
            );
        }
    }

    fn feed_profile(input: f64) -> ModelProfile {
        entry("", "", input, input * 5.0, 1_000_000).profile
    }

    /// The feed is maintained and the static table is a snapshot, so a model
    /// the feed prices differently must follow the feed — that is the whole
    /// point of fetching it.
    #[test]
    fn feed_outranks_the_built_in_table() {
        let _g = model_state_lock();
        set_feed_prices(HashMap::from([(
            "claude-opus-5".to_string(),
            feed_profile(7.5),
        )]));
        let r = resolve("claude-opus-5");
        assert_eq!(r.source, ModelProfileSource::Feed);
        assert_eq!(r.profile.input_per_m, 7.5);
        set_feed_prices(HashMap::new());

        // ...and with no feed loaded, the static table still answers.
        let r = resolve("claude-opus-5");
        assert_eq!(r.source, ModelProfileSource::BuiltIn);
        assert_eq!(r.profile.input_per_m, 5.0);
    }

    /// A user who has written a price into config has said what they want; the
    /// feed must not silently overrule it.
    #[test]
    fn override_outranks_the_feed() {
        let _g = model_state_lock();
        set_feed_prices(HashMap::from([(
            "claude-opus-5".to_string(),
            feed_profile(7.5),
        )]));
        set_overrides(vec![ModelOverride {
            name: "claude-opus-5".into(),
            profile: feed_profile(2.0),
        }]);
        let r = resolve("claude-opus-5");
        assert_eq!(r.source, ModelProfileSource::Override);
        assert_eq!(r.profile.input_per_m, 2.0);
        set_overrides(Vec::new());
        set_feed_prices(HashMap::new());
    }

    /// A model released after our last release has no static entry; the feed is
    /// what keeps it off the loud fallback.
    #[test]
    fn feed_prices_a_model_the_static_table_has_never_heard_of() {
        let _g = model_state_lock();
        set_feed_prices(HashMap::from([(
            "gpt-5-codex".to_string(),
            feed_profile(1.25),
        )]));
        let r = resolve("gpt-5-codex");
        assert_eq!(r.source, ModelProfileSource::Feed);
        assert_eq!(r.profile.input_per_m, 1.25);
        set_feed_prices(HashMap::new());

        // Without the feed it is an unpriced guess, as before.
        assert_eq!(resolve("gpt-5-codex").source, ModelProfileSource::Fallback);
    }

    /// An override that omits the 1h rate must not price 1h writes at zero.
    #[test]
    fn override_without_1h_rate_derives_it() {
        let _g = model_state_lock();
        set_overrides(vec![ModelOverride {
            name: "custom-model".into(),
            profile: ModelProfile {
                input_per_m: 4.0,
                output_per_m: 8.0,
                cache_read_per_m: 0.4,
                cache_write_per_m: 5.0,
                cache_write_1h_per_m: 0.0,
                context_max: 100_000,
            },
        }]);
        let r = resolve("custom-model");
        assert_eq!(r.source, ModelProfileSource::Override);
        assert_eq!(r.profile.cache_write_1h_per_m, 8.0);
        set_overrides(Vec::new());
    }
}
