//! Which agent product a session belongs to.
//!
//! Carried on the session record rather than inferred at each use, for the same
//! reason [`crate::session::SessionOrigin`] is: a value that decides how we
//! address a session must travel with it, or two call sites will eventually
//! disagree about what they are looking at.

use serde::{Deserialize, Serialize};

/// An agent product that runs sessions.
///
/// This is narrower than [`crate::shared_home::PROVIDERS`], which also covers
/// the `AGENTS.md` file convention — a render target, not something that has
/// processes or transcripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentProvider {
    Claude,
    Codex,
}

impl AgentProvider {
    /// Every product, in display order.
    pub fn all() -> &'static [AgentProvider] {
        &[Self::Claude, Self::Codex]
    }

    /// Label for the PROVIDER column.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Claude => "Claude",
            Self::Codex => "Codex",
        }
    }

    /// argv0 of the product's CLI, as it appears in a `ps` row.
    pub fn executable(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
        }
    }

    /// The product's config directory, relative to `$HOME`.
    pub fn home_dir(&self) -> &'static str {
        match self {
            Self::Claude => ".claude",
            Self::Codex => ".codex",
        }
    }

    /// Directory under [`Self::home_dir`] holding session transcripts.
    pub fn transcript_root(&self) -> &'static str {
        match self {
            Self::Claude => "projects",
            Self::Codex => "sessions",
        }
    }
}

/// Claude, because every session predating this field was one.
impl Default for AgentProvider {
    fn default() -> Self {
        Self::Claude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_product_has_distinct_identity() {
        let labels: Vec<&str> = AgentProvider::all().iter().map(|p| p.label()).collect();
        let exes: Vec<&str> = AgentProvider::all()
            .iter()
            .map(|p| p.executable())
            .collect();
        assert_eq!(labels, ["Claude", "Codex"]);
        assert_eq!(exes, ["claude", "codex"]);
        assert_eq!(
            AgentProvider::Claude.transcript_root(),
            "projects",
            "Claude keeps transcripts under ~/.claude/projects"
        );
        assert_eq!(
            AgentProvider::Codex.transcript_root(),
            "sessions",
            "Codex keeps them under ~/.codex/sessions"
        );
    }

    #[test]
    fn a_session_written_before_this_field_reads_back_as_claude() {
        assert_eq!(AgentProvider::default(), AgentProvider::Claude);
    }

    #[test]
    fn serde_round_trips_through_a_stable_lowercase_tag() {
        for provider in AgentProvider::all() {
            let json = serde_json::to_string(provider).unwrap();
            assert_eq!(
                serde_json::from_str::<AgentProvider>(&json).unwrap(),
                *provider
            );
        }
        // The wire form is the tag a registry file already holds, so pin it.
        assert_eq!(
            serde_json::to_string(&AgentProvider::Claude).unwrap(),
            "\"claude\""
        );
        assert_eq!(
            serde_json::to_string(&AgentProvider::Codex).unwrap(),
            "\"codex\""
        );
    }
}
