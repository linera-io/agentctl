//! Per-product behaviour, behind one interface.
//!
//! Every method returns plain data and performs no I/O, so a call site can be
//! switched to the adapter without changing what it does — spawning, reading
//! and writing stay in the thin wrappers that already own them. That split is
//! what makes the second product a new implementation rather than a new branch
//! in each of `discovery`, `process`, `transcript` and `terminals`.

use std::path::{Path, PathBuf};

use crate::provider::AgentProvider;
use crate::transcript::TranscriptEvent;

pub mod claude;

/// What agentctl needs to know about a product to drive it.
pub trait AgentProviderAdapter {
    /// Which product this adapter speaks for.
    fn provider(&self) -> AgentProvider;

    /// argv0 of the product's CLI.
    fn executable(&self) -> &'static str;

    /// Directory holding this product's session transcripts.
    fn transcript_root(&self, home: &Path) -> PathBuf;

    /// Is this `ps` command line one of the product's sessions?
    fn matches_process(&self, command: &str) -> bool;

    /// Arguments following the executable for a launch or resume.
    fn launch_args(&self, prompt: Option<&str>, resume: Option<&str>) -> Vec<String>;

    /// Can a session be renamed in place?
    fn supports_rename(&self) -> bool;

    /// Parse one line of the product's transcript format.
    fn parse_transcript_line(&self, line: &str) -> Option<TranscriptEvent>;
}

/// The adapter for a product, if one exists yet.
///
/// `None` for a product with no adapter, rather than a stand-in: falling back
/// to Claude would parse a foreign transcript with the wrong schema and launch
/// the wrong binary, reporting success either way. Callers must decide what to
/// do about a product they cannot drive.
///
/// Returns `&'static dyn` rather than a boxed value — adapters are stateless,
/// so there is nothing to own and no reason to allocate per call.
pub fn for_provider(provider: AgentProvider) -> Option<&'static dyn AgentProviderAdapter> {
    match provider {
        AgentProvider::Claude => Some(&claude::ClaudeProvider),
        AgentProvider::Codex => None,
    }
}
