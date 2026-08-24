//! Codex, behind [`AgentProviderAdapter`].
//!
//! Passive: this teaches agentctl to recognise and read Codex sessions. Driving
//! them — launch, resume, rename, focus — is Task 7, so `launch_args` covers
//! only the resume form already verifiable from the CLI's own surface.

use std::path::{Path, PathBuf};

use super::AgentProviderAdapter;
use crate::provider::AgentProvider;
use crate::transcript::TranscriptEvent;

pub struct CodexProvider;

impl AgentProviderAdapter for CodexProvider {
    fn provider(&self) -> AgentProvider {
        AgentProvider::Codex
    }

    fn executable(&self) -> &'static str {
        AgentProvider::Codex.executable()
    }

    fn transcript_root(&self, home: &Path) -> PathBuf {
        home.join(AgentProvider::Codex.home_dir())
            .join(AgentProvider::Codex.transcript_root())
    }

    /// Same argv0 rule as Claude, so `codex`, `codex resume <id>` and
    /// `codex --remote <url>` match while `codex-code-mode-host` — a real
    /// sibling binary the npm package ships — does not.
    fn matches_process(&self, command: &str) -> bool {
        crate::process::argv0_token_count(command, self.executable()).is_some()
    }

    /// `codex resume <id>` — a subcommand, not Claude's `--resume` flag.
    ///
    /// A prompt is not passed positionally here: Codex takes it on stdin or via
    /// `exec`, and guessing an argv shape would produce a command that looks
    /// right and fails at run time. Task 7 settles it against the real CLI.
    fn launch_args(&self, _prompt: Option<&str>, resume: Option<&str>) -> Vec<String> {
        match resume {
            Some(id) => vec!["resume".to_string(), id.to_string()],
            None => Vec::new(),
        }
    }

    /// Codex has no in-place session rename.
    fn supports_rename(&self) -> bool {
        false
    }

    /// Codex rollout lines are a different schema from Claude's transcript.
    ///
    /// Returning `None` rather than reaching for Claude's parser: the two share
    /// no field names, so parsing one as the other yields empty events that read
    /// as an idle session. [`super::codex_rollout`] is the real reader.
    fn parse_transcript_line(&self, _line: &str) -> Option<TranscriptEvent> {
        None
    }
}
