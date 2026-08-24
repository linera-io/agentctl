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

    /// `codex [resume <id>] [PROMPT]` — verified against `codex --help` and
    /// `codex resume --help` on codex-cli 0.148.0.
    ///
    /// Resume is a subcommand, not Claude's `--resume` flag, and the prompt is
    /// POSITIONAL on both forms rather than behind `-p`. Order matters:
    /// `codex resume [SESSION_ID] [PROMPT]`, so a prompt without a resume id
    /// would be read as the id.
    fn launch_args(&self, prompt: Option<&str>, resume: Option<&str>) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(id) = resume {
            args.push("resume".to_string());
            args.push(id.to_string());
        }
        if let Some(text) = prompt {
            args.push(text.to_string());
        }
        args
    }

    /// Codex exposes no rename.
    ///
    /// `codex --help` on 0.148.0 offers `archive`, `unarchive`, `delete` and
    /// `fork`, but nothing that renames a session. Writing a name into the
    /// rollout ourselves would mean agentctl mutating a file the product owns,
    /// which the ownership model forbids.
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
