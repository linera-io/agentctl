//! Claude Code, behind [`AgentProviderAdapter`].
//!
//! Every method delegates to the function that already implemented it, so this
//! extraction cannot change behaviour: there is one implementation, now reached
//! through an interface. Moving the logic in here would make parity something
//! to verify rather than something that holds by construction.

use std::path::{Path, PathBuf};

use super::AgentProviderAdapter;
use crate::provider::AgentProvider;
use crate::transcript::TranscriptEvent;

pub struct ClaudeProvider;

impl AgentProviderAdapter for ClaudeProvider {
    fn provider(&self) -> AgentProvider {
        AgentProvider::Claude
    }

    fn executable(&self) -> &'static str {
        AgentProvider::Claude.executable()
    }

    fn transcript_root(&self, home: &Path) -> PathBuf {
        home.join(AgentProvider::Claude.home_dir())
            .join(AgentProvider::Claude.transcript_root())
    }

    fn matches_process(&self, command: &str) -> bool {
        crate::process::is_claude_process(command)
    }

    fn launch_args(&self, prompt: Option<&str>, resume: Option<&str>) -> Vec<String> {
        crate::terminals::build_claude_args(prompt, resume)
    }

    fn supports_rename(&self) -> bool {
        true
    }

    fn parse_transcript_line(&self, line: &str) -> Option<TranscriptEvent> {
        crate::transcript::parse_line(line)
    }
}
