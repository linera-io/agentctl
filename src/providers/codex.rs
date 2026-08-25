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

    /// Renaming needs the App Server, which agentctl does not yet drive.
    ///
    /// The CLI has no rename — `codex --help` offers `archive`, `unarchive`,
    /// `delete` and `fork` — which is why this said "Codex cannot rename". The
    /// generated protocol schema corrects that: `thread/name/set` is a real
    /// request and `thread/name/updated` a real notification. So the capability
    /// exists and the honest answer is "not over the transport we currently
    /// use", which flips once the App Server client is wired in.
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

/// The agent's working root from a `codex` command line, if it set one.
///
/// `codex -C/--cd <DIR>` moves the agent's root WITHOUT changing the process's
/// own cwd, so `/proc/<pid>/cwd` reports where the shell was, while the rollout
/// records `<DIR>`. Correlating on the process cwd alone therefore misses every
/// session started that way — silently, as an absent row rather than a wrong
/// one. Verified against `codex --help` on codex-cli 0.148.0.
///
/// Both the separate form (`-C dir`) and the joined forms (`--cd=dir`, `-Cdir`)
/// are accepted, because all three reach the same clap argument.
pub fn working_root_override(args: &str) -> Option<String> {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];
        if let Some(dir) = token.strip_prefix("--cd=") {
            return non_empty(dir);
        }
        if token == "--cd" || token == "-C" {
            return tokens.get(i + 1).and_then(|d| non_empty(d));
        }
        if let Some(dir) = token.strip_prefix("-C") {
            // clap accepts `-C=dir` as well as `-Cdir`.
            let dir = dir.strip_prefix('=').unwrap_or(dir);
            if !dir.starts_with('-') {
                return non_empty(dir);
            }
        }
        i += 1;
    }
    None
}

fn non_empty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}
