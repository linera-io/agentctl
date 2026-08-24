//! Provider-neutral paths shared by Claude Code and Codex.
//!
//! The shared home owns durable, user-authored state. Provider configuration
//! and transcripts remain in their native homes and are only referenced by
//! generated adapters or the unified session index.

use std::path::{Path, PathBuf};

/// Instruction adapters rendered into each provider's native home, declared as
/// data so adding a provider is a registry entry rather than a `classify` arm.
const INSTRUCTION_ADAPTER_TARGETS: &[&str] =
    &[".claude/CLAUDE.md", "AGENTS.md", ".codex/AGENTS.md"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathOwnership {
    SharedSource,
    GeneratedAdapter,
    ProviderNative,
    TranscriptEvidence,
    External,
}

/// One MCP server, described once for every provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpServer {
    pub name: String,
    pub command: String,
    /// Environment for the server. Values must be `${VAR}` references, never
    /// literals — see [`McpRegistry::validated`].
    pub env: Vec<(String, String)>,
}

/// The provider-neutral MCP registry, validated on construction.
///
/// Construction is the only way in, so an unvalidated registry cannot be
/// rendered by accident.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpRegistry {
    servers: Vec<McpServer>,
}

impl McpRegistry {
    /// Build a registry, refusing any inline credential.
    ///
    /// This file is shared, version-controllable, and rendered into every
    /// provider's native config, so a literal secret here becomes a secret in
    /// all of them at once. Only `${VAR}` is accepted; the value is resolved
    /// from the environment or the host's own credential store at launch.
    ///
    /// Refuses rather than redacting. A redacting writer fails open on the one
    /// shape it did not anticipate, and the failure is silent.
    pub fn validated(mut servers: Vec<McpServer>) -> Result<Self, String> {
        for server in &servers {
            for (key, value) in &server.env {
                let trimmed = value.trim();
                let is_reference = trimmed.starts_with("${") && trimmed.ends_with('}');
                if !trimmed.is_empty() && !is_reference {
                    return Err(format!(
                        "MCP server '{}': env var {key} holds a literal value. \
                         Use ${{{key}}} and keep the secret in the environment or the \
                         host credential store — this registry is shared.",
                        server.name
                    ));
                }
            }
        }
        // Sorted at construction so every render is byte-identical for the same
        // set: an unstable order would make each reconcile look like a change
        // and hide real drift in the noise.
        servers.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { servers })
    }

    pub fn servers(&self) -> &[McpServer] {
        &self.servers
    }

    /// Render Claude Code's `.mcp.json` shape.
    pub fn render_claude(&self) -> String {
        let entries: Vec<String> = self
            .servers
            .iter()
            .map(|server| {
                let env: Vec<String> = server
                    .env
                    .iter()
                    .map(|(k, v)| format!("        {k:?}: {v:?}"))
                    .collect();
                format!(
                    "    {:?}: {{\n      \"command\": {:?},\n      \"env\": {{\n{}\n      }}\n    }}",
                    server.name,
                    server.command,
                    env.join(",\n")
                )
            })
            .collect();
        format!(
            "{{\n  \"mcpServers\": {{\n{}\n  }}\n}}\n",
            entries.join(",\n")
        )
    }
}

/// What a caller read off disk before planning.
///
/// Planning takes this rather than touching the filesystem itself, so the rules
/// are testable without arranging a home directory — and so a plan can be shown
/// to the user before anything is written.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Observed {
    /// Paths that already exist.
    pub existing: Vec<PathBuf>,
    /// Generated adapters a human has since edited. These are reported, never
    /// overwritten.
    pub drifted: Vec<PathBuf>,
    /// An established Claude auto-memory directory, if one was found. Adopted by
    /// reference rather than copied.
    pub claude_memory: Option<PathBuf>,
}

/// One step of a reconcile. There is deliberately no variant that removes
/// anything: adoption of an existing setup is the common case, and a reconcile
/// able to delete is one bad reading away from taking a user's own instructions
/// with it. Removal, when it is ever wanted, is a separate explicit operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action {
    CreateDir(PathBuf),
    /// Write a provider's native file from shared content.
    RenderAdapter {
        target: PathBuf,
        from: PathBuf,
    },
    /// Point a provider at existing content instead of copying it. This is how
    /// an established memory graph is adopted: thousands of notes stay where
    /// they are and keep their history.
    AdoptInPlace {
        target: PathBuf,
        source: PathBuf,
    },
    /// A generated file that no longer matches what we would generate.
    ReportDrift {
        target: PathBuf,
        reason: String,
    },
}

/// The full set of steps, in the order they should be applied.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Plan {
    pub actions: Vec<Action>,
}

impl Plan {
    /// Every path this plan would bring into existence. Feeding these back in as
    /// `Observed::existing` is what the idempotence test asserts on.
    pub fn targets(&self) -> Vec<PathBuf> {
        self.actions
            .iter()
            .filter_map(|action| match action {
                Action::CreateDir(path) => Some(path.clone()),
                Action::RenderAdapter { target, .. } => Some(target.clone()),
                Action::AdoptInPlace { target, .. } => Some(target.clone()),
                Action::ReportDrift { .. } => None,
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedAgentHome {
    home: PathBuf,
    root: PathBuf,
}

impl SharedAgentHome {
    pub fn from_home(home: &Path) -> Self {
        Self {
            home: home.to_path_buf(),
            root: home.join(".agents"),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn global_instructions(&self) -> PathBuf {
        self.root.join("instructions/global.md")
    }

    pub fn skills(&self) -> PathBuf {
        self.root.join("skills")
    }

    pub fn memory(&self) -> PathBuf {
        self.root.join("memory")
    }

    pub fn mcp_registry(&self) -> PathBuf {
        self.root.join("config/mcp.toml")
    }

    pub fn hook_registry(&self) -> PathBuf {
        self.root.join("config/hooks.toml")
    }

    pub fn workflow_registry(&self) -> PathBuf {
        self.root.join("workflows")
    }

    /// Holds the per-provider adapter declarations that drive rendering.
    pub fn adapters(&self) -> PathBuf {
        self.root.join("adapters")
    }

    pub fn session_index(&self) -> PathBuf {
        self.root.join("sessions/index.jsonl")
    }

    /// Directories the shared home owns, created before anything is written into
    /// them.
    fn owned_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.root.clone(),
            self.skills(),
            self.memory(),
            self.workflow_registry(),
            self.adapters(),
            self.root.join("instructions"),
            self.root.join("config"),
            self.root.join("sessions"),
        ]
    }

    /// Compute the reconcile without touching the filesystem.
    ///
    /// Pure so a plan can be shown before it is applied, and so the rules can be
    /// tested against a constructed `Observed` rather than a real home. Applying
    /// the plan and re-planning must produce nothing — that idempotence is what
    /// makes it safe to run on every hook event.
    pub fn plan(&self, observed: &Observed) -> Plan {
        let exists = |p: &Path| observed.existing.iter().any(|e| e == p);
        let has_drifted = |p: &Path| observed.drifted.iter().any(|e| e == p);
        let mut actions = Vec::new();

        for dir in self.owned_dirs() {
            if !exists(&dir) {
                actions.push(Action::CreateDir(dir));
            }
        }

        for relative in INSTRUCTION_ADAPTER_TARGETS {
            let target = self.home.join(relative);
            if has_drifted(&target) {
                actions.push(Action::ReportDrift {
                    target,
                    reason: "edited since it was generated; refusing to overwrite".to_string(),
                });
                continue;
            }
            if !exists(&target) {
                actions.push(Action::RenderAdapter {
                    target,
                    from: self.global_instructions(),
                });
            }
        }

        if let Some(source) = &observed.claude_memory {
            let target = self.memory().join("adopted-graph");
            if !exists(&target) {
                actions.push(Action::AdoptInPlace {
                    target,
                    source: source.clone(),
                });
            }
        }

        Plan { actions }
    }

    /// Read the filesystem into an `Observed`.
    ///
    /// The only function here that touches disk before `apply`. Kept separate so
    /// planning stays pure and a plan can be printed before anything is written.
    pub fn observe(&self) -> Observed {
        let mut existing = Vec::new();
        for path in self.owned_dirs() {
            if path.exists() {
                existing.push(path);
            }
        }
        for relative in INSTRUCTION_ADAPTER_TARGETS {
            let target = self.home.join(relative);
            if target.exists() {
                existing.push(target);
            }
        }
        Observed {
            existing,
            drifted: Vec::new(),
            claude_memory: self.discover_claude_memory(),
        }
    }

    /// The established Claude auto-memory directory, if there is exactly one.
    ///
    /// Ambiguity is reported as "none" rather than guessed: adopting the wrong
    /// project's graph would point every provider at someone else's notes, and a
    /// missed adoption is a no-op the user can correct.
    fn discover_claude_memory(&self) -> Option<PathBuf> {
        let projects = self.home.join(".claude/projects");
        let mut found: Vec<PathBuf> = std::fs::read_dir(projects)
            .ok()?
            .flatten()
            .map(|entry| entry.path().join("memory"))
            .filter(|path| path.is_dir())
            .collect();
        found.sort();
        match found.len() {
            1 => found.pop(),
            _ => None,
        }
    }

    /// Execute a plan.
    ///
    /// Writes are atomic (temp file then rename) so a crash mid-render cannot
    /// leave a provider reading half a file, and any file being replaced is
    /// backed up first. `ReportDrift` writes nothing by design.
    pub fn apply(&self, plan: &Plan) -> std::io::Result<()> {
        for action in &plan.actions {
            match action {
                Action::CreateDir(path) => std::fs::create_dir_all(path)?,
                Action::RenderAdapter { target, from } => {
                    let content = std::fs::read(from)?;
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    if target.exists() {
                        // Suffix, not overwrite: two renders in the same second
                        // must not have the second silently discard the first
                        // backup.
                        let mut nth = 0;
                        let backup = loop {
                            let candidate = target.with_file_name(format!(
                                "{}.bak{}",
                                target.file_name().unwrap_or_default().to_string_lossy(),
                                if nth == 0 {
                                    String::new()
                                } else {
                                    format!(".{nth}")
                                }
                            ));
                            if !candidate.exists() {
                                break candidate;
                            }
                            nth += 1;
                        };
                        std::fs::rename(target, backup)?;
                    }
                    let temp = target.with_extension("agentctl-tmp");
                    std::fs::write(&temp, &content)?;
                    std::fs::rename(&temp, target)?;
                }
                Action::AdoptInPlace { target, source } => {
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    // A pointer file, not a symlink: a symlink into another
                    // provider's tree makes that tree's layout load-bearing for
                    // everyone reading through it.
                    if !target.exists() {
                        std::fs::write(target, format!("{}\n", source.display()))?;
                    }
                }
                Action::ReportDrift { .. } => {}
            }
        }
        Ok(())
    }

    pub fn classify(&self, path: &Path) -> PathOwnership {
        let claude_home = self.home.join(".claude");
        let codex_home = self.home.join(".codex");

        // Codex discovers and rewrites this catalog implicitly, so agentctl must
        // treat it as provider-owned even though it sits inside the shared home.
        if path.starts_with(self.root.join("plugins")) {
            return PathOwnership::ProviderNative;
        }

        if path.starts_with(&self.root) {
            return PathOwnership::SharedSource;
        }

        let is_instruction_adapter = INSTRUCTION_ADAPTER_TARGETS
            .iter()
            .any(|relative| path == self.home.join(relative));
        if is_instruction_adapter {
            return PathOwnership::GeneratedAdapter;
        }

        let is_claude_transcript = path.starts_with(claude_home.join("projects"))
            && path
                .extension()
                .is_some_and(|extension| extension == "jsonl");
        let is_codex_transcript = path.starts_with(codex_home.join("sessions"))
            && path
                .extension()
                .is_some_and(|extension| extension == "jsonl");
        if is_claude_transcript || is_codex_transcript {
            return PathOwnership::TranscriptEvidence;
        }

        if path.starts_with(claude_home) || path.starts_with(codex_home) {
            return PathOwnership::ProviderNative;
        }

        PathOwnership::External
    }
}
