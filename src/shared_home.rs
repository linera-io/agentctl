//! Provider-neutral paths, rendered into each agent product's native home.
//!
//! The shared home owns durable, user-authored state. Provider configuration
//! and transcripts remain in their native homes and are only referenced by
//! generated adapters or the unified session index.

use std::path::{Path, PathBuf};

/// One agent product agentctl renders adapters for.
///
/// Every product-specific path lives here, so supporting another one is a row
/// rather than an edit in `classify`, the adapter list and the plan.
pub struct Provider {
    pub name: &'static str,
    /// Config directory relative to `$HOME`, empty for a product with none.
    pub home_dir: &'static str,
    /// Global-instructions file, relative to `$HOME`, rendered from the shared
    /// source.
    pub instructions: &'static str,
    /// Subdirectory of `home_dir` holding immutable transcripts, if any.
    pub transcripts: Option<&'static str>,
    /// Where the product keeps a per-project memory graph we can adopt,
    /// relative to `home_dir`, with `*` standing for the project segment.
    pub memory: Option<&'static str>,
}

/// Every product we render for.
///
/// `agents-md` is the cross-vendor `~/AGENTS.md` convention rather than one
/// product, so it has no config directory of its own.
pub const PROVIDERS: &[Provider] = &[
    Provider {
        name: "claude-code",
        home_dir: ".claude",
        instructions: ".claude/CLAUDE.md",
        transcripts: Some("projects"),
        memory: Some("projects/*/memory"),
    },
    Provider {
        name: "codex",
        home_dir: ".codex",
        instructions: ".codex/AGENTS.md",
        transcripts: Some("sessions"),
        memory: None,
    },
    Provider {
        name: "agents-md",
        home_dir: "",
        instructions: "AGENTS.md",
        transcripts: None,
        memory: None,
    },
];

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
    /// Adapters we rendered, unchanged since, whose shared source has moved on.
    /// These are ours to re-render — that is how an edit to the shared
    /// instructions reaches each provider.
    pub stale: Vec<PathBuf>,
    /// An established memory graph in some product's tree, if one was found.
    /// Adopted by reference rather than copied.
    pub adoptable_memory: Option<PathBuf>,
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

    /// Where the bytes last rendered to `target` are recorded.
    ///
    /// Without this, "the adapter differs from the shared source" is
    /// indistinguishable between *you edited the adapter* and *the source moved
    /// on* — which need opposite responses. The stamp is the memory that tells
    /// them apart.
    fn stamp_for(&self, target: &Path) -> PathBuf {
        let flattened = target
            .strip_prefix(&self.home)
            .unwrap_or(target)
            .to_string_lossy()
            .replace(['/', '\\'], "_");
        self.adapters().join(format!("{flattened}.rendered"))
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

        for provider in PROVIDERS {
            let target = self.home.join(provider.instructions);
            if has_drifted(&target) {
                actions.push(Action::ReportDrift {
                    target,
                    reason: "edited since it was generated; refusing to overwrite".to_string(),
                });
                continue;
            }
            let is_stale = observed.stale.iter().any(|p| p == &target);
            if !exists(&target) || is_stale {
                actions.push(Action::RenderAdapter {
                    target,
                    from: self.global_instructions(),
                });
            }
        }

        if let Some(source) = &observed.adoptable_memory {
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
        // What `RenderAdapter` would write, so an adapter can be compared
        // against it rather than merely counted as present.
        let rendered = std::fs::read(self.global_instructions()).ok();
        let mut drifted = Vec::new();
        let mut stale = Vec::new();
        for provider in PROVIDERS {
            let target = self.home.join(provider.instructions);
            if !target.exists() {
                continue;
            }
            let Ok(current) = std::fs::read(&target) else {
                existing.push(target);
                continue;
            };
            match std::fs::read(self.stamp_for(&target)) {
                // We rendered it and it still matches: ours to update.
                Ok(stamp) if stamp == current => {
                    if rendered.as_ref().is_some_and(|source| source != &current) {
                        stale.push(target.clone());
                    }
                }
                // Either a file we never wrote, or one we wrote and a human has
                // since changed. Both are theirs; report, never overwrite.
                _ => drifted.push(target.clone()),
            }
            existing.push(target);
        }

        // The adopt pointer is a plan target too. Omitting it here is what made
        // `AdoptInPlace` re-plan on every run: `plan()` tests for this exact
        // path, so an observation that can never contain it can never settle.
        let adopted = self.memory().join("adopted-graph");
        if adopted.exists() {
            existing.push(adopted);
        }

        Observed {
            existing,
            drifted,
            stale,
            adoptable_memory: self.discover_adoptable_memory(),
        }
    }

    /// The one adoptable memory graph across every product, if there is exactly
    /// one.
    ///
    /// Ambiguity is reported as "none" rather than guessed: adopting the wrong
    /// project's graph would point every provider at someone else's notes, and a
    /// missed adoption is a no-op the user can correct. The count spans all
    /// products, so one graph each in two products is ambiguous, not two
    /// answers.
    fn discover_adoptable_memory(&self) -> Option<PathBuf> {
        let mut found: Vec<PathBuf> = Vec::new();
        for provider in PROVIDERS {
            let Some(pattern) = provider.memory else {
                continue;
            };
            let Some((prefix, suffix)) = pattern.split_once("/*/") else {
                continue;
            };
            let root = self.home.join(provider.home_dir).join(prefix);
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            found.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path().join(suffix))
                    .filter(|path| path.is_dir()),
            );
        }
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
                    // Record what we wrote, so a later run can tell its own
                    // output from a human's edit.
                    let stamp = self.stamp_for(target);
                    if let Some(parent) = stamp.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(stamp, &content)?;
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
        // Codex discovers and rewrites this catalog implicitly, so agentctl must
        // treat it as provider-owned even though it sits inside the shared home.
        if path.starts_with(self.root.join("plugins")) {
            return PathOwnership::ProviderNative;
        }

        if path.starts_with(&self.root) {
            return PathOwnership::SharedSource;
        }

        if PROVIDERS
            .iter()
            .any(|provider| path == self.home.join(provider.instructions))
        {
            return PathOwnership::GeneratedAdapter;
        }

        // Transcripts before native config: a transcript lives inside the
        // provider's home, so the broader check would swallow it.
        for provider in PROVIDERS {
            let Some(transcripts) = provider.transcripts else {
                continue;
            };
            let dir = self.home.join(provider.home_dir).join(transcripts);
            if path.starts_with(&dir)
                && path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
            {
                return PathOwnership::TranscriptEvidence;
            }
        }

        for provider in PROVIDERS {
            if provider.home_dir.is_empty() {
                continue;
            }
            if path.starts_with(self.home.join(provider.home_dir)) {
                return PathOwnership::ProviderNative;
            }
        }

        PathOwnership::External
    }
}
