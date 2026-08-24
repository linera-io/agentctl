use std::path::Path;

use claudectl::shared_home::{PathOwnership, SharedAgentHome};

#[test]
fn canonical_layout_lives_under_dot_agents() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));

    assert_eq!(layout.root(), Path::new("/home/andre/.agents"));
    assert_eq!(
        layout.global_instructions(),
        Path::new("/home/andre/.agents/instructions/global.md")
    );
    assert_eq!(layout.skills(), Path::new("/home/andre/.agents/skills"));
    assert_eq!(layout.memory(), Path::new("/home/andre/.agents/memory"));
    assert_eq!(
        layout.mcp_registry(),
        Path::new("/home/andre/.agents/config/mcp.toml")
    );
    assert_eq!(
        layout.hook_registry(),
        Path::new("/home/andre/.agents/config/hooks.toml")
    );
    assert_eq!(
        layout.workflow_registry(),
        Path::new("/home/andre/.agents/workflows")
    );
    assert_eq!(
        layout.session_index(),
        Path::new("/home/andre/.agents/sessions/index.jsonl")
    );
}

#[test]
fn provider_mutable_state_is_not_owned_by_the_shared_home() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));

    assert_eq!(
        layout.classify(Path::new("/home/andre/.agents/memory/decision.md")),
        PathOwnership::SharedSource
    );
    assert_eq!(
        layout.classify(Path::new("/home/andre/.claude/CLAUDE.md")),
        PathOwnership::GeneratedAdapter
    );
    assert_eq!(
        layout.classify(Path::new("/home/andre/AGENTS.md")),
        PathOwnership::GeneratedAdapter
    );
    assert_eq!(
        layout.classify(Path::new("/home/andre/.claude/settings.json")),
        PathOwnership::ProviderNative
    );
    assert_eq!(
        layout.classify(Path::new("/home/andre/.codex/config.toml")),
        PathOwnership::ProviderNative
    );
    assert_eq!(
        layout.classify(Path::new(
            "/home/andre/.claude/projects/project/session.jsonl"
        )),
        PathOwnership::TranscriptEvidence
    );
    assert_eq!(
        layout.classify(Path::new(
            "/home/andre/.codex/sessions/2026/08/20/rollout.jsonl"
        )),
        PathOwnership::TranscriptEvidence
    );
}

#[test]
fn adapter_declarations_have_a_slot_in_the_shared_home() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));

    assert_eq!(layout.adapters(), Path::new("/home/andre/.agents/adapters"));
}

/// Codex reads `$CODEX_HOME/AGENTS.md` as its global instructions, verified
/// against codex-cli 0.148.0 with `codex debug prompt-input`.
#[test]
fn rendered_codex_instruction_adapter_is_a_generated_adapter() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));

    assert_eq!(
        layout.classify(Path::new("/home/andre/.codex/AGENTS.md")),
        PathOwnership::GeneratedAdapter
    );
}

/// codex-cli 0.148.0 implicitly discovers and rewrites `~/.agents/plugins`, so
/// the shared home cannot claim it without agentctl fighting Codex over writes.
#[test]
fn codex_plugin_catalog_inside_the_shared_home_stays_provider_native() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));

    assert_eq!(
        layout.classify(Path::new("/home/andre/.agents/plugins/marketplace.json")),
        PathOwnership::ProviderNative
    );
    assert_eq!(
        layout.classify(Path::new("/home/andre/.agents/plugins")),
        PathOwnership::ProviderNative
    );
    assert_eq!(
        layout.classify(Path::new("/home/andre/.agents/plugins-notes.md")),
        PathOwnership::SharedSource
    );
}

// ---- Step 2: adoption planning -------------------------------------------

use claudectl::shared_home::{Action, Observed};

/// A plan is computed from what was observed, never from the filesystem, so the
/// rules can be tested without arranging a home directory.
#[test]
fn a_fresh_home_is_planned_from_nothing_but_the_observation() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));
    let plan = layout.plan(&Observed::default());

    assert!(
        plan.actions
            .iter()
            .any(|a| matches!(a, Action::CreateDir(p) if p == &layout.skills())),
        "a fresh home creates its skills directory: {:?}",
        plan.actions
    );
    assert!(
        plan.actions.iter().any(|a| matches!(
            a,
            Action::RenderAdapter { target, .. } if target == Path::new("/home/andre/.claude/CLAUDE.md")
        )),
        "and renders the Claude instruction adapter"
    );
}

/// Every action is additive or informational.
///
/// There is no `Action::Remove`, so "the plan cannot delete" is enforced by the
/// type rather than asserted here — this test guards the weaker property the
/// type cannot express: that observing existing content produces *fewer*
/// actions, never compensating ones. If a `Remove` variant is ever added, this
/// match stops being exhaustive and the compiler makes someone justify it.
#[test]
fn planning_over_existing_content_only_adds_what_is_missing() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));
    let claude_md = Path::new("/home/andre/.claude/CLAUDE.md").to_path_buf();
    let observed = Observed {
        existing: vec![claude_md.clone(), layout.skills()],
        ..Observed::default()
    };

    let plan = layout.plan(&observed);
    for action in &plan.actions {
        match action {
            Action::CreateDir(_)
            | Action::RenderAdapter { .. }
            | Action::AdoptInPlace { .. }
            | Action::ReportDrift { .. } => {}
        }
    }
    assert!(
        !plan.targets().contains(&claude_md),
        "an adapter that already exists is not re-rendered"
    );
    assert!(
        !plan.targets().contains(&layout.skills()),
        "a directory that already exists is not re-created"
    );
}

/// An adapter a human edited is not ours to overwrite. The plan reports drift
/// and leaves it; silently replacing it is how a user loses instructions they
/// wrote themselves.
#[test]
fn an_independently_edited_adapter_is_reported_not_replaced() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));
    let target = Path::new("/home/andre/.claude/CLAUDE.md").to_path_buf();
    let observed = Observed {
        existing: vec![target.clone()],
        drifted: vec![target.clone()],
        ..Observed::default()
    };

    let plan = layout.plan(&observed);
    assert!(
        plan.actions
            .iter()
            .any(|a| matches!(a, Action::ReportDrift { target: t, .. } if t == &target)),
        "drift must be reported"
    );
    assert!(
        !plan
            .actions
            .iter()
            .any(|a| matches!(a, Action::RenderAdapter { target: t, .. } if t == &target)),
        "and must NOT also be rendered over"
    );
}

/// Running the same plan twice must be a no-op the second time — that is what
/// makes a reconcile safe to run on every hook event.
#[test]
fn a_plan_against_its_own_result_is_empty() {
    let layout = SharedAgentHome::from_home(Path::new("/home/andre"));
    let first = layout.plan(&Observed::default());

    let settled = Observed {
        existing: first.targets(),
        ..Observed::default()
    };
    let second = layout.plan(&settled);

    assert!(
        second.actions.is_empty(),
        "second plan should be empty, got {:?}",
        second.actions
    );
}

// ---- Steps 4-5: applying a plan, and adopting an existing memory graph ----

/// Replacing a generated adapter keeps the previous content. A reconcile that
/// overwrites with no way back is how a bad render becomes permanent.
#[test]
fn replacing_an_adapter_leaves_a_backup() {
    let home = tempfile::tempdir().unwrap();
    let layout = SharedAgentHome::from_home(home.path());
    let target = home.path().join(".claude/CLAUDE.md");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    std::fs::write(&target, "previous generated content").unwrap();
    std::fs::create_dir_all(layout.global_instructions().parent().unwrap()).unwrap();
    std::fs::write(layout.global_instructions(), "new shared content").unwrap();

    layout
        .apply(&claudectl::shared_home::Plan {
            actions: vec![Action::RenderAdapter {
                target: target.clone(),
                from: layout.global_instructions(),
            }],
        })
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "new shared content"
    );
    let backups: Vec<_> = std::fs::read_dir(target.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("CLAUDE.md.bak"))
        .collect();
    assert_eq!(backups.len(), 1, "exactly one backup");
    assert_eq!(
        std::fs::read_to_string(backups[0].path()).unwrap(),
        "previous generated content"
    );
}

/// Applying a plan twice must change nothing the second time.
#[test]
fn applying_twice_is_a_no_op() {
    let home = tempfile::tempdir().unwrap();
    let layout = SharedAgentHome::from_home(home.path());
    std::fs::create_dir_all(layout.global_instructions().parent().unwrap()).unwrap();
    std::fs::write(layout.global_instructions(), "shared").unwrap();

    // Stage a Claude memory directory so the adoption branch actually runs.
    // Without it `discover_claude_memory` returns None, the branch is skipped,
    // and this test passes while `AdoptInPlace` re-plans forever in production
    // — which is exactly what it did.
    let claude_memory = home.path().join(".claude/projects/-p/memory");
    std::fs::create_dir_all(&claude_memory).unwrap();
    std::fs::write(claude_memory.join("MEMORY.md"), "index").unwrap();

    layout.apply(&layout.plan(&layout.observe())).unwrap();
    let after_first = layout.observe();
    layout.apply(&layout.plan(&after_first)).unwrap();

    assert!(
        layout.plan(&layout.observe()).is_empty(),
        "a settled home plans nothing, adoption included"
    );
}

/// An adapter edited by hand must be observed as drifted, not merely as
/// present. `observe()` hardcoding an empty `drifted` made the whole
/// report-don't-overwrite guarantee unreachable in production while the pure
/// planning test still passed.
#[test]
fn a_hand_edited_adapter_is_observed_as_drifted() {
    let home = tempfile::tempdir().unwrap();
    let layout = SharedAgentHome::from_home(home.path());
    std::fs::create_dir_all(layout.global_instructions().parent().unwrap()).unwrap();
    std::fs::write(layout.global_instructions(), "generated content").unwrap();
    layout.apply(&layout.plan(&layout.observe())).unwrap();

    let adapter = home.path().join(".claude/CLAUDE.md");
    std::fs::write(&adapter, "a human wrote this").unwrap();

    let plan = layout.plan(&layout.observe());
    assert!(
        plan.actions
            .iter()
            .any(|a| matches!(a, Action::ReportDrift { target, .. } if target == &adapter)),
        "hand-edited adapter must be reported: {:?}",
        plan.actions
    );
}

/// The existing Claude memory graph is adopted by reference. Copying it would
/// duplicate every note and fork its history the moment either side is written.
#[test]
fn an_existing_memory_graph_is_adopted_in_place_not_copied() {
    let home = tempfile::tempdir().unwrap();
    let layout = SharedAgentHome::from_home(home.path());
    let claude_memory = home.path().join(".claude/projects/-home-andre/memory");
    std::fs::create_dir_all(&claude_memory).unwrap();
    std::fs::write(claude_memory.join("MEMORY.md"), "the index").unwrap();

    let plan = layout.plan(&Observed {
        claude_memory: Some(claude_memory.clone()),
        ..layout.observe()
    });

    assert!(
        plan.actions.iter().any(|a| matches!(
            a,
            Action::AdoptInPlace { source, .. } if source == &claude_memory
        )),
        "the graph is adopted where it lives: {:?}",
        plan.actions
    );
    assert!(
        !plan
            .actions
            .iter()
            .any(|a| matches!(a, Action::RenderAdapter { from, .. } if from == &claude_memory)),
        "and never copied"
    );
}

// ---- Step 6: provider-neutral MCP registry --------------------------------

use claudectl::shared_home::{McpRegistry, McpServer};

/// The canonical registry is shared and version-controllable, so a literal
/// secret in it is a secret in every provider's rendered config and in whatever
/// the shared home is synced to. Only an env reference is accepted.
#[test]
fn an_inline_credential_is_refused_by_the_registry() {
    let inline = McpServer {
        name: "grafana".into(),
        command: "grafana-mcp".into(),
        env: vec![("GRAFANA_TOKEN".into(), "glsa_livetokenvalue".into())],
    };
    let err = McpRegistry::validated(vec![inline]).unwrap_err();
    assert!(
        err.contains("GRAFANA_TOKEN") && err.contains("grafana"),
        "the error must name the server and the variable: {err}"
    );
}

/// `${VAR}` is a reference, not a value — that is the supported way to reach a
/// credential the host holds.
#[test]
fn an_env_reference_is_accepted() {
    let referenced = McpServer {
        name: "grafana".into(),
        command: "grafana-mcp".into(),
        env: vec![("GRAFANA_TOKEN".into(), "${GRAFANA_TOKEN}".into())],
    };
    assert!(McpRegistry::validated(vec![referenced]).is_ok());
}

/// Rendering is deterministic: same registry, same bytes. A render that
/// reordered itself would make every reconcile look like a change and mask real
/// drift.
#[test]
fn rendering_the_registry_is_deterministic() {
    let servers = vec![
        McpServer {
            name: "zeta".into(),
            command: "z".into(),
            env: vec![],
        },
        McpServer {
            name: "alpha".into(),
            command: "a".into(),
            env: vec![],
        },
    ];
    let registry = McpRegistry::validated(servers).unwrap();
    assert_eq!(registry.render_claude(), registry.render_claude());
    assert!(
        registry.render_claude().find("alpha").unwrap()
            < registry.render_claude().find("zeta").unwrap(),
        "servers render in a stable order regardless of input order"
    );
}

/// Editing the shared instructions must reach every provider.
///
/// This is the reason the tool exists, and it was broken: drift was computed as
/// "adapter differs from source", which is equally true when the source moved
/// on. Every adapter reported DRIFT and refused to update, so an edit reached
/// nobody. A render stamp is what separates "we wrote this" from "a human did".
#[test]
fn editing_the_shared_source_re_renders_every_adapter() {
    let home = tempfile::tempdir().unwrap();
    let layout = SharedAgentHome::from_home(home.path());
    std::fs::create_dir_all(layout.global_instructions().parent().unwrap()).unwrap();
    std::fs::write(layout.global_instructions(), "v1").unwrap();
    layout.apply(&layout.plan(&layout.observe())).unwrap();

    std::fs::write(layout.global_instructions(), "v2").unwrap();
    let plan = layout.plan(&layout.observe());
    assert!(
        !plan
            .actions
            .iter()
            .any(|a| matches!(a, Action::ReportDrift { .. })),
        "our own unmodified output is not drift: {:?}",
        plan.actions
    );
    layout.apply(&plan).unwrap();

    assert_eq!(
        std::fs::read_to_string(home.path().join(".claude/CLAUDE.md")).unwrap(),
        "v2"
    );
    assert_eq!(
        std::fs::read_to_string(home.path().join(".codex/AGENTS.md")).unwrap(),
        "v2"
    );
    assert!(
        layout.plan(&layout.observe()).is_empty(),
        "and then settles"
    );
}

/// A file we never rendered is the user's, even where we would have put ours.
#[test]
fn a_pre_existing_adapter_we_never_wrote_is_never_clobbered() {
    let home = tempfile::tempdir().unwrap();
    let layout = SharedAgentHome::from_home(home.path());
    std::fs::create_dir_all(layout.global_instructions().parent().unwrap()).unwrap();
    std::fs::write(layout.global_instructions(), "shared").unwrap();
    let adapter = home.path().join(".claude/CLAUDE.md");
    std::fs::create_dir_all(adapter.parent().unwrap()).unwrap();
    std::fs::write(&adapter, "instructions I wrote myself").unwrap();

    layout.apply(&layout.plan(&layout.observe())).unwrap();

    assert_eq!(
        std::fs::read_to_string(&adapter).unwrap(),
        "instructions I wrote myself",
        "a file with no stamp is not ours to overwrite"
    );
}
