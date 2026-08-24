//! CLI subcommand handlers extracted from main.rs.
//!
//! Each function implements a standalone CLI mode (--doctor, --clean, --list, etc.)
//! called from `run_main()` dispatch in main.rs.

use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::time::Duration;

use crate::Cli;
use crate::ViewFilters;
use crate::app::{App, FocusFilter, StatusFilter};
use crate::brain;
use crate::config;
use crate::demo;
use crate::discovery;
use crate::hook_state;
use crate::launch;
use crate::process;
use crate::reaper;
use crate::rules;
use crate::sandbox_registry;
use crate::session;
use crate::terminals;

pub(crate) fn launch_session(
    cwd: &str,
    prompt: Option<&str>,
    resume: Option<&str>,
) -> io::Result<()> {
    let request = launch::prepare(cwd, prompt, resume).map_err(io::Error::other)?;

    match launch::launch(&request) {
        Ok(target) => {
            println!(
                "Launched Claude session in {} at {}{}",
                target,
                request.cwd_path.display(),
                request.option_summary()
            );
            Ok(())
        }
        Err(e) => Err(io::Error::other(e)),
    }
}

/// Parse `config/mcp.toml` into servers. A missing or unreadable registry is an
/// empty one — the shared home is optional, and a first run has no registry yet.
fn read_mcp_registry(
    home: &agentctl::shared_home::SharedAgentHome,
) -> Vec<agentctl::shared_home::McpServer> {
    let Ok(text) = std::fs::read_to_string(home.mcp_registry()) else {
        return Vec::new();
    };
    let mut servers = Vec::new();
    let mut current: Option<agentctl::shared_home::McpServer> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line
            .strip_prefix("[server.")
            .and_then(|r| r.strip_suffix(']'))
        {
            if let Some(done) = current.take() {
                servers.push(done);
            }
            current = Some(agentctl::shared_home::McpServer {
                name: name.to_string(),
                command: String::new(),
                env: Vec::new(),
            });
        } else if let Some(server) = current.as_mut() {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').to_string();
                match key {
                    "command" => server.command = value,
                    _ if key.starts_with("env.") => {
                        server
                            .env
                            .push((key.trim_start_matches("env.").to_string(), value));
                    }
                    _ => {}
                }
            }
        }
    }
    servers.extend(current);
    servers
}

/// `--shared-home [--apply]`: reconcile `~/.agents` and the adapters rendered
/// from it.
///
/// Prints the plan by default and writes nothing. Adoption of an existing setup
/// is the common case, so seeing what would happen before it happens is the
/// default rather than an opt-in.
pub(crate) fn reconcile_shared_home(apply: bool) -> io::Result<()> {
    let Some(home_dir) = std::env::var_os("HOME") else {
        return Err(io::Error::other("HOME is unset; cannot locate ~/.agents"));
    };
    let home = agentctl::shared_home::SharedAgentHome::from_home(std::path::Path::new(&home_dir));

    let plan = home.plan(&home.observe());
    if plan.is_empty() {
        println!(
            "Shared agent home is up to date ({}).",
            home.root().display()
        );
        return Ok(());
    }

    println!("Shared agent home: {}", home.root().display());
    for action in &plan.actions {
        match action {
            agentctl::shared_home::Action::CreateDir(path) => {
                println!("  create    {}", path.display());
            }
            agentctl::shared_home::Action::RenderAdapter { target, from } => {
                println!("  render    {}  <- {}", target.display(), from.display());
            }
            agentctl::shared_home::Action::AdoptInPlace { target, source } => {
                println!("  adopt     {}  -> {}", target.display(), source.display());
            }
            agentctl::shared_home::Action::ReportDrift { target, reason } => {
                println!("  DRIFT     {}  ({reason})", target.display());
            }
        }
    }

    // Validating the registry here is the point at which a literal credential
    // is caught: before anything is rendered into a provider's config, and in
    // the dry run rather than only on --apply.
    match agentctl::shared_home::McpRegistry::validated(read_mcp_registry(&home)) {
        Ok(registry) => {
            if !registry.servers().is_empty() {
                println!("\nMCP servers ({}):", registry.servers().len());
                for server in registry.servers() {
                    println!("  {}  ({})", server.name, server.command);
                }
            }
        }
        Err(problem) => {
            println!();
            return Err(io::Error::other(problem));
        }
    }

    if !apply {
        println!("\nNothing written. Re-run with --apply to execute.");
        return Ok(());
    }

    home.apply(&plan)?;
    println!("\nApplied {} action(s).", plan.actions.len());
    Ok(())
}

/// `--restore-sessions`: bring back your local (laptop) Claude sessions — e.g.
/// after a Ghostty restart-to-update — spawning one window per session that was
/// live, each running `claude --resume <id>` in its recorded directory. Reads
/// the local registry (`local-sessions.json`), maintained by every host hook.
pub(crate) fn restore_local_sessions(dry_run: bool) -> io::Result<()> {
    let entries = sandbox_registry::load_local().sessions;
    if entries.is_empty() {
        println!("No local sessions registered — nothing to restore.");
        return Ok(());
    }

    // A local session is "already running" if a live process still holds its id:
    // either one a prior restore relaunched (visible via its `--resume <id>`
    // argv) or — unlike the post-`sbx rm` sandbox flow — a fresh-started `claude`
    // that never left (no `--resume` argv, so only its live pointer file betrays
    // it). Union both, or restoring while sessions are up would open a second
    // window against a live id and double-write its transcript.
    let mut running = running_resumed_ids();
    running.extend(discovery::live_sessions().into_iter().map(|s| s.session_id));

    let can_spawn = terminals::can_spawn_command_window();
    println!("Restoring {} local session(s):", entries.len());
    let (spawned, skipped) = restore_slice(&entries, "claude", &running, can_spawn, dry_run);
    report_restore(spawned, skipped, can_spawn, dry_run);
    Ok(())
}

/// `--restore-sbx-sessions [name]`: bring back sandbox Claude sessions after
/// `sbx rm`, spawning one window per session running `sc --resume <id>` in its
/// recorded directory. Resolves which sandbox(es) to restore (a name, `all`, or
/// an interactive pick). With `--dry-run` (or a terminal that can't spawn
/// windows) it prints the commands instead.
pub(crate) fn restore_sandbox_sessions(sandbox_arg: &str, dry_run: bool) -> io::Result<()> {
    let registry = sandbox_registry::load();
    if registry.sandboxes.is_empty() {
        println!("No sandbox sessions registered — nothing to restore.");
        return Ok(());
    }
    let targets = select_sandboxes(&registry, sandbox_arg)?;
    if targets.is_empty() {
        return Ok(());
    }

    let running = running_resumed_ids();
    let can_spawn = terminals::can_spawn_command_window();
    let mut spawned = 0usize;
    let mut skipped = 0usize;

    for sandbox in &targets {
        let Some(entries) = registry
            .sandboxes
            .get(sandbox)
            .filter(|entries| !entries.is_empty())
        else {
            eprintln!("  '{sandbox}': no sessions registered");
            continue;
        };
        println!("Restoring {} session(s) from '{sandbox}':", entries.len());
        let (slice_spawned, slice_skipped) =
            restore_slice(entries, "sc", &running, can_spawn, dry_run);
        spawned += slice_spawned;
        skipped += slice_skipped;
    }

    report_restore(spawned, skipped, can_spawn, dry_run);
    Ok(())
}

/// Spawn (or, with `--dry-run` / a non-spawning terminal, print) one
/// `<resume_binary> --resume <id>` window per entry, in its recorded cwd —
/// `resume_binary` is `claude` for local sessions, `sc` for sandbox ones.
/// Skips invalid ids, already-running sessions, and sessions whose transcript is
/// nowhere under `projects/`; returns `(spawned, skipped)`.
fn restore_slice(
    entries: &[sandbox_registry::SessionEntry],
    resume_binary: &str,
    running: &HashSet<String>,
    can_spawn: bool,
    dry_run: bool,
) -> (usize, usize) {
    let mut spawned = 0usize;
    let mut skipped = 0usize;
    for entry in entries {
        // The id is interpolated into a `<binary> --resume <id>` command the
        // terminal runs via `bash -lc`, so reject anything outside the Claude
        // Code UUID charset — shell-injection defense + corruption signal.
        if !is_valid_session_id(&entry.session_id) {
            eprintln!(
                "  [skip] {:?}: unexpected characters in session id",
                entry.session_id
            );
            skipped += 1;
            continue;
        }
        // Don't open a second window for a session already being resumed.
        if running.contains(&entry.session_id) {
            println!("  [skip] {}: already running", entry_label(entry));
            skipped += 1;
            continue;
        }
        // The recorded path is only a hypothesis: it is derived from the cwd, and
        // a session that lost its cwd points at `projects/-/`, which never
        // exists. Ask the session id before declaring the transcript gone.
        let site = discovery::locate_transcript(
            &entry.session_id,
            &entry.transcript,
            &|path| path.is_file(),
            &discovery::find_transcript_by_session_id,
        );
        if let discovery::TranscriptSite::Missing(tried) = &site {
            eprintln!(
                "  [skip] {} ({}): no transcript at {}, and none under any project directory",
                entry_label(entry),
                entry.cwd,
                tried.display()
            );
            skipped += 1;
            continue;
        }

        // Say so when the registry was wrong. Repairing this silently leaves the
        // bad row looking identical to a healthy one, which is how it went
        // unnoticed until a restore reported the session as lost.
        if let discovery::TranscriptSite::Recovered(found) = &site {
            let recorded = if entry.transcript.is_empty() {
                "no transcript recorded".to_string()
            } else {
                format!("recorded transcript {} is absent", entry.transcript)
            };
            println!(
                "  [note] {}: {recorded} — resuming from {}",
                entry_label(entry),
                found.display()
            );
        }

        let cwd = restore_cwd(entry, &site, &discovery::recover_cwd_from_transcript);
        let command = format!("{resume_binary} --resume {}", entry.session_id);

        if dry_run || !can_spawn {
            let note = if dry_run {
                ""
            } else {
                "   (this terminal can't spawn windows — run it manually)"
            };
            println!("  {}  {cwd}\n      $ {command}{note}", entry_label(entry));
            continue;
        }

        match terminals::spawn_command_window(&cwd, &command) {
            Ok(term) => {
                println!("  ↻ {}  {cwd}  ({term})", entry_label(entry));
                spawned += 1;
            }
            Err(error) => {
                // A nominally-supported terminal can still fail to spawn at
                // runtime (Kitty without `allow_remote_control`, no running mux,
                // a detached ssh session where the terminal binary is absent).
                // Never lose the session: surface the error AND print the command
                // so the user can paste it, exactly like the can't-spawn branch.
                eprintln!("  [fail] {}: {error}", entry_label(entry));
                println!(
                    "  {}  {cwd}\n      $ {command}   (spawn failed — run it manually)",
                    entry_label(entry)
                );
                skipped += 1;
            }
        }
    }
    (spawned, skipped)
}

/// One-line summary after a real (non-dry-run, spawning) restore.
fn report_restore(spawned: usize, skipped: usize, can_spawn: bool, dry_run: bool) {
    if !dry_run && can_spawn {
        let tail = if skipped > 0 {
            format!(" ({skipped} skipped)")
        } else {
            String::new()
        };
        println!("Restored {spawned} session(s){tail}.");
    }
}

/// First 8 chars of a session id — enough to eyeball, short enough to scan.
fn short_id(session_id: &str) -> &str {
    session_id.get(..8).unwrap_or(session_id)
}

/// Human label for a registry entry: its `/rename` name plus the short id when
/// named (e.g. `mimir-timeouts (c3df00ed)`), otherwise just the short id.
fn entry_label(entry: &sandbox_registry::SessionEntry) -> String {
    match entry.name.as_deref() {
        Some(name) if !name.is_empty() => format!("{name} ({})", short_id(&entry.session_id)),
        _ => short_id(&entry.session_id).to_string(),
    }
}

/// Session ids currently being resumed anywhere in the host process table. A
/// restored (or manually `sc --resume`d) session shows up as `sbx exec … claude
/// --resume <id>`, so `--resume <id>` in any process's args means it's already
/// up. Fresh-started sessions carry no id in their args and so aren't detected
/// here; the sandbox restore flow runs post-`sbx rm` (nothing live), while the
/// local flow unions this with `discovery::live_sessions()` to catch them.
fn running_resumed_ids() -> HashSet<String> {
    match std::process::Command::new("ps")
        .args(["-axo", "args="])
        .output()
    {
        Ok(output) => resumed_ids_in(&String::from_utf8_lossy(&output.stdout)),
        Err(_) => HashSet::new(),
    }
}

/// Pure: collect every token immediately following `--resume` in `ps` output.
fn resumed_ids_in(ps_output: &str) -> HashSet<String> {
    let mut ids = HashSet::new();
    for line in ps_output.lines() {
        let mut tokens = line.split_whitespace();
        while let Some(token) = tokens.next() {
            if token == "--resume" {
                if let Some(id) = tokens.next() {
                    ids.insert(id.to_string());
                }
            }
        }
    }
    ids
}

/// Whether `session_id` is safe to interpolate into the `sc --resume <id>`
/// command line (which the terminal runs via `bash -lc`). Claude Code session
/// ids are UUIDs; this conservative allowlist accepts them and rejects every
/// shell metacharacter.
fn is_valid_session_id(session_id: &str) -> bool {
    !session_id.is_empty()
        && session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

/// Where to reopen a restored session.
///
/// A recovered transcript carries the cwd its registry entry lost, and that
/// beats [`resolve_cwd`]'s `$HOME` fallback — which is right only by luck for a
/// session started in the home directory, and wrong for one in a worktree.
fn restore_cwd(
    entry: &sandbox_registry::SessionEntry,
    site: &discovery::TranscriptSite,
    read_cwd: &impl Fn(&Path) -> Option<String>,
) -> String {
    if entry.cwd.trim().is_empty() {
        if let discovery::TranscriptSite::Recovered(path) = site {
            if let Some(cwd) = read_cwd(path) {
                return cwd;
            }
        }
    }
    resolve_cwd(&entry.cwd)
}

/// The directory to reopen a session in, falling back to `$HOME` (then `.`)
/// when the registry has no recorded cwd.
fn resolve_cwd(cwd: &str) -> String {
    if cwd.trim().is_empty() {
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
    } else {
        cwd.to_string()
    }
}

/// Decide which sandbox slice(s) to restore. An explicit name (or `all`) is
/// honored directly; otherwise a single registered sandbox is auto-selected
/// and multiple prompt an interactive pick.
/// Registered sandboxes belonging to the family `arg` names, in registry order.
///
/// A sandbox's name encodes the image + mounts it was built from
/// (`linera-agent-a3f11b28c4d0`), so what used to be one long-lived
/// `linera-agent` is now a rolling series of them, several alive at once while
/// older ones drain. `--restore-sbx-sessions linera-agent` therefore has to
/// mean "everything in the linera-agent family", or it silently restores a
/// fraction of your sessions — and after the legacy sandbox is finally reaped,
/// nothing at all, because no sandbox is called exactly that any more.
///
/// The boundary is a literal `-`, not a bare prefix: `linera-agent` must not
/// swallow a hypothetical `linera-agentic`. An exact name still selects exactly
/// itself, so naming one rolled sandbox targets only that one, and an
/// engineer's `SANDBOX_NAME=` sandbox is never swept in by a family it doesn't
/// belong to.
pub(crate) fn matching_sandboxes(names: &[String], arg: &str) -> Vec<String> {
    let family_prefix = format!("{arg}-");
    names
        .iter()
        .filter(|name| name.as_str() == arg || name.starts_with(&family_prefix))
        .cloned()
        .collect()
}

fn select_sandboxes(
    registry: &sandbox_registry::Registry,
    sandbox_arg: &str,
) -> io::Result<Vec<String>> {
    let arg = sandbox_arg.trim();
    let names: Vec<String> = registry.sandboxes.keys().cloned().collect();

    if !arg.is_empty() {
        if arg.eq_ignore_ascii_case("all") {
            return Ok(names);
        }
        let matched = matching_sandboxes(&names, arg);
        if !matched.is_empty() {
            return Ok(matched);
        }
        eprintln!(
            "No sessions registered for sandbox '{arg}'. Registered: {}",
            names.join(", ")
        );
        return Ok(Vec::new());
    }

    if names.len() == 1 {
        return Ok(names);
    }

    // Several sandboxes registered and none named — let the user choose.
    println!("Multiple sandboxes have registered sessions:");
    for (index, name) in names.iter().enumerate() {
        let count = registry.sandboxes.get(name).map_or(0, Vec::len);
        println!("  {}) {name}  ({count} session(s))", index + 1);
    }
    println!("  a) all");
    print!("Restore which? [1-{}/a]: ", names.len());
    io::Write::flush(&mut io::stdout())?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let choice = line.trim();

    if choice.eq_ignore_ascii_case("a") || choice.eq_ignore_ascii_case("all") {
        return Ok(names);
    }
    if let Ok(number) = choice.parse::<usize>() {
        if (1..=names.len()).contains(&number) {
            return Ok(vec![names[number - 1].clone()]);
        }
    }
    if registry.sandboxes.contains_key(choice) {
        return Ok(vec![choice.to_string()]);
    }
    eprintln!("Nothing selected.");
    Ok(Vec::new())
}

fn print_doctor_transcripts() {
    println!();
    println!("Transcript Discovery");

    let sessions_dir = discovery::projects_dir().parent().unwrap().join("sessions");
    let projects_dir = discovery::projects_dir();

    // Check sessions directory
    let sessions_exists = sessions_dir.exists();
    println!(
        "  [{}] sessions dir: {}",
        if sessions_exists { "ok" } else { "!!" },
        sessions_dir.display()
    );

    // Check projects directory
    let projects_exists = projects_dir.exists();
    println!(
        "  [{}] projects dir: {}",
        if projects_exists { "ok" } else { "!!" },
        projects_dir.display()
    );

    if !sessions_exists {
        println!("      No session pointer files found — Claude Code may not have run yet");
        return;
    }

    // Scan sessions and attempt resolution
    let mut sessions = discovery::scan_sessions();
    if sessions.is_empty() {
        println!("  [--] no session pointer files found");
        return;
    }

    process::fetch_and_enrich(&mut sessions);
    let alive: Vec<_> = sessions
        .iter()
        .filter(|s| s.status != session::SessionStatus::Finished)
        .collect();

    if alive.is_empty() {
        println!("  [--] no active Claude Code sessions");
        return;
    }

    // Resolve JSONL paths for alive sessions
    let mut alive_sessions: Vec<_> = alive.into_iter().cloned().collect();
    for s in &mut alive_sessions {
        discovery::resolve_jsonl_paths(std::slice::from_mut(s));
    }

    for s in &alive_sessions {
        let found = s.jsonl_path.is_some();
        let slug = s.cwd.trim_end_matches('/').replace('/', "-");
        let expected_dir = projects_dir.join(&slug);

        println!(
            "  [{}] PID {} ({})",
            if found { "ok" } else { "!!" },
            s.pid,
            s.project_name
        );
        println!("      cwd:  {}", s.cwd);
        println!("      slug: {slug}");
        if let Some(ref path) = s.jsonl_path {
            println!("      jsonl: {}", path.display());
        } else {
            println!(
                "      expected dir: {} (exists={})",
                expected_dir.display(),
                expected_dir.exists()
            );
            let expected_file = expected_dir.join(format!("{}.jsonl", s.session_id));
            println!(
                "      expected file: {} (exists={})",
                expected_file.display(),
                expected_file.exists()
            );
            println!(
                "      fix: check that Claude Code's project directory slug matches the cwd encoding above"
            );
        }
    }
}

fn file_mtime_ms(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_millis() as u64)
}

/// Report any session whose hook channel has stopped delivering.
///
/// Nothing else claudectl shows can tell "this session is quiet" apart from
/// "this session's hooks stopped arriving", and the difference is expensive:
/// every fast path for noticing a session start, block, or *end* is a hook, and
/// the hook fails silently by construction — the command ends in
/// `2>/dev/null || true`, `claudectl-hook` writes nothing to stdout by design,
/// and its own log is opt-in behind `CLAUDECTL_HOOK_LOG`.
///
/// On 2026-08-10 one session's hooks had been silent for 12 h. Its `SessionEnd`
/// never landed, so its row outlived it by ~96 s, and no surface anywhere said
/// why. This is that surface.
fn print_doctor_hook_liveness() {
    println!();
    println!("Hook Channel");

    let running = reaper::running_sandboxes();
    let mut scopes: Vec<(String, Vec<sandbox_registry::SessionEntry>)> =
        sandbox_registry::load().sandboxes.into_iter().collect();
    scopes.sort_by(|left, right| left.0.cmp(&right.0));
    scopes.push((
        HOST_SCOPE.to_string(),
        sandbox_registry::load_local().sessions,
    ));

    let mut checked = 0_usize;
    let mut silent = 0_usize;
    // A row with no locatable transcript is unverifiable, not proven dead — the
    // silence check needs a transcript as its control, and there isn't one.
    let mut unverifiable = 0_usize;
    for (scope, entries) in scopes {
        // A stopped sandbox fires no hooks by definition; its slice is restore
        // material, not evidence of anything. `None` means we could not ask
        // (no `sbx`), in which case judge everything rather than nothing.
        if scope != HOST_SCOPE
            && running
                .as_ref()
                .is_some_and(|names| !names.contains(&scope))
        {
            println!("  [--] {scope}: not running — no hooks expected");
            continue;
        }
        for entry in entries {
            // Stamped means `SessionEnd` arrived and did its job. Nothing to say.
            if entry.departed_at_ms.is_some() {
                continue;
            }
            let newest_hook_ms = hook_state::HookState::load(&entry.session_id)
                .as_ref()
                .map_or(0, hook_state::newest_hook_event_ms);
            // Stat where the transcript actually is: a wrong recorded path stats
            // as mtime 0, the silence check fails quiet on a zero, and the row
            // then passes `[ok]` on a reading that never happened.
            let site = discovery::locate_transcript(
                &entry.session_id,
                &entry.transcript,
                &|path| path.is_file(),
                &discovery::find_transcript_by_session_id,
            );
            let transcript_ms = match &site {
                discovery::TranscriptSite::AsRecorded(path)
                | discovery::TranscriptSite::Recovered(path) => file_mtime_ms(path),
                discovery::TranscriptSite::Unrecorded | discovery::TranscriptSite::Missing(_) => 0,
            };
            checked += 1;
            let label = entry.name.as_deref().unwrap_or("unnamed");
            let short = &entry.session_id[..entry.session_id.len().min(8)];
            if let discovery::TranscriptSite::Missing(tried) = &site {
                unverifiable += 1;
                println!("  [!!] {scope} {short} ({label}): transcript not found");
                println!("      recorded:   {} (absent)", tried.display());
                println!(
                    "      reading: no project directory holds this session id, so there is no \
                     transcript to check the hook channel against. The row cannot be verified \
                     either way and cannot be restored."
                );
                continue;
            }
            if !hook_state::hook_channel_is_silent(transcript_ms, newest_hook_ms) {
                println!("  [ok] {scope} {short} ({label})");
                continue;
            }
            silent += 1;
            let behind_secs = transcript_ms.saturating_sub(newest_hook_ms) / 1_000;
            println!(
                "  [!!] {scope} {short} ({label}): transcript is {} ahead of the last hook event",
                crate::history::format_duration(behind_secs)
            );
            match &site {
                discovery::TranscriptSite::Recovered(path) => println!(
                    "      transcript: {} (mtime {transcript_ms}) — recorded as {}, which is absent",
                    path.display(),
                    entry.transcript
                ),
                _ => println!(
                    "      transcript: {} (mtime {transcript_ms})",
                    entry.transcript
                ),
            }
            if newest_hook_ms == 0 {
                println!("      last hook:  none ever recorded for this session");
            } else {
                println!("      last hook:  {newest_hook_ms}");
            }
            println!(
                "      reading: either the session is live and its hooks stopped arriving, or it \
                 ended without a SessionEnd. Both mean this row is unverifiable — no departure \
                 stamp will ever be written, so it is retired by the host tty sweep or the \
                 collector rather than instantly."
            );
            println!(
                "      next: `echo '{{\"session_id\":\"probe\",\"hook_event_name\":\"Stop\"}}' | \
                 CLAUDECTL_HOOK_LOG=/tmp/hook.log claudectl-hook` in that scope. If that logs and \
                 exits 0, the receiver is healthy and Claude Code is not invoking it — restart the \
                 session; if it fails, the receiver is the problem."
            );
        }
    }

    if checked == 0 {
        println!("  [--] no live registry sessions to check");
    } else if silent == 0 && unverifiable == 0 {
        println!("  {checked} session(s) delivering hooks");
    } else {
        if silent > 0 {
            println!("  {silent} of {checked} session(s) have a dead hook channel");
        }
        if unverifiable > 0 {
            println!("  {unverifiable} of {checked} session(s) have no locatable transcript");
        }
    }
}

/// Scope label for the laptop's own (non-sandbox) sessions. Not a sandbox name,
/// so it is never run through the running-sandbox filter.
const HOST_SCOPE: &str = "host";

pub(crate) fn print_doctor() -> io::Result<()> {
    use crate::terminals;

    let report = terminals::doctor_report();
    println!("{}", terminals::format_doctor_report(&report));

    // Transcript discovery diagnostics
    print_doctor_transcripts();

    // Is anything still delivering hooks?
    print_doctor_hook_liveness();

    // Brain diagnostics
    let cfg = config::Config::load();
    println!();
    println!("Brain (local LLM)");

    // Check curl
    let curl_ok = std::process::Command::new("curl")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    println!(
        "  [{}] curl: {}",
        if curl_ok { "ok" } else { "!!" },
        if curl_ok {
            "available (required for brain HTTP calls)"
        } else {
            "not found — brain requires curl on PATH"
        }
    );

    // Check ollama binary
    let ollama_ok = std::process::Command::new("ollama")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    println!(
        "  [{}] ollama: {}",
        if ollama_ok { "ok" } else { "--" },
        if ollama_ok {
            "installed"
        } else {
            "not found (install: brew install ollama)"
        }
    );

    // Check endpoint reachability
    if let Some(ref brain) = cfg.brain {
        println!(
            "  Config: enabled={}, model={}, auto={}, few_shot={}",
            brain.enabled, brain.model, brain.auto_mode, brain.few_shot_count
        );
        let endpoint_ok = check_brain_endpoint(&brain.endpoint, brain.timeout_ms);
        println!(
            "  [{}] endpoint {}: {}",
            if endpoint_ok { "ok" } else { "!!" },
            brain.endpoint,
            if endpoint_ok {
                "reachable"
            } else {
                "not reachable"
            }
        );
        if !endpoint_ok {
            println!("      fix: start ollama with `ollama serve`, or check --brain-endpoint URL");
        }
    } else {
        println!("  Config: not configured");
        println!("  To enable: add [brain] section to .claudectl.toml or use --brain flag");
    }

    Ok(())
}

pub(crate) fn check_brain_endpoint(endpoint: &str, timeout_ms: u64) -> bool {
    let timeout_secs = (timeout_ms / 1000).max(1);
    std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--max-time",
            &timeout_secs.to_string(),
            endpoint,
        ])
        .output()
        .is_ok_and(|o| {
            let code = String::from_utf8_lossy(&o.stdout);
            // Any HTTP response (even 404/405) means the server is up
            code.trim() != "000"
        })
}

pub(crate) fn parse_duration_str(s: &str) -> Duration {
    let s = s.trim();
    if let Some(hours) = s.strip_suffix('h') {
        if let Ok(h) = hours.parse::<u64>() {
            return Duration::from_secs(h * 3600);
        }
    }
    if let Some(mins) = s.strip_suffix('m') {
        if let Ok(m) = mins.parse::<u64>() {
            return Duration::from_secs(m * 60);
        }
    }
    if let Some(days) = s.strip_suffix('d') {
        if let Ok(d) = days.parse::<u64>() {
            return Duration::from_secs(d * 86400);
        }
    }
    Duration::from_secs(24 * 3600) // default 24h
}

pub(crate) fn parse_status_filter(value: Option<&str>) -> io::Result<StatusFilter> {
    match value {
        Some(raw) => StatusFilter::parse(raw).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Invalid --filter-status value: {raw}. Expected one of: all, needs-input, processing, waiting, unknown, idle, finished"
                ),
            )
        }),
        None => Ok(StatusFilter::All),
    }
}

pub(crate) fn parse_focus_filter(value: Option<&str>) -> io::Result<FocusFilter> {
    match value {
        Some(raw) => FocusFilter::parse(raw).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Invalid --focus value: {raw}. Expected one of: all, attention, over-budget, high-context, unknown-telemetry, conflict"
                ),
            )
        }),
        None => Ok(FocusFilter::All),
    }
}

pub(crate) fn apply_filters(app: &mut App, filters: &ViewFilters) {
    app.status_filter = filters.status_filter;
    app.focus_filter = filters.focus_filter;
    app.search_query = filters.search.trim().to_string();
    app.search_buffer.clear();
    app.search_mode = false;
    let len = app.visible_session_count();
    if len == 0 {
        app.table_state.select(None);
    } else if app.table_state.selected().is_none() {
        app.table_state.select(Some(0));
    } else if let Some(sel) = app.table_state.selected() {
        if sel >= len {
            app.table_state.select(Some(len - 1));
        }
    }
}

pub(crate) fn run_clean(
    older_than: Option<&str>,
    finished_only: bool,
    dry_run: bool,
) -> io::Result<()> {
    let min_age = older_than.map(parse_duration_str);
    let now = std::time::SystemTime::now();

    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));

    // Collect active PIDs to avoid deleting live sessions
    let active_pids: std::collections::HashSet<u32> = {
        let app = App::new();
        app.data_snapshot().sessions.iter().map(|s| s.pid).collect()
    };

    let mut removed_sessions = 0u64;
    let mut removed_jsonl = 0u64;
    let mut freed_bytes = 0u64;

    // Phase 1: Clean session JSON files in ~/.claude/sessions/
    let sessions_dir = home.join(".claude/sessions");
    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let pid: u32 = match stem.parse() {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Never delete active sessions
            if active_pids.contains(&pid) {
                continue;
            }

            // Check age if --older-than is set
            if let Some(min_age) = min_age {
                let modified = entry.metadata().ok().and_then(|m| m.modified().ok());
                if let Some(modified) = modified {
                    let age = now.duration_since(modified).unwrap_or_default();
                    if age < min_age {
                        continue;
                    }
                }
            }

            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if dry_run {
                println!("  would remove: {} ({} bytes)", path.display(), size);
            } else {
                let _ = std::fs::remove_file(&path);
            }
            removed_sessions += 1;
            freed_bytes += size;
        }
    }

    // Phase 2: Clean JSONL transcript files in ~/.claude/projects/*/
    let projects_dir = home.join(".claude/projects");
    if let Ok(project_entries) = std::fs::read_dir(&projects_dir) {
        for project_entry in project_entries.flatten() {
            let project_path = project_entry.path();
            if !project_path.is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(&project_path) else {
                continue;
            };
            for file_entry in files.flatten() {
                let file_path = file_entry.path();
                if file_path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                    continue;
                }

                let metadata = match file_entry.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                // Check age if --older-than is set
                if let Some(min_age) = min_age {
                    let modified = metadata.modified().ok();
                    if let Some(modified) = modified {
                        let age = now.duration_since(modified).unwrap_or_default();
                        if age < min_age {
                            continue;
                        }
                    }
                }

                // If --finished only, skip JSONL files whose corresponding session is still active
                if finished_only {
                    // Check if any active session is using this JSONL
                    let app = App::new();
                    let is_active = app.data_snapshot().sessions.iter().any(|s| {
                        s.jsonl_path
                            .as_ref()
                            .map(|p| p == &file_path)
                            .unwrap_or(false)
                    });
                    if is_active {
                        continue;
                    }
                }

                let size = metadata.len();
                if dry_run {
                    println!("  would remove: {} ({} bytes)", file_path.display(), size);
                } else {
                    let _ = std::fs::remove_file(&file_path);
                }
                removed_jsonl += 1;
                freed_bytes += size;
            }
        }
    }

    let freed_str = if freed_bytes >= 1_073_741_824 {
        format!("{:.1} GB", freed_bytes as f64 / 1_073_741_824.0)
    } else if freed_bytes >= 1_048_576 {
        format!("{:.1} MB", freed_bytes as f64 / 1_048_576.0)
    } else if freed_bytes >= 1024 {
        format!("{:.1} KB", freed_bytes as f64 / 1024.0)
    } else {
        format!("{freed_bytes} bytes")
    };

    if dry_run {
        println!();
        println!(
            "Dry run: would remove {} sessions + {} transcripts, freeing {}",
            removed_sessions, removed_jsonl, freed_str
        );
    } else if removed_sessions + removed_jsonl == 0 {
        println!("Nothing to clean up.");
    } else {
        println!(
            "Removed {} sessions + {} transcripts, freed {}",
            removed_sessions, removed_jsonl, freed_str
        );
    }

    Ok(())
}

pub(crate) fn print_summary(since: &str) -> io::Result<()> {
    let since_duration = parse_duration_str(since);
    let app = App::new();
    let snap = app.data_snapshot();

    if snap.sessions.is_empty() {
        println!("No active Claude sessions.");
        return Ok(());
    }

    for s in &snap.sessions {
        let status_color = match s.status {
            session::SessionStatus::Processing => "\x1b[32m",
            session::SessionStatus::Compacting => "\x1b[36m",
            session::SessionStatus::NeedsInput => "\x1b[35m",
            session::SessionStatus::WaitingInput => "\x1b[33m",
            session::SessionStatus::Unknown => "\x1b[34m",
            session::SessionStatus::Idle => "\x1b[90m",
            session::SessionStatus::Finished => "\x1b[31m",
        };
        let reset = "\x1b[0m";
        let status_text = if s.status == session::SessionStatus::Unknown {
            format!("Unknown: {}", s.telemetry_label())
        } else {
            s.status.to_string()
        };

        println!(
            "=== {} ({}, {}, {status_color}{}{reset}) ===",
            s.display_name(),
            s.format_elapsed(),
            s.format_cost(),
            status_text,
        );

        // Git stats from session's cwd
        let since_secs = since_duration.as_secs();
        let git_since = format!("{since_secs} seconds ago");

        let git_log = std::process::Command::new("git")
            .args(["log", "--oneline", &format!("--since={git_since}")])
            .current_dir(&s.cwd)
            .output();

        if let Ok(output) = git_log {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let commits: Vec<&str> = stdout.lines().collect();
            if !commits.is_empty() {
                println!("  Commits: {}", commits.len());
                for c in commits.iter().take(5) {
                    println!("    {c}");
                }
                if commits.len() > 5 {
                    println!("    ... and {} more", commits.len() - 5);
                }
            }
        }

        let git_diff = std::process::Command::new("git")
            .args(["diff", "--stat", "HEAD"])
            .current_dir(&s.cwd)
            .output();

        if let Ok(output) = git_diff {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            if !lines.is_empty() {
                let file_count = lines.len().saturating_sub(1); // last line is summary
                if file_count > 0 {
                    println!("  Files changed: {file_count}");
                }
            }
        }

        // Token summary
        let total_tokens = s.total_input_tokens + s.total_output_tokens;
        if total_tokens > 0 {
            println!(
                "  Tokens: {} in / {} out",
                format_count(s.total_input_tokens),
                format_count(s.total_output_tokens)
            );
        }

        // Model and context
        if !s.model.is_empty() {
            let context_text = if s.has_usage_metrics() {
                format!("{}%", s.context_percent() as u32)
            } else {
                "n/a".to_string()
            };
            let estimate_note = if s.cost_estimate_unverified {
                " [fallback estimate]"
            } else if s.model_profile_source == "override" {
                " [config override]"
            } else {
                ""
            };
            println!(
                "  Model: {}{} (context: {})",
                s.model, estimate_note, context_text
            );
        }
        if s.status == session::SessionStatus::Unknown || !s.has_usage_metrics() {
            println!("  Telemetry: {}", s.telemetry_label());
        }

        if s.subagent_count > 0 {
            println!("  Subagents: {}", s.format_subagent_summary());
        }

        println!();
    }

    let total_cost: f64 = snap.sessions.iter().map(|s| s.cost_usd).sum();
    println!("Total cost: ${total_cost:.2}");

    Ok(())
}

pub(crate) fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn make_app(demo: bool, filters: &ViewFilters) -> App {
    let mut app = if demo {
        let mut app = App::new();
        app.demo_mode = true;
        app.replace_data(crate::app::AppData {
            sessions: demo::generate_sessions(10),
            ..crate::app::AppData::default()
        });
        app
    } else {
        App::new()
    };
    apply_filters(&mut app, filters);
    app
}

pub(crate) fn print_json(demo: bool, filters: &ViewFilters) -> io::Result<()> {
    let app = make_app(demo, filters);
    let values: Vec<serde_json::Value> = app
        .visible_sessions()
        .iter()
        .map(|s| s.to_json_value())
        .collect();
    let json = serde_json::to_string_pretty(&values).unwrap_or_else(|_| "[]".to_string());
    println!("{json}");
    Ok(())
}

pub(crate) fn print_list(demo: bool, filters: &ViewFilters) -> io::Result<()> {
    let app = make_app(demo, filters);
    let visible_sessions = app.visible_sessions();

    if visible_sessions.is_empty() {
        if app.has_active_filters() {
            println!("No sessions match the current filters.");
        } else {
            println!("No active Claude sessions.");
        }
        if app.has_active_filters() {
            println!("  ({})", app.filter_summary());
        }
        return Ok(());
    }

    println!(
        "{:<7} {:<16} {:<12} {:<8} {:<8} {:<9} {:<10} {:<6} {:<6} TOKENS",
        "PID", "PROJECT", "STATUS", "CTX%", "COST", "$/HR", "ELAPSED", "CPU%", "MEM"
    );
    println!("{}", "-".repeat(105));

    for s in visible_sessions {
        let status_text = if s.status == session::SessionStatus::Unknown {
            s.telemetry_status.short_label().to_string()
        } else {
            s.status.to_string()
        };
        println!(
            "{:<7} {:<16} {:<12} {:<8} {:<8} {:<9} {:<10} {:<6} {:<6} {}",
            s.pid,
            s.display_name(),
            status_text,
            s.format_context(),
            s.format_cost(),
            s.format_burn_rate(),
            s.format_elapsed(),
            s.format_cpu(),
            s.format_mem(),
            s.format_tokens(),
        );
    }

    let total_cost: f64 = app.visible_sessions().iter().map(|s| s.cost_usd).sum();
    println!("{}", "-".repeat(105));
    println!("Total cost: ${total_cost:.2}");
    if app.has_active_filters() {
        println!("{}", app.filter_summary());
    }

    Ok(())
}

pub(crate) fn run_watch(
    tick_rate: Duration,
    json_mode: bool,
    format_str: &str,
    filters: &ViewFilters,
) -> io::Result<()> {
    use crate::session::SessionStatus;
    use std::collections::HashMap;

    let mut app = App::new();
    apply_filters(&mut app, filters);
    let mut prev_statuses: HashMap<u32, SessionStatus> = app
        .data_snapshot()
        .sessions
        .iter()
        .map(|s| (s.pid, s.status))
        .collect();

    // Print initial state for all sessions
    let visible = app.visible_sessions();
    for s in &visible {
        if json_mode {
            let obj = serde_json::json!({
                "event": "initial",
                "pid": s.pid,
                "project": s.display_name(),
                "status": s.status.to_string(),
                "telemetry": s.telemetry_label(),
                "cost_usd": if s.has_usage_metrics() { serde_json::json!((s.cost_usd * 100.0).round() / 100.0) } else { serde_json::Value::Null },
                "context_pct": if s.has_usage_metrics() { serde_json::json!((s.context_percent() * 100.0).round() / 100.0) } else { serde_json::Value::Null },
                "elapsed_secs": s.elapsed.as_secs(),
            });
            println!("{}", serde_json::to_string(&obj).unwrap_or_default());
        } else {
            println!("{}", format_session(format_str, s));
        }
    }

    loop {
        std::thread::sleep(tick_rate);
        app.tick();
        let visible_pids: std::collections::HashSet<u32> =
            app.visible_sessions().iter().map(|s| s.pid).collect();

        let snap = app.data_snapshot();
        for s in &snap.sessions {
            let prev = prev_statuses.get(&s.pid).copied();
            let changed = prev.is_none_or(|p| p != s.status);

            if !changed || !visible_pids.contains(&s.pid) {
                continue;
            }

            if json_mode {
                let obj = serde_json::json!({
                    "event": "status_change",
                    "pid": s.pid,
                    "project": s.display_name(),
                    "old_status": prev.map(|p| p.to_string()).unwrap_or_default(),
                    "new_status": s.status.to_string(),
                    "telemetry": s.telemetry_label(),
                    "cost_usd": if s.has_usage_metrics() { serde_json::json!((s.cost_usd * 100.0).round() / 100.0) } else { serde_json::Value::Null },
                    "context_pct": if s.has_usage_metrics() { serde_json::json!((s.context_percent() * 100.0).round() / 100.0) } else { serde_json::Value::Null },
                    "elapsed_secs": s.elapsed.as_secs(),
                });
                println!("{}", serde_json::to_string(&obj).unwrap_or_default());
            } else {
                println!("{}", format_session(format_str, s));
            }
        }

        prev_statuses = snap.sessions.iter().map(|s| (s.pid, s.status)).collect();
    }
}

pub(crate) fn format_session(fmt: &str, s: &session::AgentSession) -> String {
    let cost = if s.has_usage_metrics() {
        format!("{:.2}", s.cost_usd)
    } else {
        "n/a".to_string()
    };
    let context = if s.has_usage_metrics() {
        format!("{}", s.context_percent() as u32)
    } else {
        "n/a".to_string()
    };
    fmt.replace("{pid}", &s.pid.to_string())
        .replace("{project}", s.display_name())
        .replace("{status}", &s.status.to_string())
        .replace("{cost}", &cost)
        .replace("{context}", &context)
}

/// Path to the brain gate mode state file.
pub(crate) fn brain_gate_mode_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    crate::product::home_dot_dir(&std::path::PathBuf::from(home))
        .join("brain")
        .join("gate-mode")
}

/// Read the current brain gate mode from disk. Returns "on" if no file exists.
pub(crate) fn read_brain_gate_mode() -> String {
    let path = brain_gate_mode_path();
    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "on".into())
}

/// Set the brain gate mode (on/off/auto) and print confirmation.
pub(crate) fn run_brain_mode(mode: &str) -> io::Result<()> {
    match mode {
        "on" | "off" | "auto" => {}
        "status" | "" => {
            let current = read_brain_gate_mode();
            println!("Brain gate mode: {current}");
            println!();
            println!("Modes:");
            println!("  on   — brain evaluates tool calls, denies dangerous ones (default)");
            println!("  off  — brain disabled, all tool calls pass through");
            println!("  auto — brain auto-approves above confidence threshold");
            return Ok(());
        }
        _ => {
            eprintln!("Unknown brain mode: {mode}");
            eprintln!("Valid modes: on, off, auto, status");
            std::process::exit(1);
        }
    }

    let path = brain_gate_mode_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if mode == "on" {
        // "on" is the default — remove the file so absence = on
        let _ = std::fs::remove_file(&path);
    } else {
        std::fs::write(&path, mode)?;
    }

    let description = match mode {
        "on" => "brain evaluates tool calls, denies dangerous ones",
        "off" => "brain disabled — all tool calls pass through to normal permission flow",
        "auto" => "brain auto-approves tool calls above confidence threshold",
        _ => unreachable!(),
    };

    println!("Brain gate mode set to: {mode}");
    println!("  {description}");
    Ok(())
}

/// Handle --insights: show insights or set mode (on/off/status).
/// Requires brain to be enabled.
pub(crate) fn run_insights(cfg: &config::Config, cli: &Cli, arg: &str) -> io::Result<()> {
    let brain_enabled = cfg.brain.as_ref().map(|b| b.enabled).unwrap_or(false) || cli.brain;

    if !brain_enabled {
        eprintln!(
            "Insights requires the brain. Use --brain or set brain.enabled = true in config."
        );
        std::process::exit(1);
    }

    match arg {
        "on" => {
            let _ = brain::insights::write_insights_mode("on");
            println!("Insights mode: on");
            println!("  Auto-generating insights every 10 decisions during brain distillation.");
            println!("  Run `claudectl --brain --insights` to view.");
        }
        "off" => {
            let _ = brain::insights::write_insights_mode("off");
            println!("Insights mode: off");
            println!(
                "  Auto-generation disabled. Run `claudectl --brain --insights` to generate on demand."
            );
        }
        "status" => {
            let mode = brain::insights::read_insights_mode();
            println!("Insights mode: {mode}");
            println!();
            println!("Modes:");
            println!("  on   — auto-generate insights every 10 decisions");
            println!("  off  — disabled, generate on demand only (default)");
        }
        "" => {
            // No argument: show insights
            brain::insights::print_insights();
        }
        _ => {
            eprintln!("Unknown insights argument: {arg}");
            eprintln!("Usage: --insights [on|off|status]");
            eprintln!("  No argument: show current insights");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Standalone brain query: builds a minimal context from CLI args, calls the
/// local LLM, and prints a JSON decision to stdout. Designed to be called
/// by Claude Code plugin hooks (PreToolUse) for inline approve/deny.
pub(crate) fn run_brain_query(cfg: &config::Config, cli: &Cli) -> io::Result<()> {
    // Respect brain gate mode — if off, skip immediately
    let gate_mode = read_brain_gate_mode();
    if gate_mode == "off" {
        let result = serde_json::json!({
            "action": "abstain",
            "reasoning": "Brain gate mode is off",
            "confidence": 0.0,
            "source": "gate",
        });
        println!("{}", serde_json::to_string(&result).unwrap());
        return Ok(());
    }

    let brain_cfg = cfg.brain.clone().unwrap_or_default();

    if !brain_cfg.enabled && !cli.brain {
        eprintln!("Brain is not enabled. Use --brain or set brain.enabled = true in config.");
        std::process::exit(1);
    }

    let tool_name = cli.tool.clone().unwrap_or_else(|| "unknown".into());
    let command = cli.tool_input.clone().unwrap_or_default();
    let project = cli.project.clone().unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "unknown".into())
    });

    // Step 1: Check static deny rules first (instant, no LLM needed)
    let auto_rules = cfg.rules.clone();
    let deny_rules: Vec<_> = auto_rules
        .iter()
        .filter(|r| r.action == rules::RuleAction::Deny)
        .cloned()
        .collect();

    // Build a minimal synthetic session for rule matching
    let mut synthetic = session::AgentSession::from_raw(session::RawSession {
        pid: std::process::id(),
        session_id: "brain-query".into(),
        cwd: std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".into()),
        started_at: 0,
        name: None,
        name_source: None,
    });
    synthetic.project_name = project.clone();
    synthetic.status = session::SessionStatus::NeedsInput;
    synthetic.pending_tool_name = Some(tool_name.clone());
    synthetic.pending_tool_input = if command.is_empty() {
        None
    } else {
        Some(command.clone())
    };

    // Check deny rules
    if let Some(deny_match) = rules::evaluate(&deny_rules, &synthetic) {
        let result = serde_json::json!({
            "action": "deny",
            "reasoning": format!("Deny rule '{}' matched", deny_match.rule_name),
            "confidence": 1.0,
            "source": "rule",
        });
        println!("{}", serde_json::to_string(&result).unwrap());
        return Ok(());
    }

    // Step 2: Check approve rules
    let approve_rules: Vec<_> = auto_rules
        .iter()
        .filter(|r| r.action == rules::RuleAction::Approve)
        .cloned()
        .collect();
    if let Some(approve_match) = rules::evaluate(&approve_rules, &synthetic) {
        let result = serde_json::json!({
            "action": "approve",
            "reasoning": format!("Approve rule '{}' matched", approve_match.rule_name),
            "confidence": 1.0,
            "source": "rule",
        });
        println!("{}", serde_json::to_string(&result).unwrap());
        return Ok(());
    }

    // Step 3: Query the LLM brain
    let tool_display = if command.is_empty() {
        tool_name.clone()
    } else {
        format!("{tool_name}: {command}")
    };

    let session_summary = format!(
        "Project: {project} | Status: Needs Input | Pending tool: {tool_name} | Command: {command}"
    );

    // Load distilled preferences
    let pref_section = if let Some(prefs) = brain::decisions::load_preferences_for_project(&project)
    {
        let summary = brain::decisions::format_preference_summary(&prefs);
        format!("\n\n## Learned Preferences\n{summary}")
    } else {
        String::new()
    };

    // Load few-shot examples
    let few_shot_section = {
        let similar = brain::decisions::retrieve_similar(
            Some(&tool_name),
            &project,
            brain_cfg.few_shot_count.min(5),
            Some(brain::decisions::DecisionType::Session),
        );
        if similar.is_empty() {
            String::new()
        } else {
            let examples = brain::decisions::format_few_shot_examples(&similar);
            format!("\n\n## Past Decisions\n{examples}")
        }
    };

    let prompt = format!(
        "You are a session supervisor deciding whether to approve or deny a tool call.\n\
         \n## Session\n{session_summary}\
         {pref_section}\
         {few_shot_section}\n\
         \n## Decision\n\
         The session wants to run [{tool_display}]. \
         Should this be approved or denied? \
         Respond with JSON: {{\"action\": \"approve\"|\"deny\", \
         \"message\": \"...\", \"reasoning\": \"...\", \"confidence\": 0.0-1.0}}"
    );

    match brain::client::infer(&brain_cfg, &prompt) {
        Ok(suggestion) => {
            // Check adaptive threshold
            let threshold = brain::decisions::adaptive_threshold(Some(&tool_name)).unwrap_or(0.6);
            let below_threshold = suggestion.confidence < threshold;

            let result = serde_json::json!({
                "action": suggestion.action.label(),
                "reasoning": suggestion.reasoning,
                "confidence": suggestion.confidence,
                "message": suggestion.message,
                "source": "brain",
                "below_threshold": below_threshold,
                "threshold": threshold,
            });
            println!("{}", serde_json::to_string(&result).unwrap());
            Ok(())
        }
        Err(e) => {
            // On brain failure, output abstain (don't block the user)
            let result = serde_json::json!({
                "action": "abstain",
                "reasoning": format!("Brain query failed: {e}"),
                "confidence": 0.0,
                "source": "error",
            });
            println!("{}", serde_json::to_string(&result).unwrap());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The registry as it looks mid-roll: the legacy sandbox still draining,
    /// two rolled ones, and an engineer's own task sandbox.
    fn registered() -> Vec<String> {
        vec![
            "linera-agent".to_string(),
            "linera-agent-2a14db7ea350".to_string(),
            "linera-agent-ecc2914459c0".to_string(),
            "scylla-investigation".to_string(),
        ]
    }

    #[test]
    fn base_name_covers_the_whole_rolled_family() {
        // The reported bug: restoring 'linera-agent' recovered only the one
        // sandbox named exactly that, leaving every rolled sandbox's sessions
        // stranded.
        assert_eq!(
            matching_sandboxes(&registered(), "linera-agent"),
            vec![
                "linera-agent".to_string(),
                "linera-agent-2a14db7ea350".to_string(),
                "linera-agent-ecc2914459c0".to_string(),
            ]
        );
    }

    #[test]
    fn base_name_still_works_once_the_legacy_sandbox_is_gone() {
        // No sandbox is called exactly 'linera-agent' any more. Before this
        // change that resolved to nothing at all.
        let names = vec![
            "linera-agent-2a14db7ea350".to_string(),
            "linera-agent-ecc2914459c0".to_string(),
        ];
        assert_eq!(matching_sandboxes(&names, "linera-agent"), names);
    }

    #[test]
    fn an_exact_rolled_name_selects_only_itself() {
        assert_eq!(
            matching_sandboxes(&registered(), "linera-agent-2a14db7ea350"),
            vec!["linera-agent-2a14db7ea350".to_string()]
        );
    }

    #[test]
    fn a_named_task_sandbox_is_never_swept_into_another_family() {
        assert_eq!(
            matching_sandboxes(&registered(), "scylla-investigation"),
            vec!["scylla-investigation".to_string()]
        );
        // ...and naming the family must not drag it in.
        assert!(
            !matching_sandboxes(&registered(), "linera-agent")
                .contains(&"scylla-investigation".to_string())
        );
    }

    #[test]
    fn family_matching_stops_at_a_hyphen_boundary() {
        // A bare prefix match would make 'linera-agent' swallow these.
        let names = vec![
            "linera-agentic".to_string(),
            "linera-agentfoo".to_string(),
            "linera-agent".to_string(),
        ];
        assert_eq!(
            matching_sandboxes(&names, "linera-agent"),
            vec!["linera-agent".to_string()]
        );
    }

    #[test]
    fn an_unknown_name_matches_nothing() {
        assert!(matching_sandboxes(&registered(), "no-such-sandbox").is_empty());
        assert!(matching_sandboxes(&[], "linera-agent").is_empty());
    }

    #[test]
    fn session_id_validation_accepts_uuids_and_rejects_shell_metacharacters() {
        // Real Claude Code session ids.
        assert!(is_valid_session_id("11111111-aaaa-2222-bbbb-333333333333"));
        assert!(is_valid_session_id("abc_123.def"));
        // Injection attempts that would break out of `sc --resume <id>`.
        assert!(!is_valid_session_id(""));
        assert!(!is_valid_session_id("foo; rm -rf ~"));
        assert!(!is_valid_session_id("$(whoami)"));
        assert!(!is_valid_session_id("a b"));
        assert!(!is_valid_session_id("back`tick`"));
        assert!(!is_valid_session_id("a|b"));
        assert!(!is_valid_session_id("a&&b"));
    }

    #[test]
    fn resumed_ids_parses_resume_targets_from_ps() {
        let ps = "\
root  sbx exec -it linera-agent bash -lc sc --resume aaaa-1111\n\
root  /bin/bash -lc sc --resume bbbb-2222\n\
root  claudectl --restore-sbx-sessions linera-agent\n\
root  --resume\n";
        let ids = resumed_ids_in(ps);
        assert!(ids.contains("aaaa-1111"));
        assert!(ids.contains("bbbb-2222"));
        assert_eq!(ids.len(), 2, "a trailing --resume with no id is ignored");
    }

    fn entry_with_cwd(cwd: &str) -> sandbox_registry::SessionEntry {
        sandbox_registry::SessionEntry {
            session_id: "7deca33a".to_string(),
            cwd: cwd.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn a_recovered_transcript_supplies_the_cwd_the_entry_lost() {
        // Restoring into `$HOME` is right only by luck. The transcript stamps
        // the real cwd on every record, so a recovered one can say where the
        // session actually ran.
        let site = discovery::TranscriptSite::Recovered(PathBuf::from("/t.jsonl"));
        let cwd = restore_cwd(&entry_with_cwd(""), &site, &|_| {
            Some("/Users/ndr/worktrees/feature".to_string())
        });
        assert_eq!(cwd, "/Users/ndr/worktrees/feature");
    }

    #[test]
    fn a_recorded_cwd_is_never_overridden_by_the_transcript() {
        // No-clobber: the registry's own cwd wins whenever it has one, so a
        // recovered transcript can never move a session out of its directory.
        let site = discovery::TranscriptSite::Recovered(PathBuf::from("/t.jsonl"));
        let cwd = restore_cwd(&entry_with_cwd("/Users/ndr/pm-app"), &site, &|_| {
            panic!("must not read the transcript when the entry has a cwd")
        });
        assert_eq!(cwd, "/Users/ndr/pm-app");
    }

    #[test]
    fn an_unreadable_transcript_falls_back_to_the_home_default() {
        // Recovery is best-effort; failing it must not leave the cwd empty.
        let site = discovery::TranscriptSite::Recovered(PathBuf::from("/t.jsonl"));
        let cwd = restore_cwd(&entry_with_cwd(""), &site, &|_| None);
        assert_eq!(cwd, resolve_cwd(""));
    }

    #[test]
    fn entry_label_prefers_name_then_falls_back_to_short_id() {
        let mut entry = sandbox_registry::SessionEntry {
            session_id: "c3df00ed-83cd-45d7-8ddc-43820b6e4473".to_string(),
            cwd: "/Users/ndr".to_string(),
            transcript: String::new(),
            started_at_ms: 0,
            name: Some("mimir-timeouts".to_string()),
            pid: None,
            owner_pid: None,
            owner_started_at: None,
            ..Default::default()
        };
        assert_eq!(entry_label(&entry), "mimir-timeouts (c3df00ed)");
        entry.name = None;
        assert_eq!(entry_label(&entry), "c3df00ed");
    }
}
