//! Seam tests: what a hook writes vs what the dashboard then shows.
//!
//! Every other test in this crate exercises one side of that seam. The hook
//! writer has thorough coverage of what it records; `foreign_sessions_from`
//! has thorough coverage of how it renders a registry it is handed. Nothing
//! joined them — and all three of the session-visibility bugs shipped in
//! 2026-08 lived exactly there, with both sides individually correct and
//! individually green:
//!
//! - **#36** made the renderer read the registry as the live set, inheriting a
//!   retention policy of "keep sessions whose terminal died" that exists for
//!   `--restore-sbx-sessions`. A closed terminal kept rendering for a minute.
//! - The withdrawn prune proposal would have deleted that same retained data,
//!   destroying restore, for the mirror-image reason.
//! - A registry fixture asserted on the host file while the code under test
//!   wrote to a sandbox slice, so it passed on CI and failed in a sandbox.
//!
//! A unit test cannot catch any of those, because each is written against the
//! same mental model that produced the bug. These drive the **real**
//! `record_hook_event` and assert on the **real** renderer, so the two models
//! have to agree with each other rather than with their author.
//!
//! The invariants, stated once here so a future change has to break a named
//! rule rather than an incidental assertion:
//!
//! 1. A session that has started and not ended is VISIBLE.
//! 2. A session whose `SessionEnd` has fired is NOT VISIBLE — immediately, on
//!    the next render, with no reconcile from any other session.
//! 3. A session whose terminal died is still RESTORABLE after it stops being
//!    visible. Invariants 2 and 3 must hold *simultaneously*; that pair is what
//!    every one of the bugs above got wrong.
//! 4. Visibility never depends on an unrelated session's activity.

use crate::sandbox_registry::{self, SandboxSnapshot};

const NOW: u64 = 1_785_814_692_000;
const INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);
const SANDBOX: &str = "linera-agent-seam";

/// Render exactly what the dashboard would show for `SANDBOX`, through the
/// real renderer, from whatever is on disk right now.
fn visible_session_ids() -> Vec<String> {
    let registry = sandbox_registry::load();
    let running = crate::app::RunningFilter::Known(std::iter::once(SANDBOX.to_string()).collect());
    crate::app::foreign_sessions_from(
        &registry,
        &SandboxSnapshot::default(),
        &running,
        // Render as the LAPTOP: `here` is None, so the sandbox's slice is
        // foreign and goes through the path this seam is about.
        None,
        &[],
        NOW,
        INTERVAL,
        None,
        // This seam asserts which rows render, not what their CPU reads.
        &std::collections::HashMap::new(),
    )
    .into_iter()
    .map(|session| session.session_id)
    .collect()
}

/// Session ids `--restore-sbx-sessions` would replay for `SANDBOX`.
fn restorable_session_ids() -> Vec<String> {
    sandbox_registry::load()
        .sandboxes
        .get(SANDBOX)
        .map(|entries| {
            entries
                .iter()
                .map(|entry| entry.session_id.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Put a session in the sandbox slice the way a live session gets there.
///
/// Writes the entry directly rather than driving `SessionStart`: the hook path
/// records whatever `discovery::live_sessions()` reports, which under a temp
/// `HOME` is empty. The seam under test is what happens to an entry *between*
/// being recorded and being rendered, so seeding it is faithful.
fn seed_live(session_id: &str) {
    let mut slice = sandbox_registry::load()
        .sandboxes
        .remove(SANDBOX)
        .unwrap_or_default();
    slice.push(sandbox_registry::SessionEntry {
        session_id: session_id.to_string(),
        cwd: "/Users/ndr/repos/linera-infra".to_string(),
        transcript: String::new(),
        started_at_ms: 1,
        pid: Some(4242),
        ..Default::default()
    });
    sandbox_registry::replace_sandbox_slice(SANDBOX, slice).unwrap();
}

fn session_end(session_id: &str, reason: &str) {
    crate::hook_state::record_hook_event(&serde_json::json!({
        "hook_event_name": "SessionEnd",
        "session_id": session_id,
        "reason": reason,
    }))
    .unwrap();
}

/// Pin the registry to a temp dir AND make the process look like it is running
/// inside `SANDBOX`, so `record_hook_event` routes its writes to that slice —
/// which is what the laptop then renders.
struct SeamFixture {
    _registry: sandbox_registry::tests::TempRegistry,
    saved_marker: Option<std::ffi::OsString>,
    saved_name: Option<std::ffi::OsString>,
}

impl SeamFixture {
    fn new(tag: &str) -> Self {
        let registry = sandbox_registry::tests::TempRegistry::with_home(tag);
        let saved_marker = std::env::var_os(sandbox_registry::ENV_SANDBOX_MARKER);
        let saved_name = std::env::var_os(sandbox_registry::ENV_SANDBOX_NAME);
        // SAFETY: serialized by the env lock the TempRegistry holds.
        unsafe {
            std::env::set_var(sandbox_registry::ENV_SANDBOX_MARKER, "1");
            std::env::set_var(sandbox_registry::ENV_SANDBOX_NAME, SANDBOX);
        }
        SeamFixture {
            _registry: registry,
            saved_marker,
            saved_name,
        }
    }
}

impl Drop for SeamFixture {
    fn drop(&mut self) {
        // SAFETY: the TempRegistry still holds the env lock at this point.
        unsafe {
            match self.saved_marker.take() {
                Some(value) => std::env::set_var(sandbox_registry::ENV_SANDBOX_MARKER, value),
                None => std::env::remove_var(sandbox_registry::ENV_SANDBOX_MARKER),
            }
            match self.saved_name.take() {
                Some(value) => std::env::set_var(sandbox_registry::ENV_SANDBOX_NAME, value),
                None => std::env::remove_var(sandbox_registry::ENV_SANDBOX_NAME),
            }
        }
    }
}

#[test]
fn invariant_1_a_started_session_is_visible() {
    let _fixture = SeamFixture::new("seam-visible");
    seed_live("aaa");
    assert_eq!(visible_session_ids(), ["aaa"]);
}

#[test]
fn invariant_2_a_closed_terminal_disappears_without_any_other_session_acting() {
    // The exact reported bug. `other` is the reason a terminal-window close
    // produces, and it is precisely the one that does NOT delete the entry —
    // so before the departure stamp this row stayed visible until some
    // unrelated session in the same sandbox happened to fire a hook.
    let _fixture = SeamFixture::new("seam-closed-terminal");
    seed_live("aaa");
    assert_eq!(visible_session_ids(), ["aaa"], "visible while running");

    session_end("aaa", "other");

    assert!(
        visible_session_ids().is_empty(),
        "a session whose SessionEnd fired must vanish on the next render, \
         not when an unrelated session next fires a hook"
    );
}

#[test]
fn invariant_3_the_vanished_session_is_still_restorable() {
    // Runs together with invariant 2 on purpose: hiding the row and keeping
    // the restore material are the two halves that every bug here broke one of.
    let _fixture = SeamFixture::new("seam-still-restorable");
    seed_live("aaa");
    session_end("aaa", "other");

    assert!(visible_session_ids().is_empty(), "hidden");
    assert_eq!(
        restorable_session_ids(),
        ["aaa"],
        "--restore-sbx-sessions must still replay a session that died with its terminal"
    );
}

#[test]
fn a_deliberate_exit_also_disappears_and_is_not_restore_material() {
    // The other half of the reason matrix: a session the user closed on purpose
    // is not something restore should bring back, so here the entry really does
    // go away. Both reasons must leave the view immediately; they differ only
    // in what stays on disk.
    let _fixture = SeamFixture::new("seam-deliberate-exit");
    seed_live("aaa");
    session_end("aaa", "prompt_input_exit");

    assert!(visible_session_ids().is_empty(), "hidden");
    assert!(
        restorable_session_ids().is_empty(),
        "a deliberate close is not restore material"
    );
}

#[test]
fn clear_does_not_hide_a_session_that_is_still_running() {
    // `clear` fires SessionEnd without ending anything. Hiding it would be the
    // same class of bug, mirrored: a live session missing from the dashboard.
    let _fixture = SeamFixture::new("seam-clear");
    seed_live("aaa");
    session_end("aaa", "clear");

    assert_eq!(
        visible_session_ids(),
        ["aaa"],
        "`clear` leaves the session running, so it must stay visible"
    );
}

#[test]
fn invariant_4_one_sessions_departure_does_not_disturb_another() {
    let _fixture = SeamFixture::new("seam-independence");
    seed_live("aaa");
    seed_live("bbb");
    session_end("aaa", "other");

    assert_eq!(
        visible_session_ids(),
        ["bbb"],
        "ending one session must not hide or resurrect any other"
    );
}

#[test]
fn a_resumed_session_becomes_visible_again() {
    // `--resume` reuses the session id. A departure stamp left over from the
    // previous run would hide the resumed session indefinitely, which is the
    // bug this seam would otherwise trade for the original one.
    let _fixture = SeamFixture::new("seam-resume");
    seed_live("aaa");
    session_end("aaa", "other");
    assert!(visible_session_ids().is_empty());

    // Being recorded live again is what un-departs it.
    seed_live("aaa");

    assert_eq!(
        visible_session_ids(),
        ["aaa"],
        "a resumed session must come back into view"
    );
}
