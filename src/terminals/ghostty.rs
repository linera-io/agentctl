use super::{applescript_escape, run_osascript};
use crate::session::ClaudeSession;

/// A session with no routing data at all cannot be matched to any Ghostty
/// surface — fail with an actionable claudectl error instead of shipping an
/// AppleScript guaranteed to die with an empty lookup key (seen live as
/// `execution error: No Ghostty terminal found for  (-2700)` after a registry
/// briefly recorded sessions without cwds).
fn require_routing_data(session: &ClaudeSession) -> Result<(), String> {
    if session.terminal_id.is_none() && session.tty.is_empty() && session.cwd.is_empty() {
        return Err(format!(
            "no terminal routing data for {} (pid {}): no surface id, tty, or cwd yet — \
             give the dashboard a few refresh ticks and retry",
            session.display_name(),
            session.pid
        ));
    }
    Ok(())
}

/// Find the best matching Ghostty terminal for a session.
///
/// Matching priority (most precise first):
///   1. `session.terminal_id` — the surface's AppleScript id, captured at launch
///      by the agent-sandbox wrapper. Unambiguous.
///   2. `tty` — Ghostty >= 1.4.0 exposes a `tty` property on terminals
///      (ghostty-org/ghostty#11922): an exact 1:1 key, same as iTerm2. Matched
///      with `contains` because `ps` reports `ttysNNN` while Ghostty reports
///      `/dev/ttysNNN` (mirrors the iTerm2 matcher), and wrapped in `try` so it's
///      a harmless no-op on Ghostty <= 1.3.1 (where the property doesn't exist
///      and the query errors) — we then fall through to the CWD match.
///   3. working directory, exact then substring, + title disambiguator — the
///      fallback for older Ghostty. Breaks down when multiple claudes share a CWD.
fn find_terminal_script(session: &ClaudeSession) -> String {
    if let Some(ref id) = session.terminal_id {
        let escaped = applescript_escape(id);
        return format!(
            r#"
            set matches to every terminal whose id is "{escaped}"
            if (count of matches) = 0 then error "No Ghostty terminal with id {escaped}"
            set t to item 1 of matches
            "#,
        );
    }
    let cwd = applescript_escape(&session.cwd);
    let session_name = applescript_escape(&session.session_name);
    let tty = applescript_escape(&session.tty);

    // Build the candidate list in order of precision: tty (exact, Ghostty >=
    // 1.4.0) → working directory exact → working directory substring. Exact-cwd
    // before substring stops a shallow cwd (e.g. the home directory) from
    // matching every surface nested under it; the substring fallback preserves
    // behavior when Ghostty reports a normalized path (symlink /tmp ->
    // /private/tmp, or a trailing slash) that doesn't byte-match the cwd.
    let mut find = String::from("\n            set matches to {}\n");
    if !tty.is_empty() {
        find.push_str(&format!(
            "            try\n                set matches to every terminal whose tty contains \"{tty}\"\n            end try\n"
        ));
    }
    find.push_str(&format!(
        r#"            if (count of matches) = 0 then
                set matches to every terminal whose working directory is "{cwd}"
            end if
            if (count of matches) = 0 then
                set matches to every terminal whose working directory contains "{cwd}"
            end if
            if (count of matches) = 0 then error "No Ghostty terminal found for {cwd}"
            set t to item 1 of matches
"#
    ));
    if !session_name.is_empty() {
        // Disambiguate by title when several surfaces share a CWD. Claude Code
        // sets the title to "<spinner> <task_description>", which often contains
        // the session name. (A tty match, when available, is already unique, so
        // this loop is a no-op there.)
        find.push_str(&format!(
            r#"            repeat with candidate in matches
                if name of candidate contains "{session_name}" then
                    set t to candidate
                    exit repeat
                end if
            end repeat
"#
        ));
    }
    find
}

pub fn switch(session: &ClaudeSession) -> Result<(), String> {
    require_routing_data(session)?;
    let find = find_terminal_script(session);

    let script = format!(
        r#"
        tell application "Ghostty"
            {find}
            focus t
            activate
        end tell
        "#,
    );

    run_osascript(&script)
}

pub fn send_input(session: &ClaudeSession, text: &str) -> Result<(), String> {
    require_routing_data(session)?;
    let find = find_terminal_script(session);

    // Strip trailing newline — we append AppleScript `return` instead so the
    // newline is a proper CR rather than a literal embedded in the string.
    let trimmed = text.trim_end_matches('\n').trim_end_matches('\r');
    let escaped = applescript_escape(trimmed);
    let has_trailing_newline = text.ends_with('\n') || text.ends_with('\r');

    let text_expr = if has_trailing_newline {
        format!("\"{escaped}\" & return")
    } else {
        format!("\"{escaped}\"")
    };

    let script = format!(
        r#"
        tell application "Ghostty"
            {find}
            input text {text_expr} to t
        end tell
        "#,
    );
    run_osascript(&script)
}

pub fn approve(session: &ClaudeSession) -> Result<(), String> {
    require_routing_data(session)?;
    let find = find_terminal_script(session);

    let script = format!(
        r#"
        tell application "Ghostty"
            {find}
            send key "enter" to t
        end tell
        "#,
    );
    run_osascript(&script)
}

/// Build the AppleScript that opens a NEW window in the already-running Ghostty
/// and runs `command` in `cwd`.
///
/// Uses Ghostty's native `new window with configuration {…}` (from its scripting
/// dictionary) — no new app instance (unlike `open -na`), and pure Apple Events
/// to Ghostty, so no macOS Accessibility permission (unlike a System-Events
/// keystroke). `initial working directory` sets the cwd; `initial input` runs
/// `command` in the window's normal login shell (so PATH resolves `sc`/`sbx`),
/// with `& return` to execute it. Pure — unit-tested.
fn new_window_script(cwd: &str, command: &str) -> String {
    let cwd = applescript_escape(cwd);
    let command = applescript_escape(command);
    format!(
        r#"
        tell application "Ghostty"
            new window with configuration {{initial working directory:"{cwd}", initial input:"{command}" & return}}
        end tell
        "#,
    )
}

/// Open a new window in the running Ghostty running `command` in `cwd` (no new
/// app instance). macOS only — it drives Ghostty via AppleScript.
pub fn spawn_window(cwd: &str, command: &str) -> Result<String, String> {
    if !cfg!(target_os = "macos") {
        return Err("Ghostty window spawn is only implemented on macOS".to_string());
    }
    run_osascript(&new_window_script(cwd, command))?;
    Ok("Ghostty".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{ClaudeSession, RawSession};

    fn routed_session(
        terminal_id: Option<&str>,
        tty: &str,
        cwd: &str,
        name: &str,
    ) -> ClaudeSession {
        let mut session = ClaudeSession::from_raw(RawSession {
            pid: 42,
            session_id: "sess-42".into(),
            cwd: cwd.into(),
            started_at: 0,
            name: (!name.is_empty()).then(|| name.to_string()),
        });
        session.terminal_id = terminal_id.map(str::to_string);
        session.tty = tty.into();
        session
    }

    #[test]
    fn find_script_prefers_surface_id_over_everything() {
        let script = find_terminal_script(&routed_session(
            Some("9B65C6AC"),
            "/dev/ttys019",
            "/Users/ndr",
            "my-session",
        ));
        assert!(script.contains(r#"whose id is "9B65C6AC""#));
        assert!(!script.contains("working directory"), "id match is final");
    }

    #[test]
    fn find_script_matches_tty_then_falls_back_to_cwd() {
        let script = find_terminal_script(&routed_session(None, "/dev/ttys019", "/Users/ndr", ""));
        assert!(script.contains(r#"whose tty contains "/dev/ttys019""#));
        assert!(script.contains(r#"whose working directory is "/Users/ndr""#));
        assert!(script.contains(r#"error "No Ghostty terminal found for /Users/ndr""#));
    }

    #[test]
    fn find_script_disambiguates_shared_cwd_by_title() {
        let script = find_terminal_script(&routed_session(None, "", "/Users/ndr", "my-session"));
        assert!(!script.contains("whose tty"), "no tty -> no tty clause");
        assert!(script.contains(r#"name of candidate contains "my-session""#));
    }

    #[test]
    fn switch_fails_fast_without_any_routing_data() {
        // Regression: a session with no surface id, tty, or cwd used to ship
        // an AppleScript that could only die with an empty lookup key
        // (`No Ghostty terminal found for  (-2700)`). It must fail on our
        // side with an actionable message instead, before reaching the
        // bridge/osascript.
        let session = routed_session(None, "", "", "");
        for result in [
            switch(&session),
            send_input(&session, "hi"),
            approve(&session),
        ] {
            let err = result.expect_err("routing-data-less session must fail fast");
            assert!(
                err.contains("no terminal routing data"),
                "actionable claudectl error, got: {err}"
            );
        }
    }

    #[test]
    fn new_window_script_uses_native_new_window_with_cwd_and_command() {
        let script = new_window_script("/work/scylla", "sc --resume abc123");
        // Native Ghostty command → new window in the running instance.
        assert!(script.contains("new window with configuration"));
        assert!(script.contains("initial working directory:\"/work/scylla\""));
        assert!(script.contains("initial input:\"sc --resume abc123\" & return"));
        // No System Events (would need Accessibility) and no `open` (new instance).
        assert!(!script.contains("keystroke"));
        assert!(!script.to_lowercase().contains("system events"));
    }

    #[test]
    fn new_window_script_escapes_applescript_specials() {
        // A quote in the cwd must be escaped so the record literal stays valid.
        let script = new_window_script("/work/a\"b", "sc --resume x");
        assert!(script.contains("initial working directory:\"/work/a\\\"b\""));
    }

    fn make_session(cwd: &str, name: &str) -> ClaudeSession {
        let raw = RawSession {
            pid: 100,
            session_id: "test".into(),
            cwd: cwd.into(),
            started_at: 0,
            name: None,
        };
        let mut s = ClaudeSession::from_raw(raw);
        s.session_name = name.into();
        s
    }

    #[test]
    fn find_script_unnamed_session() {
        let s = make_session("/tmp/my-project", "");
        let script = find_terminal_script(&s);
        // Exact-first, with a substring fallback.
        assert!(script.contains("working directory is \"/tmp/my-project\""));
        assert!(script.contains("working directory contains \"/tmp/my-project\""));
        // Should NOT have name-matching logic
        assert!(!script.contains("name of candidate"));
    }

    #[test]
    fn find_script_named_session() {
        let s = make_session("/tmp/my-project", "my-task");
        let script = find_terminal_script(&s);
        // Exact-first, with a substring fallback.
        assert!(script.contains("working directory is \"/tmp/my-project\""));
        assert!(script.contains("working directory contains \"/tmp/my-project\""));
        assert!(script.contains("name of candidate contains \"my-task\""));
        // Should set fallback before loop
        assert!(script.contains("set t to item 1 of matches"));
    }

    #[test]
    fn find_script_prefers_tty_when_present() {
        let mut s = make_session("/tmp/p", "task");
        s.tty = "ttys014".into();
        let script = find_terminal_script(&s);
        // tty match first, `try`-wrapped (no-op on Ghostty <= 1.3.1), matched
        // with `contains` (ps `ttysNNN` vs Ghostty `/dev/ttysNNN`).
        assert!(script.contains("try"));
        assert!(script.contains("whose tty contains \"ttys014\""));
        // CWD fallback still present after the tty attempt.
        assert!(script.contains("working directory is \"/tmp/p\""));
    }

    #[test]
    fn find_script_omits_tty_block_when_tty_empty() {
        let s = make_session("/tmp/p", ""); // tty defaults to empty
        let script = find_terminal_script(&s);
        assert!(!script.contains("whose tty"));
    }

    #[test]
    fn find_script_escapes_quotes() {
        let s = make_session("/tmp/project \"alpha\"", "task \"beta\"");
        let script = find_terminal_script(&s);
        assert!(script.contains("project \\\"alpha\\\""));
        assert!(script.contains("task \\\"beta\\\""));
    }

    #[test]
    fn find_script_escapes_backslashes() {
        let s = make_session("/tmp/path\\with\\slashes", "name\\here");
        let script = find_terminal_script(&s);
        assert!(script.contains("path\\\\with\\\\slashes"));
        assert!(script.contains("name\\\\here"));
    }

    #[test]
    fn applescript_escape_handles_both() {
        assert_eq!(applescript_escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn send_input_trailing_newline_uses_return() {
        let s = make_session("/tmp/proj", "");
        // We can't call send_input directly (it runs osascript), but we can
        // verify the text processing logic by checking the escaping.
        let text = "continue\n";
        let trimmed = text.trim_end_matches('\n').trim_end_matches('\r');
        let escaped = applescript_escape(trimmed);
        let has_trailing = text.ends_with('\n') || text.ends_with('\r');
        assert_eq!(trimmed, "continue");
        assert_eq!(escaped, "continue");
        assert!(has_trailing);
        // The expression should use & return
        let expr = if has_trailing {
            format!("\"{escaped}\" & return")
        } else {
            format!("\"{escaped}\"")
        };
        assert_eq!(expr, "\"continue\" & return");
        let _ = s; // suppress unused
    }

    #[test]
    fn send_input_no_trailing_newline() {
        let text = "some text";
        let trimmed = text.trim_end_matches('\n').trim_end_matches('\r');
        let has_trailing = text.ends_with('\n') || text.ends_with('\r');
        assert_eq!(trimmed, "some text");
        assert!(!has_trailing);
    }
}
