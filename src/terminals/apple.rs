use super::{applescript_cd_exec, run_osascript};
use crate::session::ClaudeSession;

/// Open a new Apple Terminal (Terminal.app) window that runs `command` in `cwd`.
/// A bare `do script` (no `in` target) opens a fresh window, mirroring the
/// one-window-per-session shape of the other restore backends.
pub fn spawn_window(cwd: &str, command: &str) -> Result<String, String> {
    run_osascript(&new_window_script(cwd, command))?;
    Ok("Apple Terminal".to_string())
}

fn new_window_script(cwd: &str, command: &str) -> String {
    format!(
        r#"
        tell application "Terminal"
            do script {}
            activate
        end tell
        "#,
        applescript_cd_exec(cwd, command),
    )
}

pub fn switch(session: &ClaudeSession) -> Result<(), String> {
    let script = format!(
        r#"
        tell application "Terminal"
            repeat with w in windows
                repeat with t in tabs of w
                    if tty of t contains "{tty}" then
                        set selected tab of w to t
                        set index of w to 1
                        activate
                        return "ok"
                    end if
                end repeat
            end repeat
            error "TTY not found in Terminal.app"
        end tell
        "#,
        tty = session.tty
    );
    run_osascript(&script)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_window_script_opens_a_window_that_cds_and_execs() {
        let script = new_window_script("/work/scylla", "claude --resume abc123");
        // `do script` with no `in` target opens a new Terminal.app window.
        assert!(script.contains("do script"));
        // cwd is shell-quoted at runtime; command is exec'd after cd.
        assert!(script.contains(r#"quoted form of "/work/scylla""#));
        assert!(script.contains("exec claude --resume abc123"));
    }

    #[test]
    fn new_window_script_escapes_quotes_in_cwd() {
        let script = new_window_script("/work/a\"b", "claude --resume x");
        assert!(script.contains(r#"quoted form of "/work/a\"b""#));
    }
}
