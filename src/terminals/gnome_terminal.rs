use crate::session::ClaudeSession;

/// Open a new GNOME Terminal window running `command` in `cwd`, via a login
/// shell so PATH resolves `claude`/`sc`. `cwd` and `command` are passed as argv
/// tokens (no shell interpretation), so neither can inject.
pub fn spawn_window(cwd: &str, command: &str) -> Result<String, String> {
    let output = std::process::Command::new("gnome-terminal")
        .args(spawn_argv(cwd, command))
        .output()
        .map_err(|e| format!("gnome-terminal spawn failed: {e}"))?;

    if output.status.success() {
        Ok("gnome-terminal window".into())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn spawn_argv(cwd: &str, command: &str) -> Vec<String> {
    [
        "--window",
        "--working-directory",
        cwd,
        "--",
        "bash",
        "-lc",
        command,
    ]
    .map(String::from)
    .to_vec()
}

pub fn launch(cwd: &str, prompt: Option<&str>, resume: Option<&str>) -> Result<String, String> {
    let mut cmd = std::process::Command::new("gnome-terminal");
    cmd.args(["--window", "--working-directory", cwd, "--", "claude"]);
    for arg in super::build_claude_args(prompt, resume) {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("gnome-terminal launch failed: {e}"))?;

    if output.status.success() {
        Ok("gnome-terminal window".into())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

pub fn switch(_session: &ClaudeSession) -> Result<(), String> {
    Err(
        "GNOME Terminal launch is supported, but remote focus/input control is not yet reliable. Use tmux or Kitty for session switching and input automation."
            .into(),
    )
}

pub fn send_input(_session: &ClaudeSession, _text: &str) -> Result<(), String> {
    Err(
        "GNOME Terminal launch is supported, but remote focus/input control is not yet reliable. Use tmux or Kitty for session input automation."
            .into(),
    )
}

pub fn approve(_session: &ClaudeSession) -> Result<(), String> {
    Err(
        "GNOME Terminal launch is supported, but remote focus/input control is not yet reliable. Use tmux or Kitty for approval automation."
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_argv_opens_a_window_in_cwd_running_the_command_via_login_shell() {
        let argv = spawn_argv("/work/scylla", "claude --resume abc123");
        assert_eq!(
            argv,
            [
                "--window",
                "--working-directory",
                "/work/scylla",
                "--",
                "bash",
                "-lc",
                "claude --resume abc123",
            ]
        );
    }
}
