use crate::session::ClaudeSession;

/// Open a new Kitty OS window running `command` in `cwd`, via a login shell so
/// PATH resolves `claude`/`sc`. Needs `allow_remote_control` (like [`launch`]).
/// `cwd`/`command` are argv tokens — no shell interpretation, no injection.
pub fn spawn_window(cwd: &str, command: &str) -> Result<String, String> {
    let output = std::process::Command::new("kitty")
        .args(spawn_argv(cwd, command))
        .output()
        .map_err(|e| format!("kitty spawn failed: {e}. Is allow_remote_control enabled?"))?;

    if output.status.success() {
        Ok("kitty window".into())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn spawn_argv(cwd: &str, command: &str) -> Vec<String> {
    // `--type=os-window` gives one window per session (vs `launch`'s `--type=tab`).
    [
        "@",
        "launch",
        "--type=os-window",
        "--cwd",
        cwd,
        "bash",
        "-lc",
        command,
    ]
    .map(String::from)
    .to_vec()
}

pub fn launch(cwd: &str, prompt: Option<&str>, resume: Option<&str>) -> Result<String, String> {
    let mut cmd = std::process::Command::new("kitty");
    cmd.args(["@", "launch", "--type=tab", "--cwd", cwd, "claude"]);
    for arg in super::build_claude_args(prompt, resume) {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("kitty launch failed: {e}. Is allow_remote_control enabled?"))?;

    if output.status.success() {
        Ok("kitty tab".into())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

pub fn switch(session: &ClaudeSession) -> Result<(), String> {
    // Kitty has a powerful remote control protocol via `kitty @ focus-window`.
    // Requires `allow_remote_control yes` or `allow_remote_control socket-only` in kitty.conf.
    // Match by the PID of the foreground process in the window.
    let pid = session.pid.to_string();

    // First try matching by the foreground process PID
    let output = std::process::Command::new("kitty")
        .args(["@", "focus-window", "--match", &format!("pid:{pid}")])
        .output();

    match output {
        Ok(o) if o.status.success() => return Ok(()),
        _ => {}
    }

    // Fallback: match by cwd
    let output = std::process::Command::new("kitty")
        .args([
            "@",
            "focus-window",
            "--match",
            &format!("cwd:{}", session.cwd),
        ])
        .output()
        .map_err(|e| format!("kitty @ failed: {e}. Is allow_remote_control enabled?"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Kitty: {}", stderr.trim()))
    }
}

pub fn send_input(session: &ClaudeSession, text: &str) -> Result<(), String> {
    let output = std::process::Command::new("kitty")
        .args([
            "@",
            "send-text",
            "--match",
            &format!("pid:{}", session.pid),
            text,
        ])
        .output()
        .map_err(|e| format!("kitty send-text failed: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

pub fn approve(session: &ClaudeSession) -> Result<(), String> {
    let output = std::process::Command::new("kitty")
        .args([
            "@",
            "send-text",
            "--match",
            &format!("pid:{}", session.pid),
            "\r",
        ])
        .output()
        .map_err(|e| format!("kitty send-text failed: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_argv_opens_an_os_window_in_cwd_running_the_command_via_login_shell() {
        let argv = spawn_argv("/work/scylla", "sc --resume abc123");
        assert_eq!(
            argv,
            [
                "@",
                "launch",
                "--type=os-window",
                "--cwd",
                "/work/scylla",
                "bash",
                "-lc",
                "sc --resume abc123",
            ]
        );
    }
}
