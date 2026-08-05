//! The Claude Code hook receiver, as a binary of its own.
//!
//! Claude Code invokes this on every hook event with a JSON payload on stdin.
//! It records the event into claudectl's session registry and hook state, and
//! exits. That is the whole job.
//!
//! # Why this is separate from `claudectl`
//!
//! The agent sandbox needs a hook receiver inside it — hooks fire in the
//! sandbox, so the binary must live there — but it needs nothing else claudectl
//! does. Shipping the full binary meant the sandbox image carried the TUI, the
//! local-LLM brain, the orchestrator and every terminal backend, and therefore
//! changed whenever any of those did. Since a sandbox's identity is derived
//! from its image digest, "someone pushed a TUI tweak" was enough to roll every
//! engineer onto a fresh sandbox.
//!
//! A receiver whose contract is "read a hook payload, write a registry entry"
//! changes on the order of never, so the sandbox can pin it and bump
//! deliberately.
//!
//! # Why it is not a separate project
//!
//! It shares the library: the same `SessionEntry` shape, the same flock plus
//! temp-file-rename discipline, the same status-inference state. A standalone
//! project would have to reimplement those, and the two copies would drift —
//! which is precisely the failure mode that produced a run of "the collected
//! payload doesn't carry what the reader needs" bugs. One implementation, two
//! entry points.
//!
//! `claudectl` keeps its own hook fast-path, so settings still pointing at the
//! old command keep working and this can be rolled out without a flag day.

fn main() {
    // Opt-in diagnostic log. `claudectl` takes a `--log` flag for this, but a
    // hook receives no arguments — Claude Code invokes it as a bare command —
    // so the only channel left is the environment.
    //
    // Off by default and deliberately so: this runs once per hook event per
    // session, which on a busy sandbox is tens of writes per turn onto a
    // shared mount. Opt-in keeps that cost at zero until someone is actually
    // debugging.
    //
    // It exists because this binary is otherwise completely silent. It writes
    // nothing to stdout (that would corrupt a caller capturing output), its
    // errors are swallowed by design, and the hook command itself ends in
    // `2>/dev/null || true`. A hook that runs and quietly does the wrong thing
    // is indistinguishable from one that never ran — which is exactly the wall
    // hit on 2026-08-05, when `claudectl-hook` under `sbx exec` recorded no
    // terminal routing and three separate theories for why were all wrong.
    if let Some(path) = std::env::var_os("CLAUDECTL_HOOK_LOG") {
        let _ = claudectl::logger::init(&path.to_string_lossy());
    }

    // Mirrors the fast-path in `claudectl`'s `main`. `try_read_hook_payload`
    // returns None promptly for tty / empty / no-data stdin, so an accidental
    // interactive invocation exits quietly instead of hanging.
    //
    // Failures are swallowed on purpose: a hook that errors must never block
    // or slow the Claude Code session it fired from. Nothing is written to
    // stdout, which would corrupt a caller capturing output.
    match claudectl::hook_state::try_read_hook_payload() {
        Ok(Some(payload)) => {
            let event = payload
                .get("hook_event_name")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            claudectl::logger::log("DEBUG", &format!("hook: received {event}"));
            if let Err(e) = claudectl::hook_state::record_hook_event(&payload) {
                claudectl::logger::log("ERROR", &format!("hook: {event} failed: {e}"));
            }
        }
        Ok(None) => claudectl::logger::log("DEBUG", "hook: no payload on stdin"),
        Err(e) => claudectl::logger::log("ERROR", &format!("hook: unreadable payload: {e}")),
    }
}
