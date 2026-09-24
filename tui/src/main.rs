use std::env;
use std::io::{self, IsTerminal};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use habfly_tui::{
    app::App,
    process::{ProcessMessage, RuntimeProcess},
    protocol::{CommandKind, RuntimeCommand},
    ui,
};
use serde_json::{json, Value};

const HELP: &str = "HabFly local terminal interface\n\nRun from the HabFly repository root:\n  cargo run --manifest-path tui/Cargo.toml -- [OPTIONS]\n\nOptions:\n  --python PATH          Runtime interpreter (default .venv/bin/python)\n  --replay PATH          Open a saved JSONL trajectory through the Python runtime\n  --trace PATH           Save-trace destination (default experiments/traces/tui-session.jsonl)\n  --start-payload JSON    Runtime start settings (default expert, 1 star, seed 0)\n  --autostart            Start the configured runtime episode on launch\n  -h, --help             Print this help\n\nKeys: s start, space pause/resume, p pause, r resume, n step, a abort,\n      t save trace, v focus panels, arrows scroll controls, q or Ctrl-C quit.\n";

struct Options {
    python: String,
    replay: Option<String>,
    trace: String,
    start_payload: Value,
    autostart: bool,
}

fn options(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, String> {
    let mut options = Options {
        python: ".venv/bin/python".into(),
        replay: None,
        trace: "experiments/traces/tui-session.jsonl".into(),
        start_payload: json!({"policy":"expert", "seed":0, "stars":1}),
        autostart: false,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--autostart" => options.autostart = true,
            "--python" | "--replay" | "--trace" | "--start-payload" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                match arg.as_str() {
                    "--python" => options.python = value,
                    "--replay" => options.replay = Some(value),
                    "--trace" => options.trace = value,
                    _ => {
                        let payload: Value = serde_json::from_str(&value)
                            .map_err(|error| format!("Invalid start payload: {error}"))?;
                        let object = payload
                            .as_object()
                            .ok_or("Start payload must be a JSON object")?;
                        options
                            .start_payload
                            .as_object_mut()
                            .unwrap()
                            .extend(object.clone());
                    }
                }
            }
            _ => return Err(format!("Unknown option {arg}; use --help")),
        }
    }
    if options.replay.is_some() && options.autostart {
        return Err("Choose --replay or --autostart, not both".into());
    }
    Ok(Some(options))
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("HabFly TUI: {error}");
        std::process::exit(1);
    }
}

fn send(process: &RuntimeProcess, app: &mut App, command: CommandKind, payload: Value) {
    match process.send(RuntimeCommand::new(command, payload)) {
        Ok(()) => app.log(format!("Command: {command:?}")),
        Err(error) => app.log(error),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(options) = options(env::args().skip(1)).map_err(io::Error::other)? else {
        print!("{HELP}");
        return Ok(());
    };
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(
            "An interactive terminal is required; use `habfly runtime --jsonl` for a pipe".into(),
        );
    }
    let mut process = RuntimeProcess::spawn(&options.python).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "Cannot start {}: {error}. Run from the repository root or set --python PATH",
                options.python
            ),
        )
    })?;
    // try_init installs Ratatui's panic restoration hook. RAII covers ordinary errors.
    let mut terminal = ratatui::try_init()?;
    let _restore = TerminalGuard;
    let mut app = App::default();
    let mut exit_reported = false;
    let mut dirty = true;
    if let Some(path) = &options.replay {
        send(
            &process,
            &mut app,
            CommandKind::Replay,
            json!({"path":path}),
        );
    } else if options.autostart {
        send(
            &process,
            &mut app,
            CommandKind::Start,
            options.start_payload.clone(),
        );
    }

    loop {
        // Bound each drain so a busy trainer cannot starve rendering or keyboard input.
        for _ in 0..128 {
            match process.messages.try_recv() {
                Ok(ProcessMessage::Event(event)) => {
                    app.apply(event);
                    dirty = true;
                }
                Ok(ProcessMessage::Diagnostic(message)) => {
                    app.log(message);
                    dirty = true;
                }
                Ok(ProcessMessage::StdoutClosed) => {
                    app.connected = false;
                    app.log("Runtime stdout closed");
                    dirty = true;
                }
                Err(_) => break,
            }
        }
        if !exit_reported {
            if let Some(status) = process.exit_status()? {
                app.connected = false;
                app.state["status"] = Value::String(format!("runtime exited: {status}"));
                app.log(format!("Runtime exited: {status}; q to close"));
                exit_reported = true;
                dirty = true;
            }
        }
        if dirty {
            terminal.draw(|frame| ui::draw(frame, &app))?;
            dirty = false;
        }
        if !event::poll(Duration::from_millis(50))? {
            continue;
        }
        let input = event::read()?;
        if matches!(input, Event::Resize(_, _)) {
            dirty = true;
        }
        if let Event::Key(key) = input {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if key.code == KeyCode::Char('q')
                || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
            {
                if matches!(app.state["status"].as_str(), Some("running" | "paused")) {
                    let _ = process.send(RuntimeCommand::new(CommandKind::Abort, json!({})));
                }
                break;
            }
            match key.code {
                KeyCode::Char('s') => send(
                    &process,
                    &mut app,
                    CommandKind::Start,
                    options.start_payload.clone(),
                ),
                KeyCode::Char('p') => send(&process, &mut app, CommandKind::Pause, json!({})),
                KeyCode::Char('r') => send(&process, &mut app, CommandKind::Resume, json!({})),
                KeyCode::Char(' ') => {
                    let command = if app.paused {
                        CommandKind::Resume
                    } else {
                        CommandKind::Pause
                    };
                    send(&process, &mut app, command, json!({}));
                }
                KeyCode::Char('n')
                    if app.state["browser_execution"] == "autonomous"
                        && app.state["replay"] != true
                        && !app.paused =>
                {
                    app.log(
                        "Autonomous decisions run automatically; p pauses before manual stepping.",
                    );
                }
                KeyCode::Char('n') => send(&process, &mut app, CommandKind::Step, json!({})),
                KeyCode::Char('b')
                    if app.state["browser_phase"] == "awaiting_ready"
                        && app.state["replay"] != true =>
                {
                    send(
                        &process,
                        &mut app,
                        CommandKind::Step,
                        json!({"browser_ready":true}),
                    )
                }
                KeyCode::Char('y')
                    if app.state["browser_phase"] == "awaiting_copy"
                        && (app.state["browser_execution"] != "autonomous" || app.paused)
                        && app.state["replay"] != true =>
                {
                    send(
                        &process,
                        &mut app,
                        CommandKind::Step,
                        json!({"approve_copy":true}),
                    )
                }
                KeyCode::Char('a') => send(&process, &mut app, CommandKind::Abort, json!({})),
                KeyCode::Char('t') => send(
                    &process,
                    &mut app,
                    CommandKind::SaveTrace,
                    json!({"path":options.trace}),
                ),
                KeyCode::Up | KeyCode::Down => {
                    let offset = if app.focused_panel == 3 {
                        &mut app.observation_scroll
                    } else {
                        &mut app.controls_scroll
                    };
                    *offset = if key.code == KeyCode::Up {
                        offset.saturating_sub(1)
                    } else {
                        offset.saturating_add(1)
                    };
                }
                KeyCode::Char('v') => app.focused_panel = (app.focused_panel + 1) % 4,
                _ => {}
            }
            dirty = true;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_demo_is_explicitly_expert() {
        let options = options(Vec::<String>::new()).unwrap().unwrap();
        assert_eq!(options.start_payload["policy"], "expert");
        assert_eq!(options.python, ".venv/bin/python");
    }

    #[test]
    fn partial_start_override_preserves_defaults() {
        let options = options(["--start-payload".into(), r#"{"seed":42}"#.into()])
            .unwrap()
            .unwrap();
        assert_eq!(options.start_payload["seed"], 42);
        assert_eq!(options.start_payload["policy"], "expert");
        assert!(options.replay.is_none());
    }

    #[test]
    fn bad_arguments_are_actionable() {
        for args in [
            vec!["--python"],
            vec!["--unknown"],
            vec!["--start-payload", "[]"],
            vec!["--autostart", "--replay", "trace.jsonl"],
        ] {
            assert!(options(args.into_iter().map(String::from)).is_err());
        }
    }
}
