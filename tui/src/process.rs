use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use crate::protocol::{parse_event, RuntimeCommand, RuntimeEvent};

const MAX_LINE_BYTES: usize = 1024 * 1024;
const EVENT_QUEUE_SIZE: usize = 256;
const COMMAND_QUEUE_SIZE: usize = 32;

#[derive(Debug)]
pub enum ProcessMessage {
    Event(RuntimeEvent),
    Diagnostic(String),
    StdoutClosed,
}

pub struct RuntimeProcess {
    child: Child,
    pub messages: Receiver<ProcessMessage>,
    commands: Option<SyncSender<RuntimeCommand>>,
}

impl RuntimeProcess {
    pub fn spawn(python: &str) -> io::Result<Self> {
        let mut command = Command::new(python);
        command.args(["-u", "-m", "habfly", "runtime", "--jsonl"]);
        Self::spawn_command(&mut command)
    }

    fn spawn_command(command: &mut Command) -> io::Result<Self> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // These are guaranteed by Stdio::piped above; acquire all before creating threads.
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut stdin = child.stdin.take().expect("piped stdin");
        let (message_tx, messages) = mpsc::sync_channel(EVENT_QUEUE_SIZE);
        let (command_tx, commands) = mpsc::sync_channel::<RuntimeCommand>(COMMAND_QUEUE_SIZE);

        let stdout_tx = message_tx.clone();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                match bounded_line(&mut reader) {
                    Ok(Some(Ok(line))) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let message = match parse_event(&line) {
                            Ok(event) => ProcessMessage::Event(event),
                            Err(error) => ProcessMessage::Diagnostic(error),
                        };
                        if stdout_tx.send(message).is_err() {
                            break;
                        }
                    }
                    Ok(Some(Err(error))) => {
                        if stdout_tx.send(ProcessMessage::Diagnostic(error)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let _ = stdout_tx.send(ProcessMessage::Diagnostic(format!(
                            "Runtime stdout: {error}"
                        )));
                        break;
                    }
                }
            }
            let _ = stdout_tx.send(ProcessMessage::StdoutClosed);
        });
        let stderr_tx = message_tx.clone();
        thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            while let Ok(Some(line)) = bounded_line(&mut reader) {
                let message = match line {
                    Ok(line) => format!("stderr: {line}"),
                    Err(error) => format!("stderr: {error}"),
                };
                if stderr_tx.send(ProcessMessage::Diagnostic(message)).is_err() {
                    break;
                }
            }
        });
        thread::spawn(move || {
            while let Ok(command) = commands.recv() {
                let result = serde_json::to_writer(&mut stdin, &command)
                    .map_err(io::Error::other)
                    .and_then(|_| stdin.write_all(b"\n"))
                    .and_then(|_| stdin.flush());
                if let Err(error) = result {
                    let _ = message_tx.send(ProcessMessage::Diagnostic(format!(
                        "Runtime stdin: {error}"
                    )));
                    break;
                }
            }
        });
        Ok(Self {
            child,
            messages,
            commands: Some(command_tx),
        })
    }

    pub fn send(&self, command: RuntimeCommand) -> Result<(), String> {
        let sender = self.commands.as_ref().ok_or("Runtime input is closed")?;
        sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) => {
                "Runtime command queue is full; try again after it responds".into()
            }
            TrySendError::Disconnected(_) => "Runtime command reader disconnected".into(),
        })
    }

    pub fn exit_status(&mut self) -> io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        // Closing stdin gives Python a chance to flush its trace and terminate normally.
        self.commands.take();
        let deadline = Instant::now() + Duration::from_millis(250);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                _ if Instant::now() >= deadline => break,
                _ => thread::sleep(Duration::from_millis(10)),
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Bounded UTF-8 line reader. Oversized lines are discarded through the delimiter so
/// the next valid event can still be decoded. Memory does not grow with bad output.
fn bounded_line(reader: &mut impl BufRead) -> io::Result<Option<Result<String, String>>> {
    let mut line = Vec::new();
    let mut oversized = false;
    let mut had_bytes = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if !had_bytes {
                return Ok(None);
            }
            break;
        }
        had_bytes = true;
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(consumed) > MAX_LINE_BYTES {
            oversized = true;
            line.clear();
        }
        if !oversized {
            line.extend_from_slice(&available[..consumed]);
        }
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }
    if oversized {
        Ok(Some(Err(format!(
            "Skipped runtime line larger than {MAX_LINE_BYTES} bytes"
        ))))
    } else {
        Ok(Some(String::from_utf8(line).map_err(|_| {
            "Skipped runtime line with invalid UTF-8".into()
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::CommandKind;
    use serde_json::json;

    #[test]
    fn oversized_line_does_not_poison_following_event() {
        let mut data = vec![b'x'; MAX_LINE_BYTES + 1];
        data.extend_from_slice(b"\nvalid\n");
        let mut reader = BufReader::new(io::Cursor::new(data));
        assert!(bounded_line(&mut reader).unwrap().unwrap().is_err());
        assert_eq!(
            bounded_line(&mut reader).unwrap().unwrap().unwrap(),
            "valid\n"
        );
        assert!(bounded_line(&mut reader).unwrap().is_none());
    }

    #[test]
    fn missing_python_has_actionable_io_error() {
        let result = RuntimeProcess::spawn("/habfly-does-not-exist/python");
        assert!(matches!(result, Err(error) if error.kind() == io::ErrorKind::NotFound));
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_receives_commands_and_reports_malformed_output() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "read line; printf '%s\\n' \"$line\""]);
        let process = RuntimeProcess::spawn_command(&mut command).unwrap();
        process
            .send(RuntimeCommand::new(CommandKind::Pause, json!({})))
            .unwrap();
        match process
            .messages
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
        {
            ProcessMessage::Diagnostic(message) => {
                assert!(message.contains("Invalid runtime event"))
            }
            message => panic!("Unexpected message: {message:?}"),
        }
    }

    #[test]
    fn invalid_utf8_recovers() {
        let mut reader = BufReader::new(io::Cursor::new(vec![255, b'\n', b'a', b'\n']));
        assert!(bounded_line(&mut reader).unwrap().unwrap().is_err());
        assert_eq!(bounded_line(&mut reader).unwrap().unwrap().unwrap(), "a\n");
    }

    #[test]
    #[ignore = "requires the installed HabFly Python environment; run with --ignored"]
    fn python_runtime_bridge_start_step_pause_save_and_replay() {
        use crate::app::App;
        use crate::protocol::EventKind;
        use std::path::Path;
        use std::time::SystemTime;

        fn receive_until(process: &RuntimeProcess, app: &mut App, done: impl Fn(&App) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(30);
            while !done(app) {
                assert!(
                    Instant::now() < deadline,
                    "Runtime did not reach expected state: {:?}",
                    app.timeline
                );
                match process.messages.recv_timeout(Duration::from_secs(2)) {
                    Ok(ProcessMessage::Event(event)) => {
                        assert_ne!(
                            event.event,
                            EventKind::Error,
                            "Runtime error: {}",
                            event.payload
                        );
                        app.apply(event);
                    }
                    Ok(ProcessMessage::Diagnostic(message)) => app.log(message),
                    Ok(ProcessMessage::StdoutClosed) => {
                        panic!("Runtime closed before expected state: {:?}", app.timeline)
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(error) => panic!("Runtime disconnected: {error}"),
                }
            }
        }

        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let artifacts = repo
            .join("tui/target")
            .join(format!("runtime-test-{stamp}"));
        let mut command = Command::new(repo.join(".venv/bin/python"));
        command
            .current_dir(repo)
            .args(["-u", "-m", "habfly", "runtime", "--jsonl"]);
        let process = RuntimeProcess::spawn_command(&mut command).unwrap();
        let mut app = App::default();
        receive_until(&process, &mut app, |app| app.connected);
        process
            .send(RuntimeCommand::new(
                CommandKind::Start,
                json!({
                    "policy":"expert", "stars":1, "seed":3, "paused":true, "interval":0,
                    "artifact_dir": artifacts,
                }),
            ))
            .unwrap();
        receive_until(&process, &mut app, |app| {
            app.paused && app.observation["controls"].is_array()
        });
        process
            .send(RuntimeCommand::new(CommandKind::Step, json!({})))
            .unwrap();
        receive_until(&process, &mut app, |app| app.result["steps"] == 1);
        assert!(app.paused);
        assert_eq!(app.action["action_source"], "scripted_expert");
        assert_eq!(app.neural["activity_source"], "untrained_observer");
        assert!(app.neural["top_neurons"].as_array().unwrap().len() > 1);
        process
            .send(RuntimeCommand::new(CommandKind::Resume, json!({})))
            .unwrap();
        receive_until(&process, &mut app, |app| app.state["status"] == "running");
        process
            .send(RuntimeCommand::new(CommandKind::Pause, json!({})))
            .unwrap();
        receive_until(&process, &mut app, |app| app.paused);
        let saved = artifacts.join("saved/trace.jsonl");
        process
            .send(RuntimeCommand::new(
                CommandKind::SaveTrace,
                json!({"path":saved}),
            ))
            .unwrap();
        receive_until(&process, &mut app, |app| {
            app.state["trace_path"] == saved.to_string_lossy().as_ref()
        });
        assert!(saved.is_file());
        process
            .send(RuntimeCommand::new(
                CommandKind::Replay,
                json!({"path":saved}),
            ))
            .unwrap();
        receive_until(&process, &mut app, |app| {
            app.run_id.as_deref().unwrap_or("").starts_with("replay-")
        });
        receive_until(&process, &mut app, |app| app.state["status"] == "completed");
        assert_eq!(app.action["action_source"], "scripted_expert");
        assert!(!app.histories["sensory"].is_empty());
        let completed_sequence = app.sequence;
        let completed_summary = app.summary.clone();
        process
            .send(RuntimeCommand::new(CommandKind::Abort, json!({})))
            .unwrap();
        receive_until(&process, &mut app, |app| app.sequence > completed_sequence);
        assert_eq!(app.state["status"], "completed");
        assert_eq!(app.summary, completed_summary);
        // Aborting a genuinely active episode must still report interruption.
        process
            .send(RuntimeCommand::new(
                CommandKind::Start,
                json!({
                    "policy":"expert", "seed":4, "paused":true, "artifact_dir":artifacts,
                }),
            ))
            .unwrap();
        receive_until(&process, &mut app, |app| app.state["status"] == "paused");
        process
            .send(RuntimeCommand::new(CommandKind::Abort, json!({})))
            .unwrap();
        receive_until(&process, &mut app, |app| app.state["status"] == "aborted");
        assert_eq!(app.summary["completed"], false);
    }
}
