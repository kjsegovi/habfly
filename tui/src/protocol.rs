use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Hello,
    State,
    Observation,
    ActionProposed,
    ActionResult,
    NeuralActivity,
    EpisodeSummary,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeEvent {
    pub version: u16,
    pub event: EventKind,
    pub sequence: u64,
    pub run_id: Option<String>,
    pub payload: Value,
}

pub fn parse_event(line: &str) -> Result<RuntimeEvent, String> {
    let event: RuntimeEvent =
        serde_json::from_str(line).map_err(|error| format!("Invalid runtime event: {error}"))?;
    if event.version != PROTOCOL_VERSION {
        return Err(format!(
            "Unsupported protocol version {}; expected {PROTOCOL_VERSION}",
            event.version
        ));
    }
    if !event.payload.is_object() {
        return Err("Runtime event payload must be an object".into());
    }
    Ok(event)
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandKind {
    Start,
    Pause,
    Resume,
    Step,
    Abort,
    SaveTrace,
    Replay,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCommand {
    pub version: u16,
    pub command: CommandKind,
    pub payload: Value,
}

impl RuntimeCommand {
    pub fn new(command: CommandKind, payload: Value) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            command,
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn golden_stream_parses_all_events() {
        let events: Vec<_> = include_str!("../tests/fixtures/session.jsonl")
            .lines()
            .map(|line| parse_event(line).unwrap())
            .collect();
        assert_eq!(events.len(), 8);
        assert_eq!(events[0].event, EventKind::Hello);
        assert_eq!(events[7].event, EventKind::EpisodeSummary);
    }

    #[test]
    fn malformed_and_incompatible_events_fail_without_panicking() {
        for line in [
            "not json",
            r#"{"version":2,"event":"hello","sequence":1,"run_id":null,"payload":{}}"#,
            r#"{"version":1,"event":"mystery","sequence":1,"run_id":null,"payload":{}}"#,
            r#"{"version":1,"event":"hello","sequence":1,"run_id":null,"payload":[]}"#,
            r#"{"version":1,"event":"hello","sequence":-1,"run_id":null,"payload":{}}"#,
        ] {
            assert!(parse_event(line).is_err(), "{line}");
        }
        assert!(parse_event(
            include_str!("../tests/fixtures/session.jsonl")
                .lines()
                .next()
                .unwrap()
        )
        .is_ok());
    }

    #[test]
    fn commands_match_python_wire_format() {
        let command = RuntimeCommand::new(CommandKind::SaveTrace, json!({"path":"a.jsonl"}));
        assert_eq!(
            serde_json::to_value(command).unwrap(),
            json!({
                "version":1, "command":"save_trace", "payload":{"path":"a.jsonl"}
            })
        );
    }
}
