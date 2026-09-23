use std::collections::{BTreeMap, VecDeque};

use serde_json::{json, Value};

use crate::protocol::{EventKind, RuntimeEvent};

pub const TIMELINE_LIMIT: usize = 200;
pub const HISTORY_LIMIT: usize = 80;
pub const GROUPS: [&str; 4] = ["sensory", "intrinsic", "descending", "efferent"];

pub struct App {
    pub state: Value,
    pub observation: Value,
    pub action: Value,
    pub result: Value,
    pub neural: Value,
    pub summary: Value,
    pub timeline: VecDeque<String>,
    pub histories: BTreeMap<String, VecDeque<u64>>,
    pub run_id: Option<String>,
    pub sequence: Option<u64>,
    pub connected: bool,
    pub paused: bool,
    pub controls_scroll: u16,
    pub focused_panel: u8,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: json!({"status":"waiting for runtime"}),
            observation: json!({}),
            action: json!({}),
            result: json!({}),
            neural: json!({}),
            summary: json!({}),
            timeline: VecDeque::new(),
            histories: GROUPS
                .iter()
                .map(|group| (group.to_string(), VecDeque::new()))
                .collect(),
            run_id: None,
            sequence: None,
            connected: false,
            paused: false,
            controls_scroll: 0,
            focused_panel: 0,
        }
    }
}

impl App {
    pub fn log(&mut self, message: impl Into<String>) {
        // Terminal escape and control characters in runtime text are never interpreted.
        let text: String = message
            .into()
            .chars()
            .filter(|c| !c.is_control())
            .take(1000)
            .collect();
        self.timeline.push_back(text);
        if self.timeline.len() > TIMELINE_LIMIT {
            self.timeline.pop_front();
        }
    }

    pub fn apply(&mut self, event: RuntimeEvent) {
        let payload = event.payload;
        if event.run_id != self.run_id && event.run_id.is_some() {
            self.observation = json!({});
            self.action = json!({});
            self.result = json!({});
            self.neural = json!({});
            self.summary = json!({});
            self.histories.values_mut().for_each(VecDeque::clear);
            self.controls_scroll = 0;
        }
        self.run_id = event.run_id;
        self.sequence = Some(event.sequence);
        match event.event {
            EventKind::Hello => {
                self.connected = true;
                self.log(format!("Runtime connected, protocol {}", event.version));
                if let Some(object) = payload.as_object() {
                    self.state.as_object_mut().unwrap().extend(object.clone());
                }
            }
            EventKind::State => {
                if let Some(object) = payload.as_object() {
                    self.state.as_object_mut().unwrap().extend(object.clone());
                }
                self.paused = self.state["paused"]
                    .as_bool()
                    .unwrap_or_else(|| self.state["status"] == "paused");
                self.log(format!("State: {}", display(&self.state["status"])));
            }
            EventKind::Observation => {
                self.observation = payload.get("observation").unwrap_or(&payload).clone();
            }
            EventKind::ActionProposed => {
                self.action = payload.get("action").unwrap_or(&payload).clone();
                if let Some(source) = payload.get("action_source") {
                    self.action["action_source"] = source.clone();
                }
                self.log(format!(
                    "#{} {} → {} {}",
                    event.sequence,
                    display(&self.action["kind"]),
                    display(action_target(&self.action)),
                    display(&self.action["value"])
                ));
            }
            EventKind::ActionResult => {
                self.result = payload.get("result").unwrap_or(&payload).clone();
                if let Some(observation) = self.result.get("observation") {
                    self.observation = observation.clone();
                }
                if !self.result["failure_reason"].is_null() {
                    self.log(format!(
                        "Stop/failure: {}",
                        display(&self.result["failure_reason"])
                    ));
                }
                let feedback = display(&self.observation["feedback"]);
                self.log(format!(
                    "Reward {} · {}",
                    display(&self.result["reward"]),
                    feedback
                ));
            }
            EventKind::NeuralActivity => {
                self.neural = payload;
                for group in GROUPS {
                    if let Some(mean) = population_mean(&self.neural, group).as_f64() {
                        let history = self.histories.get_mut(group).unwrap();
                        history.push_back((mean.abs().min(1000.0) * 1000.0) as u64);
                        if history.len() > HISTORY_LIMIT {
                            history.pop_front();
                        }
                    }
                }
            }
            EventKind::EpisodeSummary => {
                self.summary = payload;
                self.log(format!("Episode summary: {}", self.summary));
            }
            EventKind::Error => {
                self.log(format!("ERROR: {}", display(&payload)));
            }
        }
    }
}

pub fn action_target(action: &Value) -> &Value {
    action
        .get("target")
        .or_else(|| action.get("target_id"))
        .unwrap_or(&Value::Null)
}

pub fn population_mean<'a>(neural: &'a Value, group: &str) -> &'a Value {
    neural
        .get("groups")
        .and_then(|groups| groups.get(group))
        .and_then(|stats| stats.get("mean"))
        .or_else(|| {
            neural
                .get("populations")
                .and_then(|groups| groups.get(group))
        })
        .unwrap_or(&Value::Null)
}

pub fn display(value: &Value) -> String {
    let text = match value {
        Value::Null => "—".into(),
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    };
    text.chars()
        .filter(|c| !c.is_control() || *c == '\n')
        .take(4000)
        .collect()
}

pub fn confidence(action: &Value, field: &str) -> String {
    let Some(probability) = action[field].as_f64() else {
        return "unknown".into();
    };
    if !(0.0..=1.0).contains(&probability) {
        return "unknown".into();
    }
    let suffix = if action["calibrated"].as_bool() == Some(true) {
        let scope = action
            .get("calibration_scope")
            .or_else(|| {
                action
                    .get("calibration")
                    .and_then(|calibration| calibration.get("scope"))
            })
            .and_then(Value::as_str)
            .unwrap_or("scope unknown");
        format!("calibrated: {scope}")
    } else {
        "uncalibrated policy p".to_string()
    };
    format!("{:.1}% ({suffix})", probability * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::parse_event;

    #[test]
    fn golden_stream_retains_actions_rewards_and_telemetry() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        assert!(app.connected);
        assert_eq!(app.observation["instruction"], "Analyze star K-12");
        assert_eq!(app.action["target_id"], "o1:c1");
        assert_eq!(app.result["cumulative_reward"], 1.5);
        assert_eq!(app.histories["sensory"].len(), 1);
        assert_eq!(app.summary["terminated"], true);
        assert!(app
            .timeline
            .iter()
            .any(|line| line.contains("fixture warning")));
    }

    #[test]
    fn bounded_timeline_and_truthful_confidence() {
        let mut app = App::default();
        for _ in 0..500 {
            app.log("hello\x1b[31m");
        }
        assert_eq!(app.timeline.len(), TIMELINE_LIMIT);
        assert!(!app.timeline[0].contains('\x1b'));
        assert_eq!(confidence(&json!({}), "action_confidence"), "unknown");
        assert_eq!(
            confidence(&json!({"action_confidence": 8.0}), "action_confidence"),
            "unknown"
        );
        assert!(
            confidence(&json!({"action_confidence":0.8}), "action_confidence")
                .contains("uncalibrated policy p")
        );
    }

    #[test]
    fn accepts_python_contract_targets_and_population_means() {
        let mut app = App::default();
        app.apply(parse_event(r#"{"version":1,"event":"action_proposed","sequence":1,"run_id":"test","payload":{"action":{"kind":"CLICK","target":"o1:c1"},"action_source":"scripted_expert"}}"#).unwrap());
        assert_eq!(action_target(&app.action), "o1:c1");
        assert_eq!(app.action["action_source"], "scripted_expert");
        app.apply(parse_event(r#"{"version":1,"event":"neural_activity","sequence":2,"run_id":"test","payload":{"populations":{"sensory":0.123}}}"#).unwrap());
        assert_eq!(app.histories["sensory"][0], 123);
    }

    #[test]
    fn calibrated_probability_names_its_evaluation_scope() {
        let value = confidence(
            &json!({"action_confidence":0.8,"calibrated":true,"calibration_scope":"grounding-held-out"}),
            "action_confidence",
        );
        assert_eq!(value, "80.0% (calibrated: grounding-held-out)");
        assert!(confidence(
            &json!({"action_confidence":0.8,"calibrated":true}),
            "action_confidence"
        )
        .contains("scope unknown"));
    }
}
