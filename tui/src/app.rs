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
    pub action_control: Value,
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
    pub observation_scroll: u16,
    pub focused_panel: u8,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: json!({"status":"waiting for runtime"}),
            observation: json!({}),
            action: json!({}),
            action_control: json!({}),
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
            observation_scroll: 0,
            focused_panel: 0,
        }
    }
}

impl App {
    pub fn project_progress(&self) -> Option<&Value> {
        self.state
            .get("project_progress")
            .filter(|value| value.is_object() && validate_project_progress(value).is_ok())
    }

    pub fn reference_measurements(&self) -> Option<&Value> {
        self.state
            .get("reference_measurements")
            .filter(|value| value.is_object() && validate_reference_measurements(value).is_ok())
    }

    fn clear_reference_measurements(&mut self) {
        self.state["reference_measurements"] = Value::Null;
        self.state["reference_measurements_error"] = Value::Null;
    }

    fn clear_save_outcome(&mut self) {
        if let Some(state) = self.state.as_object_mut() {
            state.remove("save_outcome");
            state.remove("save_outcome_uncertain");
        }
    }

    fn clear_reference_on_observation_handoff(&mut self) {
        let star = self.observation["values"]["star_name"]
            .as_str()
            .or_else(|| self.observation["chart"]["star"].as_str());
        if self.reference_measurements().is_some_and(|reference| {
            star.is_some_and(|star| {
                !reference["star"]
                    .as_str()
                    .unwrap()
                    .eq_ignore_ascii_case(star)
            })
        }) {
            self.clear_reference_measurements();
        }
    }

    fn clear_policy_diagnostics(&mut self) {
        self.action = json!({});
        self.action_control = json!({});
        self.neural = json!({});
        self.state["pending_browser_copy"] = Value::Null;
        self.state["pending_browser_color"] = Value::Null;
        self.state["action_source"] = Value::Null;
        self.clear_reference_measurements();
        self.histories.values_mut().for_each(VecDeque::clear);
    }

    pub fn is_browser_project(&self) -> bool {
        self.state["runtime_task"] == "browser_project" || self.state["task"] == "browser_project"
    }

    pub fn is_project_finalization(&self) -> bool {
        self.is_browser_project()
            && matches!(
                self.state["mode"].as_str(),
                Some("post_campaign_finalization" | "bounded_post_campaign_finalization")
            )
    }

    pub fn project_visible_star(&self) -> Option<&str> {
        self.state["current_star"]["star"]
            .as_str()
            .or_else(|| self.state["project_owner"]["star"].as_str())
            .or_else(|| self.state["initial_star"]["star"].as_str())
    }

    pub fn project_phase(&self) -> Option<&str> {
        self.state
            .get("browser_phase")
            .unwrap_or(&self.state["phase"])
            .as_str()
    }

    pub fn project_finished(&self) -> bool {
        self.state["finished"] == true
            || matches!(
                self.state["status"].as_str(),
                Some("handoff" | "stopped" | "aborted" | "completed")
            )
    }

    pub fn project_automatic_decision(&self) -> bool {
        self.is_browser_project()
            && self.state["autonomous_decisions_enabled"] == true
            && matches!(self.state["status"].as_str(), Some("paused" | "running"))
            && matches!(
                self.project_phase(),
                Some(
                    "awaiting_class_source"
                        | "awaiting_planet_class"
                        | "awaiting_gases"
                        | "awaiting_habitability"
                )
            )
    }

    pub fn project_handoff_guidance(&self) -> Option<String> {
        if self.project_automatic_decision() {
            return None;
        }
        match self.project_phase() {
            Some("awaiting_class_source") => Some(
                "External class decision required: supply reference_class with an explicit class, rationale and applicable lifetime prefix, or a validated class_source receipt for this captured star. No class is inferred or approved by TUI keys.".into(),
            ),
            Some("awaiting_inventory") => Some(
                "External inventory evidence required: supply a verified inventory_dir and inventory_sha256 through the runtime. Visible workflow completion is pending journal import.".into(),
            ),
            Some("awaiting_planet_class") => Some(
                "Explicit planet_class decision required: supply a name and reference rationale through the runtime. No planet class is inferred or approved by TUI keys.".into(),
            ),
            Some("awaiting_gases") => Some(
                "Explicit gases reference decision required: supply gases, rationale and supplied_greenhouse_increment through the runtime. TUI keys do not identify gases.".into(),
            ),
            Some("awaiting_habitability") => Some(
                "Explicit habitability reference decision required: supply choice and rationale through the runtime. A chamber phase alone does not decide habitability.".into(),
            ),
            Some("awaiting_assessment") => Some(
                "The bounded campaign's workflow target is verified. Assessment, score transfer and submission are separate gates; this is not project completion.".into(),
            ),
            Some("pagination_required") => Some(
                "Collection spans multiple visible pages. A verified live paginated inventory is required; a partial page cannot authorize continuation or scoring.".into(),
            ),
            Some("unknown_pending") => Some(
                "Submission outcome is unknown/pending. A returned click is not a grounded submission receipt; do not retry. Review the current canonical pending action and saved evidence.".into(),
            ),
            Some("assessed_score_transfer_disabled") => Some(
                "Assessment receipts are recorded; score transfer is disabled. Submission and project completion are not implied.".into(),
            ),
            Some("score_transferred_not_submitted") => Some(
                "Score transfer is verified, but the project is not submitted. This finalizer has stopped at its authorized handoff; no additional approval or retry is available through TUI keys.".into(),
            ),
            _ => None,
        }
    }

    fn project_control_guidance(&self) -> Option<String> {
        if self.project_finished() {
            return Some(format!(
                "Project run {} at {}. No further steps, approvals or automatic retries; review the evidence and any handoff. Task completed: {}; project completed: {}.{}",
                display(&self.state["status"]), display(&self.state["browser_phase"]),
                display(&self.state["task_completed"]), display(&self.state["project_completed"]),
                self.project_handoff_guidance().map_or_else(String::new, |text| format!(" {text}"))
            ));
        }
        if let Some(guidance) = self.project_handoff_guidance() {
            return Some(guidance);
        }
        if self.project_automatic_decision() {
            return None;
        }
        if self.is_project_finalization() {
            return (!(matches!(
                self.project_phase(),
                Some("finalization_pending" | "scoring_initializing" | "scoring_active"
                    | "scoring_verifying" | "submission_initializing" | "submission_active"
                    | "submission_verifying")
            ) && matches!(self.state["status"].as_str(), Some("paused" | "running"))
                && self.state["scoring_max_seconds"].as_f64() == Some(600.0)
                && self.state["submission_max_seconds"].as_f64() == Some(180.0)))
                .then(|| "Finalization lifecycle or fixed budgets unavailable; no step or resume sent. Review evidence or abort. No new authorization is inferred.".into());
        }
        if !matches!(
            self.project_phase(),
            Some(
                "launch_pending"
                    | "opening_preview"
                    | "setting_up"
                    | "capturing_initial_star"
                    | "initializing_project"
                    | "setting_reference_class"
                    | "setting_lifetime_prefix"
                    | "class_setup_handoff"
                    | "ready"
                    | "active"
                    | "inventory_initializing"
                    | "inventory_active"
                    | "inventory_import"
                    | "selecting_planet_class"
                    | "positive_initializing"
                    | "positive_active"
                    | "terrestrial_initializing"
                    | "terrestrial_active"
                    | "initializing_star"
                    | "verifying_star"
                    | "next_star_initializing"
                    | "next_star_active"
                    | "adopting_fresh_star"
            )
        ) || !matches!(self.state["status"].as_str(), Some("paused" | "running"))
        {
            return Some(
                "Project lifecycle is unavailable or unrecognized; wait for a valid state or abort. No step or approval sent.".into(),
            );
        }
        None
    }

    pub fn browser_resume_guidance(&self) -> Option<String> {
        if self.is_browser_project() && self.state["replay"] != true {
            return self.project_control_guidance();
        }
        None
    }

    pub fn browser_step_guidance(&self) -> Option<String> {
        if self.state["replay"] == true {
            return None;
        }
        if self.is_browser_project() {
            return self.project_control_guidance().or_else(|| {
                (!self.paused).then(|| {
                    "Project stages run automatically; pause before single-stepping. Pause and abort take effect between bounded browser calls, not during a synchronous call.".into()
                })
            });
        }
        if !self.state["browser_phase"].is_string() {
            return None;
        }
        if self.state["browser_execution"] == "autonomous"
            && !self.paused
            && self.state["browser_phase"] != "finished"
        {
            return Some(
                "Autonomous decisions run automatically; pause before manual stepping.".into(),
            );
        }
        if self.state["browser_phase"] != "ready"
            && self.state["browser_phase"] != "awaiting_handoff"
        {
            return Some(
                self.state["browser_guidance"]
                    .as_str()
                    .unwrap_or("No decision is available; review the browser status.")
                    .to_string(),
            );
        }
        None
    }

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
        let mut payload = event.payload;
        if event.run_id != self.run_id && event.run_id.is_some() {
            // Pending browser approvals and replay flags belong to one run only.
            self.state = json!({});
            self.paused = false;
            self.observation = json!({});
            self.action = json!({});
            self.action_control = json!({});
            self.result = json!({});
            self.neural = json!({});
            self.summary = json!({});
            self.histories.values_mut().for_each(VecDeque::clear);
            self.controls_scroll = 0;
            self.observation_scroll = 0;
        }
        self.run_id = event.run_id;
        self.sequence = Some(event.sequence);
        match event.event {
            EventKind::Hello => {
                self.connected = true;
                // A reconnect without a run_id must not retain old reference numbers.
                self.clear_reference_measurements();
                self.clear_save_outcome();
                self.state
                    .as_object_mut()
                    .unwrap()
                    .remove("score_checkpoint_completed");
                self.state.as_object_mut().unwrap().remove("reported_score");
                self.state.as_object_mut().unwrap().remove("save_strategy");
                self.state
                    .as_object_mut()
                    .unwrap()
                    .remove("persistence_verified");
                self.log(format!("Runtime connected, protocol {}", event.version));
                if let Some(object) = payload.as_object() {
                    self.state.as_object_mut().unwrap().extend(object.clone());
                }
            }
            EventKind::State => {
                if (payload["task"] == "browser_project"
                    || payload["runtime_task"] == "browser_project"
                    || payload["mode"] == "bounded_post_campaign_finalization")
                    && (payload.get("score_checkpoint_completed").is_none()
                        || payload.get("reported_score").is_none())
                {
                    self.state
                        .as_object_mut()
                        .unwrap()
                        .remove("score_checkpoint_completed");
                    self.state.as_object_mut().unwrap().remove("reported_score");
                }
                if (payload["task"] == "browser_project"
                    || payload["runtime_task"] == "browser_project")
                    && payload.get("save_strategy").is_none()
                {
                    self.state.as_object_mut().unwrap().remove("save_strategy");
                    self.state
                        .as_object_mut()
                        .unwrap()
                        .remove("persistence_verified");
                }
                // These optional failure facts belong to one star and one
                // authoritative snapshot, not every later legacy/replay state.
                if (payload["task"] == "browser_project"
                    || payload["runtime_task"] == "browser_project")
                    && (payload.get("save_outcome").is_none()
                        || payload.get("save_outcome_uncertain").is_none())
                {
                    self.clear_save_outcome();
                }
                // The journal can correctly report zero collected stars before
                // inventory import. Its null active_star is not a star change.
                let project_context = self.is_browser_project()
                    || payload["task"] == "browser_project"
                    || payload["runtime_task"] == "browser_project";
                // Nested lifecycle envelopes sometimes carry a transient null
                // stage. Retain the last known stage until a meaningful change,
                // so numeric -> null -> color still clears numeric diagnostics.
                if project_context
                    && payload["policy_stage"].is_null()
                    && self.state["policy_stage"].is_string()
                {
                    payload.as_object_mut().unwrap().remove("policy_stage");
                }
                let visible_star = project_context
                    .then(|| {
                        payload
                            .get("current_star")
                            .unwrap_or(&self.state["current_star"])["star"]
                            .as_str()
                            .or_else(|| {
                                payload
                                    .get("project_owner")
                                    .unwrap_or(&self.state["project_owner"])["star"]
                                    .as_str()
                            })
                            .or_else(|| {
                                payload
                                    .get("initial_star")
                                    .unwrap_or(&self.state["initial_star"])["star"]
                                    .as_str()
                            })
                            .map(str::to_owned)
                    })
                    .flatten();
                let visible_star_changed = project_context
                    && self.project_visible_star().is_some()
                    && self.project_visible_star() != visible_star.as_deref();
                if let Some(progress) = payload.get("project_progress") {
                    if let Err(reason) = validate_project_progress(progress) {
                        // A malformed update must not make a stale project snapshot
                        // look current. The next valid state event can recover.
                        payload["project_progress"] = Value::Null;
                        payload["project_progress_error"] = reason.into();
                        self.log(format!("Project progress unavailable: {reason}"));
                    } else {
                        payload["project_progress_error"] = Value::Null;
                    }
                }
                if let Some(reference) = payload.get("reference_measurements") {
                    let journal_star = if payload.get("project_progress").is_some() {
                        payload["project_progress"]["active_star"]["name"].as_str()
                    } else {
                        payload["star"].as_str().or_else(|| {
                            self.state["project_progress"]["active_star"]["name"]
                                .as_str()
                                .or_else(|| self.state["star"].as_str())
                        })
                    };
                    let current_star = visible_star.as_deref().or(journal_star);
                    let validation = validate_reference_measurements(reference).and_then(|()| {
                        if current_star.zip(reference["star"].as_str()).is_some_and(
                            |(current, measured)| !current.eq_ignore_ascii_case(measured),
                        ) {
                            Err("reference measurements belong to a different star")
                        } else {
                            Ok(())
                        }
                    });
                    if let Err(reason) = validation {
                        payload["reference_measurements"] = Value::Null;
                        payload["reference_measurements_error"] = reason.into();
                        self.log(format!("Reference measurements unavailable: {reason}"));
                    } else {
                        payload["reference_measurements_error"] = Value::Null;
                    }
                }
                let old_star = &self.state["project_progress"]["active_star"];
                let new_star = &payload["project_progress"]["active_star"];
                let star_changed = visible_star_changed
                    || (payload.get("project_progress").is_some()
                        && old_star["id"].is_string()
                        && old_star["id"] != new_star["id"]
                        && visible_star.is_none());
                let project_stage_changed = old_star["stage"].is_string()
                    && new_star["stage"].is_string()
                    && old_star["stage"] != new_star["stage"];
                let reference_handoff = [
                    "component",
                    "task",
                    "runtime_task",
                    "browser_task",
                    "star",
                    "policy_stage",
                ]
                .iter()
                .any(|key| payload.get(key).is_some() && payload[key] != self.state[key])
                    || (payload.get("project_progress").is_some()
                        && self.reference_measurements().is_some_and(|reference| {
                            visible_star
                                .as_deref()
                                .or_else(|| new_star["name"].as_str())
                                .is_none_or(|name| {
                                    !reference["star"]
                                        .as_str()
                                        .unwrap()
                                        .eq_ignore_ascii_case(name)
                                })
                        }));
                // A scripted policy handoff resets recurrent state. Do not keep
                // displaying the previous model's action or neural activity.
                let policy_stage_changed = payload["policy_stage"].is_string()
                    && self.state["policy_stage"].is_string()
                    && payload["policy_stage"] != self.state["policy_stage"];
                if star_changed || project_stage_changed || policy_stage_changed {
                    self.clear_policy_diagnostics();
                }
                if project_context && policy_stage_changed {
                    self.observation = json!({});
                    self.result = json!({});
                    self.controls_scroll = 0;
                    self.observation_scroll = 0;
                }
                if star_changed {
                    self.clear_save_outcome();
                    self.observation = json!({});
                    self.result = json!({});
                    self.summary = json!({});
                    // Reference choices and child summaries belong to the old
                    // visible star. The incoming state may supply fresh ones.
                    for key in [
                        "class_setup",
                        "class_handoff",
                        "component_summary",
                        "autonomous_decision",
                    ] {
                        self.state[key] = Value::Null;
                    }
                    self.observation_scroll = 0;
                    self.controls_scroll = 0;
                }
                if reference_handoff {
                    self.clear_reference_measurements();
                }
                if let Some(object) = payload.as_object() {
                    self.state.as_object_mut().unwrap().extend(object.clone());
                }
                // Project lifecycle states are authoritative. A stale legacy
                // `paused` field must not make a running project look paused.
                self.paused = if self.is_browser_project() {
                    self.state["status"] == "paused"
                } else {
                    self.state["paused"]
                        .as_bool()
                        .unwrap_or_else(|| self.state["status"] == "paused")
                };
                self.log(format!("State: {}", display(&self.state["status"])));
            }
            EventKind::Observation => {
                self.observation = payload.get("observation").unwrap_or(&payload).clone();
                self.clear_reference_on_observation_handoff();
            }
            EventKind::ActionProposed => {
                self.action = payload.get("action").unwrap_or(&payload).clone();
                self.action_control = self.observation["controls"]
                    .as_array()
                    .and_then(|controls| {
                        controls
                            .iter()
                            .find(|c| c["id"] == *action_target(&self.action))
                    })
                    .cloned()
                    .unwrap_or_else(|| json!({}));
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
                    self.clear_reference_on_observation_handoff();
                }
                if self.result["chart_sample"].is_object() {
                    if !self.observation["chart"].is_object() {
                        self.observation["chart"] = json!({});
                    }
                    self.observation["chart"]["latest_sample"] =
                        self.result["chart_sample"].clone();
                }
                if !self.result["failure_reason"].is_null() {
                    self.log(format!(
                        "Stop/failure: {}",
                        display(&self.result["failure_reason"])
                    ));
                    if let Some(explanation) = self.result["failure_reason"]
                        .as_str()
                        .and_then(failure_explanation)
                    {
                        self.log(explanation);
                    }
                }
                if self.result["chart_readout_unavailable"] == true {
                    self.log("Chart probe: tooltip not fully exposed (no sample; no answer write)");
                } else if self.result["chart_sample"].is_object() {
                    self.log(format!(
                        "Visible chart sample: day {} = {}% (no answer write)",
                        display(&self.result["chart_sample"]["day"]),
                        display(&self.result["chart_sample"]["brightness_percent"])
                    ));
                } else if self.result["replayed_native_receipt"] == true
                    && self.result["browser_action_executed"] == false
                {
                    self.log(format!(
                        "Recovered readback: {} = {} {} (no browser write)",
                        display(&self.result["destination"]),
                        display(&self.result["display"]["display_value"]),
                        display(&self.result["unit"])
                    ));
                } else if self.result["readback_verified"] == true
                    && self.result["destination"].is_string()
                {
                    self.log(format!(
                        "Native copy readback: {} = {} {} (course correctness unverified)",
                        display(&self.result["destination"]),
                        display(&self.result["display"]["display_value"]),
                        display(&self.result["unit"])
                    ));
                } else {
                    let feedback = display(&self.observation["feedback"]);
                    self.log(format!(
                        "Reward {} · {}",
                        display(&self.result["reward"]),
                        feedback
                    ));
                }
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
                if self.state["task"] == "browser_transit_sampling" {
                    self.state["chart_stop"] = payload.clone();
                }
                self.log(format!("ERROR: {}", display(&payload)));
                if let Some(explanation) = ["message", "reason", "failure_reason"]
                    .iter()
                    .find_map(|key| payload[*key].as_str().and_then(failure_explanation))
                {
                    self.log(explanation);
                }
            }
        }
    }
}

fn validate_reference_measurements(reference: &Value) -> Result<(), &'static str> {
    if reference.is_null() {
        return Ok(());
    }
    if !reference.is_object()
        || !matches!(
            reference["mode"].as_str(),
            Some(
                "approximate_reference_raster"
                    | "approximate_reference_visible_tooltips_v1"
                    | "approximate_reference_two_visible_tooltips_v1"
            )
        )
        || reference["observation_limit_days"].as_u64() != Some(5000)
        || !reference["star"].as_str().is_some_and(|star| {
            !star.trim().is_empty() && star.len() <= 80 && !star.chars().any(char::is_control)
        })
        || [
            "learned_perception",
            "scientific_verified",
            "training_label",
        ]
        .iter()
        .any(|key| reference[key].as_bool() != Some(false))
    {
        return Err("unsupported or malformed approximate-reference schema");
    }
    let number = |value: &Value| value.as_f64().filter(|n| n.is_finite() && *n >= 0.0);
    let two_events = reference["mode"] == "approximate_reference_two_visible_tooltips_v1";
    if two_events
        && (reference["confirmed_feature_count"].as_u64() != Some(2)
            || reference["observed_interval_count"].as_u64() != Some(1)
            || reference["consistency_redundancy"].as_u64() != Some(0)
            || reference["consecutive_events_assumed"].as_bool() != Some(true)
            || reference["recurrence_confirmed"].as_bool() != Some(false)
            || reference["observed_recurrence_compatible"].as_bool() != Some(false)
            || reference["single_spacing_compatible"].as_bool() != Some(true)
            || reference["period_days"]
                .get("estimate_kind")
                .is_some_and(|kind| kind != "single_spacing"))
    {
        return Err("invalid two-event reference scope or recurrence claim");
    }
    if two_events || reference["mode"] == "approximate_reference_visible_tooltips_v1" {
        let period = &reference["period_days"];
        let interval = &period["compatibility_interval"];
        let drop = &reference["brightness_drop_percent"];
        let (Some(lower), Some(value), Some(upper), Some(decline)) = (
            number(&interval["lower"]),
            number(&period["value"]),
            number(&interval["upper"]),
            number(&drop["value"]),
        ) else {
            return Err("invalid sampled-tooltip reference quantities");
        };
        if period["unit"] != "days"
            || interval["endpoints"] != "open"
            || lower <= 0.0
            || lower >= value
            || value >= upper
            || upper > 5000.0
            || drop["unit"] != "percent"
            || decline <= 0.0
            || decline > 100.0
            || drop.get("physical_bounds") != Some(&Value::Null)
            || [period, drop]
                .iter()
                .any(|item| item.get("lower").is_some() || item.get("upper").is_some())
        {
            return Err("invalid sampled-tooltip compatibility or fabricated physical bounds");
        }
    } else {
        for (key, unit, positive, maximum) in [
            ("period_days", "days", true, f64::MAX),
            ("brightness_drop_percent", "percent", false, 100.0),
        ] {
            let item = &reference[key];
            let (Some(lower), Some(value), Some(upper)) = (
                number(&item["lower"]),
                number(&item["value"]),
                number(&item["upper"]),
            ) else {
                return Err("invalid approximate-reference quantity or pixel bounds");
            };
            if !item.is_object()
                || item["unit"] != unit
                || lower > value
                || value > upper
                || upper > maximum
                || (positive && lower <= 0.0)
            {
                return Err("invalid approximate-reference quantity or pixel bounds");
            }
        }
    }
    if !reference["line_shift"].is_object()
        || reference["line_shift"]["unit"] != "nm"
        || number(&reference["line_shift"]["value"]).is_none()
    {
        return Err("invalid reference line shift");
    }
    Ok(())
}

fn validate_project_progress(progress: &Value) -> Result<(), &'static str> {
    if progress.is_null() {
        return Ok(());
    }
    if !progress.is_object() || progress["schema_version"].as_u64() != Some(1) {
        return Err("unsupported or malformed project-progress schema");
    }
    let has_invalid = |object: &Value, keys: &[&str], test: fn(&Value) -> bool| {
        keys.iter()
            .any(|key| object.get(key).is_some_and(|value| !test(value)))
    };
    if has_invalid(
        progress,
        &["target", "collected", "verified", "unresolved"],
        |v| v.as_u64().is_some(),
    ) || has_invalid(progress, &["project_id", "attempt_id"], Value::is_string)
        || has_invalid(progress, &["active_stage"], |value| {
            value.is_null() || value.is_string()
        })
        || has_invalid(
            progress,
            &["score_transfer_verified", "submitted", "project_completed"],
            Value::is_boolean,
        )
    {
        return Err("invalid project-progress scalar");
    }
    for (key, flags) in [
        ("assessment", &["data_quality", "scavenger_hunt"][..]),
        ("ladder", &["one_star", "three_star", "thirty_star"][..]),
    ] {
        if let Some(value) = progress.get(key) {
            if !value.is_object() || has_invalid(value, flags, Value::is_boolean) {
                return Err("invalid project verification flags");
            }
        }
    }
    if let Some(star) = progress.get("active_star").filter(|value| !value.is_null()) {
        if !star.is_object()
            || has_invalid(star, &["id", "name", "stage"], Value::is_string)
            || has_invalid(star, &["task_completed"], Value::is_boolean)
        {
            return Err("invalid active-star progress");
        }
        if star
            .get("stellar")
            .is_some_and(|value| !value.is_null() && !value.is_object())
        {
            return Err("invalid stellar progress");
        }
        for branch in [
            star.get("planet"),
            star.get("habitability"),
            star["stellar"].get("numeric"),
            star["stellar"].get("color"),
            star["stellar"].get("classification"),
        ]
        .into_iter()
        .flatten()
        {
            if branch.is_null() {
                continue;
            }
            if !branch.is_object()
                || has_invalid(
                    branch,
                    &[
                        "outcome",
                        "provenance",
                        "value",
                        "applicability_reason",
                        "policy_label",
                    ],
                    |value| value.is_null() || value.is_string(),
                )
                || has_invalid(branch, &["observation_limit_days"], |value| {
                    value.is_null()
                        || value
                            .as_u64()
                            .is_some_and(|days| (1..=10000).contains(&days))
                })
                || has_invalid(
                    branch,
                    &[
                        "transport_verified",
                        "scientific_verified",
                        "branch_applicability_verified",
                    ],
                    Value::is_boolean,
                )
            {
                return Err("invalid star-stage evidence");
            }
        }
    }
    if let Some(actions) = progress.get("uncertain_actions") {
        let Some(actions) = actions.as_array() else {
            return Err("invalid uncertain-action list");
        };
        if actions.iter().any(|action| {
            !action.is_object()
                || has_invalid(action, &["id", "kind"], Value::is_string)
                || action
                    .get("star_id")
                    .is_some_and(|value| !value.is_null() && !value.is_string())
                || action
                    .get("revision")
                    .is_some_and(|value| !value.is_null() && value.as_u64().is_none())
        }) {
            return Err("invalid uncertain-action entry");
        }
    }
    Ok(())
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

pub fn failure_explanation(code: &str) -> Option<&'static str> {
    match code {
        "shallow_probe_exact_two_overview_hints_required"
        | "shallow_probe_three_overview_hints_required" => Some(
            "The chart does not provide the separate possible dips needed to estimate the time between them. Evidence remains unresolved; a planet is neither confirmed nor ruled out. No automatic retry.",
        ),
        _ => None,
    }
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
    fn shallow_failure_explanations_preserve_exact_codes_payloads_and_stop_state() {
        for code in [
            "shallow_probe_exact_two_overview_hints_required",
            "shallow_probe_three_overview_hints_required",
        ] {
            for kind in [EventKind::Error, EventKind::ActionResult] {
                let mut app = App {
                    state: json!({"task":"browser_project", "status":"stopped",
                        "failure_reason":code, "task_completed":false,
                        "project_completed":false, "submitted":false}),
                    ..App::default()
                };
                let before = app.state.clone();
                let payload = if kind == EventKind::Error {
                    json!({"message":code,"type":"BrowserSafetyStop"})
                } else {
                    json!({"failure_reason":code,"task_completed":false})
                };
                app.apply(RuntimeEvent {
                    version: 1,
                    event: kind,
                    sequence: 1,
                    run_id: None,
                    payload: payload.clone(),
                });
                assert_eq!(app.state, before);
                assert!(app.timeline.iter().any(|line| line.contains(code)));
                assert!(app
                    .timeline
                    .iter()
                    .any(|line| line.contains("Evidence remains unresolved")));
                if kind == EventKind::ActionResult {
                    assert_eq!(app.result, payload);
                } else {
                    assert!(app
                        .timeline
                        .iter()
                        .any(|line| line == &format!("ERROR: {}", display(&payload))));
                }
            }
        }
    }

    #[test]
    fn shallow_failure_explanations_never_match_unknown_or_embedded_codes() {
        for code in [
            "",
            "another_failure",
            "prefix:shallow_probe_three_overview_hints_required",
            "shallow_probe_exact_two_overview_hints_required_suffix",
        ] {
            assert_eq!(failure_explanation(code), None);
        }
        let mut app = App::default();
        app.apply(RuntimeEvent {
            version: 1,
            event: EventKind::Error,
            sequence: 1,
            run_id: None,
            payload: json!({"message":"another_failure"}),
        });
        assert_eq!(app.timeline.len(), 1);
        assert_eq!(app.timeline[0], "ERROR: {\"message\":\"another_failure\"}");
    }

    #[test]
    fn autosave_scope_clears_on_legacy_root_or_hello_but_not_scoped_child_state() {
        for boundary in [EventKind::Hello, EventKind::State] {
            let mut app = App::default();
            app.apply(RuntimeEvent {
                version: 1,
                event: EventKind::State,
                sequence: 0,
                run_id: None,
                payload: json!({"task":"browser_project", "save_strategy":"autosave",
                    "persistence_verified":false,"task_completed":false}),
            });
            app.apply(RuntimeEvent {
                version: 1,
                event: EventKind::State,
                sequence: 1,
                run_id: None,
                payload: json!({"component":"planet.window", "phase":"observing"}),
            });
            assert_eq!(app.state["save_strategy"], "autosave");
            app.apply(RuntimeEvent {
                version: 1,
                event: boundary,
                sequence: 2,
                run_id: None,
                payload: json!({"task":"browser_project", "task_completed":false}),
            });
            assert!(app.state.get("save_strategy").is_none());
            assert!(app.state.get("persistence_verified").is_none());
            assert_eq!(app.state["task_completed"], false);
        }
    }

    #[test]
    fn project_fixture_steps_only_paused_boundaries_and_requires_external_handoffs() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/browser-project.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
            if app.sequence == Some(0) {
                assert!(app.browser_step_guidance().is_some());
                continue;
            }
            assert!(app.is_browser_project());
            match app.sequence.unwrap() {
                1..=5 | 7 => {
                    assert!(app.paused);
                    assert!(app.browser_step_guidance().is_none());
                    assert!(app.browser_resume_guidance().is_none());
                }
                6 | 9 => {
                    assert!(app.paused);
                    assert!(app.browser_step_guidance().unwrap().contains("External"));
                    assert!(app.browser_resume_guidance().is_some());
                }
                8 => {
                    assert!(!app.paused);
                    assert!(app.browser_step_guidance().unwrap().contains("pause"));
                    assert_eq!(app.state["component_summary"]["task_completed"], true);
                    assert_eq!(app.state["task_completed"], false);
                    assert!(!app.project_finished());
                }
                10 => {
                    assert!(app.project_finished());
                    assert!(app.browser_step_guidance().is_some());
                    assert!(app.browser_resume_guidance().is_some());
                    assert_eq!(app.state["task_completed"], true);
                    assert_eq!(app.state["project_completed"], false);
                    assert_eq!(app.project_progress().unwrap()["verified"], 1);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn project_reference_setup_stages_never_infer_a_class_from_step() {
        for phase in [
            "setting_reference_class",
            "setting_lifetime_prefix",
            "class_setup_handoff",
        ] {
            let mut app = App {
                paused: true,
                state: json!({"runtime_task":"browser_project", "status":"paused",
                    "browser_phase":phase,"automatic_classification":false}),
                ..App::default()
            };
            assert!(app.browser_step_guidance().is_none());
            assert!(app.browser_resume_guidance().is_none());
            app.state["browser_phase"] = "awaiting_class_source".into();
            let guidance = app.browser_step_guidance().unwrap();
            assert!(guidance.contains("reference_class"));
            assert!(guidance.contains("No class is inferred"));
        }
    }

    #[test]
    fn automatic_project_decisions_schedule_only_explicit_known_live_stages() {
        for phase in [
            "awaiting_class_source",
            "awaiting_planet_class",
            "awaiting_gases",
            "awaiting_habitability",
        ] {
            let mut app = App {
                paused: true,
                state: json!({"runtime_task":"browser_project", "status":"paused", "browser_phase":phase,
                    "autonomous_decisions_enabled":true, "task_completed":false,"project_completed":false}),
                ..App::default()
            };
            let before = app.state.clone();
            assert!(app.project_automatic_decision());
            assert!(app.browser_step_guidance().is_none());
            assert!(app.browser_resume_guidance().is_none());
            assert_eq!(app.state, before);
            app.paused = false;
            app.state["status"] = "running".into();
            assert!(app.browser_step_guidance().unwrap().contains("pause"));
            assert!(app.browser_resume_guidance().is_none());
            for disabled in [json!(false), json!(1), json!("true"), Value::Null] {
                app.state["autonomous_decisions_enabled"] = disabled;
                assert!(!app.project_automatic_decision());
                assert!(app.browser_resume_guidance().is_some());
            }
            app.state["autonomous_decisions_enabled"] = true.into();
            app.state["status"] = "stopped".into();
            assert!(app
                .browser_resume_guidance()
                .unwrap()
                .contains("No further"));
            app.state["replay"] = true.into();
            assert!(app.browser_step_guidance().is_none());
        }
        let app = App {
            paused: true,
            state: json!({"runtime_task":"browser_project","status":"paused",
                "browser_phase":"awaiting_inventory","autonomous_decisions_enabled":true}),
            ..App::default()
        };
        assert!(!app.project_automatic_decision());
        assert!(app.browser_step_guidance().unwrap().contains("inventory"));
    }

    #[test]
    fn project_inventory_stages_are_cooperative_not_external_approval_handoffs() {
        for phase in [
            "inventory_initializing",
            "inventory_active",
            "inventory_import",
        ] {
            let mut app = App {
                paused: true,
                state: json!({"runtime_task":"browser_project", "status":"paused",
                    "browser_phase":phase,"task_completed":false,"project_completed":false}),
                ..App::default()
            };
            assert!(app.project_handoff_guidance().is_none());
            assert!(app.browser_step_guidance().is_none());
            assert!(app.browser_resume_guidance().is_none());
            app.state["status"] = "running".into();
            app.paused = false;
            assert!(app.browser_step_guidance().is_some());
            assert!(app.browser_resume_guidance().is_none());
            app.state["status"] = "stopped".into();
            assert!(app.browser_step_guidance().is_some());
            assert!(app.browser_resume_guidance().is_some());
        }
    }

    #[test]
    fn project_planet_stages_require_an_explicit_decision_before_scheduling() {
        let mut app = App {
            paused: true,
            state: json!({"runtime_task":"browser_project", "status":"paused",
                "browser_phase":"awaiting_planet_class","task_completed":false}),
            ..App::default()
        };
        assert!(app
            .browser_step_guidance()
            .unwrap()
            .contains("planet_class"));
        assert!(app
            .browser_resume_guidance()
            .unwrap()
            .contains("No planet class is inferred"));
        for phase in [
            "selecting_planet_class",
            "positive_initializing",
            "positive_active",
            "terrestrial_initializing",
            "terrestrial_active",
        ] {
            app.state["browser_phase"] = phase.into();
            assert!(app.browser_step_guidance().is_none());
            assert!(app.browser_resume_guidance().is_none());
        }
        for phase in ["awaiting_gases", "awaiting_habitability"] {
            app.state["browser_phase"] = phase.into();
            assert!(app
                .browser_step_guidance()
                .unwrap()
                .contains("reference decision"));
            assert!(app.browser_resume_guidance().is_some());
        }
    }

    #[test]
    fn project_invalid_lifecycle_blocks_then_recovers_and_status_owns_pause() {
        let mut app = App {
            state: json!({"runtime_task":"browser_project", "task":"child_numeric", "paused":true}),
            ..App::default()
        };
        for phase in [
            Value::Null,
            json!(44),
            json!("awaiting_copy"),
            json!("new_unknown_phase"),
        ] {
            app.apply(
                parse_event(
                    &json!({"version":1,"event":"state","sequence":1,
                "payload":{"status":"paused","browser_phase":phase}})
                    .to_string(),
                )
                .unwrap(),
            );
            assert!(app.browser_step_guidance().is_some());
            assert!(app.browser_resume_guidance().is_some());
        }
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":2,"payload":{"status":"running","browser_phase":"active"}}"#).unwrap());
        assert!(!app.paused);
        assert!(app.browser_step_guidance().is_some());
        app.apply(
            parse_event(
                r#"{"version":1,"event":"state","sequence":3,"payload":{"status":"paused"}}"#,
            )
            .unwrap(),
        );
        assert!(app.paused);
        assert!(app.browser_step_guidance().is_none());
    }

    #[test]
    fn project_finished_status_cannot_step_even_with_active_phase_or_child_success() {
        let mut app = App {
            paused: true,
            state: json!({"task":"browser_project",
            "browser_phase":"active", "task_completed":false,"project_completed":false}),
            ..App::default()
        };
        for status in ["handoff", "stopped", "aborted", "completed"] {
            app.state["status"] = status.into();
            assert!(app.browser_step_guidance().is_some());
            assert!(app.browser_resume_guidance().is_some());
        }
        app.state["status"] = "paused".into();
        app.state["finished"] = true.into();
        assert!(app.browser_step_guidance().is_some());
        app.state["replay"] = true.into();
        assert!(app.browser_step_guidance().is_none());
        assert!(app.browser_resume_guidance().is_none());
    }

    #[test]
    fn project_reference_context_uses_visible_star_without_inventing_inventory() {
        let mut app = App {
            state: json!({"runtime_task":"browser_project", "initial_star":{"star":"Fixture"},
                "project_owner":{"star":"Fixture"}}),
            ..App::default()
        };
        let mut event = parse_event(
            include_str!("../tests/fixtures/reference-measurements.jsonl")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap();
        event.run_id = None;
        event.payload["project_progress"] =
            json!({"schema_version":1,"active_star":null,"collected":0});
        app.apply(event.clone());
        assert!(app.reference_measurements().is_some());
        assert_eq!(app.project_progress().unwrap()["collected"], 0);
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":2,"payload":{"project_progress":{"schema_version":1,"active_star":null,"collected":0}}}"#).unwrap());
        assert!(app.reference_measurements().is_some());
        // A journal stage can become inactive without the visible star changing.
        app.state["project_progress"]["active_star"] = json!({"id":"fixture","name":"Fixture"});
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":2,"payload":{"project_progress":{"schema_version":1,"active_star":null,"collected":0}}}"#).unwrap());
        assert!(app.reference_measurements().is_some());
        event.payload["reference_measurements"]["star"] = "Wrong".into();
        app.apply(event.clone());
        assert!(app.reference_measurements().is_none());
        event.payload["reference_measurements"]["star"] = "Fixture".into();
        app.apply(event);
        assert!(app.reference_measurements().is_some());
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":3,"payload":{"project_owner":{"star":"Other"}}}"#).unwrap());
        assert!(app.reference_measurements().is_none());
        assert_eq!(app.project_visible_star(), Some("Other"));
    }

    #[test]
    fn campaign_adoption_clears_old_star_diagnostics_but_retains_initial_provenance() {
        let mut app = App {
            state: json!({"runtime_task":"browser_project", "project_campaign":true,
                "initial_star":{"star":"First"}, "current_star":{"star":"First","ordinal":1},
                "project_owner":{"star":"First"},
                "class_setup":{"selected_class":"main_sequence"},
                "class_handoff":{"star":"First"}, "component_summary":{"task_completed":true},
                "autonomous_decision":{"star":"First","decision":{"selected_class":"main_sequence"}}}),
            observation: json!({"instruction":"First", "controls":[{"id":"old-control"}]}),
            action: json!({"target_id":"old-control"}),
            action_control: json!({"label":"old"}),
            neural: json!({"top_neurons":[1]}),
            result: json!({"task_completed":true}),
            summary: json!({"task_completed":true}),
            ..App::default()
        };
        // A transition keeps the old context until the runtime verifies adoption.
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":1,"payload":{"status":"paused","browser_phase":"next_star_active"}}"#).unwrap());
        assert_eq!(app.project_visible_star(), Some("First"));
        assert!(!app.action.is_null() && app.action != json!({}));
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":2,"payload":{"status":"paused","browser_phase":"initializing_star","current_star":{"star":"Second","ordinal":2},"project_owner":null}}"#).unwrap());
        assert_eq!(app.project_visible_star(), Some("Second"));
        assert_eq!(app.state["initial_star"]["star"], "First");
        for value in [
            &app.observation,
            &app.action,
            &app.action_control,
            &app.neural,
            &app.result,
            &app.summary,
        ] {
            assert_eq!(value, &json!({}));
        }
        for key in [
            "class_setup",
            "class_handoff",
            "component_summary",
            "autonomous_decision",
        ] {
            assert!(app.state[key].is_null());
        }
        assert!(app.browser_step_guidance().is_none());
    }

    #[test]
    fn campaign_transition_stages_are_bounded_and_target_handoff_is_not_completion() {
        let mut app = App {
            paused: true,
            state: json!({"runtime_task":"browser_project", "project_campaign":true,
                "status":"paused", "task_completed":false,"project_completed":false}),
            ..App::default()
        };
        for phase in [
            "initializing_star",
            "verifying_star",
            "next_star_initializing",
            "next_star_active",
            "adopting_fresh_star",
        ] {
            app.state["browser_phase"] = phase.into();
            assert!(app.browser_step_guidance().is_none());
            assert!(app.browser_resume_guidance().is_none());
        }
        app.state["browser_phase"] = "awaiting_assessment".into();
        app.state["target_workflows_verified"] = true.into();
        app.state["status"] = "handoff".into();
        app.state["finished"] = true.into();
        assert!(app
            .browser_step_guidance()
            .unwrap()
            .contains("not project completion"));
        assert_eq!(app.state["task_completed"], false);
        assert_eq!(app.state["project_completed"], false);
        app.state["browser_phase"] = "pagination_required".into();
        assert!(app
            .browser_resume_guidance()
            .unwrap()
            .contains("partial page"));
    }

    #[test]
    fn project_fixture_replay_does_not_block_recorded_steps_or_legacy_recovery() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/browser-project.jsonl").lines() {
            let mut event = parse_event(line).unwrap();
            event.payload["replay"] = true.into();
            app.apply(event);
            assert!(app.browser_step_guidance().is_none());
            assert!(app.browser_resume_guidance().is_none());
        }
        assert!(parse_event("malformed fixture line").is_err());
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        assert!(!app.is_browser_project());
        assert_eq!(app.observation["instruction"], "Analyze star K-12");
        assert_eq!(app.result["cumulative_reward"], 1.5);
    }

    #[test]
    fn blocked_browser_steps_show_guidance_without_sending_a_command() {
        let mut app = App {
            paused: true,
            state: json!({"browser_phase":"awaiting_color",
            "browser_guidance":"y approves one color; a aborts"}),
            ..App::default()
        };
        assert_eq!(
            app.browser_step_guidance().unwrap(),
            "y approves one color; a aborts"
        );
        for phase in ["setting_up", "awaiting_ready", "awaiting_copy", "finished"] {
            app.state["browser_phase"] = phase.into();
            assert!(app.browser_step_guidance().is_some());
        }
        app.state["browser_phase"] = "ready".into();
        assert!(app.browser_step_guidance().is_none());
        app.state["browser_phase"] = "awaiting_handoff".into();
        assert!(app.browser_step_guidance().is_none());
        app.state["browser_execution"] = "autonomous".into();
        app.paused = false;
        assert!(app.browser_step_guidance().is_some());
        app.state["replay"] = true.into();
        assert!(app.browser_step_guidance().is_none());
    }

    #[test]
    fn score_checkpoint_does_not_leak_into_legacy_hello_or_state() {
        for boundary in [EventKind::Hello, EventKind::State] {
            let mut app = App {
                state: json!({"task":"browser_project", "score_checkpoint_completed":true,
                    "reported_score":0}),
                ..App::default()
            };
            app.apply(RuntimeEvent {
                version: 1,
                event: boundary,
                sequence: 1,
                run_id: None,
                payload: json!({"task":"browser_project", "task_completed":false}),
            });
            assert!(app.state.get("score_checkpoint_completed").is_none());
            assert!(app.state.get("reported_score").is_none());
        }
    }

    #[test]
    fn finalization_fixture_preserves_campaign_boundary_and_pending_submission() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/project-finalization.jsonl").lines() {
            let event = parse_event(line).unwrap();
            let sequence = event.sequence;
            if sequence == 1 {
                app.observation = json!({"instruction":"Old star"});
                app.neural = json!({"top_neurons":[{"body_id":1}]});
                app.summary = json!({"task_completed":true});
            }
            app.apply(event);
            assert_eq!(app.state["task_completed"], false);
            assert_eq!(app.state["project_completed"], false);
            assert_eq!(app.project_progress().unwrap()["verified"], 30);
            match sequence {
                0 => {
                    assert!(!app.is_project_finalization());
                    assert!(app
                        .browser_step_guidance()
                        .unwrap()
                        .contains("not project completion"));
                }
                1 => {
                    assert!(app.is_project_finalization());
                    assert_eq!(app.project_visible_star(), None);
                    assert_eq!(app.observation, json!({}));
                    assert_eq!(app.neural, json!({}));
                    assert_eq!(app.summary, json!({}));
                    assert!(app.browser_step_guidance().is_none());
                }
                2 | 4..=7 => {
                    assert!(app.browser_step_guidance().is_none());
                    assert!(app.browser_resume_guidance().is_none());
                }
                3 => {
                    assert!(app
                        .browser_step_guidance()
                        .unwrap()
                        .contains("pause before"));
                    assert!(app.browser_resume_guidance().is_none());
                }
                8 => {
                    assert!(app
                        .browser_step_guidance()
                        .unwrap()
                        .contains("do not retry"));
                    assert!(app
                        .browser_resume_guidance()
                        .unwrap()
                        .contains("unknown/pending"));
                    assert_eq!(
                        app.project_progress().unwrap()["uncertain_actions"]
                            .as_array()
                            .unwrap()
                            .len(),
                        1
                    );
                }
                9 => {
                    assert!(app.browser_step_guidance().is_none());
                    assert_eq!(app.project_phase(), Some("unknown_pending"));
                    assert_eq!(app.state["recorded_status"], "handoff");
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn finalization_unrecognized_phases_budgets_and_terminal_flags_never_enable_steps() {
        let valid = json!({"runtime_task":"browser_project","mode":"bounded_post_campaign_finalization",
            "status":"paused","browser_phase":"scoring_active","finished":false,
            "scoring_max_seconds":600,"submission_max_seconds":180});
        for (key, value) in [
            ("browser_phase", json!("automatic_retry")),
            ("browser_phase", json!("unknown_pending")),
            ("scoring_max_seconds", json!(601)),
            ("submission_max_seconds", Value::Null),
            ("submission_max_seconds", json!("180")),
            ("status", json!("stopped")),
            ("finished", json!(true)),
            ("mode", json!("guessed_finalizer")),
        ] {
            let mut app = App {
                state: valid.clone(),
                paused: true,
                ..App::default()
            };
            app.state[key] = value;
            assert!(
                app.browser_step_guidance().is_some(),
                "{key}: {}",
                app.state
            );
            assert!(app.browser_resume_guidance().is_some());
        }
        for phase in [
            "assessed_score_transfer_disabled",
            "score_transferred_not_submitted",
        ] {
            let mut app = App {
                state: valid.clone(),
                paused: true,
                ..App::default()
            };
            app.state["browser_phase"] = phase.into();
            assert!(app.browser_step_guidance().is_some());
        }
    }

    #[test]
    fn project_stage_handoffs_across_null_envelopes_clear_only_stage_diagnostics() {
        let mut app = App::default();
        let state = |sequence, stage| {
            parse_event(
                &json!({
                    "version":1,"sequence":sequence,"run_id":"campaign","event":"state",
                    "payload":{"runtime_task":"browser_project","status":"running",
                        "current_star":{"star":"Crabiltia","ordinal":1},"policy_stage":stage,
                        "task_completed":false,"project_completed":false}
                })
                .to_string(),
            )
            .unwrap()
        };
        app.apply(state(1, json!("stellar_numeric")));
        for (index, next) in ["stellar_color", "planet_window"].iter().enumerate() {
            app.observation = json!({"instruction":"Previous stage","controls":[{"id":"old"}]});
            app.action = json!({"kind":"CLICK"});
            app.neural = json!({"populations":{"sensory":0.5}});
            app.result = json!({"terminated":true,"reward":1});
            app.histories.get_mut("sensory").unwrap().push_back(500);
            let previous = app.state["policy_stage"].clone();
            app.apply(state(2 + index as u64 * 2, Value::Null));
            assert_eq!(app.state["policy_stage"], previous);
            assert_eq!(app.histories["sensory"].len(), 1);
            assert_eq!(app.observation["instruction"], "Previous stage");
            app.apply(state(3 + index as u64 * 2, json!(next)));
            assert_eq!(app.state["policy_stage"], *next);
            assert_eq!(app.observation, json!({}));
            assert_eq!(app.action, json!({}));
            assert_eq!(app.neural, json!({}));
            assert_eq!(app.result, json!({}));
            assert!(app.histories["sensory"].is_empty());
            assert_eq!(app.project_visible_star(), Some("Crabiltia"));
            assert_eq!(app.state["task_completed"], false);
        }
    }

    #[test]
    fn failed_save_state_clears_on_star_run_hello_and_authoritative_legacy_state() {
        let failure = json!({"runtime_task":"browser_project","status":"stopped",
            "browser_phase":"stopped","current_star":{"star":"First","ordinal":1},
            "task_completed":false,"project_completed":false,
            "save_outcome_uncertain":true,
            "save_outcome":{"star":"First","dispatch_recorded":true,
                "acknowledgement_recorded":true,"canonical_receipt":false}});
        for (kind, run, payload) in [
            (
                "state",
                "same",
                json!({"current_star":{"star":"Second","ordinal":2}}),
            ),
            (
                "state",
                "new",
                json!({"runtime_task":"browser_project","status":"paused"}),
            ),
            ("hello", "same", json!({"protocol_version":1})),
            (
                "state",
                "same",
                json!({"runtime_task":"browser_project","status":"stopped"}),
            ),
        ] {
            let mut app = App::default();
            let event =
                |kind, run, sequence, payload| {
                    parse_event(&json!({
                "version":1,"sequence":sequence,"run_id":run,"event":kind,"payload":payload
            }).to_string()).unwrap()
                };
            app.apply(event("state", "same", 0, failure.clone()));
            assert_eq!(app.state["save_outcome_uncertain"], true);
            app.apply(event(kind, run, 1, payload));
            assert!(app.state.get("save_outcome").is_none());
            assert!(app.state.get("save_outcome_uncertain").is_none());
            assert_ne!(app.state["task_completed"], true);
            assert_ne!(app.state["project_completed"], true);
        }
    }

    #[test]
    fn failed_save_survives_scoped_child_event_and_replay_eof_but_not_partial_root_pair() {
        let mut app = App::default();
        let event = |sequence, payload| {
            parse_event(
                &json!({"version":1,"sequence":sequence,
            "run_id":"same","event":"state","payload":payload})
                .to_string(),
            )
            .unwrap()
        };
        let diagnostic = json!({"star":"First","dispatch_recorded":true,
            "acknowledgement_recorded":true,"final_readback_verified":false});
        let state = json!({"runtime_task":"browser_project","status":"stopped",
            "current_star":{"star":"First"},"save_outcome_uncertain":true,
            "save_outcome":diagnostic,"task_completed":false,"project_completed":false});
        app.apply(event(0, state.clone()));
        app.apply(event(
            1,
            json!({"component":"planet.window","component_state":{"finished":true}}),
        ));
        assert_eq!(app.state["save_outcome"], diagnostic);
        let mut eof = state;
        eof["status"] = "replay_completed".into();
        eof["recorded_status"] = "stopped".into();
        eof["replay_finished"] = true.into();
        app.apply(event(2, eof));
        assert_eq!(app.state["save_outcome"], diagnostic);
        assert_eq!(app.state["save_outcome_uncertain"], true);
        assert_eq!(app.state["task_completed"], false);
        app.apply(event(
            3,
            json!({"runtime_task":"browser_project","save_outcome_uncertain":null}),
        ));
        assert!(app.state.get("save_outcome").is_none());
        assert!(app.state["save_outcome_uncertain"].is_null());
    }

    #[test]
    fn replay_eof_retains_current_star_and_does_not_infer_task_success() {
        let mut app = App::default();
        app.apply(parse_event(r#"{"version":1,"sequence":1,"run_id":"replay","event":"state","payload":{"runtime_task":"browser_project","status":"running","replay":true,"current_star":{"star":"Second","ordinal":2},"browser_phase":"active","policy_stage":"planet_window","task_completed":false,"project_completed":false}}"#).unwrap());
        app.apply(parse_event(r#"{"version":1,"sequence":2,"run_id":"replay","event":"state","payload":{"status":"completed","recorded_status":"stopped","replay":true,"replay_finished":true}}"#).unwrap());
        assert_eq!(app.project_visible_star(), Some("Second"));
        assert_eq!(app.project_phase(), Some("active"));
        assert_eq!(app.state["task_completed"], false);
        assert_eq!(app.state["project_completed"], false);
        assert!(app.summary["task_completed"].is_null());
    }

    #[test]
    fn policy_handoff_clears_previous_model_diagnostics_and_approval() {
        let mut app = App::default();
        app.apply(parse_event(r#"{"version":1,"sequence":0,"run_id":"four","event":"state","payload":{"status":"paused","policy_stage":"numeric","checkpoint":"numeric.pt","pending_browser_copy":{"exact_value":"12"}}}"#).unwrap());
        app.neural = json!({"populations":{"sensory":1.2}});
        app.action = json!({"kind":"CLICK"});
        app.apply(parse_event(r#"{"version":1,"sequence":1,"run_id":"four","event":"state","payload":{"status":"paused","policy_stage":"color","checkpoint":"color.pt","pending_browser_copy":null,"pending_browser_color":null}}"#).unwrap());
        assert_eq!(app.state["checkpoint"], "color.pt");
        assert_eq!(app.neural, json!({}));
        assert_eq!(app.action, json!({}));
        assert!(app.state["pending_browser_copy"].is_null());
        assert!(app.paused);
    }

    #[test]
    fn browser_approval_state_cannot_leak_into_another_run() {
        let mut app = App::default();
        app.apply(parse_event(r#"{"version":1,"sequence":0,"run_id":"browser","event":"state","payload":{"status":"paused","browser_phase":"awaiting_copy","pending_browser_copy":{"exact_value":"12"}}}"#).unwrap());
        assert!(app.paused);
        app.state["pending_browser_color"] = serde_json::json!({"selected_color":"UV"});
        app.apply(parse_event(r#"{"version":1,"sequence":1,"run_id":"local","event":"state","payload":{"status":"running"}}"#).unwrap());
        assert!(app.state["browser_phase"].is_null());
        assert!(app.state["pending_browser_copy"].is_null());
        assert!(app.state["pending_browser_color"].is_null());
        assert!(!app.paused);
    }

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

    #[test]
    fn recovered_readbacks_are_not_presented_as_repeated_browser_writes() {
        let mut app = App::default();
        app.apply(parse_event(r#"{"version":1,"event":"action_result","sequence":1,"run_id":"recovery","payload":{"destination":"planet_mass","unit":"MEarth","display":{"display_value":"57.35"},"replayed_native_receipt":true,"browser_action_executed":false,"readback_verified":true,"task_completed":false}}"#).unwrap());
        assert_eq!(
            app.timeline.back().unwrap(),
            "Recovered readback: planet_mass = 57.35 MEarth (no browser write)"
        );
        assert_eq!(app.result["task_completed"], false);
        app.apply(parse_event(r#"{"version":1,"event":"action_result","sequence":2,"run_id":"recovery","payload":{"destination":"planet_radius","unit":"REarth","display":{"display_value":"6.627"},"readback_verified":true,"task_completed":false}}"#).unwrap());
        assert!(app
            .timeline
            .back()
            .unwrap()
            .starts_with("Native copy readback: planet_radius = 6.627 REarth"));
        assert!(app
            .timeline
            .back()
            .unwrap()
            .contains("course correctness unverified"));
    }

    #[test]
    fn chart_sensor_replay_preserves_samples_stop_and_incomplete_scope() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/chart-sensor.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        assert_eq!(app.observation["chart"]["latest_sample"]["day"], 1279);
        assert_eq!(app.state["progress"]["unique_days"], 100);
        assert_eq!(app.state["chart_stop"]["reason"], "chart_time_limit");
        assert_eq!(app.result["task_completed"], false);
        assert_eq!(app.state["learned_chart_perception"], false);
        assert!(app
            .timeline
            .iter()
            .any(|line| line.contains("1279 = 99.99% (no answer write)")));
        assert!(app.histories.values().all(|history| history.is_empty()));
        app.apply(
            parse_event(
                r#"{"version":1,"event":"hello","sequence":0,"run_id":"new","payload":{}}"#,
            )
            .unwrap(),
        );
        assert!(app.observation["chart"].is_null());
        assert!(app.state["chart_stop"].is_null());
    }

    #[test]
    fn project_fixture_keeps_negative_outcomes_and_receipts_distinct() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/project-progress.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        let progress = app.project_progress().unwrap();
        assert_eq!(progress["collected"], 3);
        assert_eq!(progress["verified"], 1);
        assert_eq!(progress["unresolved"], 2);
        assert_eq!(progress["active_star"]["planet"]["outcome"], "no_planet");
        assert_eq!(
            progress["active_star"]["habitability"]["outcome"],
            "not_applicable"
        );
        assert_eq!(
            progress["active_star"]["planet"]["scientific_verified"],
            false
        );
        assert_eq!(progress["active_star"]["task_completed"], false);
        assert_eq!(
            progress["active_star"]["planet"]["policy_label"],
            "bounded_no_dip_assumption"
        );
        assert_eq!(
            progress["active_star"]["planet"]["observation_limit_days"],
            5000
        );
        assert_eq!(progress["submitted"], true);
        assert_eq!(progress["score_transfer_verified"], false);
        assert_eq!(progress["project_completed"], false);
        assert_eq!(app.result["terminated"], true);
        assert_eq!(app.summary["task_completed"], false);
        assert!(app.neural.as_object().unwrap().is_empty());
        assert!(app.action.as_object().unwrap().is_empty());
        assert!(app.histories.values().all(VecDeque::is_empty));
        assert!(app.paused);
    }

    #[test]
    fn malformed_project_updates_clear_stale_snapshot_and_recover() {
        let valid = include_str!("../tests/fixtures/project-progress.jsonl")
            .lines()
            .nth(1)
            .unwrap();
        for malformed in [
            json!([]),
            json!({"schema_version":2}),
            json!({"schema_version":1,"verified":true}),
            json!({"schema_version":1,"assessment":{"data_quality":"yes"}}),
            json!({"schema_version":1,"active_star":{"planet":{"scientific_verified":"true"}}}),
            json!({"schema_version":1,"uncertain_actions":[null]}),
        ] {
            let mut app = App::default();
            app.apply(parse_event(valid).unwrap());
            let event = json!({"version":1,"event":"state","sequence":2,"run_id":"fixture-project",
                "payload":{"project_progress":malformed,"status":"paused"}});
            app.apply(parse_event(&event.to_string()).unwrap());
            assert!(app.project_progress().is_none());
            assert!(app.state["project_progress_error"].is_string());
            assert!(app.paused);
            app.apply(parse_event(valid).unwrap());
            assert!(app.project_progress().is_some());
            assert!(app.state["project_progress_error"].is_null());
        }
    }

    #[test]
    fn project_replay_recovers_from_malformed_lines_and_accepts_legacy_events() {
        let mut app = App::default();
        let lines = include_str!("../tests/fixtures/project-progress.jsonl").lines();
        let mut errors = 0;
        for line in lines
            .chain(["not JSON", "{\"version\":2}"])
            .chain(include_str!("../tests/fixtures/session.jsonl").lines())
        {
            match parse_event(line) {
                Ok(event) => app.apply(event),
                Err(error) => {
                    errors += 1;
                    app.log(error);
                }
            }
        }
        assert_eq!(errors, 2);
        assert!(app.project_progress().is_none());
        assert!(app.state["project_progress_error"].is_null());
        assert_eq!(app.observation["instruction"], "Analyze star K-12");
        assert_eq!(app.action["target_id"], "o1:c1");
        assert_eq!(app.result["cumulative_reward"], 1.5);
    }

    #[test]
    fn same_run_project_state_survives_ordinary_updates_but_explicit_null_clears() {
        let mut app = App::default();
        app.apply(
            parse_event(
                include_str!("../tests/fixtures/project-progress.jsonl")
                    .lines()
                    .nth(1)
                    .unwrap(),
            )
            .unwrap(),
        );
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":2,"run_id":"fixture-project","payload":{"paused":true,"policy_stage":"numeric"}}"#).unwrap());
        assert_eq!(app.project_progress().unwrap()["collected"], 3);
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":3,"run_id":"fixture-project","payload":{"project_progress":null}}"#).unwrap());
        assert!(app.project_progress().is_none());
        assert!(app.state["project_progress_error"].is_null());
    }

    #[test]
    fn malformed_observation_policy_clears_stale_progress_then_recovers() {
        let valid = parse_event(
            include_str!("../tests/fixtures/project-progress.jsonl")
                .lines()
                .nth(4)
                .unwrap(),
        )
        .unwrap();
        for branch in ["planet", "habitability"] {
            for (field, value) in [
                ("policy_label", json!(true)),
                ("policy_label", json!([])),
                ("observation_limit_days", json!(0)),
                ("observation_limit_days", json!(10001)),
                ("observation_limit_days", json!(-1)),
                ("observation_limit_days", json!(5000.0)),
                ("observation_limit_days", json!(true)),
                ("observation_limit_days", json!("5000")),
                ("branch_applicability_verified", json!("true")),
                ("branch_applicability_verified", json!(1)),
                ("branch_applicability_verified", json!(null)),
            ] {
                let mut app = App::default();
                app.apply(valid.clone());
                let mut malformed = valid.clone();
                malformed.payload["project_progress"]["active_star"][branch][field] = value;
                app.apply(malformed);
                assert!(
                    app.project_progress().is_none(),
                    "Accepted malformed {branch}.{field}"
                );
                assert_eq!(
                    app.state["project_progress_error"],
                    "invalid star-stage evidence"
                );
                app.apply(valid.clone());
                assert!(app.project_progress().is_some());
                assert!(app.state["project_progress_error"].is_null());
            }
        }
    }

    #[test]
    fn optional_observation_policy_accepts_null_legacy_fields_and_bounded_days() {
        let valid = parse_event(
            include_str!("../tests/fixtures/project-progress.jsonl")
                .lines()
                .nth(4)
                .unwrap(),
        )
        .unwrap();
        for days in [Value::Null, json!(1), json!(5000), json!(10000)] {
            let mut event = valid.clone();
            event.payload["project_progress"]["active_star"]["planet"]["policy_label"] =
                Value::Null;
            event.payload["project_progress"]["active_star"]["planet"]["observation_limit_days"] =
                days;
            let mut app = App::default();
            app.apply(event);
            assert!(app.project_progress().is_some());
        }
        let mut app = App::default();
        app.apply(
            parse_event(
                include_str!("../tests/fixtures/project-progress.jsonl")
                    .lines()
                    .nth(1)
                    .unwrap(),
            )
            .unwrap(),
        );
        assert!(app.project_progress().is_some());
        assert!(app.project_progress().unwrap()["active_star"]["planet"]["policy_label"].is_null());
    }

    #[test]
    fn project_level_handoff_clears_previous_star_approvals_without_commands() {
        let mut app = App::default();
        app.apply(
            parse_event(
                include_str!("../tests/fixtures/project-progress.jsonl")
                    .lines()
                    .nth(1)
                    .unwrap(),
            )
            .unwrap(),
        );
        app.state["pending_browser_copy"] = json!({"destination":"distance"});
        app.state["pending_browser_color"] = json!({"selected_color":"IR"});
        app.state["action_source"] = json!("learned_prediction");
        app.observation = json!({"instruction":"Old star"});
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":8,"run_id":"fixture-project","payload":{"project_progress":{"schema_version":1,"active_star":null,"active_stage":"assessment"}}}"#).unwrap());
        assert!(app.state["pending_browser_copy"].is_null());
        assert!(app.state["pending_browser_color"].is_null());
        assert!(app.state["action_source"].is_null());
        assert_eq!(app.observation, json!({}));
        assert_eq!(
            app.project_progress().unwrap()["active_stage"],
            "assessment"
        );
    }

    fn reference_event() -> RuntimeEvent {
        parse_event(
            include_str!("../tests/fixtures/reference-measurements.jsonl")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap()
    }

    fn tooltip_reference_event() -> RuntimeEvent {
        parse_event(
            include_str!("../tests/fixtures/tooltip-reference-measurements.jsonl")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap()
    }

    fn two_tooltip_reference_event() -> RuntimeEvent {
        parse_event(
            include_str!("../tests/fixtures/two-tooltip-reference-measurements.jsonl")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn two_tooltip_fixture_preserves_single_interval_and_no_completion_at_replay_eof() {
        let mut app = App::default();
        for (index, line) in
            include_str!("../tests/fixtures/two-tooltip-reference-measurements.jsonl")
                .lines()
                .enumerate()
        {
            app.apply(parse_event(line).unwrap());
            assert_eq!(
                app.reference_measurements().is_some(),
                matches!(index, 1..=3)
            );
            assert_ne!(app.state["task_completed"], true);
            assert_ne!(app.state["project_completed"], true);
            assert_ne!(app.state["submitted"], true);
            if let Some(reference) = app.reference_measurements() {
                assert_eq!(reference["consistency_redundancy"], 0);
                assert_eq!(reference["recurrence_confirmed"], false);
            }
        }
    }

    #[test]
    fn two_tooltip_scope_is_strict_and_invalid_updates_clear_stale_values() {
        let valid = two_tooltip_reference_event();
        for (path, value) in [
            ("/confirmed_feature_count", json!(3)),
            ("/confirmed_feature_count", json!(2.0)),
            ("/confirmed_feature_count", json!("2")),
            ("/observed_interval_count", json!(2)),
            ("/observed_interval_count", json!(true)),
            ("/consistency_redundancy", json!(1)),
            ("/consistency_redundancy", json!(0.0)),
            ("/consistency_redundancy", json!(false)),
            ("/consecutive_events_assumed", json!(false)),
            ("/consecutive_events_assumed", json!(1)),
            ("/recurrence_confirmed", json!(true)),
            ("/recurrence_confirmed", json!(0)),
            ("/observed_recurrence_compatible", json!(true)),
            ("/observed_recurrence_compatible", json!(0)),
            ("/single_spacing_compatible", json!(false)),
            ("/single_spacing_compatible", json!(1)),
            ("/period_days/estimate_kind", json!("recurrence")),
            ("/period_days/value", json!(999)),
            ("/period_days/compatibility_interval/upper", json!(5001)),
            ("/brightness_drop_percent/value", json!(0)),
            (
                "/brightness_drop_percent/physical_bounds",
                json!([0.1, 1.0]),
            ),
            ("/learned_perception", json!(true)),
            ("/scientific_verified", json!(true)),
            ("/training_label", json!(true)),
        ] {
            let mut app = App::default();
            app.apply(valid.clone());
            let mut malformed = valid.clone();
            *malformed.payload["reference_measurements"]
                .pointer_mut(path)
                .unwrap() = value;
            app.apply(malformed);
            assert!(app.reference_measurements().is_none(), "Accepted {path}");
            assert!(app.state["reference_measurements_error"].is_string());
            app.apply(valid.clone());
            assert!(app.reference_measurements().is_some());
        }
        for key in [
            "confirmed_feature_count",
            "observed_interval_count",
            "consistency_redundancy",
            "consecutive_events_assumed",
            "recurrence_confirmed",
            "observed_recurrence_compatible",
            "single_spacing_compatible",
        ] {
            let mut app = App::default();
            app.apply(valid.clone());
            let mut missing = valid.clone();
            missing.payload["reference_measurements"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            app.apply(missing);
            assert!(
                app.reference_measurements().is_none(),
                "Accepted missing {key}"
            );
        }
    }

    #[test]
    fn two_tooltip_replacement_does_not_change_legacy_or_retain_another_star() {
        let mut app = App::default();
        app.apply(two_tooltip_reference_event());
        app.apply(tooltip_reference_event());
        assert_eq!(
            app.reference_measurements().unwrap()["mode"],
            "approximate_reference_visible_tooltips_v1"
        );
        assert!(app
            .reference_measurements()
            .unwrap()
            .get("consistency_redundancy")
            .is_none());
        app.apply(two_tooltip_reference_event());
        app.apply(reference_event());
        assert_eq!(
            app.reference_measurements().unwrap()["mode"],
            "approximate_reference_raster"
        );
        app.apply(two_tooltip_reference_event());
        let mut handoff = two_tooltip_reference_event();
        handoff.payload = json!({"star":"Other"});
        app.apply(handoff);
        assert!(app.reference_measurements().is_none());
    }

    #[test]
    fn tooltip_fixture_preserves_replay_eof_without_claiming_completion() {
        let mut app = App::default();
        for (index, line) in include_str!("../tests/fixtures/tooltip-reference-measurements.jsonl")
            .lines()
            .enumerate()
        {
            app.apply(parse_event(line).unwrap());
            assert_eq!(
                app.reference_measurements().is_some(),
                matches!(index, 1..=3)
            );
            assert_ne!(app.summary["task_completed"], true);
            assert_ne!(app.state["project_completed"], true);
        }
    }

    #[test]
    fn malformed_tooltip_quantities_clear_stale_reference_and_can_recover() {
        let valid = tooltip_reference_event();
        for (path, value) in [
            ("/period_days/value", json!(0)),
            ("/period_days/value", json!(999)),
            ("/period_days/value", json!(1001)),
            ("/period_days/value", json!("1000")),
            ("/period_days/compatibility_interval/lower", json!(0)),
            ("/period_days/compatibility_interval/upper", json!(5001)),
            (
                "/period_days/compatibility_interval/endpoints",
                json!("closed"),
            ),
            ("/period_days/unit", json!("years")),
            ("/brightness_drop_percent/value", json!(0)),
            ("/brightness_drop_percent/value", json!(-1)),
            ("/brightness_drop_percent/value", json!(101)),
            ("/brightness_drop_percent/value", json!("0.762")),
            ("/brightness_drop_percent/unit", json!("%")),
            (
                "/brightness_drop_percent/physical_bounds",
                json!([0.7, 0.8]),
            ),
            ("/learned_perception", json!(true)),
            ("/scientific_verified", json!(true)),
            ("/training_label", json!(true)),
        ] {
            let mut app = App::default();
            app.apply(valid.clone());
            let mut malformed = valid.clone();
            *malformed.payload["reference_measurements"]
                .pointer_mut(path)
                .unwrap() = value;
            app.apply(malformed);
            assert!(app.reference_measurements().is_none(), "Accepted {path}");
            assert!(app.state["reference_measurements_error"].is_string());
            app.apply(valid.clone());
            assert!(app.reference_measurements().is_some());
        }
        for key in ["period_days", "brightness_drop_percent"] {
            let mut app = App::default();
            app.apply(valid.clone());
            let mut malformed = valid.clone();
            malformed.payload["reference_measurements"][key]["lower"] = json!(0);
            app.apply(malformed);
            assert!(app.reference_measurements().is_none());
        }
        let mut app = App::default();
        app.apply(valid.clone());
        let mut missing = valid;
        missing.payload["reference_measurements"]["brightness_drop_percent"]
            .as_object_mut()
            .unwrap()
            .remove("physical_bounds");
        app.apply(missing);
        assert!(app.reference_measurements().is_none());
    }

    #[test]
    fn tooltip_star_handoff_and_legacy_replacement_do_not_retain_stale_values() {
        let mut app = App::default();
        let event = tooltip_reference_event();
        app.apply(event.clone());
        let mut handoff = event;
        handoff.payload = json!({"star":"Other"});
        app.apply(handoff);
        assert!(app.reference_measurements().is_none());
        app.apply(reference_event());
        assert_eq!(
            app.reference_measurements().unwrap()["mode"],
            "approximate_reference_raster"
        );
        app.apply(tooltip_reference_event());
        assert_eq!(
            app.reference_measurements().unwrap()["mode"],
            "approximate_reference_visible_tooltips_v1"
        );
        assert!(parse_event("malformed event").is_err());
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        assert!(app.reference_measurements().is_none());
        assert_eq!(app.observation["instruction"], "Analyze star K-12");
    }

    #[test]
    fn reference_fixture_replays_atomic_updates_and_component_handoff() {
        let mut app = App::default();
        for (index, line) in include_str!("../tests/fixtures/reference-measurements.jsonl")
            .lines()
            .enumerate()
        {
            app.apply(parse_event(line).unwrap());
            assert_eq!(
                app.reference_measurements().is_some(),
                matches!(index, 1..=3 | 5)
            );
        }
        let reference = app.reference_measurements().unwrap();
        assert_eq!(reference["star"], "Other");
        assert_eq!(reference["period_days"]["value"], 440);
        assert_eq!(reference["line_shift"]["value"].as_f64(), Some(0.00002));
        assert_eq!(reference["learned_perception"], false);
        assert!(app.summary["task_completed"].is_null());
        assert!(app.state["reference_measurements_error"].is_null());
    }

    #[test]
    fn invalid_reference_values_clear_stale_and_valid_replacement_recovers() {
        let valid = reference_event();
        for (path, value) in [
            ("/mode", json!("learned_raster")),
            ("/star", json!("")),
            ("/star", json!("\u{1b}[31m")),
            ("/star", json!("Other")),
            ("/star", json!(true)),
            ("/observation_limit_days", json!(5000.0)),
            ("/observation_limit_days", json!(10000)),
            ("/observation_limit_days", json!("5000")),
            ("/learned_perception", json!(true)),
            ("/scientific_verified", json!(0)),
            ("/training_label", json!(null)),
            ("/period_days/value", json!(0)),
            ("/period_days/value", json!("300")),
            ("/period_days/value", json!("NaN")),
            ("/period_days/value", json!(null)),
            ("/period_days/lower", json!(0)),
            ("/period_days/lower", json!(301)),
            ("/period_days/upper", json!(299)),
            ("/period_days/unit", json!("years")),
            ("/brightness_drop_percent/value", json!(-1)),
            ("/brightness_drop_percent/lower", json!(-1)),
            ("/brightness_drop_percent/upper", json!(101)),
            ("/brightness_drop_percent/unit", json!("%")),
            ("/line_shift/value", json!(-0.1)),
            ("/line_shift/value", json!(true)),
            ("/line_shift/unit", json!("m")),
        ] {
            let mut app = App::default();
            app.apply(valid.clone());
            let mut malformed = valid.clone();
            *malformed.payload["reference_measurements"]
                .pointer_mut(path)
                .unwrap() = value;
            app.apply(malformed);
            assert!(app.reference_measurements().is_none(), "Accepted {path}");
            assert!(app.state["reference_measurements_error"].is_string());
            app.apply(valid.clone());
            assert!(app.reference_measurements().is_some());
            assert!(app.state["reference_measurements_error"].is_null());
        }
        for malformed in [json!([]), json!("reference"), json!({})] {
            let mut app = App::default();
            app.apply(valid.clone());
            let mut event = valid.clone();
            event.payload["reference_measurements"] = malformed;
            app.apply(event);
            assert!(app.reference_measurements().is_none());
            assert!(app.state["reference_measurements_error"].is_string());
        }
    }

    #[test]
    fn reference_zero_shift_and_drop_are_not_missing_and_bounds_can_coincide() {
        let mut event = reference_event();
        event.payload["reference_measurements"]["line_shift"]["value"] = json!(0);
        for key in ["value", "lower", "upper"] {
            event.payload["reference_measurements"]["brightness_drop_percent"][key] = json!(0);
            event.payload["reference_measurements"]["period_days"][key] = json!(1);
        }
        let mut app = App::default();
        app.apply(event);
        assert!(app.reference_measurements().is_some());
    }

    #[test]
    fn component_star_project_and_observation_handoffs_clear_reference_measurements() {
        for payload in [
            json!({"component":"new"}),
            json!({"policy_stage":"numeric"}),
            json!({"task":"other"}),
            json!({"browser_task":"color"}),
            json!({"star":"Other"}),
            json!({"reference_measurements":null}),
            json!({"project_progress":{"schema_version":1,"active_star":{"id":"other","name":"Other","stage":"planet"}}}),
            json!({"project_progress":{"schema_version":1,"active_star":null}}),
        ] {
            let mut app = App::default();
            app.apply(reference_event());
            let mut event = reference_event();
            event.payload = payload;
            app.apply(event);
            assert!(app.reference_measurements().is_none());
            assert!(app.state["reference_measurements_error"].is_null());
        }
        for kind in [EventKind::Observation, EventKind::ActionResult] {
            for observation in [
                json!({"values":{"star_name":"Other"}}),
                json!({"chart":{"star":"Other"}}),
            ] {
                let mut app = App::default();
                app.apply(reference_event());
                let mut event = reference_event();
                event.event = kind;
                event.payload = json!({"observation":observation});
                app.apply(event);
                assert!(app.reference_measurements().is_none());
            }
        }
    }

    #[test]
    fn reference_updates_preserve_pause_and_reject_project_star_mismatch() {
        let mut app = App::default();
        let valid = reference_event();
        app.apply(valid.clone());
        let mut status = valid.clone();
        status.payload = json!({"paused":true});
        app.apply(status);
        assert!(app.reference_measurements().is_some() && app.paused);
        let mut mismatch = valid.clone();
        mismatch.payload["project_progress"] =
            json!({"schema_version":1,"active_star":{"id":"other","name":"Other"}});
        app.apply(mismatch);
        assert!(app.reference_measurements().is_none());
        assert!(app.state["reference_measurements_error"]
            .as_str()
            .unwrap()
            .contains("different star"));
        let mut same_event_handoff = valid;
        same_event_handoff.payload["star"] = json!("Other");
        same_event_handoff.payload["component"] = json!("new");
        same_event_handoff.payload["reference_measurements"]["star"] = json!("other");
        app.apply(same_event_handoff);
        assert!(app.reference_measurements().is_some());
    }

    #[test]
    fn reference_replay_parser_recovers_and_legacy_streams_do_not_gain_reference_claims() {
        let mut app = App::default();
        app.apply(reference_event());
        for malformed in [
            "not JSON",
            r#"{"version":1,"event":"state","sequence":2,"payload":{"reference_measurements":{"value":1e400}}}"#,
        ] {
            assert!(parse_event(malformed).is_err());
        }
        app.apply(
            parse_event(r#"{"version":1,"event":"hello","sequence":0,"payload":{}}"#).unwrap(),
        );
        assert!(app.reference_measurements().is_none());
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        assert!(app.reference_measurements().is_none());
        assert_eq!(app.observation["instruction"], "Analyze star K-12");
        assert_eq!(app.result["cumulative_reward"], 1.5);
    }
}
