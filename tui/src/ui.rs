use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Sparkline, Wrap},
    Frame,
};

use crate::app::{
    action_target, confidence, display, failure_explanation, population_mean, App, GROUPS,
};

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if area.width < 65 || area.height < 18 {
        frame.render_widget(Paragraph::new("HabFly · enlarge terminal to at least 65 × 18\ns start · space pause/resume · n step · a abort · q quit")
            .block(Block::bordered().title("HabFly")), area);
        return;
    }
    if app.focused_panel != 0 {
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(2),
        ])
        .split(area);
        frame.render_widget(
            Paragraph::new(format!(
                "{} · policy {} · seed {}",
                runtime_status(app),
                display(&app.state["policy"]),
                display(&app.state["seed"])
            ))
            .block(Block::bordered().title("HabFly · focused panel · v cycles views")),
            rows[0],
        );
        if app.focused_panel == 1 {
            draw_neural(frame, app, rows[1]);
        } else if app.focused_panel == 3 {
            draw_observation(frame, app, rows[1]);
        } else {
            let sections = Layout::vertical([
                Constraint::Length(6),
                Constraint::Min(3),
                Constraint::Length(6),
            ])
            .split(rows[1]);
            draw_observation(frame, app, sections[0]);
            draw_controls(frame, app, sections[1]);
            draw_action(frame, app, sections[2]);
        }
        frame.render_widget(Paragraph::new(footer(app)), rows[2]);
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(
            5 + u16::from(app.project_progress().is_some()) * 3
                + u16::from(app.is_browser_project()) * 2,
        ),
        Constraint::Min(8),
        Constraint::Length(6),
        Constraint::Length(2),
    ])
    .split(area);
    draw_status(frame, app, rows[0]);
    let panels =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).split(rows[1]);
    let left = Layout::vertical([
        Constraint::Length(5),
        Constraint::Min(3),
        Constraint::Length(6),
    ])
    .split(panels[0]);
    draw_observation(frame, app, left[0]);
    draw_controls(frame, app, left[1]);
    draw_action(frame, app, left[2]);
    draw_neural(frame, app, panels[1]);
    let timeline: Vec<Line> = app
        .timeline
        .iter()
        .rev()
        .take(rows[2].height.saturating_sub(2) as usize)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|line| Line::from(line.as_str()))
        .collect();
    frame.render_widget(
        Paragraph::new(timeline).block(Block::bordered().title("Timeline / runtime diagnostics")),
        rows[2],
    );
    frame.render_widget(
        Paragraph::new(footer(app)).style(Style::default().fg(Color::DarkGray)),
        rows[3],
    );
}

fn footer(app: &App) -> String {
    if app.state["replay"] == true {
        if app.state["replay_finished"] == true {
            return "RECORDED REPLAY FINISHED · v views · arrows scroll · q quit\nEnd of saved events, not task success; no browser is connected".into();
        }
        return "RECORDED REPLAY · space pause/resume · n step · v views · arrows scroll · q quit\nNo browser is connected; recorded confirmations cannot perform writes".into();
    }
    if app.is_browser_project() {
        if app.is_project_finalization() {
            if app.project_finished() || app.project_handoff_guidance().is_some() {
                return format!(
                    "FINALIZATION {} · {} · submitted {} / project {}\nNo steps, approvals or retries · outcome may be unknown/pending · v evidence · q quit",
                    display(&app.state["status"]), display(&app.state["browser_phase"]),
                    display(&app.state["submitted"]), display(&app.state["project_completed"])
                );
            }
            if app.browser_resume_guidance().is_some() {
                return "FINALIZATION lifecycle/budgets unavailable · no step or resume sent\na abort · v evidence · q quit".into();
            }
            return if app.paused {
                "FINALIZATION PAUSED · n one boundary · r/space resume · a abort · v evidence · q quit\nSeparate fixed 600s scoring / 180s submission; paused time counts; no new approvals".into()
            } else {
                "FINALIZATION RUNNING · p/space pause · a abort · v evidence · q quit\nPause/abort between bounded calls only; no retry or automatic budget increase".into()
            };
        }
        if app.project_finished() {
            return format!(
                "PROJECT {} · {} · task {} / project {}\nNo steps, approvals or retries · v views · arrows scroll · q close browser and quit",
                display(&app.state["status"]), display(&app.state["browser_phase"]),
                display(&app.state["task_completed"]), display(&app.state["project_completed"])
            );
        }
        if app.project_automatic_decision() {
            return if app.paused {
                "AUTOMATIC REFERENCE PAUSED · n one local decision stage · r/space resume · a abort · q quit\nRuntime selects from source-backed rules; browser action is a later stage; NOT learned classification".into()
            } else {
                "AUTOMATIC REFERENCE RUNNING · p/space pause · a abort · v details · q quit\nNo manual decision handoff; runtime selects from source-backed rules, not TUI keys".into()
            };
        }
        let handoff = match app.project_phase() {
            Some("awaiting_class_source") => {
                Some("CLASS EVIDENCE required for captured star; no class inferred")
            }
            Some("awaiting_inventory") => {
                Some("INVENTORY EVIDENCE required; workflow journal import pending")
            }
            Some("awaiting_planet_class") => {
                Some("PLANET CLASS reference decision required; no class inferred")
            }
            Some("awaiting_gases") => Some("GAS reference decision required; no identity inferred"),
            Some("awaiting_habitability") => {
                Some("HABITABILITY reference decision required; no conclusion inferred")
            }
            _ => None,
        };
        if let Some(handoff) = handoff {
            return format!("{handoff}\nExternal runtime handoff · n/r guidance only · a abort · v details · q quit");
        }
        if app.browser_resume_guidance().is_some() {
            return "PROJECT lifecycle unavailable · no step or approval sent\na abort · v details · arrows scroll · q quit".into();
        }
        return if app.paused {
            "PROJECT PAUSED · n one bounded stage · r/space resume · a abort · v details · q quit\nPause/abort apply between bounded calls; not an in-call interruption".into()
        } else {
            "PROJECT RUNNING · p/space pause · a abort · v details · q quit\nPause/abort apply between bounded calls; not an in-call interruption".into()
        };
    }
    if app.state["browser_phase"] == "setting_up" && app.state["browser_execution"] != "autonomous"
    {
        return format!(
            "{}\na abort · q quit · scripted setup only; learned policy remains paused",
            display(&app.state["browser_guidance"])
        );
    }
    if app.state["browser_held_open"] == true {
        return format!(
            "{}\nq close browser and quit · v views · arrows scroll · no actions or retries",
            display(&app.state["browser_guidance"])
        );
    }
    if app.state["browser_phase"] == "finished" {
        return format!(
            "{}\nRun ended · q quit · v views · arrows scroll · no further steps or writes",
            display(&app.state["browser_guidance"])
        );
    }
    if app.state["browser_execution"] == "autonomous" {
        let color = app.state["browser_task"] == "color";
        let four_field = app.state["browser_task"] == "four_field";
        let controls = if four_field && app.paused {
            "PAUSED: no automatic writes or handoff · r/space resume · n step · y approve pending write"
        } else if four_field {
            "RUNNING: three numeric copies + one color selection · p/space pause"
        } else if color && app.paused {
            "PAUSED: no automatic selection · r/space resume · y approve pending color"
        } else if color {
            "RUNNING: one automatic color selection · p/space pause"
        } else if app.paused {
            "PAUSED: no automatic steps or writes · r/space resume · n one decision · y approve pending copy"
        } else {
            "RUNNING: up to three automatic copies · p/space pause"
        };
        return format!(
            "{}\n{} · a abort · q quit · v views",
            display(&app.state["browser_guidance"]),
            controls
        );
    }
    if app.state["browser_task"] == "color" || app.state["browser_task"] == "four_field" {
        let controls = match app.state["browser_phase"].as_str() {
            Some("awaiting_ready") => "b capture ready browser",
            Some("awaiting_color") => "y approve one color",
            Some("awaiting_copy") => "y approve one numeric copy",
            Some("awaiting_handoff") => "n recheck star and switch policy (no write)",
            _ => "n step",
        };
        format!(
            "{}\n{} · a abort · v views · arrows scroll · q quit",
            display(&app.state["browser_guidance"]),
            controls
        )
    } else if app.state["browser_phase"].is_string() {
        format!("{}\nb ready · n step · y approve one copy · a abort · v views · arrows scroll · q quit",
            display(&app.state["browser_guidance"]))
    } else {
        "s start · space pause/resume · p pause · r resume · n step · a abort · t save trace\nv focus panels · ↑/↓ scroll · q / Ctrl-C quit · --replay PATH".into()
    }
}

fn option_text(app: &App, option: &serde_json::Value) -> String {
    let Some(id) = option.as_str() else {
        return display(option);
    };
    let measurement = &app.observation["values"]["measurements"][id];
    let result = &app.observation["calculation"]["results"][id];
    let item = if measurement.is_object() {
        measurement
    } else {
        result
    };
    if item.is_object() {
        format!(
            "{}: {} {} {} ({})",
            display(option),
            display(&item["kind"]),
            display(&item["value"]),
            display(&item["unit"]),
            if measurement.is_object() {
                display(&item["source"])
            } else {
                format!("valid {}", display(&item["valid"]))
            }
        )
    } else {
        display(option)
    }
}

fn draw_observation(frame: &mut Frame, app: &App, area: Rect) {
    let sheet = &app.observation["spreadsheet"];
    let calc = &app.observation["calculation"];
    let measurements = app.observation["values"]["measurements"]
        .as_object()
        .map(|items| {
            items
                .keys()
                .map(|key| option_text(app, &serde_json::Value::String(key.clone())))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let extra = if app.state["task"] == "browser_transit_sampling" {
        let sample = &app.observation["chart"]["latest_sample"];
        let progress = &app.state["progress"];
        format!(
            "\nScripted visible-chart sensor; NOT learned perception\nStar {} · latest day {} · brightness {}%\nWindow {} · unique days {} · complete transits {}\nScan status: {}\nPeriod: {} days · drop: {}%\nSafety stop: {}\nNo answer writes; incomplete coverage is NOT evidence of absence.\nPlanet choice and course completion remain unverified.",
            display(&app.observation["chart"]["star"]), display(&sample["day"]),
            display(&sample["brightness_percent"]), display(&progress["window"]),
            display(&progress["unique_days"]), display(&progress["complete_transits"]),
            display(app.summary.get("status").unwrap_or(&progress["status"])),
            display(&app.summary["period_days"]), display(&app.summary["brightness_drop_percent"]),
            display(&app.state["chart_stop"])
        )
    } else if calc["calculation_mode"].as_str() == Some("learned_peak_wavelength_color_v1") {
        let scope = if app.observation["progress"]["task"] == "browser_color" {
            "Supervised browser color (readback only; not course acceptance)"
        } else {
            "Experimental local color policy (not stellar classification)"
        };
        format!("\n{}\nSelected measurement: {}\nMeasurements:\n{}\nSelected color: {}\nReference: {}\nBands (nm): {}\nAmbiguity policy: {}\nPending reference: {}\nError: {}\nBrowser color readback: {}",
            scope,
            display(&calc["source"]), measurements, display(&app.observation["values"]["answers"]["color"]),
            display(&calc["reference_card"]["description"]), display(&calc["reference_card"]["bands"]),
            display(&calc["reference_card"]["ambiguity_policy"]), display(&calc["reference_card"]["pending"]),
            display(&calc["tool_error"]), display(&app.observation["values"]["color_readback"]))
    } else if calc.get("calculation_mode").is_some() {
        format!("\nLocal tool: {}\nSupplied class: {} · required fields {}\nOperation: {} · {}\nInputs: {}\nSelected input: {} ← {} · bindings {}\nMeasurements:\n{}\nResults: {}\nCopy: {} → {}\nLast: {}\nTool error: {}\nAnswers: {} · units {}\nBrowser readbacks: {}",
            display(&calc["calculation_mode"]), display(&app.observation["values"]["star_class"]),
            display(&app.observation["values"]["required_fields"]), display(&calc["operation"]),
            display(&calc["reference_card"]["description"]), display(&calc["reference_card"]["inputs"]),
            display(&calc["parameter"]), display(&calc["source"]), display(&calc["bindings"]), measurements,
            display(&calc["results"]), display(&calc["selected_result"]), display(&calc["destination"]),
            display(&calc["last_operation"]), display(&calc["tool_error"]),
            display(&app.observation["values"]["answers"]),
            display(&app.observation["values"]["units"]),
            display(&app.observation["values"]["numeric_readbacks"]))
    } else if sheet.get("calculation_mode").is_some() {
        format!("\nSheet: {} · generation {}\nInput: {} → {} · bindings {}\nResult: {} → {}\nValues: {}\nLast: {}\nMeasurements: {}\nAnswers: {} · units {}",
            display(&sheet["calculation_mode"]), display(&sheet["generation"]),
            display(&sheet["selected_measurement"]), display(&sheet["selected_input"]),
            display(&sheet["bindings"]), display(&sheet["selected_output"]),
            display(&sheet["selected_destination"]), display(&sheet["results"]),
            display(&sheet["last_operation"]), display(&app.observation["values"]["measurements"]),
            display(&app.observation["values"]["answers"]), display(&app.observation["values"]["units"]))
    } else {
        String::new()
    };
    let scope_notice = if app.observation["progress"]["task"] == "planet_calculations"
        || app.observation["progress"]["task"] == "browser_planet_derived"
    {
        format!(
            "\nPlanet tool-use experiment: supplied measurements; NOT course acceptance\nPhysics assumptions: {}\nPending knowledge: {}",
            display(&app.observation["values"]["physics_assumptions"]),
            display(&calc["pending_knowledge"])
        )
    } else if app.observation["progress"]["task"] == "habitability_calculations" {
        "\nSupplied-temperature exercise only: NOT gas identification, water phase or habitability classification.\nCourse acceptance remains unverified.".into()
    } else {
        String::new()
    };
    let text = format!(
        "{}{}{}{}\nFeedback: {}\nProgress: {}\nChart: {} · crop {}{}{}",
        project_runtime_text(app),
        reference_measurements_text(app),
        project_text(app),
        display(&app.observation["instruction"]),
        display(&app.observation["feedback"]),
        display(&app.observation["progress"]),
        display(&app.observation["chart"]),
        display(&app.observation["chart_crop"]),
        scope_notice,
        extra
    );
    let paragraph = Paragraph::new(text).wrap(Wrap { trim: false });
    let lines = paragraph.line_count(area.width.saturating_sub(2).max(1));
    let max_scroll = lines
        .saturating_sub(area.height.saturating_sub(2) as usize)
        .min(u16::MAX as usize) as u16;
    let scroll = if app.focused_panel == 3 {
        app.observation_scroll.min(max_scroll)
    } else {
        0
    };
    frame.render_widget(
        paragraph.scroll((scroll, 0)).block(Block::bordered().title(
            if app.reference_measurements().is_some() {
                "Approx. reference / observation · v to full view"
            } else if app.project_progress().is_some() || app.is_browser_project() {
                "Project progress / observation · v to full view"
            } else {
                "Observation"
            },
        )),
        area,
    );
}

fn project_runtime_text(app: &App) -> String {
    if !app.is_browser_project() {
        return String::new();
    }
    if app.is_project_finalization() {
        return finalization_text(app);
    }
    let owner = &app.state["project_owner"];
    let child = &owner["star_component"];
    let scope = if app.state["project_campaign"] == true {
        "bounded multi-star campaign"
    } else {
        "one active star bridge"
    };
    let mut text = format!(
        "REFERENCE-ASSISTED browser project · {scope}\nLifecycle {} · status {} · browser {}\nScripted setup: {} · advances {}\nVisible star {} (not a collection receipt) · owner {} / {} · component {}\nLearned calculation/color stages and explicit reference decisions remain distinct.\nTask completed: {} · project completed: {}\nChild summaries and termination do NOT prove project completion or course correctness.\nPause/abort apply between bounded scheduled calls; no automatic retries.\n",
        display(&app.state["browser_phase"]), runtime_status(app),
        display(&app.state["browser_status"]), display(&app.state["setup_stage"]),
        display(&app.state["setup_advances"]), display(&serde_json::json!(app.project_visible_star())),
        display(&owner["status"]), display(&owner["phase"]),
        display(child.get("phase").unwrap_or(&app.state["policy_stage"])),
        display(&app.state["task_completed"]), display(&app.state["project_completed"]),
    );
    if app.state["autonomous_decisions_enabled"] == true {
        text.push_str("AUTOMATIC REFERENCE decisions enabled · source-backed rules, NOT learned classification.\nTUI keys only schedule stages; they do not select classes, gases or habitability.\n");
        if app.state["autonomous_decision"].is_object() {
            text.push_str(&format!(
                "Last recorded decision provenance (not a success receipt): {}\n",
                display(&app.state["autonomous_decision"])
                    .chars()
                    .take(1200)
                    .collect::<String>()
            ));
        }
    }
    if app.state["supplied_stellar_inputs_enabled"] == true {
        text.push_str("SUPPLIED-INPUT TRANSFER enabled for non-main stars · visible stellar mass/radius, not learned stellar estimates.\nFrozen policy uses a versioned class-omitting inference view; actual class remains in tool observations and logs.\nMain-sequence stages retain their original path; transfer confidence is uncalibrated and scientific correctness is unverified.\n");
    }
    if app.state["save_strategy"] == "autosave" {
        if app
            .project_progress()
            .is_some_and(|progress| progress["active_star"]["task_completed"] == true)
        {
            text.push_str("Autosave assumed; visible answers verified; persistence unverified.\n");
        } else {
            text.push_str("Autosave assumed; persistence unverified. Fresh visible answers and workflow/inventory readback are still required before a task is counted.\n");
        }
        text.push_str("No explicit Save click, banner, or elapsed timer is treated as persistence proof. This strategy alone does not complete a task or project.\n");
    }
    if app.state["baseline_edge_reference_enabled"] == true {
        text.push_str("BASELINE-RENDERING COMPATIBILITY explicitly enabled · guarded original-pixel evidence only.\nThe same 5,000-day approximate No assumption remains NOT proven absence or learned perception; shallow or long-period planets may be missed.\nThis option does not claim task or project completion.\n");
    }
    if app.state.get("save_outcome").is_some() || app.state.get("save_outcome_uncertain").is_some()
    {
        let outcome = &app.state["save_outcome"];
        let status = match app.state["save_outcome_uncertain"].as_bool() {
            Some(true) => {
                "Save may have occurred; outcome uncertain and not counted as completed work"
            }
            Some(false)
                if outcome["dispatch_recorded"] == false
                    && outcome["acknowledgement_recorded"] == false
                    && outcome["evidence_status"] == "recorded_stop" =>
            {
                "recorded pre-dispatch stop; no Save dispatch recorded"
            }
            _ => "outcome unknown; failure records unavailable or invalid",
        };
        text.push_str(&format!(
            "FAILED SAVE: {status}. No retry.\nReservation retained: {} · dispatch marker: {} · acknowledgement recorded: {}\n",
            display(&outcome["reservation_retained"]), display(&outcome["dispatch_recorded"]),
            display(&outcome["acknowledgement_recorded"]),
        ));
        if outcome["acknowledgement_recorded"] == true {
            text.push_str("Data saved notice recorded, but final readback NOT verified; this is not verified persistence or task completion.\n");
        }
        text.push_str(
            "Noncanonical child diagnostic only; canonical project counts are not changed.\n",
        );
    }
    if app.state["project_campaign"] == true {
        let campaign = &app.state["campaign"];
        let uncapped = campaign["max_seconds"] == "uncapped"
            || app.state["campaign_max_seconds"] == "uncapped";
        let timer = if uncapped {
            "no overall timer".to_string()
        } else {
            format!("fixed budget {} seconds", display(&campaign["max_seconds"]))
        };
        text.push_str(&format!(
            "Campaign: star {} · verified {}/{} · phase {} · {}\nWorkflow target verified: {} — not assessment, scoring or submission.\n",
            display(&app.state["current_star"]["ordinal"]),
            display(&campaign["verified_stars"]), display(&campaign["target_stars"]),
            display(&campaign["phase"]), timer,
            display(&app.state["target_workflows_verified"]),
        ));
        if uncapped {
            text.push_str("The 30-star target, per-star/action limits and scoring guards remain. Manual abort is available; no automatic retry or unlimited looping.\n");
        }
        if campaign["next_star"].is_object() {
            text.push_str(&format!(
                "Next-star transition: {} · no completed task is implied by opening a star.\n",
                display(&campaign["next_star"]["phase"]),
            ));
        }
    }
    if let Some(guidance) = app.browser_resume_guidance() {
        text.push_str(&format!("Handoff / controls: {guidance}\n"));
    }
    let class_setup = &app.state["class_setup"];
    if class_setup.is_object() {
        text.push_str(&format!(
            "Explicit reference setup: {} · class {} · lifetime prefix {} · setup verified {}\n{}\n",
            display(&class_setup["phase"]), display(&class_setup["selected_class"]),
            display(&class_setup["lifetime_prefix"]), display(&class_setup["setup_verified"]),
            if app.state["autonomous_decisions_enabled"] == true {
                "Classification uses automatic reference rules, not a learned classifier."
            } else { "Classification is supplied, not learned or inferred by this scheduler." },
        ));
    }
    let inventory = &owner["inventory_component"];
    let window = &child["component"];
    if window["mode"] == "bounded_planet_window" {
        text.push_str(&format!(
            "Visible chart window: {} days · phase {} · polls {}/{} · evidence {}\n",
            display(&window["observation_limit_days"]),
            display(&window["phase"]),
            display(&window["polls"]),
            display(&window["max_polls"]),
            display(&window["window_status"]),
        ));
        if window["policy"]["version"] == "user_approved_5000_day_single_event_no_planet_v1"
            && window["observation_limit_days"] == 5000
        {
            text.push_str("Single-event No shortcut is enabled by the recorded 5,000-day reference policy; not confirmed absence or learned perception. This does not claim task or project completion.\n");
            let analysis = &window["analysis"];
            if analysis["reason"] == "user_approved_single_event_shortcut"
                && analysis["approximation"] == "user_approved_single_event_no_planet_shortcut"
                && analysis["visible_candidate_events"].as_u64() == Some(1)
                && analysis["possible_planet_ignored"] == true
                && analysis["status"] == "assume_no_planet"
            {
                text.push_str("Recorded shortcut: one possible dip was deliberately ignored for an assumed No. A planet is not ruled out; no repeat interval was measured.\n");
            }
        }
        if window["phase"] == "observing" && window["waiting_for_clean_partial_chart"] == true {
            text.push_str("Waiting for a clean partial-chart capture within the original limits; no planet decision authorized.\n");
        }
        if window["phase"] == "observing"
            && window["scheduling_rule"] == "complete_rendered_window_before_positive_handoff_v1"
            && window["prior_dip_observed"] == true
            && window["waiting_for_positive_endpoint"] == true
        {
            text.push_str("Dip previously observed; waiting for the complete rendered 5,000-day window before positive handoff. No planet answer or task completion is authorized by this wait.\n");
        }
    }
    let terrestrial = &owner["terrestrial_component"];
    if terrestrial.is_object() {
        text.push_str(&format!(
            "Terrestrial workflow: {} · temperature and chamber evidence remain separate from reference gas/habitability decisions\n",
            display(&terrestrial["phase"]),
        ));
    }
    let positive = &owner["positive_component"];
    if positive.is_object() {
        text.push_str(&format!(
            "Positive-planet verification: {} · status {}\nReference planet classification is separate from learned calculation; Save alone does not complete a task.\n",
            display(&positive["phase"]), display(&positive["status"]),
        ));
    }
    if owner["planet_class_decision"].is_object() {
        text.push_str(&format!(
            "Explicit planet class: {}\n",
            display(&owner["planet_class_decision"]["name"])
        ));
    }
    if inventory.is_object() {
        text.push_str(&format!(
            "Collection verification: {} · navigations {} · count verified {}\nAn inventory receipt verifies collection only; task completion still requires workflow import.\n",
            display(&inventory["phase"]), display(&inventory["navigation_clicks"]),
            display(&inventory["collection_count_verified"]),
        ));
    }
    text.push_str(&format!(
        "Failure / stop: {}\nAssessment enabled: {} · submission enabled: {}\nArtifact root: {}\nStage artifacts: {}\nRuntime trace: {}\nProject trace: {}\n",
        display(&app.state["failure_reason"]), display(&app.state["assessment_enabled"]),
        display(&app.state["submission_enabled"]), display(&app.state["artifact_root"]),
        display(&app.state["artifact_paths"]), display(&app.state["trace_path"]),
        display(&app.state["project_trace_path"]),
    ));
    if let Some(explanation) = app.state["failure_reason"]
        .as_str()
        .and_then(failure_explanation)
    {
        text.push_str(explanation);
        text.push('\n');
    }
    text
}

fn finalization_text(app: &App) -> String {
    let state = &app.state;
    let child = &state["finalization_component"];
    let scoring = &child["scoring_component"];
    let pending = if scoring.is_object() { scoring } else { child };
    let scope = |owner, queued| state.get(owner).unwrap_or(&state[queued]);
    let mut text = format!(
        "POST-CAMPAIGN FINALIZATION · the historical star campaign remains closed\nLifecycle {} · {} · policy learning is not running\nSeparate fixed budgets: scoring {} seconds / {} advances; submission {} seconds / {} advances\nPaused time counts toward the active stage deadline; no automatic budget increase or retry.\nConfigured authorization: score transfer {} · submission {}\nCurrent canonical revision {} · journal {}\nChild phase {} · scoring phase {} · assessment outcome uncertain {}\nPending canonical action {} · pending acknowledgement {}\nScore transfer receipt {} · submission outcome {}\nReadiness readback {} · Submit click returned {} · Submit may have occurred {}\nChild clicks/readiness are NOT a grounded submission or completion receipt.\nParent submitted {} · task completed {} · project completed {}\nFinalization trace: {}\nScoring artifacts: {} · submission artifacts: {}\n",
        display(&state["browser_phase"]), runtime_status(app),
        display(&state["scoring_max_seconds"]), display(scope("scoring_max_advances", "max_scoring_advances")),
        display(&state["submission_max_seconds"]), display(scope("submission_max_advances", "max_submission_advances")),
        display(&state["allow_score_transfer"]), display(&state["allow_submission"]),
        display(&state["project_progress"]["revision"]), display(&state["journal_sha256"]),
        display(&child["phase"]), display(&scoring["phase"]), display(&pending["assessment_outcome_uncertain"]),
        display(&pending["pending_canonical_action"]), display(&pending["pending_acknowledgement"]),
        verification(&state["score_transfer_verified"]), display(&state["submission_outcome"]),
        verification(&child["readiness_readback_verified"]), display(&child["submit_click_returned"]),
        display(&child["submit_write_may_have_occurred"]), display(&state["submitted"]),
        display(&state["task_completed"]), display(&state["project_completed"]),
        display(&state["finalization_trace_path"]), display(&state["scoring_dir"]), display(&state["submission_dir"]),
    );
    if let Some(score) = score_checkpoint_score(app) {
        text.push_str(&format!(
            "UPDATE SCORE CHECKPOINT COMPLETE · Update Score verified · reported score {}\n30 workflows and both assessments are verified at the current revision. A perfect score is not required.\nFormal Submit is separate and was not performed; task/project submission flags remain false.\n",
            display(score),
        ));
    }
    if app.project_phase() == Some("finalization_pending") {
        text.push_str(&format!(
            "Queued run options: assessment enabled {} · submission enabled {} (owner authorization not yet reported).\n",
            display(&state["assessment_enabled"]), display(&state["submission_enabled"]),
        ));
        text.push_str("Queued finalization: next boundary initializes its owner only; it is not an assessment or Submit click.\n");
    }
    if state["submission_feedback"].is_object() {
        text.push_str(&format!(
            "Observed submission feedback: {} · {}\nFeedback is not a success receipt; no automatic retry.\n",
            display(&state["submission_feedback"]["status"]),
            display(&state["submission_feedback"]["reason"]),
        ));
    }
    if let Some(guidance) = app.project_handoff_guidance() {
        text.push_str(&format!("Handoff: {guidance}\n"));
    }
    text
}

fn score_checkpoint_score(app: &App) -> Option<&serde_json::Value> {
    let state = &app.state;
    let progress = &state["project_progress"];
    let score = state.get("reported_score")?;
    let numeric = score.as_f64()?;
    (state["score_checkpoint_completed"] == true
        && app.is_project_finalization()
        && app.project_phase() == Some("score_transferred_not_submitted")
        && (state["status"] == "handoff"
            || (state["replay"] == true && state["recorded_status"] == "handoff"))
        && state["finished"] == true
        && state["failure_reason"].is_null()
        && state["event_forwarding_failed"] == false
        && state["cleanup_failed"] == false
        && state["task_completed"] == false
        && state["project_completed"] == false
        && state["submitted"] == false
        && progress["target"].as_u64() == Some(30)
        && progress["collected"].as_u64() == Some(30)
        && progress["verified"].as_u64() == Some(30)
        && progress["unresolved"].as_u64() == Some(0)
        && progress["assessment"]["data_quality"] == true
        && progress["assessment"]["scavenger_hunt"] == true
        && progress["score_transfer_verified"] == true
        && progress["uncertain_actions"]
            .as_array()
            .is_some_and(Vec::is_empty)
        && numeric.is_finite()
        && numeric >= 0.0)
        .then_some(score)
}

fn reference_measurements_text(app: &App) -> String {
    let Some(reference) = app.reference_measurements() else {
        return app.state["reference_measurements_error"]
            .as_str()
            .map_or_else(String::new, |error| {
                format!(
                    "Reference measurements unavailable: {}\n",
                    display(&serde_json::json!(error))
                )
            });
    };
    let period = &reference["period_days"];
    let drop = &reference["brightness_drop_percent"];
    if reference["mode"] == "approximate_reference_two_visible_tooltips_v1" {
        return format!(
            "APPROXIMATE visible-tooltip reference · two-event estimate / single interval\nStar {} · fixed 5000-day window · 2 baseline-bracketed sampled declines\nAssumed consecutive-event spacing ≈ {} days · bracket compatibility ({}, {}) days (open)\nConsistency redundancy: 0 · recurrence NOT confirmed\nMaximum sampled decline: {} percent · physical depth bounds unavailable\nLine shift (reference): {} nm\nCompatibility is NOT physical uncertainty or model confidence.\nNOT a verified physical period or confirmed transit; missed/aliased events remain possible.\nNOT a planet-absence finding; NOT learned perception; NOT scientific verification; NOT training labels.\nNo task-completion or submission claim.\n",
            display(&reference["star"]), display(&period["value"]),
            display(&period["compatibility_interval"]["lower"]),
            display(&period["compatibility_interval"]["upper"]), display(&drop["value"]),
            display(&reference["line_shift"]["value"])
        );
    }
    if reference["mode"] == "approximate_reference_visible_tooltips_v1" {
        return format!(
            "APPROXIMATE visible-tooltip reference · star {} · fixed 5000-day window\nObserved recurrence ≈ {} days · bracket compatibility ({}, {}) days (open)\nMaximum sampled decline: {} percent · physical depth bounds unavailable\nLine shift (reference): {} nm\nCompatibility is NOT physical uncertainty or model confidence.\nNOT a verified physical period or resolved transit minimum; missed/aliased events remain possible.\nNOT learned perception; NOT scientific verification; NOT training labels.\nNo task-completion or submission claim.\n",
            display(&reference["star"]), display(&period["value"]),
            display(&period["compatibility_interval"]["lower"]),
            display(&period["compatibility_interval"]["upper"]), display(&drop["value"]),
            display(&reference["line_shift"]["value"])
        );
    }
    format!(
        "APPROXIMATE reference raster · star {} · fixed 5000-day window\nPeriod ≈ {} days · pixel bounds [{}, {}]\nBrightness drop ≈ {} percent · pixel bounds [{}, {}]\nLine shift (reference): {} nm\nPixel bounds are NOT model confidence.\nNOT learned perception; NOT scientific verification; NOT training labels.\nNo task-completion or submission claim.\n",
        display(&reference["star"]), display(&period["value"]), display(&period["lower"]),
        display(&period["upper"]), display(&drop["value"]), display(&drop["lower"]),
        display(&drop["upper"]), display(&reference["line_shift"]["value"])
    )
}

fn verification(value: &serde_json::Value) -> &'static str {
    match value.as_bool() {
        Some(true) => "verified",
        Some(false) => "unverified",
        None => "unknown",
    }
}

fn provenance(value: &serde_json::Value) -> String {
    match value.as_str() {
        Some("learned_prediction") => "learned prediction (not correctness)".into(),
        Some("reference_prediction") => "reference prediction (NOT learned)".into(),
        Some("scientific_observation") => "scientific observation (NOT learned)".into(),
        Some("course_feedback") => "course feedback (NOT learned)".into(),
        Some("tool_verified") => "tool verified (NOT learned)".into(),
        None => "unknown".into(),
        _ => format!("unrecognized: {}", display(value)),
    }
}

fn branch_outcome(value: &serde_json::Value, planet: bool) -> String {
    match (planet, value.as_str()) {
        (true, Some("candidate")) => "candidate (not confirmed)".into(),
        (true, Some("no_planet")) => "no planet".into(),
        (true, Some("planet")) => "planet".into(),
        (false, Some("habitable")) => "habitable".into(),
        (false, Some("not_habitable")) => "not habitable".into(),
        (false, Some("not_applicable")) => "not applicable".into(),
        (_, None | Some("unresolved")) => "unknown / unresolved".into(),
        _ => format!("unrecognized: {}", display(value)),
    }
}

fn project_text(app: &App) -> String {
    let Some(project) = app.project_progress() else {
        return app.state["project_progress_error"]
            .as_str()
            .map_or_else(String::new, |error| {
                format!(
                    "Project progress unavailable: {}\n",
                    display(&serde_json::json!(error))
                )
            });
    };
    let active = &project["active_star"];
    let mut text = format!(
        "Project: collected {}/{} · verified tasks {} · unresolved {}\nProject {} · attempt {} · active stage {}\nVerified task counts require explicit task receipts, not filled answers or termination.\n",
        display(&project["collected"]), display(&project["target"]),
        display(&project["verified"]), display(&project["unresolved"]),
        display(&project["project_id"]), display(&project["attempt_id"]), project_stage(project)
    );
    if active.is_object() {
        text.push_str(&format!(
            "Active star {} ({}) · stage {} · task {}\n",
            display(&active["name"]),
            display(&active["id"]),
            display(&active["stage"]),
            verification(&active["task_completed"])
        ));
        for (key, label) in [
            ("numeric", "Stellar numeric"),
            ("color", "Stellar color"),
            ("classification", "Stellar classification"),
        ] {
            let evidence = &active["stellar"][key];
            text.push_str(&format!(
                "{label}: {} · source {} · readback {} · scientific evidence {}\n",
                display(&evidence["value"]),
                provenance(&evidence["provenance"]),
                verification(&evidence["transport_verified"]),
                verification(&evidence["scientific_verified"])
            ));
        }
        for (key, label, planet) in [
            ("planet", "Planet", true),
            ("habitability", "Habitability", false),
        ] {
            let evidence = &active[key];
            text.push_str(&format!(
                "{label} reported outcome: {} · class/value {}\n  Source {} · readback {} · scientific evidence {}\n",
                branch_outcome(&evidence["outcome"], planet), display(&evidence["value"]), provenance(&evidence["provenance"]),
                verification(&evidence["transport_verified"]), verification(&evidence["scientific_verified"])
            ));
            if evidence["applicability_reason"].is_string() {
                text.push_str(&format!(
                    "  Applicability: {}\n",
                    display(&evidence["applicability_reason"])
                ));
            }
            if evidence["branch_applicability_verified"].is_boolean() {
                text.push_str(&format!(
                    "  Branch applicability {} · native readback {} (distinct evidence)\n",
                    verification(&evidence["branch_applicability_verified"]),
                    verification(&evidence["transport_verified"])
                ));
            }
            if evidence["policy_label"].is_string() || evidence["observation_limit_days"].is_u64() {
                let limit = evidence["observation_limit_days"]
                    .as_u64()
                    .map_or("unknown".into(), |days| format!("{days} days"));
                text.push_str(&format!(
                    "  Observation policy: {} · observation limit {limit}\n",
                    display(&evidence["policy_label"])
                ));
                if evidence["observation_limit_days"] == 5000
                    && evidence["policy_label"]
                        == "user_approved_5000_day_single_event_no_planet_v1"
                {
                    text.push_str("  Single-event No shortcut: explicit reference assumption, not proof of absence or learned perception.\n");
                } else if evidence["observation_limit_days"] == 5000
                    && (evidence["outcome"] == "no_planet"
                        || (evidence["outcome"] == "not_applicable"
                            && active["planet"]["outcome"] == "no_planet"))
                {
                    text.push_str(
                        "  5000-day no-dip rule: reference assumption, not proof of absence.\n",
                    );
                }
            }
        }
    } else {
        text.push_str("Active star: none reported\n");
    }
    text.push_str("No planet / not habitable can be valid outcomes; unknown is NOT absence.\nReadback verifies the UI write, not scientific or course correctness.\n");
    if let Some(actions) = project["uncertain_actions"].as_array() {
        text.push_str(&format!(
            "Uncertain actions: {} (do not retry blindly)\n",
            actions.len()
        ));
        for action in actions.iter().take(30) {
            text.push_str(&format!(
                "  {} · {} · star {} · revision {}\n",
                display(&action["id"]),
                display(&action["kind"]),
                display(&action["star_id"]),
                display(&action["revision"])
            ));
        }
        if actions.len() > 30 {
            text.push_str("  Further uncertain actions retained in the event log.\n");
        }
    } else {
        text.push_str("Uncertain actions: unknown\n");
    }
    text.push_str(&format!(
        "Assessment receipts: data quality {} · scavenger hunt {}\nScore transfer {} · submission {} · project completion {}\nAcceptance ladder: one star {} · three stars {} · thirty stars {}\nSubmission is not assessment or score transfer.\n--- Current observation ---\n",
        verification(&project["assessment"]["data_quality"]), verification(&project["assessment"]["scavenger_hunt"]),
        verification(&project["score_transfer_verified"]), verification(&project["submitted"]), verification(&project["project_completed"]),
        verification(&project["ladder"]["one_star"]), verification(&project["ladder"]["three_star"]), verification(&project["ladder"]["thirty_star"])
    ));
    text
}

fn project_stage(project: &serde_json::Value) -> String {
    display(
        project
            .get("active_stage")
            .filter(|value| value.is_string())
            .unwrap_or(&project["active_star"]["stage"]),
    )
}

fn metric(value: &serde_json::Value) -> String {
    value
        .as_f64()
        .map_or_else(|| display(value), |number| format!("{number:.4}"))
}

fn runtime_status(app: &App) -> String {
    if app.state["replay"] == true {
        format!(
            "Playback {} · recorded {}",
            if app.state["replay_finished"] == true {
                "finished".into()
            } else {
                display(&app.state["status"])
            },
            display(&app.state["recorded_status"])
        )
    } else {
        display(&app.state["status"])
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let status = runtime_status(app);
    let connected = if app.connected {
        "connected"
    } else {
        "disconnected"
    };
    let mut text = format!(
        "{status} · {connected} · policy {} · seed {} · browser {}\nStage {} · graph {} · {} {}\nStep {} · reward {} · total {} · terminated {} · truncated {}",
        display(&app.state["policy"]), display(&app.state["seed"]), display(&app.state["browser_status"]),
        display(&app.state["stage"]), display(&app.state["graph"]),
        if app.is_browser_project() { "configured stellar checkpoint" } else { "checkpoint" },
        display(&app.state["checkpoint"]),
        display(app.result.get("steps").or_else(|| app.result.get("step")).unwrap_or(&serde_json::Value::Null)),
        display(&app.result["reward"]), display(&app.result["cumulative_reward"]),
        display(&app.result["terminated"]), display(&app.result["truncated"]),
    );
    if app.is_browser_project() {
        text.push_str(&format!(
            "\nReference-assisted project · phase {} · component {}\nSetup {} · task completed {} / project completed {} · v for lifecycle details",
            display(&app.state["browser_phase"]), display(&app.state["policy_stage"]),
            display(&app.state["setup_stage"]), display(&app.state["task_completed"]),
            display(&app.state["project_completed"])
        ));
    }
    if let Some(project) = app.project_progress() {
        text.push_str(&format!(
            "\nProject {}/{} collected · {} verified tasks · {} unresolved\nActive {} · stage {} · uncertain actions {}\nAssessments DQ {} / scavenger {} · score transfer {} · submission {}",
            display(&project["collected"]), display(&project["target"]), display(&project["verified"]), display(&project["unresolved"]),
            display(&project["active_star"]["name"]), project_stage(project),
            project["uncertain_actions"].as_array().map_or("unknown".into(), |actions| actions.len().to_string()),
            verification(&project["assessment"]["data_quality"]), verification(&project["assessment"]["scavenger_hunt"]),
            verification(&project["score_transfer_verified"]), verification(&project["submitted"])
        ));
    }
    frame.render_widget(
        Paragraph::new(text).block(Block::bordered().title(format!(
            "HabFly · run {} · event {}",
            app.run_id.as_deref().unwrap_or("—"),
            app.sequence.map_or("—".into(), |seq| seq.to_string())
        ))),
        area,
    );
}

fn draw_controls(frame: &mut Frame, app: &App, area: Rect) {
    let selected = action_target(&app.action).as_str();
    let controls = app.observation["controls"].as_array();
    let mut lines = Vec::new();
    if let Some(controls) = controls {
        for control in controls {
            // IDs are observation-local. Retain the last chosen control by a
            // unique visible label/role/surface, never by parsing an opaque ID.
            let same_control = |c: &&serde_json::Value| {
                !app.action_control["label"].is_null()
                    && ["label", "role", "surface"]
                        .iter()
                        .all(|key| c[*key] == app.action_control[*key])
            };
            let chosen = (selected.is_some() && control["id"].as_str() == selected)
                || (same_control(&control) && controls.iter().filter(same_control).count() == 1);
            let enabled = control["enabled"].as_bool() != Some(false);
            let marker = if chosen { "▶" } else { " " };
            let mut text = format!(
                "{marker} {} [{}] {}",
                display(&control["id"]),
                display(&control["role"]),
                display(&control["label"])
            );
            if !control["value"].is_null() {
                text.push_str(&format!(" = {}", display(&control["value"])));
            }
            if !enabled {
                text.push_str(" (disabled)");
            }
            let style = if chosen {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else if !enabled {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(text, style)));
            if let Some(options) = control["options"].as_array() {
                if !options.is_empty() {
                    if control["label"] == "Measurement or result" {
                        for option in options {
                            lines.push(Line::from(format!("    {}", option_text(app, option))));
                        }
                    } else {
                        lines.push(Line::from(format!(
                            "    options: {}",
                            display(&control["options"])
                        )));
                    }
                }
            }
        }
    }
    if lines.is_empty() {
        lines.push(Line::from(
            if app.state["task"] == "browser_transit_sampling" {
                "Sensor only: HOVER / DRAG / SCROLL; no answer controls"
            } else if app.observation["progress"]["task_completed"] == true {
                "No controls: task completed"
            } else {
                "Waiting for visible controls"
            },
        ));
    }
    let max_scroll = lines
        .len()
        .saturating_sub(area.height.saturating_sub(2) as usize)
        .min(u16::MAX as usize) as u16;
    frame.render_widget(
        Paragraph::new(lines)
            .scroll((app.controls_scroll.min(max_scroll), 0))
            .block(Block::bordered().title("Visible controls / available options")),
        area,
    );
}

fn draw_action(frame: &mut Frame, app: &App, area: Rect) {
    if app.action["surface"] == "chart" && app.state["task"] == "browser_transit_sampling" {
        let text = format!(
            "{} on visible chart · x {} · y {}\nDrag {} -> {} · wheel {}\nScripted sensing; no model confidence or answer write",
            display(&app.action["kind"]), display(&app.action["x_fraction"]),
            display(&app.action["y_fraction"]), display(&app.action["start_fraction"]),
            display(&app.action["end_fraction"]), display(&app.action["delta_y"])
        );
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(Block::bordered().title("Chart sensor action")),
            area,
        );
        return;
    }
    let color = &app.state["pending_browser_color"];
    if color.is_object() && !app.is_browser_project() {
        let measurement = &color["measurement"];
        let authorization = if app.state["replay"] == true {
            "RECORDED proposal; approval disabled"
        } else if app.state["browser_execution"] == "autonomous" {
            if app.paused {
                "PAUSED: no automatic selection; r resumes, y approves once, a aborts"
            } else {
                "AUTONOMOUS: next running tick selects; p pauses, a aborts"
            }
        } else {
            "y approves this color once; a aborts"
        };
        let text = format!("COLOR: {} -> {} (current: {})\nSource {}: {} {} {} ({})\n{}\nUncalibrated browser probabilities; no Save/score/submit",
            display(&color["selected_color"]), display(&color["destination"]), display(&color["current_value"]),
            display(&color["source_id"]), display(&measurement["kind"]), display(&measurement["value"]),
            display(&measurement["unit"]), display(&measurement["source"]), authorization);
        frame.render_widget(
            Paragraph::new(text).block(Block::bordered().title("Pending color selection")),
            area,
        );
        return;
    }
    let pending = &app.state["pending_browser_copy"];
    if pending.is_object() && !app.is_browser_project() {
        let authorization =
            if app.state["browser_execution"] == "autonomous" && app.state["replay"] != true {
                if app.paused {
                    "PAUSED: no automatic write; r resumes, y approves once, a aborts"
                } else {
                    "AUTONOMOUS: next running tick copies; p pauses, a aborts"
                }
            } else {
                "y approves this copy only; a aborts"
            };
        let text = format!("COPY: {} {} → {}\nCurrent value: {} · commit: Tab\n{}\nBrowser probabilities uncalibrated; no Save/score/submit",
            display(&pending["exact_value"]), display(&pending["unit"]),
            display(&pending["destination"]), display(&pending["current_value"]), authorization);
        frame.render_widget(
            Paragraph::new(text).block(Block::bordered().title("Pending browser write")),
            area,
        );
        return;
    }
    let text = format!(
        "{} → {} [{}] · value {}\nAction {}\nTarget {}\nSource {} · reward parts {}",
        display(&app.action["kind"]),
        display(&app.action_control["label"]),
        display(action_target(&app.action)),
        display(&app.action["value"]),
        confidence(&app.action, "action_confidence"),
        confidence(&app.action, "target_confidence"),
        display(
            app.action
                .get("action_source")
                .or_else(|| app.state.get("action_source"))
                .unwrap_or(&serde_json::Value::Null)
        ),
        display(&app.result["reward_components"])
    );
    frame.render_widget(
        Paragraph::new(text).block(Block::bordered().title("Proposed action / probabilities")),
        area,
    );
}

fn draw_neural(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered().title("Neural activity");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(8),
        Constraint::Min(0),
    ])
    .split(inner);
    let source = app.neural["source"]
        .as_str()
        .or_else(|| app.neural["activity_source"].as_str())
        .unwrap_or("unavailable");
    let source_text = if source == "untrained_observer"
        || app.neural["activity_source"] == "untrained_observer"
    {
        "untrained observer · observation only\nActions come from the scripted expert".to_string()
    } else {
        let pathway = app.neural["activity_pathway"]
            .as_str()
            .unwrap_or("workflow");
        format!("Activity source: {source}\nPath: {pathway}")
    };
    frame.render_widget(
        Paragraph::new(source_text).style(Style::default().fg(Color::Yellow)),
        rows[0],
    );
    let sparks = Layout::vertical([Constraint::Length(2); 4]).split(rows[1]);
    for (index, group) in GROUPS.iter().enumerate() {
        let values: Vec<u64> = app.histories[*group].iter().copied().collect();
        let label = format!(
            "{group} · mean {} · max {}",
            metric(population_mean(&app.neural, group)),
            metric(&app.neural["groups"][group]["max"])
        );
        frame.render_widget(
            Sparkline::default()
                .block(Block::default().title(label))
                .data(&values)
                .style(Style::default().fg(Color::Cyan)),
            sparks[index],
        );
    }
    let mut lines = vec![Line::from("Body ID · activity · type / class · polarity")];
    if let Some(neurons) = app.neural["top_neurons"].as_array() {
        for neuron in neurons {
            lines.push(Line::from(format!(
                "{} · {} · {} / {} · {}",
                display(&neuron["body_id"]),
                metric(&neuron["activity"]),
                display(&neuron["type"]),
                display(&neuron["class"]),
                display(&neuron["polarity"])
            )));
        }
    }
    frame.render_widget(Paragraph::new(lines), rows[2]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::parse_event;
    use ratatui::{backend::TestBackend, Terminal};

    #[test]
    fn shallow_stops_show_explanation_without_claiming_planet_or_task_outcome() {
        for code in [
            "shallow_probe_exact_two_overview_hints_required",
            "shallow_probe_three_overview_hints_required",
        ] {
            let app = App {
                state: serde_json::json!({"task":"browser_project", "status":"stopped",
                    "failure_reason":code, "task_completed":false,
                    "project_completed":false, "submitted":false}),
                ..App::default()
            };
            let text = project_runtime_text(&app);
            assert!(text.contains(&format!("Failure / stop: {code}")));
            assert!(
                text.contains("separate possible dips needed to estimate the time between them")
            );
            assert!(text.contains("Evidence remains unresolved"));
            assert!(text.contains("a planet is neither confirmed nor ruled out"));
            assert!(text.contains("No automatic retry"));
            assert!(text.contains("Task completed: false · project completed: false"));
            assert!(footer(&app).contains("No steps, approvals or retries"));
        }
        for code in [serde_json::Value::Null, serde_json::json!("unrelated_stop")] {
            let app = App {
                state: serde_json::json!({"task":"browser_project", "status":"stopped",
                    "failure_reason":code}),
                ..App::default()
            };
            assert!(!project_runtime_text(&app).contains("separate possible dips"));
        }
    }

    #[test]
    fn finalization_fixture_displays_scoring_submission_and_unknown_as_separate_receipts() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/project-finalization.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
            if app.sequence == Some(0) {
                continue;
            }
            let text = project_runtime_text(&app);
            assert!(text.contains("historical star campaign remains closed"));
            assert!(text.contains("scoring 600 seconds / 46 advances"));
            assert!(text.contains("submission 180 seconds / 4 advances"));
            assert!(text.contains(
                "Parent submitted false · task completed false · project completed false"
            ));
            assert!(!text.contains("LastFixture"));
            assert!(!footer(&app).contains("y approve"));
            match app.sequence.unwrap() {
                1 => {
                    assert!(text.contains("initializes its owner only"));
                    assert!(text.contains("assessment enabled true · submission enabled true"));
                    assert!(text.contains("owner authorization not yet reported"));
                    assert!(footer(&app).contains("n one boundary"));
                }
                3 => {
                    assert!(text.contains("assessment outcome uncertain true"));
                    assert!(footer(&app).contains("p/space pause"));
                }
                6 | 7 => {
                    assert!(text.contains("Submit click returned true"));
                    assert!(text.contains("NOT a grounded submission"));
                }
                8 => {
                    assert!(text.contains("unknown/pending"));
                    assert!(footer(&app).contains("No steps, approvals or retries"));
                }
                9 => {
                    assert!(footer(&app).contains("REPLAY FINISHED"));
                    assert!(runtime_status(&app).contains("recorded handoff"));
                }
                _ => (),
            }
            for (width, height) in [(65, 18), (120, 40)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| draw(frame, &app)).unwrap();
            }
        }
        app.focused_panel = 3;
        let mut terminal = Terminal::new(TestBackend::new(180, 50)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("uncertain actions") || text.contains("Uncertain actions"));
        assert!(text.contains("submission"));
    }

    #[test]
    fn replay_end_labels_playback_not_task_success_and_configured_checkpoint() {
        let app = App {
            state: serde_json::json!({"runtime_task":"browser_project","status":"completed",
                "recorded_status":"running","replay":true,"replay_finished":true,
                "current_star":{"star":"Crabiltia"},"browser_phase":"active",
                "policy_stage":"planet_window","checkpoint":"stellar.pt",
                "task_completed":false,"project_completed":false}),
            ..App::default()
        };
        assert_eq!(runtime_status(&app), "Playback finished · recorded running");
        assert!(footer(&app).contains("not task success"));
        assert!(footer(&app).contains("No browser") || footer(&app).contains("no browser"));
        let mut terminal = Terminal::new(TestBackend::new(180, 12)).unwrap();
        terminal
            .draw(|frame| draw_status(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("configured stellar checkpoint stellar.pt"));
        assert!(text.contains("task completed false / project completed false"));
        assert!(!text.contains("task completed true"));
    }

    #[test]
    fn neural_source_supports_current_and_legacy_events_without_inventing_activity() {
        for (neural, expected) in [
            (
                serde_json::json!({"source":"recurrent_hidden_state_rms"}),
                "Activity source: recurrent_hidden_state_rms",
            ),
            (
                serde_json::json!({"activity_source":"checkpoint"}),
                "Activity source: checkpoint",
            ),
            (
                serde_json::json!({"source":null,"activity_source":"checkpoint"}),
                "Activity source: checkpoint",
            ),
            (serde_json::json!({}), "Activity source: unavailable"),
            (
                serde_json::json!({"source":"recurrent_hidden_state_rms","activity_source":"untrained_observer"}),
                "untrained observer · observation only",
            ),
        ] {
            let app = App {
                neural,
                ..App::default()
            };
            let mut terminal = Terminal::new(TestBackend::new(100, 16)).unwrap();
            terminal
                .draw(|frame| draw_neural(frame, &app, frame.area()))
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(text.contains(expected), "{text}");
            assert!(app.neural["top_neurons"].is_null());
        }
    }

    #[test]
    fn project_lifecycle_footer_never_offers_legacy_approvals_or_guessed_handoffs() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/browser-project.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
            let text = footer(&app);
            assert!(!text.contains("y approve"));
            assert!(!text.contains("b ready"));
            match app.sequence.unwrap() {
                1..=5 | 7 => {
                    assert!(text.contains("n one bounded stage"));
                    assert!(text.contains("between bounded calls"));
                }
                6 => {
                    assert!(text.contains("CLASS EVIDENCE"));
                    assert!(project_runtime_text(&app).contains("class_source"));
                    assert!(!text.contains("r/space resume"));
                }
                8 => {
                    assert!(text.contains("PROJECT RUNNING"));
                    assert!(!text.contains("n one bounded stage"));
                    assert!(project_runtime_text(&app).contains("Task completed: false"));
                }
                9 => {
                    assert!(text.contains("INVENTORY EVIDENCE"));
                    assert!(project_runtime_text(&app).contains("inventory_sha256"));
                }
                10 => {
                    assert!(text.contains("task true / project false"));
                    assert!(text.contains("No steps, approvals or retries"));
                }
                _ => (),
            }
        }
        app.state["replay"] = true.into();
        assert!(footer(&app).contains("RECORDED REPLAY"));
        assert!(footer(&app).contains("No browser is connected"));
    }

    #[test]
    fn project_reference_setup_details_keep_classification_separate_from_learning() {
        let app = App {
            paused: true,
            state: serde_json::json!({"runtime_task":"browser_project", "status":"paused",
                "browser_phase":"setting_reference_class","task_completed":false,"project_completed":false,
                "class_setup":{"phase":"class_pending","selected_class":"main_sequence",
                    "lifetime_prefix":"Ga","setup_verified":false}}),
            ..App::default()
        };
        let text = project_runtime_text(&app);
        assert!(text.contains("class main_sequence"));
        assert!(text.contains("setup verified false"));
        assert!(text.contains("supplied, not learned or inferred"));
        assert!(footer(&app).contains("n one bounded stage"));
    }

    #[test]
    fn automatic_reference_stages_are_scheduling_not_manual_approvals_or_learning() {
        for phase in [
            "awaiting_class_source",
            "awaiting_planet_class",
            "awaiting_gases",
            "awaiting_habitability",
        ] {
            let mut app = App {
                paused: true,
                state: serde_json::json!({"runtime_task":"browser_project", "status":"paused", "browser_phase":phase,
                    "autonomous_decisions_enabled":true, "task_completed":false, "project_completed":false,
                    "autonomous_decision":{"star":"Fixture","source":"reference_rules","decision":{"class":"main_sequence"}}}),
                ..App::default()
            };
            assert!(footer(&app).contains("AUTOMATIC REFERENCE PAUSED"));
            assert!(footer(&app).contains("n one local decision stage"));
            assert!(!footer(&app).contains("External runtime handoff"));
            let text = project_runtime_text(&app);
            assert!(text.contains("NOT learned classification"));
            assert!(text.contains("not a success receipt"));
            assert!(text.contains("reference_rules"));
            app.paused = false;
            app.state["status"] = "running".into();
            assert!(footer(&app).contains("AUTOMATIC REFERENCE RUNNING"));
            app.state["status"] = "stopped".into();
            assert!(footer(&app).contains("No steps"));
            app.state["replay"] = true.into();
            assert!(footer(&app).contains("No browser is connected"));
        }
    }

    #[test]
    fn supplied_input_transfer_label_is_true_only_and_not_a_success_claim() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"running",
                "browser_phase":"active", "task_completed":false,"project_completed":false}),
            ..App::default()
        };
        let legacy = project_runtime_text(&app);
        for value in [
            serde_json::json!(false),
            serde_json::Value::Null,
            serde_json::json!(1),
            serde_json::json!("true"),
        ] {
            app.state["supplied_stellar_inputs_enabled"] = value;
            assert_eq!(project_runtime_text(&app), legacy);
        }
        app.state["supplied_stellar_inputs_enabled"] = true.into();
        let text = project_runtime_text(&app);
        assert!(text.contains("SUPPLIED-INPUT TRANSFER enabled for non-main stars"));
        assert!(text.contains("visible stellar mass/radius, not learned stellar estimates"));
        assert!(text.contains("versioned class-omitting inference view"));
        assert!(text.contains("actual class remains in tool observations and logs"));
        assert!(text.contains("Main-sequence stages retain their original path"));
        assert!(text.contains("uncalibrated and scientific correctness is unverified"));
        assert!(text.contains("Task completed: false · project completed: false"));
        app.state["replay"] = true.into();
        assert!(project_runtime_text(&app).contains("SUPPLIED-INPUT TRANSFER"));
    }

    #[test]
    fn baseline_edge_reference_label_is_true_only_and_not_a_success_claim() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"running",
                "browser_phase":"active", "task_completed":false,"project_completed":false}),
            ..App::default()
        };
        let legacy = project_runtime_text(&app);
        for value in [
            serde_json::json!(false),
            serde_json::Value::Null,
            serde_json::json!(1),
            serde_json::json!("true"),
        ] {
            app.state["baseline_edge_reference_enabled"] = value;
            assert_eq!(project_runtime_text(&app), legacy);
        }
        app.state["baseline_edge_reference_enabled"] = true.into();
        let text = project_runtime_text(&app);
        assert!(text.contains("BASELINE-RENDERING COMPATIBILITY explicitly enabled"));
        assert!(text.contains("guarded original-pixel evidence only"));
        assert!(text.contains("5,000-day approximate No assumption"));
        assert!(text.contains("NOT proven absence or learned perception"));
        assert!(text.contains("shallow or long-period planets may be missed"));
        assert!(text.contains("does not claim task or project completion"));
        assert!(text.contains("Task completed: false · project completed: false"));
        app.state["replay"] = true.into();
        assert!(project_runtime_text(&app).contains("BASELINE-RENDERING COMPATIBILITY"));
    }

    #[test]
    fn failed_save_display_distinguishes_uncertain_predispatch_and_unknown() {
        for (value, expected) in [
            (
                serde_json::json!(true),
                "Save may have occurred; outcome uncertain",
            ),
            (serde_json::json!(false), "recorded pre-dispatch stop"),
            (serde_json::Value::Null, "outcome unknown"),
        ] {
            let attempted = value == true;
            let app = App {
                state: serde_json::json!({"runtime_task":"browser_project", "status":"stopped",
                    "browser_phase":"stopped", "task_completed":false, "project_completed":false,
                    "save_outcome_uncertain":value,
                    "save_outcome":{"evidence_status":"recorded_stop", "reservation_retained":true,
                        "dispatch_recorded":attempted,"acknowledgement_recorded":attempted,
                        "final_readback_verified":false,"canonical_receipt":false}}),
                ..App::default()
            };
            let text = project_runtime_text(&app);
            assert!(text.contains(expected));
            assert!(text.contains("No retry"));
            assert!(text.contains("Noncanonical child diagnostic only"));
            assert!(text.contains("canonical project counts are not changed"));
            assert!(text.contains("Task completed: false · project completed: false"));
            assert_eq!(text.contains("final readback NOT verified"), attempted);
            assert_eq!(
                text.contains("not verified persistence or task completion"),
                attempted
            );
        }
    }

    #[test]
    fn legacy_and_cleared_save_payloads_do_not_display_a_failed_save_warning() {
        let app = App {
            state: serde_json::json!({"runtime_task":"browser_project","status":"stopped",
                "browser_phase":"stopped","task_completed":false,"project_completed":false}),
            ..App::default()
        };
        assert!(!project_runtime_text(&app).contains("FAILED SAVE"));
        assert!(!project_runtime_text(&app).contains("acknowledgement recorded"));
    }

    #[test]
    fn project_inventory_footer_permits_only_bounded_scheduling() {
        for phase in [
            "inventory_initializing",
            "inventory_active",
            "inventory_import",
        ] {
            let mut app = App {
                paused: true,
                state: serde_json::json!({"runtime_task":"browser_project", "status":"paused",
                    "browser_phase":phase,"task_completed":false,"project_completed":false}),
                ..App::default()
            };
            let text = footer(&app);
            assert!(text.contains("n one bounded stage"));
            assert!(!text.contains("External runtime handoff"));
            assert!(!text.contains("y approve"));
            app.state["status"] = "running".into();
            app.paused = false;
            assert!(footer(&app).contains("PROJECT RUNNING"));
        }
    }

    #[test]
    fn project_planet_handoff_and_verification_remain_reference_assisted() {
        let mut app = App {
            paused: true,
            state: serde_json::json!({"runtime_task":"browser_project", "status":"paused",
                "browser_phase":"awaiting_planet_class","task_completed":false,"project_completed":false,
                "project_owner":{"planet_class_decision":{"name":"ice_giant","classification_learned":false},
                    "terrestrial_component":{"phase":"awaiting_gases"},
                    "positive_component":{"phase":"save","status":"ready"}}}),
            ..App::default()
        };
        assert!(footer(&app).contains("PLANET CLASS reference decision required"));
        assert!(footer(&app).contains("guidance only"));
        let text = project_runtime_text(&app);
        assert!(text.contains("Explicit planet class: ice_giant"));
        assert!(text.contains("Save alone does not complete a task"));
        assert!(text.contains("Task completed: false"));
        assert!(text.contains("Terrestrial workflow: awaiting_gases"));
        for phase in [
            "selecting_planet_class",
            "positive_initializing",
            "positive_active",
            "terrestrial_initializing",
            "terrestrial_active",
        ] {
            app.state["browser_phase"] = phase.into();
            assert!(footer(&app).contains("n one bounded stage"));
        }
        for phase in ["awaiting_gases", "awaiting_habitability"] {
            app.state["browser_phase"] = phase.into();
            assert!(footer(&app).contains("reference decision required"));
            assert!(footer(&app).contains("guidance only"));
        }
    }

    #[test]
    fn partial_chart_wait_is_not_a_planet_decision_or_completion() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"running",
                "browser_phase":"active","task_completed":false,"project_completed":false,
                "project_owner":{"star_component":{"component":{
                    "mode":"bounded_planet_window","observation_limit_days":5000,
                    "phase":"observing","polls":2,"max_polls":60,
                    "window_status":"insufficient_visual_evidence",
                    "waiting_for_clean_partial_chart":true}}}}),
            ..App::default()
        };
        let text = project_runtime_text(&app);
        assert!(text.contains("Visible chart window: 5000 days"));
        assert!(text.contains("polls 2/60"));
        assert!(text.contains("Waiting for a clean partial-chart capture"));
        assert!(text.contains("no planet decision authorized"));
        assert!(text.contains("Task completed: false"));
        app.state["project_owner"]["star_component"]["component"]["phase"] = "stopped".into();
        assert!(!project_runtime_text(&app).contains("Waiting for a clean partial-chart capture"));
        app.state["project_owner"]["star_component"]["component"] = serde_json::Value::Null;
        assert!(!project_runtime_text(&app).contains("Visible chart window:"));
    }

    #[test]
    fn autosave_label_never_treats_strategy_or_elapsed_time_as_verified_answers() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"running",
                "browser_phase":"active", "task_completed":false,"project_completed":false}),
            ..App::default()
        };
        let legacy = project_runtime_text(&app);
        for value in [
            serde_json::Value::Null,
            "explicit".into(),
            "unknown".into(),
            true.into(),
        ] {
            app.state["save_strategy"] = value;
            assert_eq!(project_runtime_text(&app), legacy);
        }
        app.state["save_strategy"] = "autosave".into();
        app.state["elapsed_seconds"] = 120.into();
        let text = project_runtime_text(&app);
        assert!(text.contains("Autosave assumed; persistence unverified"));
        assert!(text.contains("readback are still required before a task is counted"));
        assert!(text.contains("No explicit Save click, banner, or elapsed timer"));
        assert!(!text.contains("visible answers verified; persistence unverified"));
        assert!(text.contains("Task completed: false · project completed: false"));
        app.state["project_progress"] = serde_json::json!({"schema_version":1,
            "active_star":{"task_completed":true}});
        let text = project_runtime_text(&app);
        assert!(text.contains("Autosave assumed; visible answers verified; persistence unverified"));
        assert!(text.contains("This strategy alone does not complete a task or project"));
        assert!(!text.contains("Saved"));
    }

    #[test]
    fn single_event_shortcut_requires_recorded_policy_and_actual_application_fields() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"running",
                "browser_phase":"active", "task_completed":false,"project_completed":false,
                "single_event_reference_enabled":true,
                "project_owner":{"star_component":{"component":{
                    "mode":"bounded_planet_window", "observation_limit_days":5000,
                    "phase":"observing", "window_status":"assume_no_planet"}}}}),
            ..App::default()
        };
        let legacy = project_runtime_text(&app);
        assert!(!legacy.contains("Single-event No shortcut"));
        app.state["project_owner"]["star_component"]["component"]["policy"] =
            serde_json::json!({"version":"user_approved_5000_day_single_event_no_planet_v1"});
        let enabled = project_runtime_text(&app);
        assert!(enabled.contains("Single-event No shortcut is enabled"));
        assert!(enabled.contains("not confirmed absence or learned perception"));
        assert!(!enabled.contains("Recorded shortcut:"));
        let applied = serde_json::json!({"status":"assume_no_planet",
            "reason":"user_approved_single_event_shortcut",
            "approximation":"user_approved_single_event_no_planet_shortcut",
            "visible_candidate_events":1,"possible_planet_ignored":true});
        app.state["project_owner"]["star_component"]["component"]["analysis"] = applied.clone();
        let before = app.state.clone();
        let text = project_runtime_text(&app);
        assert!(text.contains("one possible dip was deliberately ignored for an assumed No"));
        assert!(text.contains("A planet is not ruled out; no repeat interval was measured"));
        assert!(text.contains("Task completed: false · project completed: false"));
        assert_eq!(app.state, before);
        for (key, value) in [
            ("reason", "older_reason".into()),
            ("approximation", serde_json::Value::Null),
            ("possible_planet_ignored", false.into()),
            ("possible_planet_ignored", 1.into()),
            ("visible_candidate_events", 1.0.into()),
            ("visible_candidate_events", true.into()),
            ("status", "still_collecting".into()),
        ] {
            app.state["project_owner"]["star_component"]["component"]["analysis"] = applied.clone();
            app.state["project_owner"]["star_component"]["component"]["analysis"][key] = value;
            assert!(!project_runtime_text(&app).contains("Recorded shortcut:"));
        }
        app.state["project_owner"]["star_component"]["component"]["policy"]["version"] =
            "older_policy".into();
        assert_eq!(project_runtime_text(&app), legacy);
    }

    #[test]
    fn positive_endpoint_wait_is_explicit_true_only_and_not_completion() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"running",
                "browser_phase":"active","task_completed":false,"project_completed":false,
                "project_owner":{"star_component":{"component":{
                    "mode":"bounded_planet_window","observation_limit_days":5000,
                    "phase":"observing","polls":3,"max_polls":60,
                    "window_status":"dip_observed"}}}}),
            ..App::default()
        };
        let legacy = project_runtime_text(&app);
        app.state["project_owner"]["star_component"]["component"]["scheduling_rule"] =
            "complete_rendered_window_before_positive_handoff_v1".into();
        app.state["project_owner"]["star_component"]["component"]["prior_dip_observed"] =
            true.into();
        assert_eq!(project_runtime_text(&app), legacy);
        for value in [
            serde_json::Value::Null,
            false.into(),
            1.into(),
            "true".into(),
        ] {
            app.state["project_owner"]["star_component"]["component"]
                ["waiting_for_positive_endpoint"] = value;
            assert_eq!(project_runtime_text(&app), legacy);
        }
        app.state["project_owner"]["star_component"]["component"]
            ["waiting_for_positive_endpoint"] = true.into();
        let before = app.state.clone();
        let text = project_runtime_text(&app);
        assert!(text.contains(
            "Dip previously observed; waiting for the complete rendered 5,000-day window"
        ));
        assert!(text.contains("No planet answer or task completion is authorized by this wait"));
        assert!(text.contains("Task completed: false · project completed: false"));
        assert_eq!(app.state, before);
        for (key, value) in [
            ("scheduling_rule", "unrecognized_rule".into()),
            ("prior_dip_observed", false.into()),
            ("prior_dip_observed", 1.into()),
            ("phase", "stopped".into()),
            ("mode", "another_component".into()),
        ] {
            app.state = before.clone();
            app.state["project_owner"]["star_component"]["component"][key] = value;
            assert!(!project_runtime_text(&app).contains("Dip previously observed; waiting"));
        }
    }

    #[test]
    fn explicit_uncapped_campaign_label_preserves_finite_wording_and_false_completion() {
        let mut app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"paused",
                "project_campaign":true, "browser_phase":"awaiting_class_source",
                "task_completed":false,"project_completed":false,
                "campaign":{"target_stars":30,"max_seconds":10800}}),
            ..App::default()
        };
        assert!(project_runtime_text(&app).contains("fixed budget 10800 seconds"));
        app.state["campaign"]["max_seconds"] = "uncapped".into();
        let text = project_runtime_text(&app);
        assert!(text.contains("no overall timer"));
        assert!(text.contains("per-star/action limits and scoring guards remain"));
        assert!(!text.contains("fixed budget uncapped"));
        assert!(text.contains("Task completed: false · project completed: false"));
        app.state["campaign"] = serde_json::Value::Null;
        app.state["campaign_max_seconds"] = "uncapped".into();
        assert!(project_runtime_text(&app).contains("no overall timer"));
    }

    #[test]
    fn campaign_lifecycle_shows_current_star_and_keeps_completion_gates_separate() {
        let mut app = App {
            paused: true,
            state: serde_json::json!({"runtime_task":"browser_project", "status":"paused",
                "project_campaign":true, "browser_phase":"next_star_active",
                "initial_star":{"star":"Original"}, "current_star":{"star":"Current","ordinal":2},
                "project_owner":null,"task_completed":false,"project_completed":false,
                "target_workflows_verified":false,
                "campaign":{"phase":"next_star_active","verified_stars":2,"target_stars":3,
                    "max_seconds":5400,"next_star":{"phase":"picking"}}}),
            ..App::default()
        };
        let text = project_runtime_text(&app);
        assert!(text.contains("bounded multi-star campaign"));
        assert!(text.contains("Visible star Current"));
        assert!(!text.contains("Visible star Original"));
        assert!(text.contains("verified 2/3"));
        assert!(text.contains("fixed budget 5400 seconds"));
        assert!(text.contains("Next-star transition: picking"));
        assert!(footer(&app).contains("n one bounded stage"));
        app.state["browser_phase"] = "awaiting_assessment".into();
        app.state["status"] = "handoff".into();
        app.state["target_workflows_verified"] = true.into();
        let text = project_runtime_text(&app);
        assert!(text.contains("Workflow target verified: true — not assessment"));
        assert!(text.contains("Task completed: false · project completed: false"));
        assert!(!footer(&app).contains("n one bounded stage"));
    }

    #[test]
    fn project_inventory_details_do_not_turn_collection_into_task_success() {
        let app = App {
            state: serde_json::json!({"runtime_task":"browser_project", "status":"paused",
                "browser_phase":"inventory_import","task_completed":false,"project_completed":false,
                "project_owner":{"inventory_component":{"phase":"inventory_verified",
                    "navigation_clicks":2,"collection_count_verified":true}}}),
            ..App::default()
        };
        let text = project_runtime_text(&app);
        assert!(text.contains("Collection verification: inventory_verified"));
        assert!(text.contains("count verified true"));
        assert!(text.contains("task completion still requires workflow import"));
        assert!(text.contains("Task completed: false"));
    }

    #[test]
    fn project_lifecycle_details_distinguish_visible_star_journal_and_artifacts() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/browser-project.jsonl")
            .lines()
            .take(9)
        {
            app.apply(parse_event(line).unwrap());
        }
        app.state["project_progress"] = serde_json::json!({"schema_version":1,"collected":0,
            "verified":0,"target":30,"unresolved":0,"active_star":null});
        app.focused_panel = 3;
        let mut terminal = Terminal::new(TestBackend::new(170, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "REFERENCE-ASSISTED",
            "one active star bridge",
            "Visible star FIXTURE",
            "not a collection receipt",
            "component numeric_copy",
            "Task completed: false",
            "project completed: false",
            "collected 0/30",
            "NOT prove project completion",
            "explicit reference decisions",
            "star/numeric",
            "experiments/traces/fixture-project-runtime.jsonl",
            "experiments/fixture-project-runtime/events.jsonl",
            "Assessment enabled: false",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
        app.focused_panel = 0;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("phase active"));
        assert!(text.contains("Project 0/30 collected"));
    }

    #[test]
    fn project_stale_legacy_approval_fields_and_small_views_remain_safe() {
        let mut app = App {
            paused: true,
            state: serde_json::json!({"task":"browser_project",
            "status":"paused","browser_phase":"active","pending_browser_copy":{"exact_value":"42"},
            "pending_browser_color":{"selected_color":"UV"}}),
            ..App::default()
        };
        for (width, height) in [(64, 17), (65, 18), (80, 24), (150, 45)] {
            for focus in 0..=3 {
                app.focused_panel = focus;
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| draw(frame, &app)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(!text.contains("y approves"));
                assert!(!text.contains("Pending browser write"));
                assert!(!text.contains("Pending color selection"));
            }
        }
        app.state["status"] = "handoff".into();
        app.state["browser_phase"] = "classification_required".into();
        assert!(footer(&app).contains("classification_required"));
        assert!(!footer(&app).contains("n one"));
        app.state["status"] = "paused".into();
        app.state["browser_phase"] = serde_json::Value::Null;
        assert!(footer(&app).contains("lifecycle unavailable"));
    }

    #[test]
    fn automatic_setup_does_not_offer_model_steps_or_copy_approval() {
        let app = App {
            state: serde_json::json!({"browser_phase":"setting_up",
                "browser_guidance":"Scripted setup: selecting_visible_star"}),
            ..App::default()
        };
        let text = footer(&app);
        assert!(text.contains("selecting_visible_star"));
        assert!(text.contains("learned policy remains paused"));
        assert!(text.contains("a abort"));
        assert!(!text.contains("y approve"));
        assert!(!text.contains("n step"));
    }

    #[test]
    fn stopped_setup_keeps_inspection_guidance_without_action_keys() {
        let app = App {
            state: serde_json::json!({"browser_phase":"finished", "browser_held_open":true,
                "browser_guidance":"STOP: setup_capture_frame_count_mismatch:widgets"}),
            ..App::default()
        };
        let text = footer(&app);
        assert!(text.contains("frame_count_mismatch:widgets"));
        assert!(text.contains("q close browser"));
        assert!(!text.contains("y approve"));
        assert!(!text.contains("n step"));
    }

    #[test]
    fn finished_browser_run_does_not_offer_step_or_approval() {
        let app = App {
            state: serde_json::json!({"browser_phase":"finished", "browser_held_open":false,
                "browser_guidance":"STOP: stale_numeric_observation"}),
            ..App::default()
        };
        let text = footer(&app);
        assert!(text.contains("stale_numeric_observation"));
        assert!(text.contains("Run ended"));
        assert!(text.contains("q quit"));
        assert!(!text.contains("y approve"));
        assert!(!text.contains("n step"));
    }

    #[test]
    fn autonomous_footer_distinguishes_running_paused_and_replay() {
        let mut app = App {
            state: serde_json::json!({"browser_execution":"autonomous",
                "browser_phase":"setting_up", "browser_guidance":"Scripted setup"}),
            ..App::default()
        };
        assert!(footer(&app).contains("RUNNING"));
        assert!(footer(&app).contains("p/space pause"));
        assert!(!footer(&app).contains("learned policy remains paused"));
        app.paused = true;
        app.state["browser_phase"] = "awaiting_copy".into();
        assert!(footer(&app).contains("no automatic steps or writes"));
        app.state["replay"] = true.into();
        assert!(footer(&app).contains("No browser is connected"));
        assert!(!footer(&app).contains("AUTONOMOUS"));
    }

    #[test]
    fn four_field_footer_exposes_only_the_current_phase() {
        let mut app = App {
            state: serde_json::json!({"browser_task":"four_field",
                "browser_phase":"awaiting_handoff", "browser_guidance":"Numeric readbacks verified"}),
            ..App::default()
        };
        assert!(footer(&app).contains("n recheck star and switch policy (no write)"));
        assert!(!footer(&app).contains("y approve"));
        app.state["browser_phase"] = "awaiting_copy".into();
        assert!(footer(&app).contains("y approve one numeric copy"));
        app.state["browser_phase"] = "awaiting_color".into();
        assert!(footer(&app).contains("y approve one color"));
        assert!(!footer(&app).contains("numeric copy"));
        app.state["browser_phase"] = "finished".into();
        assert!(!footer(&app).contains("y approve"));
        app.state["replay"] = true.into();
        assert!(footer(&app).contains("No browser is connected"));
    }

    #[test]
    fn autonomous_four_field_footer_names_all_writes_and_pause() {
        let mut app = App {
            state: serde_json::json!({"browser_task":"four_field", "browser_execution":"autonomous",
                "browser_phase":"awaiting_handoff", "browser_guidance":"Next tick rechecks star"}),
            ..App::default()
        };
        assert!(footer(&app).contains("three numeric copies + one color selection"));
        assert!(!footer(&app).contains("y approve"));
        app.paused = true;
        assert!(footer(&app).contains("no automatic writes or handoff"));
        assert!(footer(&app).contains("r/space resume"));
    }

    #[test]
    fn autonomous_color_names_its_one_write_limit_and_pending_authorization() {
        let mut app = App {
            state: serde_json::json!({"browser_task":"color", "browser_execution":"autonomous",
                "browser_phase":"awaiting_color", "pending_browser_color":{"selected_color":"UV"}}),
            ..App::default()
        };
        assert!(footer(&app).contains("one automatic color selection"));
        assert!(!footer(&app).contains("three"));
        let mut terminal = Terminal::new(TestBackend::new(120, 8)).unwrap();
        terminal
            .draw(|frame| draw_action(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("next running tick selects"));
        app.paused = true;
        assert!(footer(&app).contains("PAUSED: no automatic selection"));
        app.state["browser_execution"] = "supervised".into();
        assert!(footer(&app).contains("y approve one color"));
        assert!(!footer(&app).contains("n step"));
    }

    #[test]
    fn pending_color_approval_is_distinct_and_disabled_in_replay() {
        let mut app = App {
            state: serde_json::json!({"browser_task":"color", "browser_phase":"awaiting_color",
                "browser_execution":"supervised", "pending_browser_color":{
                    "selected_color":"UV", "destination":"Peak wavelength color", "current_value":null,
                    "source_id":"browser_wavelength", "measurement":{"kind":"wavelength",
                        "value":370, "unit":"nm", "source":"current star"}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(120, 8)).unwrap();
        terminal
            .draw(|frame| draw_action(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "COLOR: UV",
            "370 nm",
            "current star",
            "y approves this color once",
            "no Save/score/submit",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(!text.contains("Tab"));
        assert!(footer(&app).contains("y approve one color"));
        assert!(!footer(&app).contains("copy"));
        app.state["replay"] = true.into();
        terminal
            .draw(|frame| draw_action(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("approval disabled"));
        assert!(!text.contains("y approves"));
        assert!(!footer(&app).contains("y approve"));
    }

    #[test]
    fn pending_browser_copy_shows_exact_value_and_separate_approval() {
        let mut app = App {
            state: serde_json::json!({"browser_phase":"awaiting_copy",
                "browser_guidance":"Review before writing",
                "pending_browser_copy":{"exact_value":"0.0001445975820837288", "unit":"Lsun",
                    "destination":"luminosity", "current_value":"0"}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(120, 10)).unwrap();
        terminal
            .draw(|frame| draw_action(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "0.0001445975820837288",
            "luminosity",
            "Tab",
            "y approves",
            "uncalibrated",
            "no Save/score/submit",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(footer(&app).contains("b ready"));
        assert!(footer(&app).contains("y approve one copy"));
        app.state["replay"] = serde_json::json!(true);
        assert!(footer(&app).contains("RECORDED REPLAY"));
        assert!(!footer(&app).contains("y approve"));
    }

    #[test]
    fn golden_render_shows_replay_evidence_and_expert_boundary() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let buffer = terminal.backend().buffer();
        let text: String = buffer.content.iter().map(|cell| cell.symbol()).collect();
        for expected in [
            "Analyze star K-12",
            "CLICK",
            "o1:c1",
            "1.5",
            "untrained observer",
            "observation only",
            "42",
            "fixture warning",
            "expert",
        ] {
            assert!(text.contains(expected), "Missing {expected} from render");
        }
    }

    #[test]
    fn approximate_reference_fixture_renders_pixel_bounds_not_confidence_or_learning() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/reference-measurements.jsonl")
            .lines()
            .take(4)
        {
            app.apply(parse_event(line).unwrap());
        }
        let mut terminal = Terminal::new(TestBackend::new(150, 30)).unwrap();
        terminal
            .draw(|frame| draw_observation(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "APPROXIMATE reference raster",
            "star Fixture",
            "fixed 5000-day window",
            "Period ≈ 300 days",
            "pixel bounds [295, 305]",
            "Brightness drop ≈ 2 percent",
            "pixel bounds [1.5, 2.5]",
            "Line shift (reference): 0.0001 nm",
            "NOT model confidence",
            "NOT learned perception",
            "NOT scientific verification",
            "NOT training labels",
            "No task-completion or submission claim",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(app.summary["task_completed"].is_null());
        for (width, height) in [(64, 17), (65, 18), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &app)).unwrap();
        }
    }

    #[test]
    fn malformed_reference_render_removes_prior_values_and_legacy_is_unchanged() {
        let mut app = App::default();
        let mut event = parse_event(
            include_str!("../tests/fixtures/reference-measurements.jsonl")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap();
        app.apply(event.clone());
        event.payload["reference_measurements"]["period_days"]["upper"] = serde_json::json!(1);
        app.apply(event);
        let text = reference_measurements_text(&app);
        assert!(text.contains("Reference measurements unavailable"));
        assert!(!text.contains("300") && !text.contains("0.0001"));
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        assert!(reference_measurements_text(&app).is_empty());
    }

    #[test]
    fn tooltip_reference_renders_sampled_depth_without_physical_bounds_or_confidence() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/tooltip-reference-measurements.jsonl")
            .lines()
            .take(4)
        {
            app.apply(parse_event(line).unwrap());
        }
        let mut terminal = Terminal::new(TestBackend::new(150, 30)).unwrap();
        terminal
            .draw(|frame| draw_observation(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "APPROXIMATE visible-tooltip reference",
            "star Fixture",
            "fixed 5000-day window",
            "Observed recurrence ≈ 1000 days",
            "bracket compatibility (999, 1001) days (open)",
            "Maximum sampled decline: 0.762 percent",
            "physical depth bounds unavailable",
            "NOT physical uncertainty or model confidence",
            "NOT a verified physical period",
            "resolved transit minimum",
            "missed/aliased events remain possible",
            "NOT learned perception",
            "NOT scientific verification",
            "NOT training labels",
            "No task-completion or submission claim",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(!text.contains("pixel bounds"));
        for (width, height) in [(64, 17), (65, 18), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &app)).unwrap();
        }
    }

    #[test]
    fn two_tooltip_reference_labels_single_interval_without_recurrence_or_success_claims() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/two-tooltip-reference-measurements.jsonl")
            .lines()
            .take(4)
        {
            app.apply(parse_event(line).unwrap());
        }
        let mut terminal = Terminal::new(TestBackend::new(160, 35)).unwrap();
        terminal
            .draw(|frame| draw_observation(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "two-event estimate / single interval",
            "fixed 5000-day window",
            "2 baseline-bracketed sampled declines",
            "Assumed consecutive-event spacing ≈ 1000 days",
            "bracket compatibility (999, 1001) days (open)",
            "Consistency redundancy: 0",
            "recurrence NOT confirmed",
            "Maximum sampled decline: 0.762 percent",
            "physical depth bounds unavailable",
            "NOT physical uncertainty or model confidence",
            "NOT a verified physical period or confirmed transit",
            "NOT a planet-absence finding",
            "NOT learned perception",
            "NOT scientific verification",
            "No task-completion or submission claim",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(!text.contains("Observed recurrence ≈"));
        assert!(!text.contains("pixel bounds"));
        assert_eq!(app.state["task_completed"], false);
        assert_eq!(app.state["project_completed"], false);
        for (width, height) in [(64, 17), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &app)).unwrap();
        }
    }

    fn project_fixture_app() -> App {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/project-progress.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        app
    }

    #[test]
    fn project_replay_renders_negative_branch_provenance_and_separate_receipts() {
        let mut app = project_fixture_app();
        app.focused_panel = 3;
        let mut terminal = Terminal::new(TestBackend::new(180, 50)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "collected 3/30",
            "verified tasks 1",
            "unresolved 2",
            "FIXTURE-NO-PLANET",
            "stage assessment",
            "main_sequence",
            "learned prediction (not correctness)",
            "reference prediction (NOT learned)",
            "Planet reported outcome: no planet",
            "Habitability reported outcome: not applicable",
            "bounded_no_dip_assumption",
            "no_planet_branch_assumption",
            "observation limit 5000 days",
            "5000-day no-dip rule: reference assumption, not proof of absence",
            "scientific evidence unverified",
            "absence remains unproven",
            "Uncertain actions: 1",
            "fixture-uncertain-save",
            "do not retry blindly",
            "data quality verified",
            "scavenger hunt unverified",
            "Score transfer unverified",
            "submission verified",
            "project completion unverified",
            "three stars unverified",
            "thirty stars unverified",
            "Submission is not assessment",
            "No browser is connected",
        ] {
            assert!(text.contains(expected), "Missing {expected} from render");
        }
        assert!(!text.contains("project completion verified"));
        assert!(!text.contains("scientific evidence verified"));
    }

    #[test]
    fn not_habitable_is_a_reported_outcome_not_a_failed_episode() {
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
        let text = project_text(&app);
        assert!(text.contains("Habitability reported outcome: not habitable"));
        assert!(text.contains("stage habitability · task verified"));
        assert!(text.contains("Planet reported outcome: planet · class/value terrestrial"));
        assert!(text.contains("No planet / not habitable can be valid outcomes"));
        assert!(!text.contains("failed episode"));
    }

    #[test]
    fn partial_project_state_does_not_invent_zero_counts_absence_or_verification() {
        let app = App {
            state: serde_json::json!({"project_progress":{"schema_version":1,
                "active_star":{"name":"UNKNOWN", "planet":{"outcome":"unresolved"},
                    "habitability":{"outcome":"new_future_branch","provenance":"new_future_source"}}}}),
            ..App::default()
        };
        let text = project_text(&app);
        for expected in [
            "collected —/—",
            "verified tasks —",
            "unknown / unresolved",
            "unrecognized: new_future_branch",
            "unrecognized: new_future_source",
            "Uncertain actions: unknown",
            "Score transfer unknown",
            "project completion unknown",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(!text.contains("Planet reported outcome: no planet"));
        assert!(!text.contains("verified tasks 0"));
        assert!(project_text(&App::default()).is_empty());
    }

    #[test]
    fn project_overview_and_pending_copy_both_remain_visible() {
        let mut app = project_fixture_app();
        app.state["pending_browser_copy"] = serde_json::json!({"exact_value":"0.0001445975820837288",
            "unit":"Lsun", "destination":"luminosity", "current_value":"0"});
        let mut terminal = Terminal::new(TestBackend::new(180, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for expected in [
            "3/30 collected",
            "1 verified tasks",
            "2 unresolved",
            "Active FIXTURE-NO-PLANET",
            "COPY: 0.0001445975820837288 Lsun",
            "luminosity",
            "terminated true",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn project_full_view_can_scroll_past_summary_to_current_observation() {
        let mut app = project_fixture_app();
        app.focused_panel = 3;
        app.observation_scroll = u16::MAX;
        app.observation = serde_json::json!({"instruction":"Current calculation", "chart_crop":"current-observation-end"});
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("current-observation-end"));
    }

    #[test]
    fn project_small_terminals_render_and_control_characters_are_not_interpreted() {
        let mut app = project_fixture_app();
        app.state["project_progress"]["active_star"]["name"] = "STAR\x1b[2J".into();
        assert!(!project_text(&app).contains('\x1b'));
        for (width, height) in [(1, 1), (60, 15), (65, 18), (80, 24)] {
            for focus in [0, 1, 2, 3] {
                app.focused_panel = focus;
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|frame| draw(frame, &app)).unwrap();
            }
        }
    }

    #[test]
    fn project_level_stages_and_nullable_component_values_remain_unknown() {
        let mut app = App::default();
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":0,"payload":{"project_progress":{"schema_version":1,"active_stage":"score_transfer","active_star":null}}}"#).unwrap());
        assert!(project_text(&app).contains("active stage score_transfer"));
        assert!(project_text(&app).contains("Active star: none reported"));
        app.apply(parse_event(r#"{"version":1,"event":"state","sequence":1,"payload":{"project_progress":{"schema_version":1,"active_stage":"stellar","active_star":{"id":"fixture","name":"FIXTURE","stage":"stellar","stellar":{"numeric":null,"color":null,"classification":{"value":null}},"planet":{"outcome":"unresolved"},"habitability":{"outcome":"unresolved"}}}}}"#).unwrap());
        assert!(app.project_progress().is_some());
        assert!(project_text(&app).contains("Stellar classification: — · source unknown"));
        assert!(project_text(&app).contains("Habitability reported outcome: unknown / unresolved"));
    }

    #[test]
    fn optional_policy_display_keeps_legacy_and_other_limits_distinct() {
        let mut app = project_fixture_app();
        assert_eq!(
            project_text(&app)
                .matches("observation limit 5000 days")
                .count(),
            2
        );
        assert_eq!(
            project_text(&app)
                .matches("reference assumption, not proof of absence")
                .count(),
            2
        );
        for key in ["planet", "habitability"] {
            app.state["project_progress"]["active_star"][key]["policy_label"] =
                serde_json::Value::Null;
            app.state["project_progress"]["active_star"][key]["observation_limit_days"] =
                serde_json::Value::Null;
        }
        assert!(!project_text(&app).contains("Observation policy:"));
        assert!(!project_text(&app).contains("5000-day no-dip rule"));
        app.state["project_progress"]["active_star"]["planet"]["policy_label"] =
            "other_observation_policy".into();
        app.state["project_progress"]["active_star"]["planet"]["observation_limit_days"] =
            10000.into();
        assert!(
            project_text(&app).contains("other_observation_policy · observation limit 10000 days")
        );
        assert!(!project_text(&app).contains("5000-day no-dip rule"));
        app.state["project_progress"]["active_star"]["planet"]["observation_limit_days"] =
            serde_json::Value::Null;
        assert!(project_text(&app).contains("observation limit unknown"));
    }

    #[test]
    fn recorded_single_event_policy_is_not_mislabeled_as_no_dip() {
        let mut app = project_fixture_app();
        for key in ["planet", "habitability"] {
            app.state["project_progress"]["active_star"][key]["policy_label"] =
                "user_approved_5000_day_single_event_no_planet_v1".into();
        }
        let text = project_text(&app);
        assert_eq!(
            text.matches("Single-event No shortcut: explicit reference assumption")
                .count(),
            2
        );
        assert!(!text.contains("5000-day no-dip rule"));
        assert!(!text.contains("one possible dip was deliberately ignored"));
    }

    #[test]
    fn branch_applicability_is_not_a_native_field_readback() {
        let mut app = project_fixture_app();
        assert!(!project_text(&app).contains("Branch applicability"));
        app.state["project_progress"]["active_star"]["habitability"]
            ["branch_applicability_verified"] = true.into();
        app.state["project_progress"]["active_star"]["habitability"]["transport_verified"] =
            false.into();
        let text = project_text(&app);
        assert!(text.contains("Branch applicability verified · native readback unverified"));
        assert!(text.contains("reference assumption, not proof of absence"));
        app.state["project_progress"]["active_star"]["habitability"]
            ["branch_applicability_verified"] = false.into();
        assert!(project_text(&app).contains("Branch applicability unverified"));
    }

    #[test]
    fn chart_sensor_replay_renders_progress_without_learned_or_completion_claims() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/chart-sensor.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        let mut terminal = Terminal::new(TestBackend::new(150, 25)).unwrap();
        terminal
            .draw(|frame| draw_observation(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "NOT learned perception",
            "FIXTURE",
            "1279",
            "99.99%",
            "unique days 100",
            "complete transits 1",
            "chart_time_limit",
            "NOT evidence of absence",
            "completion remain unverified",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        terminal
            .draw(|frame| draw_action(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in ["Chart sensor action", "HOVER", "0.2", "no model confidence"] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn small_terminals_render_without_panicking() {
        for (width, height) in [(1, 1), (60, 15), (65, 18), (100, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &App::default())).unwrap();
        }
    }

    #[test]
    fn measurement_options_and_last_target_survive_observation_local_ids() {
        let mut app = App {
            action: serde_json::json!({"target":"2:source"}),
            action_control: serde_json::json!({"label":"Measurement or result", "role":"combobox", "surface":"calculation"}),
            observation: serde_json::json!({"controls":[{"id":"3:source", "label":"Measurement or result",
                "role":"combobox", "surface":"calculation", "options":["m1"]}],
                "values":{"measurements":{"m1":{"kind":"parallax", "value":0.1, "unit":"arcsec", "source":"current star"}}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
        terminal
            .draw(|frame| draw_controls(frame, &app, frame.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("▶ 3:source"));
        assert!(text.contains("m1: parallax 0.1 arcsec (current star)"));
        let mut duplicate = app.observation["controls"][0].clone();
        duplicate["id"] = serde_json::json!("3:ambiguous");
        app.observation["controls"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        terminal
            .draw(|frame| draw_controls(frame, &app, frame.area()))
            .unwrap();
        assert!(!terminal
            .backend()
            .buffer()
            .content
            .iter()
            .any(|c| c.symbol() == "▶"));
    }

    #[test]
    fn full_observation_scroll_reaches_wrapped_final_lines() {
        let app = App {
            focused_panel: 3,
            observation_scroll: u16::MAX,
            observation: serde_json::json!({"instruction":"a long instruction with wrapping ".repeat(200),
                "feedback":"final feedback", "progress":{"task_completed":true}, "chart_crop":"last-line-marker"}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("last-line-marker"));
    }

    #[test]
    fn stellar_observation_shows_real_sheet_selection_and_results() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Stellar task", "spreadsheet": {
            "calculation_mode":"google_sheets", "generation":3, "selected_measurement":"m2",
            "selected_input":"A2", "selected_output":"C2", "selected_destination":"distance",
            "bindings":{"A2":"m2"}, "results":{"distance":101.875},
            "last_operation":{"kind":"copy", "value":101.875}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in ["google_sheets", "A2", "C2", "distance", "101.875", "copy"] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn color_reference_and_selected_measurement_render() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Peak wavelength color", "values": {
                "measurements":{"m1":{"kind":"wavelength","unit":"nm","source":"current star","value":580}},
                "answers":{"color":"Yellow"}}, "calculation": {
                "calculation_mode":"learned_peak_wavelength_color_v1", "source":"m1",
                "reference_card":{"description":"Peak-wavelength band", "bands":[{"label":"Yellow","minimum":570,"maximum":590}],
                  "ambiguity_policy":"abstain", "pending":["494-495 gap"]},
                "tool_error":"ambiguous_color_boundary"}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "Experimental local color",
            "Yellow",
            "580",
            "abstain",
            "ambiguous_color_boundary",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn local_calculation_reference_bindings_results_and_errors_render() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Local stellar task", "calculation": {
                "calculation_mode":"local_tool_assisted", "operation":"distance",
                "reference_card":{"description":"Parallax distance", "inputs":{"parallax":{"unit":"arcsec"}}},
                "parameter":"parallax", "source":"m2", "bindings":{"parallax":"m2"},
                "results":{"r1":{"value":101.875,"unit":"ly"}}, "selected_result":"r1",
                "destination":"distance", "last_operation":{"kind":"copy","value":101.875},
                "tool_error":"incompatible_unit"}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "local_tool_assisted",
            "Parallax distance",
            "arcsec",
            "m2",
            "101.875",
            "copy",
            "incompatible_unit",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn planet_tool_replay_preserves_units_and_pending_course_scope() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({
                "instruction":"Derived planet properties",
                "progress":{"task":"planet_calculations","course_acceptance_passed":false},
                "values":{"physics_assumptions":["circular orbit"],
                    "required_fields":["planet_mass"], "star_class":"main_sequence",
                    "answers":{"planet_mass":22.354}, "units":{"planet_mass":"MEarth"}},
                "calculation":{"calculation_mode":"local_tool_assisted",
                    "operation":"planet_mass", "pending_knowledge":["planet classification"],
                    "reference_card":{"description":"Mass from radial velocity",
                        "inputs":{"radial_velocity":{"unit":"m/s"}}},
                    "parameter":"radial_velocity", "source":"r2", "bindings":{"radial_velocity":"r2"},
                    "results":{"r3":{"kind":"planet_mass","value":22.354,"unit":"MEarth","valid":true}},
                    "selected_result":"r3", "destination":"planet_mass",
                    "last_operation":{"kind":"copy","value":22.354}, "tool_error":"incompatible_unit"}
            }),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(160, 60)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "NOT course acceptance",
            "circular orbit",
            "planet classification",
            "Mass from radial velocity",
            "m/s",
            "22.354",
            "MEarth",
            "r3 → planet_mass",
            "incompatible_unit",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn supplied_temperature_scope_and_exact_copy_are_visible() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({
                "instruction":"Supplied temperatures",
                "progress":{"task":"habitability_calculations","course_acceptance_passed":false},
                "values":{"answers":{"surface_temp":789.4},"units":{"surface_temp":"K"}},
                "calculation":{"calculation_mode":"local_tool_assisted","operation":"surface_temp",
                    "reference_card":{"description":"Surface temperature with supplied warming",
                        "inputs":{"equilibrium_temp":{"unit":"K"},"greenhouse_increment":{"unit":"K"}}},
                    "bindings":{"equilibrium_temp":"r1","greenhouse_increment":"m3"},
                    "results":{"r2":{"kind":"surface_temp","value":789.4,"unit":"K","valid":true}},
                    "selected_result":"r2","destination":"surface_temp",
                    "last_operation":{"kind":"copy","value":789.4}}
            }),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(160, 60)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "NOT gas identification",
            "habitability classification",
            "Course acceptance remains unverified",
            "Surface temperature with supplied warming",
            "greenhouse_increment",
            "789.4",
            "r2 → surface_temp",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn luminosity_chain_shows_reused_distance_and_both_result_options() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Distance and luminosity", "calculation": {
                "calculation_mode":"local_tool_assisted", "operation":"luminosity",
                "reference_card":{"description":"Flux and distance luminosity", "inputs":{"distance":{"unit":"ly"},"flux":{"unit":"W/m2"}}},
                "parameter":"distance", "source":"r1", "bindings":{"distance":"r1","flux":"m2"},
                "results":{"r1":{"kind":"distance","value":101.875,"unit":"ly","valid":true},
                    "r2":{"kind":"luminosity","value":0.0157,"unit":"Lsun","valid":true}},
                "selected_result":"r2", "destination":"luminosity", "tool_error":null},
                "values":{"answers":{"distance":101.875,"luminosity":0.0157},"units":{"distance":"ly","luminosity":"Lsun"}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "Flux and distance luminosity",
            "W/m2",
            "101.875",
            "Lsun",
            "r2 → luminosity",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert_eq!(
            option_text(&app, &serde_json::json!("r1")),
            "r1: distance 101.875 ly (valid true)"
        );
        assert_eq!(
            option_text(&app, &serde_json::json!("r2")),
            "r2: luminosity 0.0157 Lsun (valid true)"
        );
    }

    #[test]
    fn temperature_chain_shows_peak_wavelength_kelvin_and_all_answers() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Distance, luminosity and temperature", "calculation": {
                "calculation_mode":"local_tool_assisted", "operation":"temperature",
                "reference_card":{"description":"Temperature from peak wavelength", "inputs":{"wavelength":{"unit":"nm"}}},
                "parameter":"wavelength", "source":"m4", "bindings":{"wavelength":"m4"},
                "results":{"r1":{"kind":"distance","value":101.875,"unit":"ly","valid":true},
                    "r2":{"kind":"luminosity","value":0.0157,"unit":"Lsun","valid":true},
                    "r3":{"kind":"temperature","value":13668.719339622641,"unit":"K","valid":true}},
                "selected_result":"r3", "destination":"temperature", "tool_error":null},
                "values":{"measurements":{"m4":{"kind":"wavelength","value":212,"unit":"nm","source":"current star"}},
                    "answers":{"distance":101.875,"luminosity":0.0157,"temperature":13668.719339622641},
                    "units":{"distance":"ly","luminosity":"Lsun","temperature":"K"}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "Temperature from peak wavelength",
            "212 nm (current star)",
            "101.875",
            "Lsun",
            "13668.719339622641",
            "r3 → temperature",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert_eq!(
            option_text(&app, &serde_json::json!("r3")),
            "r3: temperature 13668.719339622641 K (valid true)"
        );
    }

    #[test]
    fn mass_chain_shows_supplied_class_luminosity_binding_and_solar_mass() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Main-sequence mass", "calculation": {
                "calculation_mode":"local_tool_assisted", "operation":"mass",
                "reference_card":{"description":"Main-sequence mass from luminosity", "inputs":{"luminosity":{"unit":"Lsun"}}},
                "parameter":"luminosity", "source":"r2", "bindings":{"luminosity":"r2"},
                "results":{"r4":{"kind":"mass","value":4,"unit":"Msun","valid":true}},
                "selected_result":"r4", "destination":"mass", "tool_error":null},
                "values":{"star_class":"main_sequence", "required_fields":["distance","luminosity","temperature","mass"],
                    "answers":{"mass":4}, "units":{"mass":"Msun"}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "Supplied class: main_sequence",
            "Main-sequence mass from luminosity",
            "luminosity ← r2",
            "r4 → mass",
            "Msun",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn radius_chain_shows_both_result_bindings_and_solar_radius() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Main-sequence radius", "calculation": {
                "calculation_mode":"local_tool_assisted", "operation":"radius",
                "reference_card":{"description":"Radius from luminosity and temperature", "inputs":{"luminosity":{"unit":"Lsun"},"temperature":{"unit":"K"}}},
                "parameter":"temperature", "source":"r3", "bindings":{"luminosity":"r2","temperature":"r3"},
                "results":{"r5":{"kind":"radius","value":8,"unit":"Rsun","valid":true}},
                "selected_result":"r5", "destination":"radius", "tool_error":null},
                "values":{"star_class":"main_sequence", "required_fields":["distance","luminosity","temperature","mass","radius"],
                    "answers":{"radius":8}, "units":{"radius":"Rsun"}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "Supplied class: main_sequence",
            "Radius from luminosity and temperature",
            "temperature ← r3",
            "\"luminosity\":\"r2\"",
            "r5 → radius",
            "Rsun",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
    }

    #[test]
    fn lifetime_shows_mass_binding_and_years_not_gigayears() {
        let app = App {
            focused_panel: 3,
            observation: serde_json::json!({"instruction":"Main-sequence lifetime", "calculation": {
                "calculation_mode":"local_tool_assisted", "operation":"lifetime",
                "reference_card":{"description":"Lifetime from mass; returns years, not billions of years", "inputs":{"mass":{"unit":"Msun"}}},
                "parameter":"mass", "source":"r4", "bindings":{"mass":"r4"},
                "results":{"r6":{"kind":"lifetime","value":10000000000.0,"unit":"yr","valid":true}},
                "selected_result":"r6", "destination":"lifetime", "tool_error":null},
                "values":{"star_class":"main_sequence", "required_fields":["distance","luminosity","temperature","mass","radius","lifetime"],
                    "answers":{"lifetime":10000000000.0}, "units":{"lifetime":"yr"}}}),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(150, 45)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        for expected in [
            "Supplied class: main_sequence",
            "returns years, not billions of years",
            "mass ← r4",
            "r6 → lifetime",
            "10000000000",
            "\"lifetime\":\"yr\"",
        ] {
            assert!(text.contains(expected), "Missing {expected}");
        }
        assert!(!text.contains("Gyr"));
    }

    #[test]
    fn isolated_color_activity_path_is_visible() {
        let app = App {
            focused_panel: 1,
            neural: serde_json::json!({
                "activity_source": "checkpoint",
                "activity_pathway": "selected_measurement_color_graph"
            }),
            ..App::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("Path: selected_measurement_color_graph"));
    }

    #[test]
    fn score_checkpoint_display_requires_explicit_clean_terminal_context() {
        let valid = serde_json::json!({
            "task":"browser_project", "mode":"bounded_post_campaign_finalization",
            "browser_phase":"score_transferred_not_submitted", "status":"handoff", "finished":true,
            "score_checkpoint_completed":true, "reported_score":0,
            "failure_reason":null, "event_forwarding_failed":false, "cleanup_failed":false,
            "task_completed":false, "project_completed":false, "submitted":false,
            "project_progress":{"schema_version":1, "target":30,"collected":30,"verified":30,
                "unresolved":0, "assessment":{"data_quality":true,"scavenger_hunt":true},
                "score_transfer_verified":true,"uncertain_actions":[]}
        });
        let mut app = App {
            state: valid.clone(),
            ..App::default()
        };
        let text = finalization_text(&app);
        assert!(text.contains("Update Score verified · reported score 0"));
        assert!(text.contains("Formal Submit is separate and was not performed"));
        app.state["status"] = "completed".into();
        app.state["replay"] = true.into();
        app.state["recorded_status"] = "handoff".into();
        assert!(score_checkpoint_score(&app).is_some());
        for (field, value) in [
            ("score_checkpoint_completed", serde_json::json!(null)),
            ("score_checkpoint_completed", serde_json::json!(1)),
            ("reported_score", serde_json::json!(null)),
            ("reported_score", serde_json::json!("0")),
            ("reported_score", serde_json::json!(-1)),
            ("status", serde_json::json!("stopped")),
            ("status", serde_json::json!("aborted")),
            ("browser_phase", serde_json::json!("unknown_pending")),
            ("event_forwarding_failed", serde_json::json!(true)),
            ("failure_reason", serde_json::json!("source_changed")),
            ("submitted", serde_json::json!(true)),
        ] {
            app.state = valid.clone();
            app.state[field] = value;
            assert!(score_checkpoint_score(&app).is_none(), "{field}");
            assert!(!finalization_text(&app).contains("UPDATE SCORE CHECKPOINT COMPLETE"));
        }
        app.state = valid;
        app.state["project_progress"]["uncertain_actions"] = serde_json::json!([{}]);
        assert!(score_checkpoint_score(&app).is_none());
    }

    #[test]
    fn submission_refusal_is_not_displayed_as_project_success() {
        let app = App {
            state: serde_json::json!({
                "runtime_task": "browser_project", "browser_phase": "unknown_pending",
                "status": "handoff", "submission_outcome": "unknown",
                "submitted": false, "task_completed": false, "project_completed": false,
                "submission_feedback": {
                    "status": "course_refusal",
                    "reason": "at_least_thirty_analyzed_submitted_stars_required"
                }
            }),
            ..App::default()
        };
        let text = finalization_text(&app);
        assert!(text.contains("course_refusal"));
        assert!(text.contains("at_least_thirty_analyzed_submitted_stars_required"));
        assert!(text.contains("not a success receipt"));
        assert!(text
            .contains("Parent submitted false · task completed false · project completed false"));
    }

    #[test]
    fn focused_neural_view_exposes_top_neurons_in_standard_terminal() {
        let mut app = App::default();
        for line in include_str!("../tests/fixtures/session.jsonl").lines() {
            app.apply(parse_event(line).unwrap());
        }
        app.focused_panel = 1;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("Body ID"));
        assert!(text.contains("42 · 0.6000"));
        assert!(text.contains("observation only"));
        app.focused_panel = 2;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("Collect star"));
    }
}
