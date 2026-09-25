use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Sparkline, Wrap},
    Frame,
};

use crate::app::{action_target, confidence, display, population_mean, App, GROUPS};

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
                display(&app.state["status"]),
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
        Constraint::Length(5),
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
        return "RECORDED REPLAY · space pause/resume · n step · v views · arrows scroll · q quit\nNo browser is connected; recorded confirmations cannot perform writes".into();
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
        let controls = if app.paused {
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
    if app.state["browser_phase"].is_string() {
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
    let extra = if calc["calculation_mode"].as_str() == Some("learned_peak_wavelength_color_v1") {
        format!("\nExperimental local color policy (not stellar classification)\nSelected measurement: {}\nMeasurements:\n{}\nSelected color: {}\nReference: {}\nBands (nm): {}\nAmbiguity policy: {}\nPending reference: {}\nError: {}",
            display(&calc["source"]), measurements, display(&app.observation["values"]["answers"]["color"]),
            display(&calc["reference_card"]["description"]), display(&calc["reference_card"]["bands"]),
            display(&calc["reference_card"]["ambiguity_policy"]), display(&calc["reference_card"]["pending"]),
            display(&calc["tool_error"]))
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
    let text = format!(
        "{}\nFeedback: {}\nProgress: {}\nChart: {} · crop {}{}",
        display(&app.observation["instruction"]),
        display(&app.observation["feedback"]),
        display(&app.observation["progress"]),
        display(&app.observation["chart"]),
        display(&app.observation["chart_crop"]),
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
        paragraph
            .scroll((scroll, 0))
            .block(Block::bordered().title("Observation")),
        area,
    );
}

fn metric(value: &serde_json::Value) -> String {
    value
        .as_f64()
        .map_or_else(|| display(value), |number| format!("{number:.4}"))
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let status = display(&app.state["status"]);
    let connected = if app.connected {
        "connected"
    } else {
        "disconnected"
    };
    let text = format!(
        "{status} · {connected} · policy {} · seed {} · browser {}\nStage {} · graph {} · checkpoint {}\nStep {} · reward {} · total {} · terminated {} · truncated {}",
        display(&app.state["policy"]), display(&app.state["seed"]), display(&app.state["browser_status"]),
        display(&app.state["stage"]), display(&app.state["graph"]), display(&app.state["checkpoint"]),
        display(app.result.get("steps").or_else(|| app.result.get("step")).unwrap_or(&serde_json::Value::Null)),
        display(&app.result["reward"]), display(&app.result["cumulative_reward"]),
        display(&app.result["terminated"]), display(&app.result["truncated"]),
    );
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
            if app.observation["progress"]["task_completed"] == true {
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
    let pending = &app.state["pending_browser_copy"];
    if pending.is_object() {
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
    let source = app.neural["activity_source"]
        .as_str()
        .unwrap_or("unavailable");
    let source_text = if source == "untrained_observer" {
        "untrained observer · observation only\nActions come from the scripted expert".to_string()
    } else {
        format!("Activity source: {source}")
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
