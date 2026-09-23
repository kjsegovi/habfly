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
        frame.render_widget(Paragraph::new("v overview/neural/controls/full observation · ↑/↓ controls · n step\ns start · space pause/resume · a abort · t save trace · q quit"), rows[2]);
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
    frame.render_widget(Paragraph::new("s start · space pause/resume · p pause · r resume · n step · a abort · t save trace\nv focus panels · ↑/↓ scroll controls · q / Ctrl-C quit · --replay PATH")
        .style(Style::default().fg(Color::DarkGray)), rows[3]);
}

fn draw_observation(frame: &mut Frame, app: &App, area: Rect) {
    let sheet = &app.observation["spreadsheet"];
    let calc = &app.observation["calculation"];
    let extra = if calc.get("calculation_mode").is_some() {
        format!("\nLocal tool: {}\nOperation: {} · {}\nInputs: {}\nSelected input: {} ← {} · bindings {}\nResults: {}\nCopy: {} → {}\nLast: {}\nTool error: {}\nMeasurements: {}\nAnswers: {} · units {}",
            display(&calc["calculation_mode"]), display(&calc["operation"]),
            display(&calc["reference_card"]["description"]), display(&calc["reference_card"]["inputs"]),
            display(&calc["parameter"]), display(&calc["source"]), display(&calc["bindings"]),
            display(&calc["results"]), display(&calc["selected_result"]), display(&calc["destination"]),
            display(&calc["last_operation"]), display(&calc["tool_error"]),
            display(&app.observation["values"]["measurements"]), display(&app.observation["values"]["answers"]),
            display(&app.observation["values"]["units"]))
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
    frame.render_widget(
        Paragraph::new(format!(
            "{}\nFeedback: {}\nProgress: {}\nChart: {} · crop {}{}",
            display(&app.observation["instruction"]),
            display(&app.observation["feedback"]),
            display(&app.observation["progress"]),
            display(&app.observation["chart"]),
            display(&app.observation["chart_crop"]),
            extra
        ))
        .block(Block::bordered().title("Observation"))
        .wrap(Wrap { trim: false }),
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
            let chosen = selected.is_some() && control["id"].as_str() == selected;
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
                    lines.push(Line::from(format!(
                        "    options: {}",
                        display(&control["options"])
                    )));
                }
            }
        }
    }
    if lines.is_empty() {
        lines.push(Line::from("Waiting for visible controls"));
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
    let text = format!(
        "{} → {} · value {}\nAction {}\nTarget {}\nSource {} · reward parts {}",
        display(&app.action["kind"]),
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
