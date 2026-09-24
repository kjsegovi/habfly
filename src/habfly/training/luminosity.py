"""Recorded chain diagnostics; all grading and stage labels stay outside observations."""

import math
from collections import Counter

from habfly.environments.luminosity import MAX_STEPS, LuminosityEnv
from habfly.environments.stellar_common import UNITS
from habfly.model.tool_state import visible_sources

from .stellar import record_episode, write_json

ERRORS = ("invalid_actions", "tool_errors", "api_failures", "infrastructure_failures")


def audit_chain(case, trajectory):
    reused = correct_flux = correct_wavelength = exact_copy = mass_reused = mass_attempts = 0
    radius_reused = radius_attempts = 0
    lifetime_reused = lifetime_attempts = 0
    selections, correct = Counter(), Counter()
    for step in trajectory:
        before, action, after = step["observation"], step["action"], step["result"]["observation"]
        control = next((c for c in before["controls"] if c["id"] == action["target"]), {})
        key = control.get("id", "").partition(":")[2]
        state = after["calculation"]
        sources = visible_sources(after)
        if key == "operation" and action["value"] == "lifetime":
            lifetime_attempts += 1
        if key == "execute" and state["operation"] == "lifetime" and not state["tool_error"]:
            mass = state["bindings"].get("mass")
            result = sources.get(mass, {})
            lifetime_reused += int(
                mass in before["calculation"]["results"]
                and result.get("kind") == "mass"
                and result.get("unit") == "Msun"
                and result.get("source") == "current star"
            )
        if key == "operation" and action["value"] == "radius":
            radius_attempts += 1
        if key == "execute" and state["operation"] == "radius" and not state["tool_error"]:
            radius_reused += int(
                all(
                    state["bindings"].get(key) in before["calculation"]["results"]
                    and sources.get(state["bindings"].get(key), {}).get("kind") == key
                    and sources.get(state["bindings"].get(key), {}).get("unit") == unit
                    and sources.get(state["bindings"].get(key), {}).get("source") == "current star"
                    for key, unit in (("luminosity", "Lsun"), ("temperature", "K"))
                )
            )
        if key == "operation" and action["value"] == "mass":
            mass_attempts += 1
        if key == "execute" and state["operation"] == "mass" and not state["tool_error"]:
            luminosity = state["bindings"].get("luminosity")
            result = sources.get(luminosity, {})
            mass_reused += int(
                luminosity in before["calculation"]["results"]
                and result.get("kind") == "luminosity"
                and result.get("unit") == "Lsun"
                and result.get("source") == "current star"
            )
        if key == "execute" and state["operation"] == "luminosity" and not state["tool_error"]:
            distance = state["bindings"].get("distance")
            flux = sources.get(state["bindings"].get("flux"), {})
            reused += int(
                distance in before["calculation"]["results"]
                and sources.get(distance, {}).get("kind") == "distance"
            )
            correct_flux += int(flux.get("kind") == "flux" and flux.get("source") == "current star")
        if key == "execute" and state["operation"] == "temperature" and not state["tool_error"]:
            wavelength = sources.get(state["bindings"].get("wavelength"), {})
            correct_wavelength += int(
                wavelength.get("kind") == "wavelength"
                and wavelength.get("unit") == "nm"
                and wavelength.get("source") == "current star"
            )
        if key == "copy":
            result = state["results"].get(state["selected_result"], {})
            exact_copy += int(
                after["values"]["answers"].get(state["destination"]) == result.get("value") and bool(result)
            )
        if key == "operation":
            selections["calculation"] += 1
            correct["calculation"] += int(action["value"] in case["required"])
        if key == "bind":
            selections["input_binding"] += 1
            reference = state["reference_card"]["inputs"].get(state["parameter"], {})
            selected = sources.get(state["source"], {})
            correct["input_binding"] += int(
                selected.get("kind") == reference.get("quantity")
                and selected.get("unit") == reference.get("unit")
                and selected.get("source") == "current star"
            )
    last = trajectory[-1]["result"]["observation"]
    for field in case["required"]:
        selections["numeric_answer"] += 1
        correct["numeric_answer"] += int(
            math.isclose(last["values"]["answers"].get(field, -1), case["expected"][field], rel_tol=1e-12)
        )
        selections["units"] += 1
        correct["units"] += int(last["values"]["units"].get(field) == UNITS[field])
    return {
        "distance_result_reuses": reused,
        "current_star_flux_uses": correct_flux,
        "current_star_wavelength_uses": correct_wavelength,
        "exact_copies": exact_copy,
        "mass_luminosity_reuses": mass_reused,
        "mass_operation_selections": mass_attempts,
        "radius_result_pair_reuses": radius_reused,
        "radius_operation_selections": radius_attempts,
        "lifetime_mass_reuses": lifetime_reused,
        "lifetime_operation_selections": lifetime_attempts,
        "selection_attempts": dict(selections),
        "selection_correct": dict(correct),
    }


def rollout(policy, calculator, cases, directory, *, environment=LuminosityEnv):
    directory.mkdir(parents=True, exist_ok=False)
    episodes = []
    for case in cases:
        trajectory, summary = record_episode(
            environment(calculator, [case], max_steps=MAX_STEPS),
            case["seed"],
            policy,
            event_path=directory / f"{case['seed']}.events.jsonl",
        )
        write_json(directory / f"{case['seed']}.trajectory.json", trajectory)
        if trajectory:
            summary.update(audit_chain(case, trajectory))
        summary["required_fields"] = list(case["required"])
        summary["star_class"] = case["star_class"]
        summary["mass_applicability_correct"] = (
            summary.get("mass_luminosity_reuses") == 1 and summary.get("mass_operation_selections") == 1
            if "mass" in case["required"]
            else summary.get("mass_operation_selections") == 0
        )
        summary["radius_applicability_correct"] = (
            summary.get("radius_result_pair_reuses") == 1 and summary.get("radius_operation_selections") == 1
            if "radius" in case["required"]
            else summary.get("radius_operation_selections") == 0
        )
        summary["lifetime_applicability_correct"] = (
            summary.get("lifetime_mass_reuses") == 1 and summary.get("lifetime_operation_selections") == 1
            if "lifetime" in case["required"]
            else summary.get("lifetime_operation_selections") == 0
        )
        episodes.append(summary)
    attempts, correct = Counter(), Counter()
    for row in episodes:
        attempts.update(row.get("selection_attempts", {}))
        correct.update(row.get("selection_correct", {}))
    report = {
        "requested": len(cases),
        "completed": sum(e["completed"] for e in episodes),
        "episodes": episodes,
        **{key: sum(e.get(key, 0) for e in episodes) for key in ERRORS},
        "accuracy": {key: correct[key] / count for key, count in attempts.items()},
        "chained": sum(
            e.get("distance_result_reuses") == 1
            and e.get("current_star_flux_uses") == 1
            and e.get("exact_copies") == len(e["required_fields"])
            and ("temperature" not in e["required_fields"] or e.get("current_star_wavelength_uses") == 1)
            and e["mass_applicability_correct"]
            and e["radius_applicability_correct"]
            and e["lifetime_applicability_correct"]
            for e in episodes
        ),
    }
    report["by_class"] = {
        cls: {
            "requested": sum(e["star_class"] == cls for e in episodes),
            "completed": sum(e["star_class"] == cls and e["completed"] for e in episodes),
            "mass_applicability_correct": sum(
                e["star_class"] == cls and e["mass_applicability_correct"] for e in episodes
            ),
            "radius_applicability_correct": sum(
                e["star_class"] == cls and e["radius_applicability_correct"] for e in episodes
            ),
            "lifetime_applicability_correct": sum(
                e["star_class"] == cls and e["lifetime_applicability_correct"] for e in episodes
            ),
        }
        for cls in sorted({e["star_class"] for e in episodes})
    }
    write_json(directory / "report.json", report)
    return report


def clean(report, count, *, steps=22):
    return (
        report["completed"] == count
        and report["chained"] == count
        and len(report["episodes"]) == count
        and not any(report[key] for key in ERRORS)
        and all(e["steps"] == (steps(e) if callable(steps) else steps) for e in report["episodes"])
    )
