"""Local development, training, evaluation, and JSONL runtime commands."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="HabFly local connectome experiment", no_args_is_help=True)
data_app = typer.Typer(help="Inspect, build, and validate connectome artifacts")
train_app = typer.Typer(help="Train reproducible local curricula")
evaluate_app = typer.Typer(help="Evaluate model, baselines, simulator, and browser")
spreadsheet_app = typer.Typer(help="Inspect or verify the dedicated Google Sheets working copy")
knowledge_app = typer.Typer(help="Inspect and independently validate the offline stellar knowledge pack")
diagnose_app = typer.Typer(help="Bounded local learning diagnostics; not browser acceptance")
browser_app = typer.Typer(help="Read-only preflight and human-stepped numeric diagnostic; no learned policy")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(spreadsheet_app, name="spreadsheet")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(diagnose_app, name="diagnose")
app.add_typer(browser_app, name="browser")

CANONICAL = Path("data/processed/canonical_neurons.feather")
EDGES = Path("data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str))


def graph_for(path):
    from .data import load_graph, make_demo_graph

    return load_graph(path) if path else make_demo_graph()


@browser_app.command("inspect")
def browser_inspect(
    config: Path, output: Path, url: str | None = None, check: bool = False, new_test_session: bool = False
):
    """Open isolated Chromium; capture only after the human selects the activity screen.

    No Firefox profile/cookies are copied and no login state is saved. The human
    owns any login and navigation. This command never clicks, fills or submits.
    --check validates configuration without opening a browser or writing files.
    """
    import sys

    from .browser import BrowserSafetyStop
    from .browser_probe import BrowserProbeConfig, capture_interactively, public_url, save_probe

    try:
        raw = json.loads(config.read_text())
        if url is not None:
            raw["url"] = url
        settings = BrowserProbeConfig.model_validate(raw)
    except (ValueError, TypeError):
        # Pydantic's default error includes input values, possibly a pasted secret URL.
        raise typer.BadParameter(
            "Invalid probe configuration: use a credential-free loopback URL, "
            "only a pinned preview_sequence_id query, and explicit HTTP(S) frame URLs"
        ) from None
    if check:
        emit({"valid": True, "url": public_url(settings.url), "mode": "read_only_browser_preflight"})
        return
    if output.exists():
        raise typer.BadParameter("Output already exists; choose a new inspection directory")
    if not sys.stdin.isatty():
        raise typer.BadParameter("Run in an interactive terminal; capture requires a human checkpoint")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = browser.new_context()
            try:
                page = context.new_page()
                page.goto(settings.url, wait_until="domcontentloaded", timeout=30000)
                if new_test_session:
                    typer.echo(
                        "Independent Chromium test session: leave the Firefox attempt alone. "
                        "You may manually select/collect ONE star here and open View Star Data. "
                        "It does not need to be the Firefox star. This is setup, not learned behavior."
                    )
                else:
                    typer.echo(
                        "Existing-state capture: Chromium does not share Firefox's attempt state. "
                        "If the intended star is absent, cancel; do not collect a replacement. "
                        "Use --new-test-session only for an explicitly separate test setup."
                    )
                typer.echo(
                    "In Chromium, sign in if needed and navigate to the selected star's stellar tab. "
                    "Do not reset, edit answers, assess, update score or submit. "
                    "Close help panels before capture. This probe performs no UI actions; "
                    "the site's own autosave may still run. No browser trace or credentials are saved."
                )
                report = capture_interactively(
                    page, settings, confirm=typer.confirm, echo=typer.echo, new_test_session=new_test_session
                )
                if report is None:
                    raise typer.Abort()
                manifest = save_probe(report, output)
                emit({"output": str(output), **manifest})
            finally:
                context.close()
                browser.close()
    except (BrowserSafetyStop, PlaywrightError) as exc:
        # Playwright messages can include full session URLs; do not echo them.
        reason = str(exc) if isinstance(exc, BrowserSafetyStop) else type(exc).__name__
        typer.echo(f"No capture saved: {reason}", err=True)
        raise typer.Exit(1) from None


@browser_app.command("map-stellar")
def browser_map_stellar(capture: Path, output: Path):
    """Map a hashed saved capture offline. No model, browser, answers, or training."""
    from .browser_stellar import StellarMappingError, load_and_map_capture, save_mapping

    try:
        mapping = load_and_map_capture(capture)
        save_mapping(mapping, output)
    except (StellarMappingError, OSError, ValueError, KeyError, TypeError) as exc:
        reason = str(exc) if isinstance(exc, StellarMappingError) else type(exc).__name__
        typer.echo(f"No mapping saved: {reason}", err=True)
        raise typer.Exit(1) from None
    emit(
        {
            "output": str(output),
            "star": mapping["star_name"],
            "capture_sha256": mapping["capture_sha256"],
            "browser_acceptance_passed": False,
        }
    )


@browser_app.command("test-numeric")
def browser_test_numeric(
    config: Path, output: Path, url: str | None = None, check: bool = False, new_test_session: bool = False
):
    """Confirm up to three numeric writes in a SEPARATE Chromium test attempt.

    Human-selected local calculations, not learned inference or star completion.
    Each confirmed fill includes Tab and a separate committed-display check.
    Never clicks Save/Assess/Update Score/Submit. Native input may trigger autosave.
    --check is offline and read-only. Requires --new-test-session for live use.
    """
    import sys

    from .browser import BrowserSafetyStop
    from .browser_probe import BrowserProbeConfig, capture_interactively, public_url
    from .browser_stellar import SIMULATION_URL

    try:
        raw = json.loads(config.read_text())
        if url is not None:
            raw["url"] = url
        settings = BrowserProbeConfig.model_validate(raw)
        if len([f for f in settings.frames if f.url == SIMULATION_URL and f.count == 1]) != 1:
            raise ValueError("Required stellar frame missing")
    except (OSError, ValueError, TypeError):
        raise typer.BadParameter(
            "Invalid numeric diagnostic config; use the stellar probe config and a plain loopback preview URL"
        ) from None
    if check:
        emit(
            {
                "valid": True,
                "url": public_url(settings.url),
                "mode": "human_stepped_numeric_diagnostic",
                "max_writes": 3,
                "learned_policy": False,
            }
        )
        return
    if not new_test_session:
        raise typer.BadParameter("Requires --new-test-session; do not use the preserved Firefox attempt")
    if output.exists():
        raise typer.BadParameter("Output already exists; choose a new diagnostic directory")
    if not sys.stdin.isatty():
        raise typer.BadParameter("Run in an interactive terminal; every write requires confirmation")

    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from .browser_numeric import run_numeric_diagnostic

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = browser.new_context()
            try:
                page = context.new_page()
                page.goto(settings.url, wait_until="domcontentloaded", timeout=30000)
                typer.echo(
                    "Independent Chromium numeric TEST: leave Firefox unchanged. Sign in if needed, "
                    "return to the ORIGINAL --url, manually select/collect ONE star, and open its stellar tab. "
                    "Close help panels. Do not choose a class/color, save, assess, update score or submit. "
                    "This session can change three numeric answers after confirmation, then uses Tab to verify each displayed value; the site may autosave. "
                    "No profiles, credentials, screenshots or browser traces are saved."
                )
                ready = capture_interactively(
                    page, settings, confirm=typer.confirm, echo=typer.echo, new_test_session=True
                )
                if ready is None:
                    raise typer.Abort()
                report = run_numeric_diagnostic(
                    page, settings, output, prompt=typer.prompt, confirm=typer.confirm, echo=typer.echo
                )
                emit({"output": str(output), **report})
                typer.prompt(
                    "Inspect the test window without saving/scoring. Press Enter to close Chromium",
                    default="",
                    show_default=False,
                )
                if report["outcome"] not in {"numeric_transport_verified", "cancelled"}:
                    raise typer.Exit(1)
            finally:
                context.close()
                browser.close()
    except (BrowserSafetyStop, PlaywrightError) as exc:
        reason = str(exc) if isinstance(exc, BrowserSafetyStop) else "browser_operation_failed"
        typer.echo(f"Stopped: {reason}. No automatic retry. Check any saved diagnostic evidence.", err=True)
        raise typer.Exit(1) from None


@diagnose_app.command("distance")
def diagnose_distance(output: Path, profile: Path = Path("configs/distance_diagnostic.yaml")):
    """One capped fresh distance-only run, then training-fit and gated unseen checks."""
    from .training.distance_diagnostic import load_distance_config, run_distance_diagnostic

    report = run_distance_diagnostic(output, load_distance_config(profile))
    emit(
        {
            "report": str(output / "report.json"),
            "outcome": report["outcome"],
            "training_fit_gate_passed": report["training_fit_gate_passed"],
            "seen_completed": report["seen"]["completed"],
            "seen_cases": report["seen"]["requested_episodes"],
            "unseen_status": report["unseen"]["status"],
            "distance_gate_passed": report["distance_gate_passed"],
            "stellar_acceptance_gate_passed": False,
        }
    )


@knowledge_app.command("inspect")
def knowledge_inspect(pack: Path | None = None):
    from .knowledge import load_knowledge_pack

    knowledge = load_knowledge_pack(pack)
    emit({"sha256": knowledge.checksum, "pack": knowledge.model_dump(mode="json")})


@knowledge_app.command("validate")
def knowledge_validate(pack: Path | None = None):
    from .knowledge import LocalCalculator, load_knowledge_pack

    emit(LocalCalculator(load_knowledge_pack(pack)).verify())


@spreadsheet_app.command("inspect")
def spreadsheet_inspect(config: Path):
    """Read-only: display the actual headers/formula fingerprint for pinning."""
    from .spreadsheet import SpreadsheetAdapter, load_spreadsheet_config

    adapter = SpreadsheetAdapter(load_spreadsheet_config(config))
    try:
        result = adapter.inspect()
        result.pop("cells")
        emit(result)
    finally:
        adapter.close()


@spreadsheet_app.command("verify")
def spreadsheet_verify(config: Path):
    """Write golden inputs in the COPY and restore its three previous inputs."""
    from .spreadsheet import SpreadsheetAdapter, load_spreadsheet_config

    adapter = SpreadsheetAdapter(load_spreadsheet_config(config))
    try:
        emit(adapter.verify())
    finally:
        adapter.close()


@data_app.command("demonstrations")
def data_demonstrations(
    output: Path,
    profile: Annotated[Path, typer.Option("--profile")],
    task: str = "stellar",
    spreadsheet_config: Path | None = None,
):
    """Collect tool-assisted demonstrations after a 100-case expert gate."""
    from .config import calculation_config, load_profile
    from .model import graph_fingerprint
    from .training.stellar import collect_demonstrations

    if task != "stellar":
        raise typer.BadParameter("This collection command supports --task stellar")
    settings = load_profile(profile)
    emit(
        collect_demonstrations(
            calculation_config(settings, spreadsheet_config),
            output,
            {
                "train": settings.training_episodes,
                "calibration": settings.calibration_episodes,
                "development": settings.development_episodes,
                "test": settings.test_episodes,
            },
            seed=settings.seed,
            graph_hash=graph_fingerprint(graph_for(settings.graph)),
        )
    )


@data_app.command("inspect")
def data_inspect(canonical: Path = CANONICAL, edges: Path = EDGES):
    from .data import inspect_sources

    emit(inspect_sources(canonical, edges))


@data_app.command("build")
def data_build(
    canonical: Path = CANONICAL,
    edges: Path = EDGES,
    output: Path = Path("data/processed/graphs"),
    sizes: str = "2000,5000,10000,30000",
    seed: int = 0,
):
    from .data import build_graphs

    emit(
        {
            "graphs": build_graphs(
                canonical, edges, output, sizes=tuple(int(s) for s in sizes.split(",")), seed=seed
            )
        }
    )


@data_app.command("validate")
def data_validate(graph: Path):
    from .data import validate_graph

    emit(validate_graph(graph))


def curriculum(stage, output, profile, graph, epochs, examples, seed, checkpoint=None, transfer_graph=False):
    import torch

    from .config import load_profile
    from .training.train import run_curriculum

    torch.set_num_threads(1)
    config = load_profile(profile)
    selected_graph = graph_for(graph or config.graph)
    policy = None
    if checkpoint:
        from .training.checkpoints import load_checkpoint

        policy, _ = load_checkpoint(checkpoint, selected_graph, allow_graph_transfer=transfer_graph)
    result = run_curriculum(
        stage,
        output,
        selected_graph,
        seed=config.seed if seed is None else seed,
        epochs=epochs or config.epochs,
        count=examples or config.examples,
        hidden_size=config.hidden_size,
        policy=policy,
    )
    emit(result)


def _curriculum_command(stage):
    def command(
        output: Path,
        profile: Path | None = None,
        graph: Path | None = None,
        epochs: int | None = None,
        examples: int | None = None,
        seed: int | None = None,
        checkpoint: Path | None = None,
        transfer_graph: bool = False,
    ):
        """Train and evaluate; specify a new output directory for each experiment."""
        curriculum(stage, output, profile, graph, epochs, examples, seed, checkpoint, transfer_graph)

    return command


for _stage in ("recognize", "ground", "arithmetic"):
    train_app.command(_stage)(_curriculum_command(_stage))


@train_app.command("policy")
def train_policy(
    output: Path,
    graph: Path | None = None,
    profile: Path | None = None,
    stars: int | None = None,
    episodes: int = 8,
    epochs: int | None = None,
    seed: int | None = None,
    method: str = "bc",
    checkpoint: Path | None = None,
    transfer_graph: bool = False,
    task: str | None = None,
    dataset: Path | None = None,
):
    """Clone recorded stellar demonstrations or generated Mini-HabWorlds episodes."""
    import torch

    from .config import load_profile
    from .content import load_content_pack
    from .environments import generate_demonstrations
    from .training.train import train_behavioral_cloning

    torch.set_num_threads(1)
    config = load_profile(profile)
    run_seed = config.seed if seed is None else seed
    selected_task = task or config.task
    if selected_task == "stellar":
        from .config import calculation_config
        from .training.stellar import train_stellar

        if method != "bc" or checkpoint or transfer_graph:
            raise typer.BadParameter("Stellar v1 uses fresh offline behavioral cloning, not PPO/resume")
        source = dataset or config.dataset
        if not source:
            raise typer.BadParameter("Collect and pass a stellar --dataset first")
        emit(
            train_stellar(
                source,
                graph_for(graph or config.graph),
                output,
                epochs=epochs or config.epochs,
                hidden_size=config.hidden_size,
                seed=run_seed,
                expected_content=calculation_config(config).content_identity(),
            )
        )
        return
    if selected_task != "mini_habworlds":
        raise typer.BadParameter("task must be mini_habworlds or stellar")
    pack = load_content_pack(config.content_pack)
    selected_graph = graph_for(graph or config.graph)
    policy = None
    if checkpoint:
        from .training.checkpoints import load_checkpoint

        policy, _ = load_checkpoint(checkpoint, selected_graph, allow_graph_transfer=transfer_graph)
    if method == "ppo":
        from .environments import MiniHabWorlds, expert_action
        from .training.ppo import train_ppo

        if policy is None:
            raise typer.BadParameter("PPO requires --checkpoint from behavioral cloning")
        emit(
            train_ppo(
                policy,
                lambda: MiniHabWorlds(stars=stars or config.stars, pack=pack),
                output,
                expert=lambda obs: expert_action(obs, pack),
                seed=run_seed,
                updates=epochs or config.epochs,
                content_pack=pack.model_dump(mode="json"),
            )
        )
        return
    if method != "bc":
        raise typer.BadParameter("method must be bc or ppo")
    if episodes < 2:
        raise typer.BadParameter("Use at least two demonstration episodes")
    examples = generate_demonstrations(range(run_seed, run_seed + episodes), stars or config.stars, pack)
    validation = generate_demonstrations(
        range(run_seed + 100000, run_seed + 100004), stars or config.stars, pack
    )
    emit(
        train_behavioral_cloning(
            examples,
            selected_graph,
            output,
            validation_episodes=validation,
            epochs=epochs or config.epochs,
            seed=run_seed,
            hidden_size=config.hidden_size,
            content_pack=pack.model_dump(mode="json"),
            policy=policy,
        )
    )


@evaluate_app.command("model")
def evaluate_model(
    checkpoint: Path | None = None,
    graph: Path | None = None,
    seed: int = 0,
    stars: int = 1,
    runs: int = 3,
    output: Path | None = None,
    content_pack: Path | None = None,
    task: str | None = None,
    profile: Path | None = None,
    dataset: Path | None = None,
    spreadsheet_config: Path | None = None,
    split: str = "test",
):
    """Evaluate a checkpoint; without one, run numerical connectivity smoke checks."""
    import torch

    from .contracts import Observation
    from .environments import MiniHabWorlds
    from .model import ConnectomePolicy
    from .training.checkpoints import load_checkpoint
    from .training.train import evaluate_policy

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    from .config import load_profile

    settings = load_profile(profile)
    selected_task = task or settings.task
    data = graph_for(graph or settings.graph)
    if selected_task == "stellar":
        from .config import calculation_config
        from .training.stellar import evaluate_stellar, load_dataset, training_content

        source = dataset or settings.dataset
        if not checkpoint or not source or not output:
            raise typer.BadParameter("Stellar evaluation requires checkpoint, dataset, and output")
        manifest, _ = load_dataset(source)
        policy, _ = load_checkpoint(checkpoint, data, content_pack=training_content(manifest))
        emit(
            evaluate_stellar(
                policy,
                calculation_config(settings, spreadsheet_config),
                source,
                output,
                split=split,
                runs=runs,
                seed=seed,
            )
        )
        return
    if selected_task != "mini_habworlds":
        raise typer.BadParameter("task must be mini_habworlds or stellar")
    if checkpoint:
        from .content import load_content_pack

        pack = load_content_pack(content_pack)
        policy, _ = load_checkpoint(checkpoint, data, content_pack=pack.model_dump(mode="json"))
        emit(
            evaluate_policy(
                policy,
                lambda: MiniHabWorlds(stars=stars, pack=pack),
                range(seed, seed + runs),
                output_dir=output,
            )
        )
    else:
        policy = ConnectomePolicy(data, hidden_size=8).eval()
        with torch.no_grad():
            baseline = policy([Observation(instruction="click next")])
            responsive = policy([Observation(instruction="calculate 2+3")])
            changed = float((baseline.action_logits - responsive.action_logits).abs().max())
            for _ in range(100):
                current = policy([Observation(instruction="click next")])
                if not torch.isfinite(current.state).all() or not torch.equal(current.state, baseline.state):
                    raise RuntimeError("Numerical smoke test failed")
        emit(
            {
                "forward_passes": 100,
                "finite": True,
                "deterministic": True,
                "input_response_max_delta": changed,
                "input_responsive": changed > 0,
                "nodes": len(data.body_ids),
                "trained": False,
            }
        )


@evaluate_app.command("simulator")
def evaluate_simulator(stars: int = 1, seed: int = 0, runs: int = 3):
    """Verify the deterministic expert; this is not a learned model evaluation."""
    from .environments import generate_demonstrations

    episodes = generate_demonstrations(range(seed, seed + runs), stars)
    emit({"policy": "scripted_expert", "synthetic": True, "episodes": [e[-1]["result"] for e in episodes]})


@evaluate_app.command("baseline")
def evaluate_baseline(
    output: Path,
    stage: str = "ground",
    graph: Path | None = None,
    seeds: str = "0,1,2",
    epochs: int = 3,
    examples: int = 96,
):
    """Train matched controls with the same examples, optimizer, and budget."""
    import torch

    from .training.baselines import run_baselines

    torch.set_num_threads(1)

    emit(
        run_baselines(
            graph_for(graph),
            output,
            stage=stage,
            seeds=[int(s) for s in seeds.split(",")],
            epochs=epochs,
            count=examples,
        )
    )


@evaluate_app.command("browser")
def evaluate_browser(config: Path, checkpoint: Path, graph: Path, content_pack: Path, seed: int = 0):
    """One fresh-account local acceptance run; requires supplied lesson pack."""
    from .runtime import Runtime

    runtime = Runtime()
    try:
        runtime.start(
            {
                "seed": seed,
                "policy": "checkpoint",
                "graph": str(graph),
                "checkpoint": str(checkpoint),
                "content_pack": str(content_pack),
                "environment": "browser",
                "browser_config": str(config),
            }
        )
        while runtime.status == "running":
            runtime.tick()
    finally:
        runtime.close()


@app.command("runtime")
def runtime(jsonl: Annotated[bool, typer.Option("--jsonl")] = False):
    """Read v1 commands from stdin and emit v1 events on stdout."""
    if not jsonl:
        raise typer.BadParameter("Use --jsonl for the versioned runtime protocol")
    from .runtime import serve

    serve()


@app.command("replay")
def replay(path: Path):
    """Validate and replay a saved event stream without acting in a browser."""
    from .runtime import read_trace

    for event in read_trace(path):
        typer.echo(event.model_dump_json())


@app.command("demo")
def demo(stars: int = 1, seed: int = 0, artifact_dir: Path = Path("experiments/runs")):
    """Run the synthetic expert and real untrained neuron observer to JSONL."""
    from .runtime import Runtime

    runtime = Runtime()
    try:
        runtime.start({"stars": stars, "seed": seed, "artifact_dir": str(artifact_dir), "interval": 0})
        while runtime.status == "running":
            runtime.tick()
    finally:
        runtime.close()


def main():
    app()
