from typer.testing import CliRunner

from habfly.cli import app


def test_all_public_commands_parse():
    runner = CliRunner()
    for command in (
        [],
        ["data"],
        ["data", "inspect"],
        ["data", "build"],
        ["data", "validate"],
        ["train", "recognize"],
        ["train", "ground"],
        ["train", "arithmetic"],
        ["train", "policy"],
        ["evaluate", "model"],
        ["evaluate", "baseline"],
        ["evaluate", "browser"],
        ["runtime"],
        ["replay"],
    ):
        result = runner.invoke(app, command + ["--help"])
        assert result.exit_code == 0, result.output


def test_data_validation_cli(tmp_path):
    from habfly.data import make_demo_graph, save_graph

    target = save_graph(make_demo_graph(), tmp_path / "graph")
    result = CliRunner().invoke(app, ["data", "validate", str(target)])
    assert result.exit_code == 0 and '"valid": true' in result.stdout


def test_runtime_requires_explicit_protocol():
    result = CliRunner().invoke(app, ["runtime"])
    assert result.exit_code != 0 and "--jsonl" in result.output


def test_curriculum_environments_share_contract():
    from habfly.contracts import Observation
    from habfly.environments.curriculum import CurriculumEnvironment

    for stage in ("recognize", "ground", "arithmetic"):
        env = CurriculumEnvironment(stage=stage, count=8)
        first, _ = env.reset(seed=1)
        assert Observation.model_validate(first)
        output, reward, done, truncated, info = env.step(env.examples[env.index].action)
        assert done and not truncated and reward == 1
        assert info["result"]["observation"] == output
