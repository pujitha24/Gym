# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_4 import prepare as preparation
from nemo_gym.data import TaskDataValidator, load_task_data_schema


SERVER_DIR = Path(__file__).resolve().parents[1]


def task_validator() -> TaskDataValidator:
    return TaskDataValidator("terminal_bench_4", load_task_data_schema(SERVER_DIR), "test.jsonl")


def test_prepared_names_match_pinned_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    output = preparation.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads((preparation.BENCHMARK_DIR / "manifest.json").read_text())
    tasks = {"terminal-bench/" + task["name"]: task["ref"] for task in manifest["tasks"]}
    assert len(rows) == len(tasks) == 66
    assert {row["task_name"] for row in rows} == tasks.keys()
    validator = task_validator()
    for index, row in enumerate(rows):
        validator.validate_row(index, row)
        assert row["task_id"] == row["task_name"]
        assert row["task_ref"] == tasks[row["task_name"]]
        assert row["dataset_ref"] == manifest["ref"]
        assert "path" not in row
    assert validator.report.clean, validator.report.summary()


@pytest.mark.parametrize("missing", ["task_name", "task_ref", "dataset_ref"])
def test_schema_rejects_missing_task_identity(missing: str) -> None:
    row = {"task_name": "terminal-bench/ks-solver-cpp", "task_ref": "sha256:task", "dataset_ref": "sha256:dataset"}
    del row[missing]
    validator = task_validator()
    validator.validate_row(0, row)
    assert validator.report.error_rows == 1
    assert missing in validator.report.summary()


def test_example_rollouts_match_pinned_tasks() -> None:
    examples = [json.loads(line) for line in (SERVER_DIR / "data/example.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (SERVER_DIR / "data/example_rollouts.jsonl").read_text().splitlines()]
    manifest = json.loads((preparation.BENCHMARK_DIR / "manifest.json").read_text())
    tasks = {"terminal-bench/" + task["name"]: task for task in manifest["tasks"]}
    assert len(examples) == len(rollouts) == 5
    assert len({row["task_name"] for row in examples}) == 5
    assert {tasks[row["task_name"]]["category"] for row in examples} == {"cpu", "compose", "gpu"}
    validator = task_validator()
    for index, (example, rollout) in enumerate(zip(examples, rollouts, strict=True)):
        validator.validate_row(index, example)
        assert example["task_ref"] == tasks[example["task_name"]]["ref"]
        assert example["dataset_ref"] == manifest["ref"]
        for key in ("task_name", "task_ref", "dataset_ref"):
            assert rollout[key] == example[key]
        assert rollout["evaluation_completed"] is True
        assert rollout["infrastructure_error"] is None
        assert rollout["response"]["output"]
        assert rollout["harness_version"] == "2.4.6"
    assert validator.report.clean, validator.report.summary()


@pytest.mark.parametrize("category,count", [("cpu", 52), ("compose", 11), ("gpu", 3)])
def test_category_selections(tmp_path, monkeypatch, category, count):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    assert len(preparation.prepare(category=category).read_text().splitlines()) == count


def test_explicit_names_and_invalid_selections(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    row = json.loads(preparation.prepare(task_names=["ks-solver-cpp"]).read_text())
    assert row["task_name"] == "terminal-bench/ks-solver-cpp"
    with pytest.raises(ValueError, match="Unknown TB4 task"):
        preparation.prepare(task_names=["missing"])
    with pytest.raises(ValueError, match="Unknown TB4 category"):
        preparation.prepare(category="missing")
    with pytest.raises(ValueError, match="empty"):
        preparation.prepare(task_names=[])


@pytest.mark.parametrize(
    "overrides,steps,timeout", [({}, 500, 30), ({"tb4_max_steps": 0, "tb4_step_timeout_sec": 45}, 0, 45)]
)
def test_benchmark_limits_resolve_defaults_and_client_overrides(overrides, steps, timeout):
    root = Path(preparation.__file__).resolve().parents[2]
    config = OmegaConf.merge(OmegaConf.load(root / "benchmarks/terminal_bench_4/miniswe.yaml"), overrides)
    agent = config.terminal_bench_4_miniswe_agent.responses_api_agents.miniswe_sandboxed_agent
    harness = agent.harness
    assert harness.step_limit == steps
    assert harness.step_timeout_sec == timeout
    resources = config.terminal_bench_4_miniswe_resources_server.resources_servers.terminal_bench_4
    assert resources.datasets[0].num_repeats == 1


def test_profile_composes_lending_resources_miniswe_and_the_environment_server():
    config = OmegaConf.merge(
        OmegaConf.load(preparation.BENCHMARK_DIR / "resources.yaml"),
        OmegaConf.load(preparation.BENCHMARK_DIR / "miniswe.yaml"),
    )
    resources = config.terminal_bench_4_miniswe_resources_server
    agent = config.terminal_bench_4_miniswe_agent.responses_api_agents.miniswe_sandboxed_agent
    environment = config.terminal_bench_4_miniswe.environment_servers.single_agent_turn_legacy
    assert resources._inherit_from == "terminal_bench_4"
    # Rows carry only the task identity; the seed's observation is the official instruction.
    assert resources.resources_servers.terminal_bench_4.instruction_observation is True
    # The inherited server lends through one block, or each pool's with split endpoints, as its sandboxes run.
    lending = config.terminal_bench_4.resources_servers.terminal_bench_4
    blocks = [lending.sandbox_provider_ref, *lending.sandbox_provider_refs.values()]
    assert dict(lending.sandbox_provider_refs).keys() == {"cpu", "gpu"}
    for name, suffix in zip(blocks, ["", "_CPU", "_GPU"], strict=True):
        connection = config[name].opensandbox.connection
        assert OmegaConf.to_container(connection)["domain"] == f"${{oc.env:OPENSANDBOX_DOMAIN{suffix},localhost}}"
        assert connection.request_timeout_s == 300
        assert config[name].opensandbox.operations.background_exec is True
    assert agent.resources_server.name == "terminal_bench_4_miniswe_resources_server"
    assert agent.agent_timeout_sec == 28800
    assert environment.resources_server.name == "terminal_bench_4_miniswe_resources_server"
    assert environment.agent_server.name == "terminal_bench_4_miniswe_agent"
    # The 8-hour agent budget and the slowest verifier (18000 seconds) fit in one episode.
    assert environment.default_episode_timeout_seconds >= 28800 + 18000


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, "results/terminal_bench_4/agent"),
        ({"tb4_jobs_dir": "/run/resources"}, "/run/resources/agent"),
        ({"tb4_agent_artifacts_dir": "/run/agent"}, "/run/agent"),
        ({"tb4_jobs_dir": "/run/resources", "tb4_agent_artifacts_dir": "/run/agent"}, "/run/agent"),
    ],
)
def test_agent_artifacts_follow_run_directory_unless_overridden(overrides, expected):
    config = OmegaConf.merge(OmegaConf.load(preparation.BENCHMARK_DIR / "miniswe.yaml"), overrides)
    assert config.terminal_bench_4_miniswe_agent.responses_api_agents.miniswe_sandboxed_agent.artifacts_dir == expected
