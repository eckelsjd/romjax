from __future__ import annotations

import os
import signal
import subprocess
import tomllib
from pathlib import Path

import pytest

from romjax.task_cli import (
    agent_environment,
    archive_task,
    build_phase_prompt,
    clean_task,
    cli,
    codex_rollout_files,
    collect_plan_prompt,
    create_task,
    current_git_branch,
    ensure_task_layout,
    ensure_task_ready,
    find_task_document,
    get_task_paths,
    git_output,
    load_manifest,
    load_or_create_manifest,
    move_task_to_state,
    parse_headless_codex_output,
    phase_entry,
    phase_new_command,
    phase_paths,
    plan_task,
    review_task,
    rollout_token_count,
    run_phase_session,
    shared_venv_bin_dir,
    shared_venv_path,
    start_task,
    stop_task,
    task_runtime_status,
    terminate_process,
    validate_task_slug,
    write_manifest,
)


def init_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "romjax"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "romjax tests"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo_root / ".gitignore").write_text("htmlcov/\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("Test instructions.\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "test setup"], cwd=repo_root, check=True, capture_output=True, text=True)
    return repo_root


def prepare_repo(repo_root: Path) -> Path:
    paths = get_task_paths(repo_root)
    ensure_task_layout(paths)
    (paths.templates_dir / "feat.md").write_text("# {task_slug}\n\n## Summary\nTODO: fill me\n", encoding="utf-8")
    (paths.templates_dir / "fix.md").write_text("# {task_slug}\n\n## Problem\nTODO: fill me\n", encoding="utf-8")
    (paths.templates_dir / "review.md").write_text(
        "# Review Summary: {task_slug}\n\n## High-Level Summary\nTODO: fill me\n",
        encoding="utf-8",
    )
    return repo_root


def assert_task_state(paths, task_slug: str, expected_state: str) -> Path:
    actual_state, task_path = find_task_document(paths, task_slug)
    assert actual_state == expected_state
    return task_path


def write_ready_task(paths, task_slug: str, summary: str = "Done") -> Path:
    task_path = create_task(paths, task_slug)
    task_path.write_text(f"# {task_slug}\n\n## Summary\n{summary}\n", encoding="utf-8")
    return task_path


def stub_phase_agent(
    monkeypatch: pytest.MonkeyPatch,
    paths,
    *,
    phase_name: str,
    body: str,
    exit_code: int = 0,
) -> None:
    def fake_agent(**kwargs):
        Path(kwargs["output_path"]).write_text(body, encoding="utf-8")
        manifest = load_or_create_manifest(paths, kwargs["task_slug"])
        phase = phase_entry(manifest, phase_name)
        extra_args = kwargs.get("extra_args") or []
        phase["status"] = "succeeded" if exit_code == 0 else "failed"
        phase["exit_code"] = exit_code
        phase["codex_args"] = list(extra_args)
        if "--model" in extra_args:
            phase["model"] = extra_args[extra_args.index("--model") + 1]
        elif "-m" in extra_args:
            phase["model"] = extra_args[extra_args.index("-m") + 1]
        else:
            phase["model"] = next(
                (arg.split("=", 1)[1] for arg in extra_args if arg.startswith("--model=")),
                None,
            )
        phase["reasoning_effort"] = None
        index = 0
        while index < len(extra_args):
            arg = extra_args[index]
            config_arg: str | None = None
            if arg in {"-c", "--config"}:
                if index + 1 < len(extra_args):
                    config_arg = extra_args[index + 1]
                    index += 2
                else:
                    break
            elif arg.startswith("--config="):
                config_arg = arg.removeprefix("--config=")
                index += 1
            elif arg.startswith("-c") and arg != "-c":
                config_arg = arg[2:]
                index += 1
            else:
                index += 1
                continue
            if config_arg is None or not config_arg.startswith("model_reasoning_effort="):
                continue
            raw_value = config_arg.split("=", 1)[1]
            try:
                phase["reasoning_effort"] = tomllib.loads(f"value = {raw_value}")["value"]
            except tomllib.TOMLDecodeError:
                phase["reasoning_effort"] = raw_value
        write_manifest(paths, kwargs["task_slug"], manifest)
        return exit_code

    mapping = {
        "planning": "romjax.task_cli.run_planning_agent",
        "implementation": "romjax.task_cli.run_implementation_agent",
        "review": "romjax.task_cli.run_review_agent",
    }
    monkeypatch.setattr(mapping[phase_name], fake_agent)


def test_task_cli_support_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    assert validate_task_slug("feat-galerkin-rom") == ("feat", "galerkin-rom")
    with pytest.raises(Exception):
        validate_task_slug("FeatBad")
    with pytest.raises(Exception):
        validate_task_slug("feat")

    assert current_git_branch(repo_root) == "main"

    task_path = create_task(paths, "feat-galerkin-rom")
    with pytest.raises(Exception):
        ensure_task_ready(task_path)
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nImplemented\n", encoding="utf-8")
    ensure_task_ready(task_path)

    monkeypatch.setattr("romjax.task_cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "   ")
    with pytest.raises(Exception):
        collect_plan_prompt()

    output_path = phase_paths(paths, "feat-galerkin-rom", "implementation")["output_path"]
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    interactive_prompt = build_phase_prompt(
        "implementation",
        paths=paths,
        task_slug="feat-galerkin-rom",
        task_path=task_path,
        manifest=manifest,
        output_path=output_path,
        headless=False,
    )
    headless_prompt = build_phase_prompt(
        "implementation",
        paths=paths,
        task_slug="feat-galerkin-rom",
        task_path=task_path,
        manifest=manifest,
        output_path=output_path,
        headless=True,
    )
    assert f"any blockers to {output_path}" in interactive_prompt
    assert "Return a concise final summary as your last assistant message" in headless_prompt

    interactive_argv, interactive_stdin = phase_new_command(
        "implementation",
        cwd=repo_root,
        repo_root=repo_root,
        prompt_text="do the work",
        output_path=output_path,
        headless=False,
        extra_args=["--model", "gpt-5.5", "-c", "model_reasoning_effort=\"high\""],
    )
    assert interactive_stdin is False
    assert interactive_argv[-5:-1] == [
        "--model",
        "gpt-5.5",
        "-c",
        'model_reasoning_effort="high"',
    ]
    assert interactive_argv[-1] == "do the work"

    headless_argv, headless_stdin = phase_new_command(
        "planning",
        cwd=repo_root,
        repo_root=repo_root,
        prompt_text="plan it",
        output_path=output_path,
        headless=True,
        extra_args=["--model=gpt-5.4-mini", "-c", "model_reasoning_effort=\"medium\""],
    )
    assert headless_stdin is True
    assert headless_argv[-4:] == [
        "--model=gpt-5.4-mini",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]
    assert headless_argv[-1] == "-"

    (repo_root / "example.txt").write_text("hello\n", encoding="utf-8")
    assert "example.txt" in git_output(repo_root, ["status", "--short"])

    venv_path = shared_venv_path(paths)
    bin_dir = shared_venv_bin_dir(paths)
    bin_dir.mkdir(parents=True)
    env = agent_environment(paths)
    assert env["VIRTUAL_ENV"] == str(venv_path)
    assert env["UV_PROJECT_ENVIRONMENT"] == str(venv_path)
    assert str(bin_dir) in env["PATH"].split(os.pathsep)

    calls: list[tuple[int, int]] = []
    monkeypatch.delattr("romjax.task_cli.os.killpg", raising=False)
    monkeypatch.setattr("romjax.task_cli.os.kill", lambda pid, sig: calls.append((pid, sig)))
    terminate_process(1234)
    assert calls == [(1234, signal.SIGTERM)]

    thread_id, usage = parse_headless_codex_output(
        '\n'.join(
            [
                '{"type":"thread.started","thread_id":"thread-123"}',
                '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":4,"output_tokens":3}}',
            ]
        )
    )
    assert thread_id == "thread-123"
    assert usage == {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 3}

    codex_home = tmp_path / "codex-home"
    rollout_path = codex_home / "sessions" / "2026" / "05" / "08" / "rollout-test.jsonl"
    rollout_path.parent.mkdir(parents=True)
    rollout_path.write_text(
        '{"token_count":{"total_token_usage":{"input_tokens":10,"output_tokens":3,"reasoning_output_tokens":2,"total_tokens":13}}}\n',
        encoding="utf-8",
    )
    assert codex_rollout_files({"CODEX_HOME": str(codex_home)}) == [rollout_path]
    assert rollout_token_count(rollout_path) == {
        "total_token_usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "reasoning_output_tokens": 2,
            "total_tokens": 13,
        }
    }


def test_run_phase_session_records_usage_and_token_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    load_or_create_manifest(paths, "feat-galerkin-rom")
    info = phase_paths(paths, "feat-galerkin-rom", "planning")
    info["prompt_path"].write_text("Plan the work.\n", encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def fake_run(*args, **kwargs):
        rollout_path = codex_home / "sessions" / "2026" / "05" / "08" / "rollout-thread-123.jsonl"
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_path.write_text(
            '{"token_count":{"total_token_usage":{"input_tokens":11,"cached_input_tokens":5,"output_tokens":7,'
            '"reasoning_output_tokens":3,"total_tokens":18}}}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=kwargs.get("args", args[0] if args else []),
            returncode=0,
            stdout=(
                '{"type":"thread.started","thread_id":"thread-123"}\n'
                '{"type":"turn.completed","usage":{"input_tokens":11,"cached_input_tokens":5,"output_tokens":7}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr("romjax.task_cli.subprocess.run", fake_run)

    exit_code = run_phase_session(
        paths=paths,
        task_slug="feat-galerkin-rom",
        phase_name="planning",
        task_path=task_path,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=True,
        extra_args=["-m", "gpt-5.4-mini", "-c", 'model_reasoning_effort="medium"'],
    )

    assert exit_code == 0
    phase = load_manifest(paths, "feat-galerkin-rom")["phases"]["planning"]
    assert phase["thread_id"] == "thread-123"
    assert phase["usage"] == {"input_tokens": 11, "cached_input_tokens": 5, "output_tokens": 7}
    assert phase["token_count"] == {
        "total_token_usage": {
            "input_tokens": 11,
            "cached_input_tokens": 5,
            "output_tokens": 7,
            "reasoning_output_tokens": 3,
            "total_tokens": 18,
        }
    }
    assert phase["session_path"].endswith("rollout-thread-123.jsonl")


def test_create_task_command(tmp_path: Path) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    task_path = create_task(paths, "feat-galerkin-rom")

    assert task_path == paths.open_dir / "feat-galerkin-rom.md"
    assert "TODO: fill me" in task_path.read_text(encoding="utf-8")

    move_task_to_state(paths, "feat-galerkin-rom", "archive")
    with pytest.raises(Exception):
        create_task(paths, "feat-galerkin-rom")


def test_plan_task_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    stub_phase_agent(
        monkeypatch,
        paths,
        phase_name="planning",
        body="# feat-galerkin-rom\n\n## Summary\nImplement a Galerkin ROM feature.\n",
    )

    task_path = plan_task(
        paths,
        "feat-galerkin-rom",
        user_prompt="Add a Galerkin ROM feature",
        extra_args=["--model", "gpt-5.5", "-c", "model_reasoning_effort=\"high\""],
    )

    assert task_path.exists()
    assert "Implement a Galerkin ROM feature." in task_path.read_text(encoding="utf-8")
    planning_phase = load_manifest(paths, "feat-galerkin-rom")["phases"]["planning"]
    assert planning_phase["status"] == "succeeded"
    assert planning_phase["model"] == "gpt-5.5"
    assert planning_phase["reasoning_effort"] == "high"
    assert_task_state(paths, "feat-galerkin-rom", "open")

    create_task(paths, "fix-bug-1127")
    info = phase_paths(paths, "fix-bug-1127", "planning")
    info["output_path"].parent.mkdir(parents=True, exist_ok=True)
    info["output_path"].write_text("existing plan\n", encoding="utf-8")
    with pytest.raises(Exception):
        plan_task(paths, "fix-bug-1127", user_prompt="Fix issue 1127.")


def test_start_task_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    create_task(paths, "feat-unready-task")
    with pytest.raises(Exception):
        start_task(paths, "feat-unready-task")

    write_ready_task(paths, "feat-galerkin-rom")
    stub_phase_agent(monkeypatch, paths, phase_name="implementation", body="summary\n")

    manifest = start_task(
        paths,
        "feat-galerkin-rom",
        extra_args=["-m", "gpt-5.4", "-c", "model_reasoning_effort=\"low\""],
    )

    assert Path(manifest["worktree_path"]).exists()
    assert_task_state(paths, "feat-galerkin-rom", "finished")
    stored = load_manifest(paths, "feat-galerkin-rom")
    assert stored["phases"]["implementation"]["status"] == "succeeded"
    assert stored["phases"]["implementation"]["model"] == "gpt-5.4"
    assert stored["phases"]["implementation"]["reasoning_effort"] == "low"
    info = phase_paths(paths, "feat-galerkin-rom", "implementation")
    assert info["prompt_path"].name == "feat-galerkin-rom-implementation-prompt.md"
    assert info["output_path"].name == "feat-galerkin-rom-implementation.md"
    assert info["log_path"].name == "feat-galerkin-rom-implementation.log"

    write_ready_task(paths, "feat-existing-output")
    existing = phase_paths(paths, "feat-existing-output", "implementation")
    existing["output_path"].parent.mkdir(parents=True, exist_ok=True)
    existing["output_path"].write_text("existing implementation\n", encoding="utf-8")
    with pytest.raises(Exception):
        start_task(paths, "feat-existing-output")


def test_review_task_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    write_ready_task(paths, "feat-not-finished", summary="Implemented feature.")
    with pytest.raises(Exception):
        review_task(paths, "feat-not-finished")

    task_path = write_ready_task(paths, "feat-galerkin-rom", summary="Implemented feature.")
    task_path.rename(paths.finished_dir / task_path.name)
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    Path(manifest["worktree_path"]).mkdir()
    write_manifest(paths, "feat-galerkin-rom", manifest)
    stub_phase_agent(
        monkeypatch,
        paths,
        phase_name="review",
        body="# Review Summary: feat-galerkin-rom\n\n## High-Level Summary\nUpdated the model.\n",
    )

    review_path = review_task(paths, "feat-galerkin-rom")

    assert "Updated the model." in review_path.read_text(encoding="utf-8")
    assert_task_state(paths, "feat-galerkin-rom", "review")

    task_path = write_ready_task(paths, "feat-existing-review", summary="Implemented feature.")
    task_path.rename(paths.finished_dir / task_path.name)
    manifest = load_or_create_manifest(paths, "feat-existing-review")
    Path(manifest["worktree_path"]).mkdir()
    write_manifest(paths, "feat-existing-review", manifest)
    info = phase_paths(paths, "feat-existing-review", "review")
    info["output_path"].parent.mkdir(parents=True, exist_ok=True)
    info["output_path"].write_text("existing review\n", encoding="utf-8")
    with pytest.raises(Exception):
        review_task(paths, "feat-existing-review")


def test_stop_task_command_and_runtime_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    create_task(paths, "fix-bug-1127")
    status = task_runtime_status(paths, "fix-bug-1127")
    assert status["task_state"] == "open"
    assert status["run_status"] == "open"

    move_task_to_state(paths, "fix-bug-1127", "running")
    manifest = load_or_create_manifest(paths, "fix-bug-1127")
    phase = phase_entry(manifest, "implementation")
    phase["status"] = "running"
    phase["pid"] = 4321
    write_manifest(paths, "fix-bug-1127", manifest)

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("romjax.task_cli.process_exists", lambda pid: True)
    monkeypatch.setattr("romjax.task_cli.terminate_process", lambda pid: killed.append((pid, signal.SIGTERM)))

    updated = stop_task(paths, "fix-bug-1127")

    assert killed == [(4321, signal.SIGTERM)]
    assert updated["phases"]["implementation"]["status"] == "stopped"
    assert_task_state(paths, "fix-bug-1127", "stopped")
    assert task_runtime_status(paths, "fix-bug-1127")["task_state"] == "stopped"


def test_archive_task_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    task_path = write_ready_task(paths, "feat-running-archive", summary="Done")
    task_path.rename(paths.running_dir / task_path.name)
    manifest = load_or_create_manifest(paths, "feat-running-archive")
    phase_entry(manifest, "implementation")["status"] = "running"
    write_manifest(paths, "feat-running-archive", manifest)
    with pytest.raises(Exception):
        archive_task(paths, "feat-running-archive")

    task_path = write_ready_task(paths, "feat-galerkin-rom", summary="Done")
    task_path.write_text(
        "# feat-galerkin-rom\n\n- Worktree: "
        f"`{repo_root.parent / 'feat-galerkin-rom'}`\n\n## Summary\nDone\n",
        encoding="utf-8",
    )
    stub_phase_agent(monkeypatch, paths, phase_name="implementation", body="summary\n")
    manifest = start_task(paths, "feat-galerkin-rom")
    original_worktree_path = Path(manifest["worktree_path"])
    archived_worktree_path = repo_root.parent / "archive" / "feat-galerkin-rom"

    archived_task_path = archive_task(paths, "feat-galerkin-rom")

    assert archived_task_path == paths.archive_dir / "feat-galerkin-rom.md"
    assert not original_worktree_path.exists()
    assert archived_worktree_path.exists()
    stored = load_manifest(paths, "feat-galerkin-rom")
    assert Path(stored["worktree_path"]) == archived_worktree_path
    assert str(archived_worktree_path) in archived_task_path.read_text(encoding="utf-8")
    worktree_list = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"worktree {archived_worktree_path.resolve()}" in worktree_list


def test_clean_task_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    stub_phase_agent(monkeypatch, paths, phase_name="implementation", body="summary\n")

    running_task = write_ready_task(paths, "feat-running-clean")
    running_task.rename(paths.running_dir / running_task.name)
    manifest = load_or_create_manifest(paths, "feat-running-clean")
    phase_entry(manifest, "implementation")["status"] = "running"
    write_manifest(paths, "feat-running-clean", manifest)
    with pytest.raises(Exception):
        clean_task(paths, "feat-running-clean")

    write_ready_task(paths, "feat-galerkin-rom")
    manifest = start_task(paths, "feat-galerkin-rom")
    worktree_path = Path(manifest["worktree_path"])
    run_dir = paths.runs_dir / "feat-galerkin-rom"

    summary = clean_task(paths, "feat-galerkin-rom")

    assert summary["removed_task_doc"] is True
    assert summary["removed_run_dir"] is True
    assert summary["removed_worktree"] is True
    assert summary["removed_branch"] is True
    assert not worktree_path.exists()
    assert not run_dir.exists()
    with pytest.raises(Exception):
        find_task_document(paths, "feat-galerkin-rom")
    assert subprocess.run(
        ["git", "branch", "--list", "feat-galerkin-rom"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == ""

    write_ready_task(paths, "feat-archived-clean")
    manifest = start_task(paths, "feat-archived-clean")
    active_worktree_path = Path(manifest["worktree_path"])
    archived_worktree_path = repo_root.parent / "archive" / "feat-archived-clean"
    archived_worktree_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "move", str(active_worktree_path), str(archived_worktree_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    move_task_to_state(paths, "feat-archived-clean", "archive")
    stale_manifest = load_manifest(paths, "feat-archived-clean")
    stale_manifest["worktree_path"] = str(active_worktree_path)
    write_manifest(paths, "feat-archived-clean", stale_manifest)

    archived_summary = clean_task(paths, "feat-archived-clean")

    assert archived_summary["removed_worktree"] is True
    assert archived_summary["worktree_path"] == str(archived_worktree_path.resolve())
    assert not archived_worktree_path.exists()
    assert not (paths.runs_dir / "feat-archived-clean").exists()


def test_cli_command_dispatch_and_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    create_path = paths.open_dir / "feat-create.md"
    create_path.parent.mkdir(parents=True, exist_ok=True)
    create_path.write_text("# feat-create\n", encoding="utf-8")
    monkeypatch.setattr("romjax.task_cli.create_task", lambda _paths, task_slug: create_path)
    assert cli(["create", "feat-create"], repo_root=repo_root) == 0
    assert "feat-create.md" in capsys.readouterr().out

    monkeypatch.setattr("romjax.task_cli.collect_plan_prompt", lambda: "Plan a new Galerkin ROM feature")

    def fake_plan_task(
        _paths,
        task_slug: str,
        user_prompt: str | None = None,
        headless: bool = False,
        extra_args: list[str] | None = None,
    ):
        assert task_slug == "feat-galerkin-rom"
        assert user_prompt == "Plan a new Galerkin ROM feature"
        assert headless is True
        assert extra_args == ["--model", "gpt-5.5"]
        task_path = paths.open_dir / f"{task_slug}.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text("# feat-galerkin-rom\n", encoding="utf-8")
        return task_path

    monkeypatch.setattr("romjax.task_cli.plan_task", fake_plan_task)
    assert cli(["plan", "feat-galerkin-rom", "--headless", "--model", "gpt-5.5"], repo_root=repo_root) == 0
    assert "feat-galerkin-rom.md" in capsys.readouterr().out

    monkeypatch.setattr(
        "romjax.task_cli.start_task",
        lambda _paths, task_slug, headless=False, extra_args=None: {"task_slug": task_slug, "exit_code": 7},
    )
    assert cli(["start", "feat-galerkin-rom", "--headless", "--model", "gpt-5.5"], repo_root=repo_root) == 7

    def fake_review_task(_paths, task_slug: str, headless: bool = False, extra_args: list[str] | None = None):
        assert headless is False
        assert extra_args == ["-c", 'model_reasoning_effort="medium"']
        review_path = paths.runs_dir / task_slug / f"{task_slug}-review.md"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text("# Review Summary: feat-galerkin-rom\n", encoding="utf-8")
        return review_path

    monkeypatch.setattr("romjax.task_cli.review_task", fake_review_task)
    assert cli(["review", "feat-galerkin-rom", "-c", 'model_reasoning_effort="medium"'], repo_root=repo_root) == 0
    assert "feat-galerkin-rom-review.md" in capsys.readouterr().out

    monkeypatch.setattr(
        "romjax.task_cli.stop_task",
        lambda _paths, task_slug: {"task_slug": task_slug, "phases": {"implementation": {"status": "stopped"}}},
    )
    assert cli(["stop", "feat-galerkin-rom"], repo_root=repo_root) == 0
    assert '"status": "stopped"' in capsys.readouterr().out

    archive_path = paths.archive_dir / "feat-galerkin-rom.md"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text("# feat-galerkin-rom\n", encoding="utf-8")
    monkeypatch.setattr("romjax.task_cli.archive_task", lambda _paths, task_slug: archive_path)
    assert cli(["archive", "feat-galerkin-rom"], repo_root=repo_root) == 0
    assert "feat-galerkin-rom.md" in capsys.readouterr().out

    monkeypatch.setattr("romjax.task_cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "no")
    assert cli(["clean", "feat-galerkin-rom"], repo_root=repo_root) == 2
    captured = capsys.readouterr()
    assert "Warning: task clean" in captured.out
    assert "Task clean cancelled." in captured.err

    create_task(paths, "feat-open")
    create_task(paths, "feat-running")
    create_task(paths, "feat-finished")
    create_task(paths, "feat-stopped")
    create_task(paths, "feat-review")
    create_task(paths, "feat-archived")
    move_task_to_state(paths, "feat-running", "running")
    move_task_to_state(paths, "feat-finished", "finished")
    move_task_to_state(paths, "feat-stopped", "stopped")
    move_task_to_state(paths, "feat-review", "review")
    move_task_to_state(paths, "feat-archived", "archive")

    assert cli(["status", "feat-open"], repo_root=repo_root) == 0
    status_output = capsys.readouterr().out
    assert "task: feat-open" in status_output
    assert "state: open" in status_output

    assert cli(["list"], repo_root=repo_root) == 0
    list_output = capsys.readouterr().out
    assert "TASK" in list_output
    assert "STATE" in list_output
    assert "feat-open" in list_output and "open" in list_output
    assert "feat-running" in list_output and "running" in list_output
    assert "feat-finished" in list_output and "finished" in list_output
    assert "feat-stopped" in list_output and "stopped" in list_output
    assert "feat-review" in list_output and "review" in list_output
    assert "feat-archived" not in list_output
