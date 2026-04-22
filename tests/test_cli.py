from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest

from romjax.cli import (
    agent_environment,
    archive_task,
    build_phase_prompt,
    clean_task,
    cli,
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
    phase_entry,
    phase_paths,
    plan_task,
    review_task,
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


def test_validate_task_slug():
    assert validate_task_slug("feat-galerkin-rom") == ("feat", "galerkin-rom")
    with pytest.raises(Exception):
        validate_task_slug("FeatBad")
    with pytest.raises(Exception):
        validate_task_slug("feat")


def test_create_task_uses_prefix_template(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    assert task_path == paths.open_dir / "feat-galerkin-rom.md"
    assert "TODO: fill me" in task_path.read_text(encoding="utf-8")


def test_create_task_rejects_existing_task_in_any_state(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "feat-galerkin-rom")
    move_task_to_state(paths, "feat-galerkin-rom", "archive")

    with pytest.raises(Exception):
        create_task(paths, "feat-galerkin-rom")


def test_current_git_branch(tmp_path: Path):
    assert current_git_branch(init_repo(tmp_path)) == "main"


def test_ensure_task_ready_rejects_unfilled_template(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    with pytest.raises(Exception):
        ensure_task_ready(task_path)
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nImplemented\n", encoding="utf-8")
    ensure_task_ready(task_path)


def test_collect_plan_prompt_raises_on_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("romjax.cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "   ")
    with pytest.raises(Exception):
        collect_plan_prompt()


def test_plan_task_creates_missing_task_and_fills_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    def fake_run_planning_agent(**kwargs):
        Path(kwargs["output_path"]).write_text(
            "# feat-galerkin-rom\n\n## Summary\nImplement a Galerkin ROM feature.\n",
            encoding="utf-8",
        )
        manifest = load_or_create_manifest(paths, kwargs["task_slug"])
        phase = phase_entry(manifest, "planning")
        phase["status"] = "succeeded"
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0

    monkeypatch.setattr("romjax.cli.run_planning_agent", fake_run_planning_agent)

    task_path = plan_task(paths, "feat-galerkin-rom", user_prompt="Add a Galerkin ROM feature")

    assert task_path.exists()
    assert "Implement a Galerkin ROM feature." in task_path.read_text(encoding="utf-8")
    manifest = load_manifest(paths, "feat-galerkin-rom")
    assert manifest["phases"]["planning"]["status"] == "succeeded"
    assert_task_state(paths, "feat-galerkin-rom", "open")


def test_plan_task_rejects_existing_planning_output(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "fix-bug-1127")
    info = phase_paths(paths, "fix-bug-1127", "planning")
    info["output_path"].parent.mkdir(parents=True, exist_ok=True)
    info["output_path"].write_text("existing plan\n", encoding="utf-8")

    with pytest.raises(Exception):
        plan_task(paths, "fix-bug-1127", user_prompt="Fix issue 1127.")


def test_start_task_creates_worktree_and_runs_implementation_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nDone\n", encoding="utf-8")

    def fake_run_implementation_agent(**kwargs):
        Path(kwargs["output_path"]).write_text("summary\n", encoding="utf-8")
        manifest = load_or_create_manifest(paths, kwargs["task_slug"])
        phase = phase_entry(manifest, "implementation")
        phase["status"] = "succeeded"
        phase["exit_code"] = 0
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0

    monkeypatch.setattr("romjax.cli.run_implementation_agent", fake_run_implementation_agent)

    manifest = start_task(paths, "feat-galerkin-rom")

    assert Path(manifest["worktree_path"]).exists()
    assert_task_state(paths, "feat-galerkin-rom", "finished")
    stored = load_manifest(paths, "feat-galerkin-rom")
    assert stored["phases"]["implementation"]["status"] == "succeeded"
    info = phase_paths(paths, "feat-galerkin-rom", "implementation")
    assert info["prompt_path"].name == "feat-galerkin-rom-implementation-prompt.md"
    assert info["output_path"].name == "feat-galerkin-rom-implementation.md"
    assert info["log_path"].name == "feat-galerkin-rom-implementation.log"


def test_start_task_rejects_existing_implementation_output(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nDone\n", encoding="utf-8")
    info = phase_paths(paths, "feat-galerkin-rom", "implementation")
    info["output_path"].parent.mkdir(parents=True, exist_ok=True)
    info["output_path"].write_text("existing implementation\n", encoding="utf-8")

    with pytest.raises(Exception):
        start_task(paths, "feat-galerkin-rom")


def test_review_task_creates_review_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nImplemented feature.\n", encoding="utf-8")
    task_path.rename(paths.finished_dir / task_path.name)
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    worktree_path = Path(manifest["worktree_path"])
    worktree_path.mkdir()
    write_manifest(paths, "feat-galerkin-rom", manifest)

    def fake_run_review_agent(**kwargs):
        Path(kwargs["output_path"]).write_text(
            "# Review Summary: feat-galerkin-rom\n\n## High-Level Summary\nUpdated the model.\n",
            encoding="utf-8",
        )
        manifest = load_or_create_manifest(paths, kwargs["task_slug"])
        phase = phase_entry(manifest, "review")
        phase["status"] = "succeeded"
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0

    monkeypatch.setattr("romjax.cli.run_review_agent", fake_run_review_agent)

    review_path = review_task(paths, "feat-galerkin-rom")

    assert "Updated the model." in review_path.read_text(encoding="utf-8")
    assert_task_state(paths, "feat-galerkin-rom", "review")


def test_review_task_rejects_existing_review_output(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nImplemented feature.\n", encoding="utf-8")
    task_path.rename(paths.finished_dir / task_path.name)
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    Path(manifest["worktree_path"]).mkdir()
    write_manifest(paths, "feat-galerkin-rom", manifest)
    info = phase_paths(paths, "feat-galerkin-rom", "review")
    info["output_path"].parent.mkdir(parents=True, exist_ok=True)
    info["output_path"].write_text("existing review\n", encoding="utf-8")

    with pytest.raises(Exception):
        review_task(paths, "feat-galerkin-rom")


def test_implementation_prompt_uses_mode_appropriate_summary_instruction(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nImplemented feature.\n", encoding="utf-8")
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    output_path = phase_paths(paths, "feat-galerkin-rom", "implementation")["output_path"]

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

    expected = (
        "Write a concise final summary describing what changed and any blockers "
        f"to {output_path}."
    )
    assert expected in interactive_prompt
    assert "Return a concise final summary as your last assistant message" in headless_prompt
    assert f"any blockers to {output_path}" not in headless_prompt


def test_git_output_reads_git_state(tmp_path: Path):
    repo_root = init_repo(tmp_path)
    (repo_root / "example.txt").write_text("hello\n", encoding="utf-8")
    assert "example.txt" in git_output(repo_root, ["status", "--short"])


def test_agent_environment_uses_shared_repo_venv(tmp_path: Path):
    repo_root = init_repo(tmp_path)
    paths = get_task_paths(repo_root)
    venv_path = shared_venv_path(paths)
    bin_dir = shared_venv_bin_dir(paths)
    bin_dir.mkdir(parents=True)
    env = agent_environment(paths)
    assert env["VIRTUAL_ENV"] == str(venv_path)
    assert env["UV_PROJECT_ENVIRONMENT"] == str(venv_path)
    assert str(bin_dir) in env["PATH"].split(os.pathsep)


def test_terminate_process_uses_kill_on_platform_without_killpg(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, int]] = []
    monkeypatch.delattr("romjax.cli.os.killpg", raising=False)
    monkeypatch.setattr("romjax.cli.os.kill", lambda pid, sig: calls.append((pid, sig)))
    terminate_process(1234)
    assert calls == [(1234, signal.SIGTERM)]


def test_task_runtime_status_reports_not_started(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "fix-bug-1127")
    status = task_runtime_status(paths, "fix-bug-1127")
    assert status["task_state"] == "open"
    assert status["run_status"] == "open"


def test_stop_task_updates_running_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "feat-galerkin-rom")
    move_task_to_state(paths, "feat-galerkin-rom", "running")
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    phase = phase_entry(manifest, "implementation")
    phase["status"] = "running"
    phase["pid"] = 4321
    write_manifest(paths, "feat-galerkin-rom", manifest)

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("romjax.cli.process_exists", lambda pid: True)
    monkeypatch.setattr("romjax.cli.terminate_process", lambda pid: killed.append((pid, signal.SIGTERM)))

    updated = stop_task(paths, "feat-galerkin-rom")

    assert killed and killed[0][0] == 4321
    assert updated["phases"]["implementation"]["status"] == "stopped"
    assert_task_state(paths, "feat-galerkin-rom", "stopped")


def test_archive_task_rejects_running_phase(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.rename(paths.running_dir / task_path.name)
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    phase = phase_entry(manifest, "implementation")
    phase["status"] = "running"
    write_manifest(paths, "feat-galerkin-rom", manifest)

    with pytest.raises(Exception):
        archive_task(paths, "feat-galerkin-rom")


def test_clean_task_removes_task_doc_run_dir_worktree_and_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nDone\n", encoding="utf-8")

    def fake_run_implementation_agent(**kwargs):
        Path(kwargs["output_path"]).write_text("summary\n", encoding="utf-8")
        manifest = load_or_create_manifest(paths, kwargs["task_slug"])
        phase = phase_entry(manifest, "implementation")
        phase["status"] = "succeeded"
        phase["exit_code"] = 0
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0

    monkeypatch.setattr("romjax.cli.run_implementation_agent", fake_run_implementation_agent)
    manifest = start_task(paths, "feat-galerkin-rom")
    worktree_path = Path(manifest["worktree_path"])
    run_dir = paths.runs_dir / "feat-galerkin-rom"

    assert worktree_path.exists()
    assert run_dir.exists()
    assert "feat-galerkin-rom" in subprocess.run(
        ["git", "branch", "--list", "feat-galerkin-rom"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

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


def test_clean_task_rejects_running_task(tmp_path: Path):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "feat-galerkin-rom")
    move_task_to_state(paths, "feat-galerkin-rom", "running")
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    phase_entry(manifest, "implementation")["status"] = "running"
    write_manifest(paths, "feat-galerkin-rom", manifest)

    with pytest.raises(Exception):
        clean_task(paths, "feat-galerkin-rom")


def test_cli_plan_command_passes_headless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    repo_root = prepare_repo(init_repo(tmp_path))
    monkeypatch.setattr("romjax.cli.collect_plan_prompt", lambda: "Plan a new Galerkin ROM feature")

    def fake_plan_task(paths, task_slug: str, user_prompt: str | None = None, headless: bool = False):
        assert task_slug == "feat-galerkin-rom"
        assert user_prompt == "Plan a new Galerkin ROM feature"
        assert headless is True
        task_path = paths.open_dir / f"{task_slug}.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text("# feat-galerkin-rom\n", encoding="utf-8")
        return task_path

    monkeypatch.setattr("romjax.cli.plan_task", fake_plan_task)

    exit_code = cli(["plan", "feat-galerkin-rom", "--headless"], repo_root=repo_root)

    assert exit_code == 0
    assert "feat-galerkin-rom.md" in capsys.readouterr().out


def test_cli_start_command_returns_agent_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    monkeypatch.setattr(
        "romjax.cli.start_task",
        lambda _paths, task_slug, headless=False: {"task_slug": task_slug, "exit_code": 7},
    )
    assert cli(["start", "feat-galerkin-rom", "--headless"], repo_root=repo_root) == 7


def test_cli_clean_command_requires_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    repo_root = prepare_repo(init_repo(tmp_path))
    monkeypatch.setattr("romjax.cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "no")

    exit_code = cli(["clean", "feat-galerkin-rom"], repo_root=repo_root)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Warning: task clean" in captured.out
    assert "Task clean cancelled." in captured.err


def test_cli_review_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    repo_root = prepare_repo(init_repo(tmp_path))

    def fake_review_task(paths, task_slug: str, headless: bool = False):
        assert headless is False
        review_path = paths.runs_dir / task_slug / f"{task_slug}-review.md"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text("# Review Summary: feat-galerkin-rom\n", encoding="utf-8")
        return review_path

    monkeypatch.setattr("romjax.cli.review_task", fake_review_task)

    assert cli(["review", "feat-galerkin-rom"], repo_root=repo_root) == 0
    assert "feat-galerkin-rom-review.md" in capsys.readouterr().out


def test_cli_list_hides_archived_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    open_task = create_task(paths, "feat-galerkin-rom")
    archived_task = create_task(paths, "fix-bug-1127")
    archived_task.rename(paths.archive_dir / archived_task.name)

    exit_code = cli(["list"], repo_root=repo_root)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert open_task.stem in output
    assert "fix-bug-1127" not in output


def test_list_includes_new_directory_backed_states(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    for slug, state in (
        ("feat-open", "open"),
        ("feat-running", "running"),
        ("feat-finished", "finished"),
        ("feat-stopped", "stopped"),
        ("feat-review", "review"),
    ):
        create_task(paths, slug)
        if state != "open":
            move_task_to_state(paths, slug, state)

    exit_code = cli(["list"], repo_root=repo_root)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TASK" in output
    assert "STATE" in output
    assert "feat-open" in output and "open" in output
    assert "feat-running" in output and "running" in output
    assert "feat-finished" in output and "finished" in output
    assert "feat-stopped" in output and "stopped" in output
    assert "feat-review" in output and "review" in output
