from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest

from romjax.task_cli import (
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
        phase["status"] = "succeeded" if exit_code == 0 else "failed"
        phase["exit_code"] = exit_code
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

    task_path = plan_task(paths, "feat-galerkin-rom", user_prompt="Add a Galerkin ROM feature")

    assert task_path.exists()
    assert "Implement a Galerkin ROM feature." in task_path.read_text(encoding="utf-8")
    assert load_manifest(paths, "feat-galerkin-rom")["phases"]["planning"]["status"] == "succeeded"
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

    manifest = start_task(paths, "feat-galerkin-rom")

    assert Path(manifest["worktree_path"]).exists()
    assert_task_state(paths, "feat-galerkin-rom", "finished")
    stored = load_manifest(paths, "feat-galerkin-rom")
    assert stored["phases"]["implementation"]["status"] == "succeeded"
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

    def fake_plan_task(_paths, task_slug: str, user_prompt: str | None = None, headless: bool = False):
        assert task_slug == "feat-galerkin-rom"
        assert user_prompt == "Plan a new Galerkin ROM feature"
        assert headless is True
        task_path = paths.open_dir / f"{task_slug}.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text("# feat-galerkin-rom\n", encoding="utf-8")
        return task_path

    monkeypatch.setattr("romjax.task_cli.plan_task", fake_plan_task)
    assert cli(["plan", "feat-galerkin-rom", "--headless"], repo_root=repo_root) == 0
    assert "feat-galerkin-rom.md" in capsys.readouterr().out

    monkeypatch.setattr(
        "romjax.task_cli.start_task",
        lambda _paths, task_slug, headless=False: {"task_slug": task_slug, "exit_code": 7},
    )
    assert cli(["start", "feat-galerkin-rom", "--headless"], repo_root=repo_root) == 7

    def fake_review_task(_paths, task_slug: str, headless: bool = False):
        assert headless is False
        review_path = paths.runs_dir / task_slug / f"{task_slug}-review.md"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text("# Review Summary: feat-galerkin-rom\n", encoding="utf-8")
        return review_path

    monkeypatch.setattr("romjax.task_cli.review_task", fake_review_task)
    assert cli(["review", "feat-galerkin-rom"], repo_root=repo_root) == 0
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
