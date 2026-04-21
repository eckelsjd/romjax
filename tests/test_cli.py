from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from romjax.cli import (
    archive_task,
    cli,
    collect_plan_prompt,
    create_task,
    ensure_task_layout,
    ensure_task_ready,
    get_task_paths,
    load_manifest,
    plan_task,
    run_agent_once,
    start_task,
    stop_task,
    task_runtime_status,
    validate_task_slug,
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
    subprocess.run(
        ["git", "commit", "-m", "test setup"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo_root


def prepare_templates(repo_root: Path) -> Path:
    paths = get_task_paths(repo_root)
    ensure_task_layout(paths)
    (paths.templates_dir / "feat.md").write_text(
        "# {task_slug}\n\n## Summary\nTODO: fill me\n",
        encoding="utf-8",
    )
    (paths.templates_dir / "fix.md").write_text(
        "# {task_slug}\n\n## Problem\nTODO: fill me\n",
        encoding="utf-8",
    )
    return repo_root


def test_validate_task_slug():
    assert validate_task_slug("feat-galerkin-rom") == ("feat", "galerkin-rom")

    with pytest.raises(Exception):
        validate_task_slug("FeatBad")

    with pytest.raises(Exception):
        validate_task_slug("feat")


def test_create_task_uses_prefix_template(tmp_path: Path):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    task_path = create_task(paths, "feat-galerkin-rom")

    assert task_path == paths.open_dir / "feat-galerkin-rom.md"
    text = task_path.read_text(encoding="utf-8")
    assert "# feat-galerkin-rom" in text
    assert "TODO: fill me" in text


def test_ensure_task_ready_rejects_unfilled_template(tmp_path: Path):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")

    with pytest.raises(Exception):
        ensure_task_ready(task_path)

    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nImplemented\n", encoding="utf-8")
    ensure_task_ready(task_path)


def test_start_task_creates_worktree_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = init_repo(tmp_path)
    paths = get_task_paths(repo_root)
    ensure_task_layout(paths)
    (paths.templates_dir / "feat.md").write_text(
        "# {task_slug}\n\n## Summary\nDone\n",
        encoding="utf-8",
    )
    create_task(paths, "feat-galerkin-rom")

    launched: dict[str, str] = {}

    def fake_launch(_, task_slug: str) -> int:
        launched["task_slug"] = task_slug
        return 4242

    monkeypatch.setattr("romjax.cli.launch_agent_runner", fake_launch)

    manifest = start_task(paths, "feat-galerkin-rom")

    assert launched["task_slug"] == "feat-galerkin-rom"
    assert manifest["pid"] == 4242
    assert Path(manifest["worktree_path"]).exists()
    assert not (paths.open_dir / "feat-galerkin-rom.md").exists()
    assert (paths.in_progress_dir / "feat-galerkin-rom.md").exists()

    stored = load_manifest(paths, "feat-galerkin-rom")
    assert stored["status"] == "running"
    assert stored["branch_name"] == "feat-galerkin-rom"
    assert Path(stored["prompt_path"]).exists()


def test_run_agent_once_updates_manifest_and_exit_code(tmp_path: Path):
    repo_root = init_repo(tmp_path)
    paths = get_task_paths(repo_root)
    ensure_task_layout(paths)
    run_dir = paths.runs_dir / "feat-galerkin-rom"
    run_dir.mkdir(parents=True)
    prompt_path = run_dir / "feat-galerkin-rom-prompt.md"
    prompt_path.write_text("agent prompt\n", encoding="utf-8")

    worktree_path = repo_root.parent / "feat-galerkin-rom"
    worktree_path.mkdir()

    manifest = {
        "task_slug": "feat-galerkin-rom",
        "task_path": str(paths.in_progress_dir / "feat-galerkin-rom.md"),
        "repo_root": str(repo_root),
        "worktree_path": str(worktree_path),
        "branch_name": "feat-galerkin-rom",
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "feat-galerkin-rom.log"),
        "prompt_path": str(prompt_path),
        "summary_path": str(run_dir / "feat-galerkin-rom-summary.md"),
        "exit_code_path": str(run_dir / "feat-galerkin-rom.exitcode"),
        "agent_command_template": (
            f"{sys.executable} -c "
            "\"from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text('summary\\n', encoding='utf-8'); print('agent ran')\" "
            "{summary_path}"
        ),
        "status": "running",
        "pid": 9999,
    }
    (paths.in_progress_dir / "feat-galerkin-rom.md").write_text("Task body\n", encoding="utf-8")
    (run_dir / "feat-galerkin-rom.json").write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = run_agent_once(paths, "feat-galerkin-rom")

    assert exit_code == 0
    updated = load_manifest(paths, "feat-galerkin-rom")
    assert updated["status"] == "succeeded"
    assert Path(updated["summary_path"]).read_text(encoding="utf-8") == "summary\n"
    assert Path(updated["exit_code_path"]).read_text(encoding="utf-8") == "0"
    assert "agent ran" in Path(updated["log_path"]).read_text(encoding="utf-8")


def test_task_runtime_status_reports_not_started(tmp_path: Path):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "fix-bug-1127")

    status = task_runtime_status(paths, "fix-bug-1127")

    assert status["task_state"] == "open"
    assert status["run_status"] == "not_started"


def test_collect_plan_prompt_raises_on_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("romjax.cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "   ")

    with pytest.raises(Exception):
        collect_plan_prompt()


def test_plan_task_creates_missing_task_and_fills_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)

    def fake_run_planning_agent(**kwargs):
        plan_prompt = Path(kwargs["plan_prompt_path"]).read_text(encoding="utf-8")
        assert "Add a Galerkin ROM feature" in plan_prompt
        Path(kwargs["plan_output_path"]).write_text(
            "# feat-galerkin-rom\n\n## Summary\nImplement a Galerkin ROM feature.\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("romjax.cli.run_planning_agent", fake_run_planning_agent)

    task_path = plan_task(paths, "feat-galerkin-rom", "Add a Galerkin ROM feature")

    assert task_path.exists()
    text = task_path.read_text(encoding="utf-8")
    assert "Implement a Galerkin ROM feature." in text
    assert "TODO:" not in text


def test_plan_task_reuses_existing_open_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    existing_task = create_task(paths, "fix-bug-1127")

    def fake_run_planning_agent(**kwargs):
        Path(kwargs["plan_output_path"]).write_text(
            "# fix-bug-1127\n\n## Problem\nFix issue 1127.\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("romjax.cli.run_planning_agent", fake_run_planning_agent)

    planned_task = plan_task(paths, "fix-bug-1127", "Plan a bug fix for issue 1127")

    assert planned_task == existing_task
    assert "Fix issue 1127." in planned_task.read_text(encoding="utf-8")


def test_stop_task_updates_manifest_and_terminates_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = init_repo(tmp_path)
    paths = get_task_paths(repo_root)
    ensure_task_layout(paths)
    run_dir = paths.runs_dir / "feat-galerkin-rom"
    run_dir.mkdir(parents=True)
    (paths.in_progress_dir / "feat-galerkin-rom.md").write_text("Task body\n", encoding="utf-8")
    manifest = {
        "task_slug": "feat-galerkin-rom",
        "task_path": str(paths.in_progress_dir / "feat-galerkin-rom.md"),
        "repo_root": str(repo_root),
        "worktree_path": str(repo_root.parent / "feat-galerkin-rom"),
        "branch_name": "feat-galerkin-rom",
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "feat-galerkin-rom.log"),
        "prompt_path": str(run_dir / "feat-galerkin-rom-prompt.md"),
        "summary_path": str(run_dir / "feat-galerkin-rom-summary.md"),
        "exit_code_path": str(run_dir / "feat-galerkin-rom.exitcode"),
        "agent_command_template": "echo ignored",
        "status": "running",
        "pid": 4321,
    }
    (run_dir / "feat-galerkin-rom.json").write_text(json.dumps(manifest), encoding="utf-8")

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("romjax.cli.process_exists", lambda pid: True)
    monkeypatch.setattr("romjax.cli.os.killpg", lambda pid, sig: killed.append((pid, sig)))

    updated = stop_task(paths, "feat-galerkin-rom")

    assert updated["status"] == "stopped"
    assert killed and killed[0][0] == 4321
    stored = load_manifest(paths, "feat-galerkin-rom")
    assert stored["status"] == "stopped"
    assert "stopped_at" in stored


def test_archive_task_moves_doc_and_list_excludes_archived(tmp_path: Path):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "fix-bug-1127")
    task_path.rename(paths.done_dir / task_path.name)

    archive_path = archive_task(paths, "fix-bug-1127")

    assert archive_path == paths.archive_dir / "fix-bug-1127.md"
    assert archive_path.exists()
    listed = cli(["list"], repo_root=repo_root)
    assert listed == 0


def test_archive_task_rejects_running_task(tmp_path: Path):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.rename(paths.in_progress_dir / task_path.name)
    run_dir = paths.runs_dir / "feat-galerkin-rom"
    run_dir.mkdir(parents=True)
    manifest = {
        "task_slug": "feat-galerkin-rom",
        "task_path": str(paths.in_progress_dir / "feat-galerkin-rom.md"),
        "repo_root": str(repo_root),
        "worktree_path": str(repo_root.parent / "feat-galerkin-rom"),
        "branch_name": "feat-galerkin-rom",
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "feat-galerkin-rom.log"),
        "prompt_path": str(run_dir / "feat-galerkin-rom-prompt.md"),
        "summary_path": str(run_dir / "feat-galerkin-rom-summary.md"),
        "exit_code_path": str(run_dir / "feat-galerkin-rom.exitcode"),
        "agent_command_template": "echo ignored",
        "status": "running",
        "pid": 9999,
    }
    (run_dir / "feat-galerkin-rom.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Exception):
        archive_task(paths, "feat-galerkin-rom")


def test_cli_create_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo_root = prepare_templates(init_repo(tmp_path))

    exit_code = cli(["create", "feat-galerkin-rom"], repo_root=repo_root)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "feat-galerkin-rom.md" in captured.out


def test_cli_plan_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    repo_root = prepare_templates(init_repo(tmp_path))
    monkeypatch.setattr("romjax.cli.collect_plan_prompt", lambda: "Plan a new Galerkin ROM feature")

    def fake_plan_task(paths, task_slug: str, user_prompt: str):
        assert task_slug == "feat-galerkin-rom"
        assert user_prompt == "Plan a new Galerkin ROM feature"
        task_path = paths.open_dir / f"{task_slug}.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text("# feat-galerkin-rom\n\n## Summary\nPlanned.\n", encoding="utf-8")
        return task_path

    monkeypatch.setattr("romjax.cli.plan_task", fake_plan_task)

    exit_code = cli(["plan", "feat-galerkin-rom"], repo_root=repo_root)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "feat-galerkin-rom.md" in captured.out


def test_cli_list_hides_archived_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo_root = prepare_templates(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    open_task = create_task(paths, "feat-galerkin-rom")
    archived_task = create_task(paths, "fix-bug-1127")
    archived_task.rename(paths.archive_dir / archived_task.name)

    exit_code = cli(["list"], repo_root=repo_root)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert open_task.stem in captured.out
    assert "fix-bug-1127" not in captured.out
