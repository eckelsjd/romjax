from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from romjax.cli import (
    agent_environment,
    archive_task,
    cli,
    collect_plan_prompt,
    create_task,
    current_git_branch,
    ensure_task_layout,
    ensure_task_ready,
    find_task_document,
    get_task_paths,
    git_output,
    latest_codex_session_id_for_home,
    load_manifest,
    load_or_create_manifest,
    move_task_to_state,
    phase_codex_home,
    phase_entry,
    phase_paths,
    phase_resume_command,
    plan_task,
    review_task,
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
        phase["session_id"] = "plan-session"
        phase["status"] = "succeeded"
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0, "plan-session"

    monkeypatch.setattr("romjax.cli.run_planning_agent", fake_run_planning_agent)

    task_path = plan_task(paths, "feat-galerkin-rom", user_prompt="Add a Galerkin ROM feature")

    assert task_path.exists()
    assert "Implement a Galerkin ROM feature." in task_path.read_text(encoding="utf-8")
    manifest = load_manifest(paths, "feat-galerkin-rom")
    assert manifest["phases"]["planning"]["session_id"] == "plan-session"
    assert_task_state(paths, "feat-galerkin-rom", "open")


def test_plan_task_resumes_existing_session_without_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "fix-bug-1127")
    manifest = load_or_create_manifest(paths, "fix-bug-1127")
    phase = phase_entry(manifest, "planning")
    phase["session_id"] = "existing-plan-session"
    write_manifest(paths, "fix-bug-1127", manifest)
    info = phase_paths(paths, "fix-bug-1127", "planning")
    info["prompt_path"].write_text("resume plan prompt\n", encoding="utf-8")

    def fake_run_planning_agent(**kwargs):
        Path(kwargs["output_path"]).write_text("# fix-bug-1127\n\n## Problem\nFix issue 1127.\n", encoding="utf-8")
        return 0, "existing-plan-session"

    monkeypatch.setattr("romjax.cli.run_planning_agent", fake_run_planning_agent)

    planned = plan_task(paths, "fix-bug-1127")

    assert planned == task_path
    assert "Fix issue 1127." in planned.read_text(encoding="utf-8")


def test_start_task_creates_worktree_and_runs_implementation_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    task_path = create_task(paths, "feat-galerkin-rom")
    task_path.write_text("# feat-galerkin-rom\n\n## Summary\nDone\n", encoding="utf-8")

    def fake_run_implementation_agent(**kwargs):
        Path(kwargs["output_path"]).write_text("summary\n", encoding="utf-8")
        manifest = load_or_create_manifest(paths, kwargs["task_slug"])
        phase = phase_entry(manifest, "implementation")
        phase["session_id"] = "impl-session"
        phase["status"] = "succeeded"
        phase["exit_code"] = 0
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0, "impl-session"

    monkeypatch.setattr("romjax.cli.run_implementation_agent", fake_run_implementation_agent)

    manifest = start_task(paths, "feat-galerkin-rom")

    assert Path(manifest["worktree_path"]).exists()
    assert_task_state(paths, "feat-galerkin-rom", "finished")
    stored = load_manifest(paths, "feat-galerkin-rom")
    assert stored["phases"]["implementation"]["session_id"] == "impl-session"
    info = phase_paths(paths, "feat-galerkin-rom", "implementation")
    assert info["prompt_path"].name == "feat-galerkin-rom-implementation-prompt.md"
    assert info["output_path"].name == "feat-galerkin-rom-implementation.md"
    assert info["log_path"].name == "feat-galerkin-rom-implementation.log"


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
        phase["session_id"] = "review-session"
        phase["status"] = "succeeded"
        write_manifest(paths, kwargs["task_slug"], manifest)
        return 0, "review-session"

    monkeypatch.setattr("romjax.cli.run_review_agent", fake_run_review_agent)

    review_path = review_task(paths, "feat-galerkin-rom")

    assert "Updated the model." in review_path.read_text(encoding="utf-8")
    assert_task_state(paths, "feat-galerkin-rom", "review")


def test_git_output_reads_git_state(tmp_path: Path):
    repo_root = init_repo(tmp_path)
    (repo_root / "example.txt").write_text("hello\n", encoding="utf-8")
    assert "example.txt" in git_output(repo_root, ["status", "--short"])


def test_agent_environment_uses_shared_repo_venv(tmp_path: Path):
    repo_root = init_repo(tmp_path)
    paths = get_task_paths(repo_root)
    venv_path = shared_venv_path(paths)
    (venv_path / "bin").mkdir(parents=True)
    env = agent_environment(paths)
    assert env["VIRTUAL_ENV"] == str(venv_path)
    assert env["UV_PROJECT_ENVIRONMENT"] == str(venv_path)
    assert str(venv_path / "bin") in env["PATH"]


def test_phase_codex_home_is_isolated_per_phase(tmp_path: Path):
    repo_root = init_repo(tmp_path)
    paths = get_task_paths(repo_root)

    planning_home = phase_codex_home(paths, "feat-galerkin-rom", "planning")
    review_home = phase_codex_home(paths, "feat-galerkin-rom", "review")

    assert planning_home != review_home
    assert planning_home.name == "feat-galerkin-rom-planning-codex-home"
    assert review_home.name == "feat-galerkin-rom-review-codex-home"


def test_latest_codex_session_id_for_home_falls_back_to_sessions(tmp_path: Path):
    codex_home = tmp_path / "codex-home"
    session_dir = codex_home / "sessions" / "2026" / "04" / "21"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "rollout-test.jsonl"
    session_file.write_text('{"id":"session-from-file"}\n{"type":"turn.started"}\n', encoding="utf-8")

    assert latest_codex_session_id_for_home(codex_home) == "session-from-file"


def test_phase_resume_command_uses_last_without_session_id():
    interactive = phase_resume_command(
        "planning",
        session_id=None,
        cwd=Path("/tmp"),
        repo_root=Path("/tmp/repo"),
        headless=False,
    )
    headless = phase_resume_command(
        "planning",
        session_id=None,
        cwd=Path("/tmp"),
        repo_root=Path("/tmp/repo"),
        headless=True,
    )

    assert "--last" in interactive
    assert "--last" in headless


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


def test_cli_plan_command_resumes_without_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    repo_root = prepare_repo(init_repo(tmp_path))
    paths = get_task_paths(repo_root)
    create_task(paths, "feat-galerkin-rom")
    manifest = load_or_create_manifest(paths, "feat-galerkin-rom")
    phase_entry(manifest, "planning")["codex_home"] = str(phase_codex_home(paths, "feat-galerkin-rom", "planning"))
    write_manifest(paths, "feat-galerkin-rom", manifest)
    monkeypatch.setattr(
        "romjax.cli.collect_plan_prompt",
        lambda: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )
    monkeypatch.setattr(
        "romjax.cli.plan_task",
        lambda paths, task_slug, user_prompt=None, headless=False: paths.open_dir / f"{task_slug}.md",
    )

    exit_code = cli(["plan", "feat-galerkin-rom"], repo_root=repo_root)

    assert exit_code == 0
    assert "feat-galerkin-rom.md" in capsys.readouterr().out


def test_cli_start_command_returns_agent_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = prepare_repo(init_repo(tmp_path))
    monkeypatch.setattr(
        "romjax.cli.start_task",
        lambda _paths, task_slug, headless=False: {"task_slug": task_slug, "exit_code": 7},
    )
    assert cli(["start", "feat-galerkin-rom", "--headless"], repo_root=repo_root) == 7


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
    assert "feat-open\topen\topen" in output
    assert "feat-running\trunning\trunning" in output
    assert "feat-finished\tfinished\tfinished" in output
    assert "feat-stopped\tstopped\tstopped" in output
    assert "feat-review\treview\treview" in output
