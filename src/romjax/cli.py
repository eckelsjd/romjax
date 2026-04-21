"""Task CLI for local AI-assisted development workflows."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_AGENT_CMD = (
    "codex exec --full-auto --cd {worktree_path} "
    "--add-dir {repo_root} "
    "--color never "
    "--output-last-message {summary_path} -"
)
DEFAULT_PLAN_CMD = (
    "codex exec --sandbox read-only --cd {repo_root} "
    "--output-last-message {plan_output_path} -"
)
DEFAULT_REVIEW_CMD = "codex exec --sandbox read-only --cd {repo_root} --output-last-message {review_output_path} -"
TASK_STATES = ("open", "running", "finished", "stopped", "review", "archive")
PHASE_NAMES = ("planning", "implementation", "review")


class TaskWorkflowError(RuntimeError):
    """Raised when the task workflow encounters an invalid local state."""


@dataclass(slots=True)
class TaskPaths:
    """Common repository paths used by the local task workflow."""

    repo_root: Path
    tasks_root: Path
    templates_dir: Path
    open_dir: Path
    running_dir: Path
    finished_dir: Path
    stopped_dir: Path
    review_dir: Path
    archive_dir: Path
    runs_dir: Path


def utc_now() -> str:
    """Return an ISO-8601 timestamp in UTC."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_task_paths(repo_root: Path | None = None) -> TaskPaths:
    """Return workflow paths for a repository root.

    :param repo_root: root of the git repository
    :return: normalized workflow paths
    """
    root = (repo_root or Path.cwd()).resolve()
    tasks_root = root / ".agents" / "tasks"
    return TaskPaths(
        repo_root=root,
        tasks_root=tasks_root,
        templates_dir=tasks_root / "templates",
        open_dir=tasks_root / "open",
        running_dir=tasks_root / "running",
        finished_dir=tasks_root / "finished",
        stopped_dir=tasks_root / "stopped",
        review_dir=tasks_root / "review",
        archive_dir=tasks_root / "archive",
        runs_dir=tasks_root / "runs",
    )


def ensure_task_layout(paths: TaskPaths) -> None:
    """Create the task workflow directory layout if needed."""
    for directory in (
        paths.tasks_root,
        paths.templates_dir,
        paths.open_dir,
        paths.running_dir,
        paths.finished_dir,
        paths.stopped_dir,
        paths.review_dir,
        paths.archive_dir,
        paths.runs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def validate_task_slug(task_slug: str) -> tuple[str, str]:
    """Validate a task slug and return its template type and short name.

    Slugs must follow ``<task-type>-<task-name>`` using lowercase letters,
    digits, and hyphens.

    :param task_slug: task identifier supplied on the CLI
    :return: task type and remainder of the slug
    """
    if not task_slug or task_slug.startswith("-") or task_slug.endswith("-"):
        raise TaskWorkflowError("Task slug must follow <task-type>-<task-name>.")
    parts = task_slug.split("-")
    if len(parts) < 2:
        raise TaskWorkflowError("Task slug must follow <task-type>-<task-name>.")
    if any(not part or not part.isalnum() or part.lower() != part for part in parts):
        raise TaskWorkflowError("Task slug may only contain lowercase letters, numbers, and hyphens.")
    return parts[0], "-".join(parts[1:])


def task_metadata(paths: TaskPaths, task_slug: str) -> dict[str, str]:
    """Build standard metadata available to task templates."""
    task_type, task_name = validate_task_slug(task_slug)
    task_doc = paths.open_dir / f"{task_slug}.md"
    worktree_path = paths.repo_root.parent / task_slug
    return {
        "task_slug": task_slug,
        "task_type": task_type,
        "task_name": task_name,
        "branch_name": task_slug,
        "created_at": utc_now(),
        "status": "open",
        "task_doc": str(task_doc),
        "worktree_path": str(worktree_path),
        "repo_root": str(paths.repo_root),
    }


def initial_manifest(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Build a default manifest structure for a task."""
    worktree_path = paths.repo_root.parent / task_slug
    return {
        "task_slug": task_slug,
        "repo_root": str(paths.repo_root),
        "worktree_path": str(worktree_path),
        "branch_name": task_slug,
        "base_ref": current_git_branch(paths.repo_root),
        "phases": {},
    }


def create_task(paths: TaskPaths, task_slug: str) -> Path:
    """Create a task document from its matching template.

    :param paths: workflow paths
    :param task_slug: slug following ``<task-type>-<task-name>``
    :return: path to the newly created task markdown document
    """
    ensure_task_layout(paths)
    task_type, _ = validate_task_slug(task_slug)
    template_path = paths.templates_dir / f"{task_type}.md"
    if not template_path.exists():
        raise TaskWorkflowError(f"No task template found for '{task_type}'. Expected {template_path}.")

    task_path = paths.open_dir / f"{task_slug}.md"
    if task_path.exists():
        raise TaskWorkflowError(f"Task document already exists: {task_path}")

    rendered = template_path.read_text(encoding="utf-8").format(**task_metadata(paths, task_slug))
    task_path.write_text(rendered, encoding="utf-8")
    return task_path


def get_or_create_open_task(paths: TaskPaths, task_slug: str) -> Path:
    """Return an open task document, creating it from the template if needed."""
    try:
        state, task_path = find_task_document(paths, task_slug)
    except TaskWorkflowError:
        return create_task(paths, task_slug)
    if state != "open":
        raise TaskWorkflowError(f"Task '{task_slug}' is not open; found state '{state}'.")
    return task_path


def state_directory(paths: TaskPaths, state: str) -> Path:
    """Return the directory for a task state."""
    mapping = {
        "open": paths.open_dir,
        "running": paths.running_dir,
        "finished": paths.finished_dir,
        "stopped": paths.stopped_dir,
        "review": paths.review_dir,
        "archive": paths.archive_dir,
    }
    try:
        return mapping[state]
    except KeyError as exc:
        raise TaskWorkflowError(f"Unsupported task state '{state}'.") from exc


def move_task_to_state(paths: TaskPaths, task_slug: str, state: str) -> Path:
    """Move a task document to the directory representing its new state."""
    _, task_path = find_task_document(paths, task_slug)
    target_path = state_directory(paths, state) / task_path.name
    if task_path == target_path:
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.rename(target_path)
    return target_path


def find_task_document(paths: TaskPaths, task_slug: str) -> tuple[str, Path]:
    """Locate a task markdown document across known task states."""
    candidates = {
        "open": paths.open_dir / f"{task_slug}.md",
        "running": paths.running_dir / f"{task_slug}.md",
        "finished": paths.finished_dir / f"{task_slug}.md",
        "stopped": paths.stopped_dir / f"{task_slug}.md",
        "review": paths.review_dir / f"{task_slug}.md",
        "archive": paths.archive_dir / f"{task_slug}.md",
    }
    for state, candidate in candidates.items():
        if candidate.exists():
            return state, candidate
    raise TaskWorkflowError(f"Could not find task document for '{task_slug}'.")


def unresolved_template_lines(text: str) -> list[str]:
    """Return lines that still look like unfilled template placeholders."""
    unresolved: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("TODO:") or line.startswith("[ ]"):
            unresolved.append(line)
    return unresolved


def ensure_task_ready(task_path: Path) -> None:
    """Check that a task markdown document is ready for execution."""
    text = task_path.read_text(encoding="utf-8")
    unresolved = unresolved_template_lines(text)
    if unresolved:
        preview = ", ".join(unresolved[:3])
        raise TaskWorkflowError(
            "Task document still contains unfinished template placeholders: "
            f"{preview}"
        )


def run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command inside the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def current_git_branch(repo_root: Path) -> str:
    """Return the current branch name for the repository root."""
    result = run_git(repo_root, ["branch", "--show-current"]).stdout.strip()
    return result or "main"


def ensure_git_worktree(paths: TaskPaths, task_slug: str) -> Path:
    """Create the git branch and sibling worktree for a task.

    :param paths: workflow paths
    :param task_slug: task identifier
    :return: worktree path
    """
    worktree_path = paths.repo_root.parent / task_slug
    if worktree_path.exists():
        raise TaskWorkflowError(f"Worktree path already exists: {worktree_path}")

    branch_lookup = run_git(paths.repo_root, ["branch", "--list", task_slug]).stdout.strip()
    if branch_lookup:
        raise TaskWorkflowError(f"Git branch already exists: {task_slug}")

    run_git(paths.repo_root, ["worktree", "add", "-b", task_slug, str(worktree_path), "HEAD"])
    return worktree_path


def build_agent_prompt(
    paths: TaskPaths,
    task_slug: str,
    task_path: Path,
    worktree_path: Path,
    summary_path: Path,
) -> str:
    """Build the default prompt given to the implementation agent."""
    return (
        "Read the repository instructions and execute the task.\n\n"
        f"Repository root: {paths.repo_root}\n"
        f"Task slug: {task_slug}\n"
        f"Task document: {task_path}\n"
        f"Worktree path: {worktree_path}\n"
        f"Required summary output file: {summary_path}\n"
        f"AGENTS instructions: {paths.repo_root / 'AGENTS.md'}\n\n"
        "Requirements:\n"
        "1. Read AGENTS.md before making changes.\n"
        "2. Read the task markdown document and implement only that scope.\n"
        "3. Use the task worktree as your working directory.\n"
        "4. Run `uv run rr lint` and `uv run rr test` before finishing if feasible.\n"
        f"5. Write a concise final summary describing what changed and any blockers to {summary_path}.\n"
    )


def build_plan_prompt(
    paths: TaskPaths,
    task_slug: str,
    task_path: Path,
    user_prompt: str,
    plan_output_path: Path,
) -> str:
    """Build the planning prompt used to fill a task template."""
    template_text = task_path.read_text(encoding="utf-8")
    return (
        "You are preparing a task markdown document for later implementation work.\n\n"
        f"Repository root: {paths.repo_root}\n"
        f"Task slug: {task_slug}\n"
        f"Task document path: {task_path}\n"
        f"Required output file: {plan_output_path}\n"
        f"AGENTS instructions: {paths.repo_root / 'AGENTS.md'}\n\n"
        "Use the following task template as the starting point. Replace placeholder TODO items "
        "with concrete, actionable recommendations based on the user prompt. Preserve markdown "
        "structure and keep the result suitable for a later `task start` run.\n\n"
        f"Before you finish, write the final markdown document to {plan_output_path}.\n"
        "Return only the final markdown document in your last assistant message. Do not wrap it in code fences.\n\n"
        "<user_prompt>\n"
        f"{user_prompt}\n"
        "</user_prompt>\n\n"
        "<task_template>\n"
        f"{template_text}\n"
        "</task_template>\n"
    )


def build_review_prompt(
    *,
    paths: TaskPaths,
    task_slug: str,
    task_path: Path,
    review_template_path: Path,
    diff_summary: str,
    committed_diff: str,
    working_tree_diff: str,
    status_short: str,
    review_output_path: Path,
) -> str:
    """Build the review prompt used to summarize implementation changes."""
    template_text = review_template_path.read_text(encoding="utf-8")
    task_text = task_path.read_text(encoding="utf-8")
    return (
        "You are preparing a concise human-facing code review summary for a completed task.\n\n"
        f"Repository root: {paths.repo_root}\n"
        f"Task slug: {task_slug}\n"
        f"Task document path: {task_path}\n"
        f"Review template path: {review_template_path}\n"
        f"Required output file: {review_output_path}\n"
        f"AGENTS instructions: {paths.repo_root / 'AGENTS.md'}\n\n"
        "Fill the review template using the task document and the available git diff context. "
        "Keep the summary concise and useful for a pull request or human review. "
        "If a section has no relevant changes, say so briefly rather than inventing details. "
        f"Before you finish, write the final markdown document to {review_output_path}. "
        "Return only the final markdown document and do not wrap it in code fences.\n\n"
        "<task_document>\n"
        f"{task_text}\n"
        "</task_document>\n\n"
        "<review_template>\n"
        f"{template_text}\n"
        "</review_template>\n\n"
        "<git_diff_summary>\n"
        f"{diff_summary}\n"
        "</git_diff_summary>\n\n"
        "<git_status_short>\n"
        f"{status_short}\n"
        "</git_status_short>\n\n"
        "<committed_diff_against_base>\n"
        f"{committed_diff}\n"
        "</committed_diff_against_base>\n\n"
        "<working_tree_diff>\n"
        f"{working_tree_diff}\n"
        "</working_tree_diff>\n"
    )


def run_dir_for_task(paths: TaskPaths, task_slug: str) -> Path:
    """Return the run directory for a task."""
    return paths.runs_dir / task_slug


def manifest_path(paths: TaskPaths, task_slug: str) -> Path:
    """Return the manifest path for a task run."""
    return run_dir_for_task(paths, task_slug) / f"{task_slug}.json"


def load_or_create_manifest(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Load a task run manifest, creating a default one if needed."""
    path = manifest_path(paths, task_slug)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = initial_manifest(paths, task_slug)
        write_manifest(paths, task_slug, manifest)
    manifest.setdefault("phases", {})
    manifest.setdefault("task_slug", task_slug)
    manifest.setdefault("repo_root", str(paths.repo_root))
    manifest.setdefault("worktree_path", str(paths.repo_root.parent / task_slug))
    manifest.setdefault("branch_name", task_slug)
    manifest.setdefault("base_ref", current_git_branch(paths.repo_root))
    return manifest


def load_manifest(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Load a task run manifest from disk."""
    path = manifest_path(paths, task_slug)
    if not path.exists():
        raise TaskWorkflowError(f"No run manifest exists for '{task_slug}'.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(paths: TaskPaths, task_slug: str, data: dict[str, Any]) -> Path:
    """Write a task run manifest to disk."""
    run_dir = run_dir_for_task(paths, task_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(paths, task_slug)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def phase_entry(manifest: dict[str, Any], phase_name: str) -> dict[str, Any]:
    """Return a mutable manifest entry for a task phase."""
    phases = manifest.setdefault("phases", {})
    return phases.setdefault(phase_name, {})


def render_agent_command(command_template: str, manifest: dict[str, Any]) -> list[str]:
    """Render the agent command template into argv form."""
    command = command_template.format(**manifest)
    if os.name == "nt":
        return split_windows_command(command)
    return shlex.split(command, posix=True)


def split_windows_command(command: str) -> list[str]:
    """Split a Windows command line into argv using CommandLineToArgvW."""
    shell32 = ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32
    shell32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    argc = ctypes.c_int()
    argv_ptr = shell32.CommandLineToArgvW(command, ctypes.byref(argc))
    if not argv_ptr:
        raise OSError("CommandLineToArgvW failed to parse command.")
    try:
        return [argv_ptr[index] for index in range(argc.value)]
    finally:
        kernel32.LocalFree(argv_ptr)


def normalize_cwd(path_value: str | Path) -> Path:
    """Normalize a cwd/path value loaded from task metadata or manifests."""
    return Path(path_value).expanduser().resolve()


def shared_venv_path(paths: TaskPaths) -> Path:
    """Return the canonical shared project virtual environment path."""
    return paths.repo_root / ".venv"


def agent_environment(paths: TaskPaths) -> dict[str, str]:
    """Build an environment that lets task worktrees reuse the main repo .venv."""
    env = os.environ.copy()
    venv_path = shared_venv_path(paths)
    if venv_path.exists():
        env["VIRTUAL_ENV"] = str(venv_path)
        env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
        bin_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return env


def terminate_process(pid: int) -> None:
    """Terminate a process or process group in a cross-platform way."""
    if hasattr(os, "killpg"):
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def collect_plan_prompt() -> str:
    """Collect a planning prompt from the user."""
    if sys.stdin.isatty():
        prompt = input("Planning prompt: ").strip()
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise TaskWorkflowError("Planning prompt cannot be empty.")
    return prompt


def latest_codex_session_id() -> str | None:
    """Return the most recent Codex session id from the local session index."""
    return latest_codex_session_id_for_home(Path.home() / ".codex")


def latest_codex_session_id_for_home(codex_home: Path) -> str | None:
    """Return the most recent Codex session id from a specific CODEX_HOME."""
    index_path = codex_home / "session_index.jsonl"
    if index_path.exists():
        latest: str | None = None
        for raw_line in index_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest = record.get("id") or latest
        if latest is not None:
            return latest

    sessions_root = codex_home / "sessions"
    if not sessions_root.exists():
        return None

    latest_file: Path | None = None
    for candidate in sessions_root.rglob("*.jsonl"):
        if latest_file is None or candidate.stat().st_mtime > latest_file.stat().st_mtime:
            latest_file = candidate
    if latest_file is None:
        return None

    try:
        first_line = latest_file.read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
    except (IndexError, OSError, json.JSONDecodeError):
        return None
    return record.get("id")


def phase_target_directory(phase_name: str, paths: TaskPaths, manifest: dict[str, Any]) -> Path:
    """Return the working directory used for a task phase."""
    if phase_name == "implementation":
        return normalize_cwd(manifest["worktree_path"])
    return paths.repo_root


def phase_codex_home(paths: TaskPaths, task_slug: str, phase_name: str) -> Path:
    """Return the isolated CODEX_HOME used for a task phase."""
    return run_dir_for_task(paths, task_slug) / f"{task_slug}-{phase_name}-codex-home"


def prepare_phase_codex_home(paths: TaskPaths, task_slug: str, phase_name: str) -> Path:
    """Prepare an isolated CODEX_HOME for a task phase with shared auth/config links."""
    codex_home = phase_codex_home(paths, task_slug, phase_name)
    codex_home.mkdir(parents=True, exist_ok=True)
    source_home = Path.home() / ".codex"
    for name in ("auth.json", "config.toml", "version.json"):
        source = source_home / name
        target = codex_home / name
        if source.exists() and not target.exists():
            try:
                target.symlink_to(source)
            except OSError:
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("rules", "skills"):
        source = source_home / name
        target = codex_home / name
        if source.exists() and not target.exists():
            try:
                target.symlink_to(source, target_is_directory=True)
            except OSError:
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
    return codex_home


def ensure_task_for_start(paths: TaskPaths, task_slug: str) -> Path:
    """Return an implementation-ready task document."""
    state, task_path = find_task_document(paths, task_slug)
    if state == "open":
        ensure_task_ready(task_path)
        return move_task_to_state(paths, task_slug, "running")
    if state in {"finished", "stopped", "review"}:
        return move_task_to_state(paths, task_slug, "running")
    if state == "running":
        return task_path
    raise TaskWorkflowError(f"Task '{task_slug}' cannot be started from state '{state}'.")


def write_phase_log(log_path: Path, lines: list[str]) -> None:
    """Append concise lifecycle lines to a phase log."""
    with log_path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip() + "\n")


def phase_paths(paths: TaskPaths, task_slug: str, phase_name: str) -> dict[str, Path]:
    """Return prompt/output/log paths for a phase."""
    run_dir = run_dir_for_task(paths, task_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = task_slug
    suffix = {
        "planning": "plan",
        "implementation": "implementation",
        "review": "review",
    }[phase_name]
    prompt_name = f"{stem}-{suffix}-prompt.md"
    output_name = f"{stem}-{suffix}.md"
    log_name = f"{stem}-{suffix}.log"
    return {
        "run_dir": run_dir,
        "prompt_path": run_dir / prompt_name,
        "output_path": run_dir / output_name,
        "log_path": run_dir / log_name,
    }


def build_phase_prompt(
    phase_name: str,
    *,
    paths: TaskPaths,
    task_slug: str,
    task_path: Path,
    manifest: dict[str, Any],
    output_path: Path,
    user_prompt: str | None = None,
) -> str:
    """Build the prompt for a given phase."""
    if phase_name == "planning":
        assert user_prompt is not None
        return build_plan_prompt(paths, task_slug, task_path, user_prompt, output_path)
    if phase_name == "implementation":
        return build_agent_prompt(
            paths,
            task_slug,
            task_path,
            normalize_cwd(manifest["worktree_path"]),
            output_path,
        )
    if phase_name == "review":
        review_template_path = paths.templates_dir / "review.md"
        if not review_template_path.exists():
            raise TaskWorkflowError(f"Review template not found: {review_template_path}")
        worktree_path = normalize_cwd(manifest["worktree_path"])
        base_ref = str(manifest.get("base_ref") or "main")
        return build_review_prompt(
            paths=paths,
            task_slug=task_slug,
            task_path=task_path,
            review_template_path=review_template_path,
            diff_summary=git_output(worktree_path, ["diff", "--stat", f"{base_ref}...HEAD"]),
            committed_diff=git_output(worktree_path, ["diff", f"{base_ref}...HEAD"]),
            working_tree_diff=git_output(worktree_path, ["diff"]),
            status_short=git_output(worktree_path, ["status", "--short"]),
            review_output_path=output_path,
        )
    raise TaskWorkflowError(f"Unsupported phase '{phase_name}'.")


def parse_headless_session_id(stdout_text: str) -> str | None:
    """Extract the thread id from codex exec JSON output."""
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            return event.get("thread_id")
    return None


def run_codex_interactive(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    codex_home: Path,
) -> tuple[int, str | None]:
    """Run Codex in interactive mode in the current terminal."""
    process = subprocess.Popen(argv, cwd=cwd, env=env)
    exit_code = process.wait()
    session_id = latest_codex_session_id_for_home(codex_home)
    write_phase_log(
        log_path,
        [
            f"[{utc_now()}] interactive command: {' '.join(argv[:4])} ...",
            f"[{utc_now()}] exit_code: {exit_code}",
            f"[{utc_now()}] session_id: {session_id or 'unknown'}",
        ],
    )
    return exit_code, session_id


def run_codex_headless(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    prompt_path: Path | None,
    log_path: Path,
) -> tuple[int, str | None]:
    """Run Codex in non-interactive exec mode and capture the session id."""
    stdin_handle = prompt_path.open("r", encoding="utf-8") if prompt_path is not None else None
    try:
        process = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdin=stdin_handle,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if stdin_handle is not None:
            stdin_handle.close()
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    return process.returncode, parse_headless_session_id(process.stdout)


def phase_resume_command(
    phase_name: str,
    *,
    session_id: str | None,
    cwd: Path,
    repo_root: Path,
    headless: bool,
) -> list[str]:
    """Build the argv for a resumed phase session."""
    if headless:
        argv = ["codex", "exec", "resume"]
        if session_id:
            argv.append(session_id)
        else:
            argv.append("--last")
        argv.append("--json")
        return argv
    argv = ["codex", "resume"]
    if session_id:
        argv.append(session_id)
    else:
        argv.append("--last")
    argv.extend(["--no-alt-screen", "-C", str(cwd)])
    if phase_name == "implementation":
        argv.extend(["--add-dir", str(repo_root)])
    return argv


def phase_new_command(
    phase_name: str,
    *,
    cwd: Path,
    repo_root: Path,
    prompt_text: str,
    output_path: Path,
    headless: bool,
) -> tuple[list[str], bool]:
    """Build the argv for a new phase session and whether stdin should supply the prompt."""
    if headless:
        if phase_name == "planning":
            argv = [
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "read-only",
                "--cd",
                str(repo_root),
                "--output-last-message",
                str(output_path),
                "-",
            ]
        elif phase_name == "review":
            argv = [
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "read-only",
                "--cd",
                str(cwd),
                "--output-last-message",
                str(output_path),
                "-",
            ]
        else:
            argv = [
                "codex",
                "exec",
                "--json",
                "--full-auto",
                "--cd",
                str(cwd),
                "--add-dir",
                str(repo_root),
                "--color",
                "never",
                "--output-last-message",
                str(output_path),
                "-",
            ]
        return argv, True

    argv = ["codex", "--no-alt-screen", "-C", str(cwd)]
    if phase_name == "planning":
        argv.extend(["-s", "read-only"])
    elif phase_name == "review":
        argv.extend(["-s", "read-only"])
    else:
        argv.extend(["--add-dir", str(repo_root)])
    argv.append(prompt_text)
    return argv, False


def run_phase_session(
    *,
    paths: TaskPaths,
    task_slug: str,
    phase_name: str,
    task_path: Path,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    headless: bool,
) -> tuple[int, str | None]:
    """Run or resume a task phase session."""
    manifest = load_or_create_manifest(paths, task_slug)
    phase = phase_entry(manifest, phase_name)
    target_cwd = phase_target_directory(phase_name, paths, manifest)
    env = agent_environment(paths)
    codex_home = prepare_phase_codex_home(paths, task_slug, phase_name)
    env["CODEX_HOME"] = str(codex_home)

    if phase.get("status") == "running":
        pid = phase.get("pid")
        if isinstance(pid, int) and process_exists(pid):
            raise TaskWorkflowError(f"Task '{task_slug}' phase '{phase_name}' is already running.")
        phase["status"] = "stopped"

    session_id = phase.get("session_id")
    resume_available = bool(session_id or phase.get("codex_home"))
    if resume_available:
        argv = phase_resume_command(
            phase_name,
            session_id=str(session_id) if session_id else None,
            cwd=target_cwd,
            repo_root=paths.repo_root,
            headless=headless,
        )
        prompt_for_stdin = None
    else:
        prompt_text = prompt_path.read_text(encoding="utf-8")
        argv, use_stdin = phase_new_command(
            phase_name,
            cwd=target_cwd,
            repo_root=paths.repo_root,
            prompt_text=prompt_text,
            output_path=output_path,
            headless=headless,
        )
        prompt_for_stdin = prompt_path if use_stdin else None

    phase.update(
        {
            "mode": "headless" if headless else "interactive",
            "status": "running",
            "prompt_path": str(prompt_path),
            "output_path": str(output_path),
            "log_path": str(log_path),
            "started_at": utc_now(),
        }
    )
    write_manifest(paths, task_slug, manifest)

    if headless:
        exit_code, new_session_id = run_codex_headless(
            argv=argv,
            cwd=target_cwd,
            env=env,
            prompt_path=prompt_for_stdin,
            log_path=log_path,
        )
        if new_session_id is None:
            new_session_id = latest_codex_session_id_for_home(codex_home)
    else:
        process = subprocess.Popen(argv, cwd=target_cwd, env=env)
        phase["pid"] = process.pid
        write_manifest(paths, task_slug, manifest)
        exit_code = process.wait()
        new_session_id = latest_codex_session_id_for_home(codex_home)
        write_phase_log(
            log_path,
            [
                f"[{utc_now()}] mode=interactive phase={phase_name}",
                f"[{utc_now()}] cwd={target_cwd}",
                f"[{utc_now()}] exit_code={exit_code}",
                f"[{utc_now()}] session_id={new_session_id or phase.get('session_id') or 'unknown'}",
            ],
        )

    if new_session_id:
        phase["session_id"] = new_session_id
    phase["exit_code"] = exit_code
    phase["status"] = "succeeded" if exit_code == 0 else "failed"
    phase["completed_at"] = utc_now()
    phase["codex_home"] = str(codex_home)
    phase.pop("pid", None)
    write_manifest(paths, task_slug, manifest)
    return exit_code, phase.get("session_id")


def git_output(worktree_path: Path, args: list[str]) -> str:
    """Return git command output from a task worktree."""
    process = subprocess.run(
        ["git", *args],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return process.stderr.strip() or process.stdout.strip()
    return process.stdout.strip()


def plan_task(paths: TaskPaths, task_slug: str, user_prompt: str | None = None, headless: bool = False) -> Path:
    """Create or update an open task document using Codex planning guidance."""
    ensure_task_layout(paths)
    task_path = get_or_create_open_task(paths, task_slug)
    manifest = load_or_create_manifest(paths, task_slug)
    phase = phase_entry(manifest, "planning")
    info = phase_paths(paths, task_slug, "planning")

    if not phase.get("session_id"):
        if user_prompt is None:
            raise TaskWorkflowError("Planning prompt is required for a new planning session.")
        info["prompt_path"].write_text(
            build_phase_prompt(
                "planning",
                paths=paths,
                task_slug=task_slug,
                task_path=task_path,
                manifest=manifest,
                output_path=info["output_path"],
                user_prompt=user_prompt,
            ),
            encoding="utf-8",
        )

    exit_code, _ = run_planning_agent(
        paths=paths,
        task_slug=task_slug,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=headless,
    )
    if exit_code != 0:
        raise TaskWorkflowError(f"Planning command failed for '{task_slug}'. See {info['log_path']}.")
    planned_text = info["output_path"].read_text(encoding="utf-8").strip()
    if not planned_text:
        raise TaskWorkflowError(f"Planning command produced an empty task document for '{task_slug}'.")
    task_path.write_text(planned_text + "\n", encoding="utf-8")
    return move_task_to_state(paths, task_slug, "open")


def start_task(paths: TaskPaths, task_slug: str, headless: bool = False) -> dict[str, Any]:
    """Create a worktree if needed, then run or resume the implementation agent."""
    ensure_task_layout(paths)
    task_path = ensure_task_for_start(paths, task_slug)
    manifest = load_or_create_manifest(paths, task_slug)

    worktree_path = normalize_cwd(manifest["worktree_path"])
    if not worktree_path.exists():
        worktree_path = ensure_git_worktree(paths, task_slug)
        manifest["worktree_path"] = str(worktree_path)
        manifest["branch_name"] = task_slug
        manifest["base_ref"] = current_git_branch(paths.repo_root)
        write_manifest(paths, task_slug, manifest)

    phase = phase_entry(manifest, "implementation")
    info = phase_paths(paths, task_slug, "implementation")
    if not phase.get("session_id"):
        info["prompt_path"].write_text(
            build_phase_prompt(
                "implementation",
                paths=paths,
                task_slug=task_slug,
                task_path=task_path,
                manifest=manifest,
                output_path=info["output_path"],
            ),
            encoding="utf-8",
        )

    exit_code, _ = run_implementation_agent(
        paths=paths,
        task_slug=task_slug,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=headless,
    )
    manifest = load_or_create_manifest(paths, task_slug)
    manifest["exit_code"] = exit_code
    write_manifest(paths, task_slug, manifest)
    target_state = "finished" if exit_code == 0 else "stopped"
    task_path = move_task_to_state(paths, task_slug, target_state)
    return manifest


def review_task(paths: TaskPaths, task_slug: str, headless: bool = False) -> Path:
    """Summarize a task implementation using the git diff from its worktree."""
    ensure_task_layout(paths)
    state, task_path = find_task_document(paths, task_slug)
    if state not in {"finished", "review"}:
        raise TaskWorkflowError(f"Task '{task_slug}' must be finished before review; found state '{state}'.")
    task_path = move_task_to_state(paths, task_slug, "review")
    manifest = load_or_create_manifest(paths, task_slug)
    worktree_path = normalize_cwd(manifest["worktree_path"])
    if not worktree_path.exists():
        raise TaskWorkflowError(f"Task worktree does not exist: {worktree_path}")

    phase = phase_entry(manifest, "review")
    info = phase_paths(paths, task_slug, "review")
    if not phase.get("session_id"):
        info["prompt_path"].write_text(
            build_phase_prompt(
                "review",
                paths=paths,
                task_slug=task_slug,
                task_path=task_path,
                manifest=manifest,
                output_path=info["output_path"],
            ),
            encoding="utf-8",
        )

    exit_code, _ = run_review_agent(
        paths=paths,
        task_slug=task_slug,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=headless,
    )
    if exit_code != 0:
        raise TaskWorkflowError(f"Review command failed for '{task_slug}'. See {info['log_path']}.")
    reviewed_text = info["output_path"].read_text(encoding="utf-8").strip()
    if not reviewed_text:
        raise TaskWorkflowError(f"Review command produced an empty summary for '{task_slug}'.")
    info["output_path"].write_text(reviewed_text + "\n", encoding="utf-8")
    return info["output_path"]


def run_planning_agent(
    *,
    paths: TaskPaths,
    task_slug: str,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    headless: bool = False,
) -> tuple[int, str | None]:
    """Run or resume the planning agent."""
    return run_phase_session(
        paths=paths,
        task_slug=task_slug,
        phase_name="planning",
        task_path=get_or_create_open_task(paths, task_slug),
        prompt_path=prompt_path,
        output_path=output_path,
        log_path=log_path,
        headless=headless,
    )


def run_implementation_agent(
    *,
    paths: TaskPaths,
    task_slug: str,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    headless: bool = False,
) -> tuple[int, str | None]:
    """Run or resume the implementation agent."""
    return run_phase_session(
        paths=paths,
        task_slug=task_slug,
        phase_name="implementation",
        task_path=ensure_task_for_start(paths, task_slug),
        prompt_path=prompt_path,
        output_path=output_path,
        log_path=log_path,
        headless=headless,
    )


def run_review_agent(
    *,
    paths: TaskPaths,
    task_slug: str,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    headless: bool = False,
) -> tuple[int, str | None]:
    """Run or resume the review agent."""
    _, task_path = find_task_document(paths, task_slug)
    return run_phase_session(
        paths=paths,
        task_slug=task_slug,
        phase_name="review",
        task_path=task_path,
        prompt_path=prompt_path,
        output_path=output_path,
        log_path=log_path,
        headless=headless,
    )


def process_exists(pid: int) -> bool:
    """Return whether a PID currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def task_runtime_status(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Return the current status of a task and its latest run metadata."""
    state, task_path = find_task_document(paths, task_slug)
    result: dict[str, Any] = {"task_slug": task_slug, "task_state": state, "task_path": str(task_path)}
    try:
        manifest = load_or_create_manifest(paths, task_slug)
    except TaskWorkflowError:
        result["run_status"] = state
        return result

    result["run_status"] = state
    result["manifest"] = manifest
    return result


def stop_task(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Stop a running task runner process if one exists."""
    manifest = load_or_create_manifest(paths, task_slug)
    for phase_name, phase in manifest.get("phases", {}).items():
        pid = phase.get("pid")
        if phase.get("status") == "running" and isinstance(pid, int):
            if process_exists(pid):
                terminate_process(pid)
            phase["status"] = "stopped"
            phase["stopped_at"] = utc_now()
            write_manifest(paths, task_slug, manifest)
            move_task_to_state(paths, task_slug, "stopped")
            return manifest
    write_manifest(paths, task_slug, manifest)
    return manifest


def archive_task(paths: TaskPaths, task_slug: str) -> Path:
    """Move a task document into the archive area.

    Running tasks must be stopped before archiving.
    """
    state, task_path = find_task_document(paths, task_slug)
    if state == "archive":
        return task_path

    try:
        manifest = load_manifest(paths, task_slug)
    except TaskWorkflowError:
        manifest = None

    if isinstance(manifest, dict):
        for phase in manifest.get("phases", {}).values():
            if phase.get("status") == "running":
                raise TaskWorkflowError(f"Task '{task_slug}' is still running. Stop it before archiving.")

    return move_task_to_state(paths, task_slug, "archive")


def list_tasks(paths: TaskPaths) -> list[dict[str, Any]]:
    """List all known tasks grouped by current state."""
    tasks: dict[str, dict[str, Any]] = {}
    for state, directory in (
        ("open", paths.open_dir),
        ("running", paths.running_dir),
        ("finished", paths.finished_dir),
        ("stopped", paths.stopped_dir),
        ("review", paths.review_dir),
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            task_slug = path.stem
            tasks[task_slug] = {
                "task_slug": task_slug,
                "task_state": state,
                "task_path": str(path),
            }

    for task_slug, item in tasks.items():
        try:
            status = task_runtime_status(paths, task_slug)
        except TaskWorkflowError:
            item["run_status"] = item["task_state"]
        else:
            item["run_status"] = status.get("run_status", item["task_state"])
    return [tasks[key] for key in sorted(tasks)]


def format_status(result: dict[str, Any]) -> str:
    """Render a human-readable status summary."""
    lines = [
        f"task: {result['task_slug']}",
        f"state: {result['task_state']}",
        f"doc: {result['task_path']}",
    ]
    manifest = result.get("manifest")
    if isinstance(manifest, dict):
        for field in ("branch_name", "worktree_path", "base_ref"):
            if field in manifest:
                lines.append(f"{field}: {manifest[field]}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the task CLI argument parser."""
    parser = argparse.ArgumentParser(description="romjax local task workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a task document from a template")
    create.add_argument("task_slug")

    plan = subparsers.add_parser("plan", help="Create or update an open task using a Codex planning prompt")
    plan.add_argument("task_slug")
    plan.add_argument("--headless", action="store_true", help="Use codex exec instead of interactive codex")

    review = subparsers.add_parser("review", help="Summarize a completed task implementation for review")
    review.add_argument("task_slug")
    review.add_argument("--headless", action="store_true", help="Use codex exec instead of interactive codex")

    start = subparsers.add_parser("start", help="Start an agent on an open task")
    start.add_argument("task_slug")
    start.add_argument("--headless", action="store_true", help="Use codex exec instead of interactive codex")

    stop = subparsers.add_parser("stop", help="Stop a running task")
    stop.add_argument("task_slug")

    archive = subparsers.add_parser("archive", help="Archive a task document")
    archive.add_argument("task_slug")

    status = subparsers.add_parser("status", help="Show task status")
    status.add_argument("task_slug")

    subparsers.add_parser("list", help="List known tasks")

    return parser


def cli(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    """Run the task workflow CLI.

    :param argv: CLI arguments excluding the interpreter name
    :param repo_root: optional repository root override, useful for tests
    :return: process exit code
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = get_task_paths(repo_root)

    try:
        if args.command == "create":
            task_path = create_task(paths, args.task_slug)
            print(task_path)
            return 0
        if args.command == "plan":
            manifest = load_or_create_manifest(paths, args.task_slug)
            existing_phase = phase_entry(manifest, "planning")
            user_prompt = (
                None
                if (existing_phase.get("session_id") or existing_phase.get("codex_home"))
                else collect_plan_prompt()
            )
            task_path = plan_task(paths, args.task_slug, user_prompt=user_prompt, headless=args.headless)
            print(task_path)
            return 0
        if args.command == "review":
            review_path = review_task(paths, args.task_slug, headless=args.headless)
            print(review_path)
            return 0
        if args.command == "start":
            manifest = start_task(paths, args.task_slug, headless=args.headless)
            return int(manifest.get("exit_code", 0))
        if args.command == "stop":
            manifest = stop_task(paths, args.task_slug)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.command == "archive":
            archive_path = archive_task(paths, args.task_slug)
            print(archive_path)
            return 0
        if args.command == "status":
            print(format_status(task_runtime_status(paths, args.task_slug)))
            return 0
        if args.command == "list":
            for item in list_tasks(paths):
                print(f"{item['task_slug']}\t{item['task_state']}\t{item['run_status']}")
            return 0
    except TaskWorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    parser.error(f"Unhandled command: {args.command}")
    return 2


def main() -> None:
    """Console-script entrypoint for the task CLI."""
    raise SystemExit(cli())


__all__ = [
    "DEFAULT_AGENT_CMD",
    "DEFAULT_PLAN_CMD",
    "DEFAULT_REVIEW_CMD",
    "PHASE_NAMES",
    "TASK_STATES",
    "TaskPaths",
    "TaskWorkflowError",
    "agent_environment",
    "archive_task",
    "build_agent_prompt",
    "build_phase_prompt",
    "cli",
    "create_task",
    "collect_plan_prompt",
    "current_git_branch",
    "ensure_task_layout",
    "ensure_task_ready",
    "find_task_document",
    "format_status",
    "get_or_create_open_task",
    "get_task_paths",
    "git_output",
    "list_tasks",
    "load_manifest",
    "load_or_create_manifest",
    "main",
    "manifest_path",
    "phase_entry",
    "phase_paths",
    "plan_task",
    "render_agent_command",
    "review_task",
    "run_implementation_agent",
    "run_planning_agent",
    "run_review_agent",
    "shared_venv_path",
    "start_task",
    "stop_task",
    "task_runtime_status",
    "terminate_process",
    "validate_task_slug",
    "write_manifest",
]
