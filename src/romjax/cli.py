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
    "--output-last-message {summary_path} -"
)
DEFAULT_PLAN_CMD = (
    "codex exec --sandbox read-only --cd {repo_root} "
    "--output-last-message {plan_output_path} -"
)
DEFAULT_REVIEW_CMD = (
    "codex exec --sandbox read-only --cd {worktree_path} "
    "--add-dir {repo_root} "
    "--output-last-message {review_output_path} -"
)
TASK_STATES = ("open", "in_progress", "done", "archive")


class TaskWorkflowError(RuntimeError):
    """Raised when the task workflow encounters an invalid local state."""


@dataclass(slots=True)
class TaskPaths:
    """Common repository paths used by the local task workflow."""

    repo_root: Path
    tasks_root: Path
    templates_dir: Path
    open_dir: Path
    in_progress_dir: Path
    done_dir: Path
    archive_dir: Path
    runs_dir: Path
    queue_dir: Path


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
        in_progress_dir=tasks_root / "in_progress",
        done_dir=tasks_root / "done",
        archive_dir=tasks_root / "archive",
        runs_dir=tasks_root / "runs",
        queue_dir=tasks_root / "queue",
    )


def ensure_task_layout(paths: TaskPaths) -> None:
    """Create the task workflow directory layout if needed."""
    for directory in (
        paths.tasks_root,
        paths.templates_dir,
        paths.open_dir,
        paths.in_progress_dir,
        paths.done_dir,
        paths.archive_dir,
        paths.runs_dir,
        paths.queue_dir,
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


def find_task_document(paths: TaskPaths, task_slug: str) -> tuple[str, Path]:
    """Locate a task markdown document across known task states."""
    candidates = {
        "open": paths.open_dir / f"{task_slug}.md",
        "in_progress": paths.in_progress_dir / f"{task_slug}.md",
        "done": paths.done_dir / f"{task_slug}.md",
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


def build_agent_prompt(paths: TaskPaths, task_slug: str, task_path: Path, worktree_path: Path) -> str:
    """Build the default prompt given to the implementation agent."""
    return (
        "Read the repository instructions and execute the task.\n\n"
        f"Repository root: {paths.repo_root}\n"
        f"Task slug: {task_slug}\n"
        f"Task document: {task_path}\n"
        f"Worktree path: {worktree_path}\n"
        f"AGENTS instructions: {paths.repo_root / 'AGENTS.md'}\n\n"
        "Requirements:\n"
        "1. Read AGENTS.md before making changes.\n"
        "2. Read the task markdown document and implement only that scope.\n"
        "3. Use the task worktree as your working directory.\n"
        "4. Run `uv run rr lint` and `uv run rr test` before finishing if feasible.\n"
        "5. Write a concise final summary describing what changed and any blockers.\n"
    )


def build_plan_prompt(paths: TaskPaths, task_slug: str, task_path: Path, user_prompt: str) -> str:
    """Build the planning prompt used to fill a task template."""
    template_text = task_path.read_text(encoding="utf-8")
    return (
        "You are preparing a task markdown document for later implementation work.\n\n"
        f"Repository root: {paths.repo_root}\n"
        f"Task slug: {task_slug}\n"
        f"Task document path: {task_path}\n"
        f"AGENTS instructions: {paths.repo_root / 'AGENTS.md'}\n\n"
        "Use the following task template as the starting point. Replace placeholder TODO items "
        "with concrete, actionable recommendations based on the user prompt. Preserve markdown "
        "structure and keep the result suitable for a later `task start` run.\n\n"
        "Return only the final markdown document. Do not wrap it in code fences.\n\n"
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
        f"AGENTS instructions: {paths.repo_root / 'AGENTS.md'}\n\n"
        "Fill the review template using the task document and the available git diff context. "
        "Keep the summary concise and useful for a pull request or human review. "
        "If a section has no relevant changes, say so briefly rather than inventing details. "
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


def render_agent_command(command_template: str, manifest: dict[str, Any]) -> list[str]:
    """Render the agent command template into argv form."""
    command = command_template.format(**manifest)
    if os.name == "nt":
        return split_windows_command(command)
    return shlex.split(command, posix=True)


def split_windows_command(command: str) -> list[str]:
    """Split a Windows command line into argv using CommandLineToArgvW."""
    argc = ctypes.c_int()
    argv_ptr = ctypes.windll.shell32.CommandLineToArgvW(command, ctypes.byref(argc))
    if not argv_ptr:
        raise OSError("CommandLineToArgvW failed to parse command.")
    try:
        return [argv_ptr[index] for index in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv_ptr)


def normalize_cwd(path_value: str | Path) -> Path:
    """Normalize a cwd/path value loaded from task metadata or manifests."""
    return Path(path_value).expanduser().resolve()


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


def plan_task(paths: TaskPaths, task_slug: str, user_prompt: str) -> Path:
    """Create or update an open task document using Codex planning guidance."""
    ensure_task_layout(paths)
    task_path = get_or_create_open_task(paths, task_slug)
    run_dir = run_dir_for_task(paths, task_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_prompt_path = run_dir / f"{task_slug}-plan-prompt.md"
    plan_output_path = run_dir / f"{task_slug}-plan.md"
    plan_log_path = run_dir / f"{task_slug}-plan.log"
    plan_command = os.environ.get("ROMJAX_TASK_PLAN_CMD", DEFAULT_PLAN_CMD)

    prompt = build_plan_prompt(paths, task_slug, task_path, user_prompt)
    plan_prompt_path.write_text(prompt, encoding="utf-8")
    run_planning_agent(
        plan_command=plan_command,
        repo_root=paths.repo_root,
        plan_prompt_path=plan_prompt_path,
        plan_output_path=plan_output_path,
        plan_log_path=plan_log_path,
    )
    planned_text = plan_output_path.read_text(encoding="utf-8").strip()
    if not planned_text:
        raise TaskWorkflowError(f"Planning command produced an empty task document for '{task_slug}'.")
    task_path.write_text(planned_text + "\n", encoding="utf-8")
    return task_path


def review_task(paths: TaskPaths, task_slug: str) -> Path:
    """Summarize a task implementation using the git diff from its worktree."""
    ensure_task_layout(paths)
    _, task_path = find_task_document(paths, task_slug)
    manifest = load_manifest(paths, task_slug)
    worktree_path = Path(manifest.get("worktree_path", paths.repo_root.parent / task_slug))
    if not worktree_path.exists():
        raise TaskWorkflowError(f"Task worktree does not exist: {worktree_path}")

    base_ref = str(manifest.get("base_ref") or "main")
    review_template_path = paths.templates_dir / "review.md"
    if not review_template_path.exists():
        raise TaskWorkflowError(f"Review template not found: {review_template_path}")

    run_dir = run_dir_for_task(paths, task_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    review_prompt_path = run_dir / f"{task_slug}-review-prompt.md"
    review_output_path = run_dir / f"{task_slug}-review.md"
    review_log_path = run_dir / f"{task_slug}-review.log"
    review_command = os.environ.get("ROMJAX_TASK_REVIEW_CMD", DEFAULT_REVIEW_CMD)

    diff_summary = git_output(worktree_path, ["diff", "--stat", f"{base_ref}...HEAD"])
    committed_diff = git_output(worktree_path, ["diff", f"{base_ref}...HEAD"])
    status_short = git_output(worktree_path, ["status", "--short"])
    working_tree_diff = git_output(worktree_path, ["diff"])
    prompt = build_review_prompt(
        paths=paths,
        task_slug=task_slug,
        task_path=task_path,
        review_template_path=review_template_path,
        diff_summary=diff_summary,
        committed_diff=committed_diff,
        working_tree_diff=working_tree_diff,
        status_short=status_short,
    )
    review_prompt_path.write_text(prompt, encoding="utf-8")
    run_review_agent(
        review_command=review_command,
        repo_root=paths.repo_root,
        worktree_path=worktree_path,
        review_prompt_path=review_prompt_path,
        review_output_path=review_output_path,
        review_log_path=review_log_path,
    )
    reviewed_text = review_output_path.read_text(encoding="utf-8").strip()
    if not reviewed_text:
        raise TaskWorkflowError(f"Review command produced an empty summary for '{task_slug}'.")
    review_output_path.write_text(reviewed_text + "\n", encoding="utf-8")
    return review_output_path


def run_planning_agent(
    *,
    plan_command: str,
    repo_root: Path,
    plan_prompt_path: Path,
    plan_output_path: Path,
    plan_log_path: Path,
) -> None:
    """Run Codex planning once and persist its raw outputs."""
    argv = render_agent_command(
        plan_command,
        {
            "repo_root": str(repo_root),
            "plan_output_path": str(plan_output_path),
        },
    )
    with plan_prompt_path.open("r", encoding="utf-8") as prompt_file, plan_log_path.open(
        "a", encoding="utf-8"
    ) as log_file:
        process = subprocess.run(
            argv,
            cwd=repo_root,
            stdin=prompt_file,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise TaskWorkflowError(
            f"Planning command failed for '{plan_prompt_path.stem.removesuffix('-plan-prompt')}'. "
            f"See {plan_log_path}."
        )
    if not plan_output_path.exists():
        raise TaskWorkflowError(f"Planning command did not create {plan_output_path}.")


def run_review_agent(
    *,
    review_command: str,
    repo_root: Path,
    worktree_path: Path,
    review_prompt_path: Path,
    review_output_path: Path,
    review_log_path: Path,
) -> None:
    """Run Codex review once and persist its raw outputs."""
    argv = render_agent_command(
        review_command,
        {
            "repo_root": str(repo_root),
            "worktree_path": str(worktree_path),
            "review_output_path": str(review_output_path),
        },
    )
    with review_prompt_path.open("r", encoding="utf-8") as prompt_file, review_log_path.open(
        "a", encoding="utf-8"
    ) as log_file:
        process = subprocess.run(
            argv,
            cwd=repo_root,
            stdin=prompt_file,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise TaskWorkflowError(
            f"Review command failed for '{review_prompt_path.stem.removesuffix('-review-prompt')}'. "
            f"See {review_log_path}."
        )
    if not review_output_path.exists():
        raise TaskWorkflowError(f"Review command did not create {review_output_path}.")


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


def start_task(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Create a worktree, move the task into progress, and launch the agent."""
    ensure_task_layout(paths)
    state, task_path = find_task_document(paths, task_slug)
    if state != "open":
        raise TaskWorkflowError(
            f"Task '{task_slug}' must be in {paths.open_dir} before starting; found state '{state}'."
        )
    ensure_task_ready(task_path)

    worktree_path = ensure_git_worktree(paths, task_slug)
    task_in_progress = paths.in_progress_dir / task_path.name
    task_path.rename(task_in_progress)

    run_dir = run_dir_for_task(paths, task_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{task_slug}.log"
    prompt_path = run_dir / f"{task_slug}-prompt.md"
    summary_path = run_dir / f"{task_slug}-summary.md"
    exit_code_path = run_dir / f"{task_slug}.exitcode"
    agent_command = os.environ.get("ROMJAX_TASK_AGENT_CMD", DEFAULT_AGENT_CMD)

    prompt = build_agent_prompt(paths, task_slug, task_in_progress, worktree_path)
    prompt_path.write_text(prompt, encoding="utf-8")

    manifest = {
        "task_slug": task_slug,
        "task_path": str(task_in_progress),
        "repo_root": str(paths.repo_root),
        "worktree_path": str(worktree_path),
        "branch_name": task_slug,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "prompt_path": str(prompt_path),
        "summary_path": str(summary_path),
        "exit_code_path": str(exit_code_path),
        "agent_command_template": agent_command,
        "base_ref": current_git_branch(paths.repo_root),
        "status": "launching",
        "created_at": utc_now(),
    }
    write_manifest(paths, task_slug, manifest)
    pid = launch_agent_runner(paths, task_slug)
    manifest["pid"] = pid
    manifest["status"] = "running"
    manifest["started_at"] = utc_now()
    write_manifest(paths, task_slug, manifest)
    return manifest


def launch_agent_runner(paths: TaskPaths, task_slug: str) -> int:
    """Launch the detached helper process that runs the implementation agent."""
    command = [sys.executable, "-m", "romjax.cli", "_run-agent", task_slug]
    process = subprocess.Popen(
        command,
        cwd=paths.repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(process.pid)


def run_agent_once(paths: TaskPaths, task_slug: str) -> int:
    """Run the configured agent command synchronously for a task."""
    manifest = load_manifest(paths, task_slug)
    log_path = Path(manifest["log_path"])
    summary_path = Path(manifest["summary_path"])
    exit_code_path = Path(manifest["exit_code_path"])
    prompt_path = Path(manifest["prompt_path"])
    repo_root = normalize_cwd(manifest["repo_root"])

    argv = render_agent_command(manifest["agent_command_template"], manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    manifest["status"] = "running"
    manifest["runner_started_at"] = utc_now()
    manifest["agent_argv"] = argv
    write_manifest(paths, task_slug, manifest)

    with prompt_path.open("r", encoding="utf-8") as prompt_file, log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.run(
            argv,
            cwd=repo_root,
            stdin=prompt_file,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    exit_code_path.write_text(str(process.returncode), encoding="utf-8")
    manifest["completed_at"] = utc_now()
    manifest["exit_code"] = process.returncode
    manifest["status"] = "succeeded" if process.returncode == 0 else "failed"
    manifest["summary_exists"] = summary_path.exists()
    write_manifest(paths, task_slug, manifest)
    return int(process.returncode)


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
        manifest = load_manifest(paths, task_slug)
    except TaskWorkflowError:
        result["run_status"] = "not_started"
        return result

    run_status = manifest.get("status", "unknown")
    pid = manifest.get("pid")
    if run_status == "running" and isinstance(pid, int) and not process_exists(pid):
        exit_code_path = Path(manifest["exit_code_path"])
        if exit_code_path.exists():
            code = int(exit_code_path.read_text(encoding="utf-8").strip())
            run_status = "succeeded" if code == 0 else "failed"
        else:
            run_status = "finished"

    result["run_status"] = run_status
    result["manifest"] = manifest
    return result


def stop_task(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Stop a running task runner process if one exists."""
    manifest = load_manifest(paths, task_slug)
    pid = manifest.get("pid")
    if not isinstance(pid, int):
        raise TaskWorkflowError(f"Task '{task_slug}' has no runner PID to stop.")

    status = manifest.get("status", "unknown")
    if status != "running":
        return manifest

    if process_exists(pid):
        terminate_process(pid)

    manifest["status"] = "stopped"
    manifest["stopped_at"] = utc_now()
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

    if isinstance(manifest, dict) and manifest.get("status") == "running":
        raise TaskWorkflowError(f"Task '{task_slug}' is still running. Stop it before archiving.")

    archive_path = paths.archive_dir / task_path.name
    if archive_path.exists():
        raise TaskWorkflowError(f"Archive path already exists: {archive_path}")
    task_path.rename(archive_path)
    return archive_path


def list_tasks(paths: TaskPaths) -> list[dict[str, Any]]:
    """List all known tasks grouped by current state."""
    tasks: dict[str, dict[str, Any]] = {}
    for state, directory in (
        ("open", paths.open_dir),
        ("in_progress", paths.in_progress_dir),
        ("done", paths.done_dir),
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
            item["run_status"] = "not_started"
        else:
            item["run_status"] = status.get("run_status", "unknown")
    return [tasks[key] for key in sorted(tasks)]


def format_status(result: dict[str, Any]) -> str:
    """Render a human-readable status summary."""
    lines = [
        f"task: {result['task_slug']}",
        f"state: {result['task_state']}",
        f"run: {result['run_status']}",
        f"doc: {result['task_path']}",
    ]
    manifest = result.get("manifest")
    if isinstance(manifest, dict):
        for field in ("branch_name", "worktree_path", "log_path", "summary_path"):
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

    review = subparsers.add_parser("review", help="Summarize a completed task implementation for review")
    review.add_argument("task_slug")

    start = subparsers.add_parser("start", help="Start an agent on an open task")
    start.add_argument("task_slug")

    stop = subparsers.add_parser("stop", help="Stop a running task")
    stop.add_argument("task_slug")

    archive = subparsers.add_parser("archive", help="Archive a task document")
    archive.add_argument("task_slug")

    status = subparsers.add_parser("status", help="Show task status")
    status.add_argument("task_slug")

    subparsers.add_parser("list", help="List known tasks")

    runner = subparsers.add_parser("_run-agent", help=argparse.SUPPRESS)
    runner.add_argument("task_slug")

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
            task_path = plan_task(paths, args.task_slug, collect_plan_prompt())
            print(task_path)
            return 0
        if args.command == "review":
            review_path = review_task(paths, args.task_slug)
            print(review_path)
            return 0
        if args.command == "start":
            manifest = start_task(paths, args.task_slug)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
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
        if args.command == "_run-agent":
            return run_agent_once(paths, args.task_slug)
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
    "TASK_STATES",
    "TaskPaths",
    "TaskWorkflowError",
    "archive_task",
    "build_agent_prompt",
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
    "main",
    "manifest_path",
    "plan_task",
    "render_agent_command",
    "review_task",
    "run_agent_once",
    "run_planning_agent",
    "run_review_agent",
    "start_task",
    "stop_task",
    "task_runtime_status",
    "validate_task_slug",
    "write_manifest",
]
