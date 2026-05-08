"""Task CLI for local AI-assisted development workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

TASK_STATES = ("open", "running", "finished", "stopped", "review", "archive")
PHASE_NAMES = ("planning", "implementation", "review")


class TaskWorkflowError(RuntimeError):
    """Raised when the task workflow encounters an invalid local state."""


@dataclass(slots=True)
class PhaseRunResult:
    """Structured result from running a Codex task phase."""

    exit_code: int
    thread_id: str | None = None
    usage: dict[str, int] | None = None
    token_count: dict[str, Any] | None = None
    session_path: str | None = None


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


def active_worktree_path(paths: TaskPaths, task_slug: str) -> Path:
    """Return the standard sibling worktree path for an active task."""
    return paths.repo_root.parent / task_slug


def archived_worktree_root(paths: TaskPaths) -> Path:
    """Return the sibling directory that stores archived task worktrees."""
    return paths.repo_root.parent / "archive"


def archived_worktree_path(paths: TaskPaths, task_slug: str) -> Path:
    """Return the standard archived worktree path for a task."""
    return archived_worktree_root(paths) / task_slug


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
    worktree_path = active_worktree_path(paths, task_slug)
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
    worktree_path = active_worktree_path(paths, task_slug)
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

    try:
        _, existing_path = find_task_document(paths, task_slug)
    except TaskWorkflowError:
        pass
    else:
        raise TaskWorkflowError(f"Task document already exists: {existing_path}")

    task_path = paths.open_dir / f"{task_slug}.md"

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
    worktree_path = active_worktree_path(paths, task_slug)
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
    *,
    headless: bool,
) -> str:
    """Build the default prompt given to the implementation agent."""
    if headless:
        summary_instruction = (
            "5. Return a concise final summary as your last assistant message describing what changed "
            "and any blockers. Do not wrap it in code fences.\n"
        )
    else:
        summary_instruction = (
            "5. Write a concise final summary describing what changed and any blockers "
            f"to {summary_path}.\n"
        )
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
        "4. Run `uv run rr lint` and targeted unit tests `uv run pytest ...` on affected files.\n"
        f"{summary_instruction}"
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
    manifest.setdefault("worktree_path", str(active_worktree_path(paths, task_slug)))
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


def codex_option_value(extra_args: list[str], *names: str) -> str | None:
    """Return the last explicit value passed for one of the given CLI flags."""
    value: str | None = None
    index = 0
    while index < len(extra_args):
        arg = extra_args[index]
        matched_name = next((name for name in names if arg == name), None)
        if matched_name is not None:
            next_index = index + 1
            if next_index < len(extra_args):
                value = extra_args[next_index]
                index = next_index + 1
                continue
            break
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                value = arg[len(prefix):]
                break
        index += 1
    return value


def codex_config_value(extra_args: list[str], key: str) -> Any:
    """Return the last explicit value passed for a Codex ``-c/--config`` override."""
    value: Any = None
    index = 0
    while index < len(extra_args):
        arg = extra_args[index]
        config_arg: str | None = None
        if arg in {"-c", "--config"}:
            next_index = index + 1
            if next_index < len(extra_args):
                config_arg = extra_args[next_index]
                index = next_index + 1
            else:
                break
        elif arg.startswith("-c") and arg not in {"-c", "--config"}:
            config_arg = arg[2:]
            index += 1
        elif arg.startswith("--config="):
            config_arg = arg.removeprefix("--config=")
            index += 1
        else:
            index += 1
            continue

        if config_arg is None or "=" not in config_arg:
            continue
        config_key, raw_value = config_arg.split("=", 1)
        if config_key != key:
            continue
        try:
            value = tomllib.loads(f"value = {raw_value}")["value"]
        except tomllib.TOMLDecodeError:
            value = raw_value
    return value


def codex_phase_metadata(extra_args: list[str]) -> dict[str, Any]:
    """Return manifest metadata for a Codex phase invocation."""
    return {
        "codex_args": list(extra_args),
        "model": codex_option_value(extra_args, "--model", "-m") or codex_config_value(extra_args, "model"),
        "reasoning_effort": codex_config_value(extra_args, "model_reasoning_effort"),
    }


def normalize_cwd(path_value: str | Path) -> Path:
    """Normalize a cwd/path value loaded from task metadata or manifests."""
    return Path(path_value).expanduser().resolve()


def shared_venv_path(paths: TaskPaths) -> Path:
    """Return the canonical shared project virtual environment path."""
    return paths.repo_root / ".venv"


def shared_venv_bin_dir(paths: TaskPaths) -> Path:
    """Return the executable directory inside the shared project virtual environment."""
    return shared_venv_path(paths) / ("Scripts" if os.name == "nt" else "bin")


def agent_environment(paths: TaskPaths) -> dict[str, str]:
    """Build an environment that lets task worktrees reuse the main repo .venv."""
    env = os.environ.copy()
    venv_path = shared_venv_path(paths)
    if venv_path.exists():
        env["VIRTUAL_ENV"] = str(venv_path)
        env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
        bin_dir = shared_venv_bin_dir(paths)
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


def phase_target_directory(phase_name: str, paths: TaskPaths, manifest: dict[str, Any]) -> Path:
    """Return the working directory used for a task phase."""
    if phase_name == "implementation":
        return normalize_cwd(manifest["worktree_path"])
    return paths.repo_root


def ensure_task_for_start(paths: TaskPaths, task_slug: str) -> Path:
    """Return an implementation-ready task document."""
    state, task_path = find_task_document(paths, task_slug)
    if state == "open":
        ensure_task_ready(task_path)
        return move_task_to_state(paths, task_slug, "running")
    if state == "stopped":
        return move_task_to_state(paths, task_slug, "running")
    if state == "running":
        return task_path
    raise TaskWorkflowError(f"Task '{task_slug}' cannot be started from state '{state}'.")


def write_phase_log(log_path: Path, lines: list[str]) -> None:
    """Append concise lifecycle lines to a phase log."""
    with log_path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip() + "\n")


def codex_home_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the Codex home directory used for session storage."""
    source = env if env is not None else os.environ
    raw_path = source.get("CODEX_HOME")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def codex_rollout_files(env: Mapping[str, str] | None = None) -> list[Path]:
    """Return known Codex rollout JSONL files sorted by modification time."""
    sessions_root = codex_home_dir(env) / "sessions"
    if not sessions_root.exists():
        return []
    files = sorted(
        sessions_root.glob("**/rollout-*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return files


def parse_json_line(raw_line: str) -> dict[str, Any] | None:
    """Parse a JSONL line emitted by Codex, returning ``None`` on non-JSON lines."""
    line = raw_line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_headless_codex_output(output_text: str) -> tuple[str | None, dict[str, int] | None]:
    """Extract thread id and usage details from ``codex exec --json`` output."""
    thread_id: str | None = None
    usage: dict[str, int] | None = None
    for raw_line in output_text.splitlines():
        event = parse_json_line(raw_line)
        if event is None:
            continue
        if event.get("type") == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str):
                thread_id = candidate
        if event.get("type") != "turn.completed":
            continue
        candidate_usage = event.get("usage")
        if not isinstance(candidate_usage, dict):
            continue
        normalized_usage = {
            key: value
            for key, value in candidate_usage.items()
            if isinstance(key, str) and isinstance(value, int)
        }
        if normalized_usage:
            usage = normalized_usage
    return thread_id, usage


def rollout_token_count(rollout_path: Path) -> dict[str, Any] | None:
    """Extract the last ``token_count`` payload from a Codex rollout file."""
    token_count: dict[str, Any] | None = None
    try:
        with rollout_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                event = parse_json_line(raw_line)
                if event is None:
                    continue
                payload = event.get("token_count")
                if isinstance(payload, dict):
                    token_count = payload
    except OSError:
        return None
    return token_count


def select_rollout_file(
    before_files: set[Path],
    after_files: list[Path],
    thread_id: str | None = None,
) -> Path | None:
    """Select the most relevant rollout file produced by the latest phase run."""
    new_files = [path for path in after_files if path not in before_files]
    candidates = new_files or after_files
    if not candidates:
        return None
    if thread_id is None:
        return candidates[-1]
    for path in reversed(candidates):
        if thread_id in path.read_text(encoding="utf-8", errors="ignore"):
            return path
    return candidates[-1]


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


def ensure_phase_output_absent(paths: TaskPaths, task_slug: str, phase_name: str, label: str) -> dict[str, Path]:
    """Return phase paths and fail if the phase output already exists."""
    info = phase_paths(paths, task_slug, phase_name)
    if info["output_path"].exists():
        raise TaskWorkflowError(
            f"Task '{task_slug}' already has {label} output at {info['output_path']}."
        )
    return info


def build_phase_prompt(
    phase_name: str,
    *,
    paths: TaskPaths,
    task_slug: str,
    task_path: Path,
    manifest: dict[str, Any],
    output_path: Path,
    user_prompt: str | None = None,
    headless: bool = False,
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
            headless=headless,
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


def run_codex_interactive(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    pid_callback: Callable[[int], None] | None = None,
) -> PhaseRunResult:
    """Run Codex in interactive mode in the current terminal."""
    before_files = set(codex_rollout_files(env))
    process = subprocess.Popen(argv, cwd=cwd, env=env)
    if pid_callback is not None:
        pid_callback(process.pid)
    exit_code = process.wait()
    after_files = codex_rollout_files(env)
    rollout_path = select_rollout_file(before_files, after_files)
    write_phase_log(
        log_path,
        [
            f"[{utc_now()}] interactive command: {' '.join(argv[:4])} ...",
            f"[{utc_now()}] exit_code: {exit_code}",
        ],
    )
    token_count = rollout_token_count(rollout_path) if rollout_path is not None else None
    return PhaseRunResult(
        exit_code=exit_code,
        token_count=token_count,
        session_path=str(rollout_path) if rollout_path is not None else None,
    )


def run_codex_headless(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    prompt_path: Path | None,
    log_path: Path,
) -> PhaseRunResult:
    """Run Codex in non-interactive exec mode."""
    before_files = set(codex_rollout_files(env))
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
    thread_id, usage = parse_headless_codex_output(process.stdout)
    rollout_path = select_rollout_file(before_files, codex_rollout_files(env), thread_id)
    token_count = rollout_token_count(rollout_path) if rollout_path is not None else None
    return PhaseRunResult(
        exit_code=process.returncode,
        thread_id=thread_id,
        usage=usage,
        token_count=token_count,
        session_path=str(rollout_path) if rollout_path is not None else None,
    )


def phase_new_command(
    phase_name: str,
    *,
    cwd: Path,
    repo_root: Path,
    prompt_text: str,
    output_path: Path,
    headless: bool,
    extra_args: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Build the argv for a new phase session and whether stdin should supply the prompt."""
    forwarded_args = list(extra_args or [])
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
            ]
        argv.extend(forwarded_args)
        argv.append("-")
        return argv, True

    argv = ["codex", "--no-alt-screen", "-C", str(cwd)]
    if phase_name == "planning":
        argv.extend(["-s", "read-only"])
    elif phase_name == "review":
        argv.extend(["-s", "read-only"])
    else:
        argv.extend(["--add-dir", str(repo_root)])
    argv.extend(forwarded_args)
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
    extra_args: list[str] | None = None,
) -> int:
    """Run a task phase session."""
    manifest = load_or_create_manifest(paths, task_slug)
    phase = phase_entry(manifest, phase_name)
    target_cwd = phase_target_directory(phase_name, paths, manifest)
    env = agent_environment(paths)

    if phase.get("status") == "running":
        pid = phase.get("pid")
        if isinstance(pid, int) and process_exists(pid):
            raise TaskWorkflowError(f"Task '{task_slug}' phase '{phase_name}' is already running.")
        phase["status"] = "stopped"

    prompt_text = prompt_path.read_text(encoding="utf-8")
    argv, use_stdin = phase_new_command(
        phase_name,
        cwd=target_cwd,
        repo_root=paths.repo_root,
        prompt_text=prompt_text,
        output_path=output_path,
        headless=headless,
        extra_args=extra_args,
    )
    prompt_for_stdin = prompt_path if use_stdin else None
    phase_metadata = codex_phase_metadata(list(extra_args or []))

    phase.update(
        {
            "mode": "headless" if headless else "interactive",
            "status": "running",
            "prompt_path": str(prompt_path),
            "output_path": str(output_path),
            "log_path": str(log_path),
            "started_at": utc_now(),
            **phase_metadata,
        }
    )
    write_manifest(paths, task_slug, manifest)

    if headless:
        run_result = run_codex_headless(
            argv=argv,
            cwd=target_cwd,
            env=env,
            prompt_path=prompt_for_stdin,
            log_path=log_path,
        )
    else:
        def record_pid(pid: int) -> None:
            phase["pid"] = pid
            write_manifest(paths, task_slug, manifest)

        run_result = run_codex_interactive(
            argv=argv,
            cwd=target_cwd,
            env=env,
            log_path=log_path,
            pid_callback=record_pid,
        )
        write_phase_log(
            log_path,
            [
                f"[{utc_now()}] mode=interactive phase={phase_name}",
                f"[{utc_now()}] cwd={target_cwd}",
            ],
        )

    phase["exit_code"] = run_result.exit_code
    phase["status"] = "succeeded" if run_result.exit_code == 0 else "failed"
    phase["completed_at"] = utc_now()
    phase.pop("pid", None)
    if run_result.thread_id is not None:
        phase["thread_id"] = run_result.thread_id
    if run_result.usage is not None:
        phase["usage"] = run_result.usage
    if run_result.token_count is not None:
        phase["token_count"] = run_result.token_count
    if run_result.session_path is not None:
        phase["session_path"] = run_result.session_path
    write_manifest(paths, task_slug, manifest)
    return run_result.exit_code


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


def plan_task(
    paths: TaskPaths,
    task_slug: str,
    user_prompt: str | None = None,
    headless: bool = False,
    extra_args: list[str] | None = None,
) -> Path:
    """Create or update an open task document using Codex planning guidance."""
    ensure_task_layout(paths)
    task_path = get_or_create_open_task(paths, task_slug)
    manifest = load_or_create_manifest(paths, task_slug)
    info = ensure_phase_output_absent(paths, task_slug, "planning", "planning")
    if user_prompt is None:
        raise TaskWorkflowError("Planning prompt is required.")
    info["prompt_path"].write_text(
        build_phase_prompt(
            "planning",
            paths=paths,
            task_slug=task_slug,
            task_path=task_path,
            manifest=manifest,
            output_path=info["output_path"],
            user_prompt=user_prompt,
            headless=headless,
        ),
        encoding="utf-8",
    )

    exit_code = run_planning_agent(
        paths=paths,
        task_slug=task_slug,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=headless,
        extra_args=extra_args,
    )
    if exit_code != 0:
        raise TaskWorkflowError(f"Planning command failed for '{task_slug}'. See {info['log_path']}.")
    planned_text = info["output_path"].read_text(encoding="utf-8").strip()
    if not planned_text:
        raise TaskWorkflowError(f"Planning command produced an empty task document for '{task_slug}'.")
    task_path.write_text(planned_text + "\n", encoding="utf-8")
    return move_task_to_state(paths, task_slug, "open")


def start_task(
    paths: TaskPaths,
    task_slug: str,
    headless: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Create a worktree if needed, then run the implementation agent."""
    ensure_task_layout(paths)
    info = ensure_phase_output_absent(paths, task_slug, "implementation", "implementation")
    task_path = ensure_task_for_start(paths, task_slug)
    manifest = load_or_create_manifest(paths, task_slug)

    worktree_path = normalize_cwd(manifest["worktree_path"])
    if not worktree_path.exists():
        worktree_path = ensure_git_worktree(paths, task_slug)
        manifest["worktree_path"] = str(worktree_path)
        manifest["branch_name"] = task_slug
        manifest["base_ref"] = current_git_branch(paths.repo_root)
        write_manifest(paths, task_slug, manifest)

    info["prompt_path"].write_text(
        build_phase_prompt(
            "implementation",
            paths=paths,
            task_slug=task_slug,
            task_path=task_path,
            manifest=manifest,
            output_path=info["output_path"],
            headless=headless,
        ),
        encoding="utf-8",
    )

    exit_code = run_implementation_agent(
        paths=paths,
        task_slug=task_slug,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=headless,
        extra_args=extra_args,
    )
    manifest = load_or_create_manifest(paths, task_slug)
    manifest["exit_code"] = exit_code
    write_manifest(paths, task_slug, manifest)
    target_state = "finished" if exit_code == 0 else "stopped"
    task_path = move_task_to_state(paths, task_slug, target_state)
    return manifest


def review_task(
    paths: TaskPaths,
    task_slug: str,
    headless: bool = False,
    extra_args: list[str] | None = None,
) -> Path:
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

    info = ensure_phase_output_absent(paths, task_slug, "review", "review")
    info["prompt_path"].write_text(
        build_phase_prompt(
            "review",
            paths=paths,
            task_slug=task_slug,
            task_path=task_path,
            manifest=manifest,
            output_path=info["output_path"],
            headless=headless,
        ),
        encoding="utf-8",
    )

    exit_code = run_review_agent(
        paths=paths,
        task_slug=task_slug,
        prompt_path=info["prompt_path"],
        output_path=info["output_path"],
        log_path=info["log_path"],
        headless=headless,
        extra_args=extra_args,
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
    extra_args: list[str] | None = None,
) -> int:
    """Run the planning agent."""
    return run_phase_session(
        paths=paths,
        task_slug=task_slug,
        phase_name="planning",
        task_path=get_or_create_open_task(paths, task_slug),
        prompt_path=prompt_path,
        output_path=output_path,
        log_path=log_path,
        headless=headless,
        extra_args=extra_args,
    )


def run_implementation_agent(
    *,
    paths: TaskPaths,
    task_slug: str,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    headless: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """Run the implementation agent."""
    return run_phase_session(
        paths=paths,
        task_slug=task_slug,
        phase_name="implementation",
        task_path=ensure_task_for_start(paths, task_slug),
        prompt_path=prompt_path,
        output_path=output_path,
        log_path=log_path,
        headless=headless,
        extra_args=extra_args,
    )


def run_review_agent(
    *,
    paths: TaskPaths,
    task_slug: str,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    headless: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """Run the review agent."""
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
        extra_args=extra_args,
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
    ensure_task_layout(paths)
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

    if isinstance(manifest, dict):
        old_worktree_path = normalize_cwd(manifest["worktree_path"])
        new_worktree_path = archived_worktree_path(paths, task_slug)
        if old_worktree_path != new_worktree_path and old_worktree_path.exists():
            new_worktree_path.parent.mkdir(parents=True, exist_ok=True)
            if new_worktree_path.exists():
                raise TaskWorkflowError(f"Archived worktree path already exists: {new_worktree_path}")
            if is_registered_worktree(paths.repo_root, old_worktree_path):
                run_git(paths.repo_root, ["worktree", "move", str(old_worktree_path), str(new_worktree_path)])
            else:
                shutil.move(str(old_worktree_path), str(new_worktree_path))
            manifest["worktree_path"] = str(new_worktree_path)
            write_manifest(paths, task_slug, manifest)
            task_text = task_path.read_text(encoding="utf-8")
            task_path.write_text(task_text.replace(str(old_worktree_path), str(new_worktree_path)), encoding="utf-8")

    return move_task_to_state(paths, task_slug, "archive")


def task_has_running_phase(paths: TaskPaths, task_slug: str) -> bool:
    """Return whether a task manifest indicates an active phase."""
    try:
        manifest = load_manifest(paths, task_slug)
    except TaskWorkflowError:
        return False
    for phase in manifest.get("phases", {}).values():
        if phase.get("status") == "running":
            return True
    return False


def is_registered_worktree(repo_root: Path, worktree_path: Path) -> bool:
    """Return whether a path is registered as a git worktree."""
    process = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = normalize_cwd(worktree_path)
    for raw_line in process.stdout.splitlines():
        if raw_line.startswith("worktree "):
            listed = normalize_cwd(raw_line.removeprefix("worktree ").strip())
            if listed == expected:
                return True
    return False


def confirm_clean(task_slug: str) -> None:
    """Ask the user to confirm a destructive task cleanup operation."""
    warning = (
        f"Warning: task clean will permanently remove the task document, run artifacts, "
        f"git worktree, and branch for '{task_slug}'."
    )
    print(warning)
    print("Type 'clean' to continue: ", end="", flush=True)
    if sys.stdin.isatty():
        response = input().strip()
    else:
        response = sys.stdin.read().strip()
    if response != "clean":
        raise TaskWorkflowError("Task clean cancelled.")


def clean_task(paths: TaskPaths, task_slug: str) -> dict[str, Any]:
    """Remove all local artifacts associated with a task."""
    state, task_path = find_task_document(paths, task_slug)
    if state == "running" or task_has_running_phase(paths, task_slug):
        raise TaskWorkflowError(f"Task '{task_slug}' is currently running. Stop it before cleaning.")

    try:
        manifest = load_manifest(paths, task_slug)
    except TaskWorkflowError:
        manifest = initial_manifest(paths, task_slug)

    run_dir = run_dir_for_task(paths, task_slug)
    worktree_path = normalize_cwd(manifest["worktree_path"])
    if state == "archive" and not worktree_path.exists():
        archived_path = archived_worktree_path(paths, task_slug)
        if archived_path.exists():
            worktree_path = archived_path
    branch_name = str(manifest.get("branch_name") or task_slug)
    summary: dict[str, Any] = {
        "task_slug": task_slug,
        "removed_task_doc": False,
        "removed_run_dir": False,
        "removed_worktree": False,
        "removed_branch": False,
        "task_doc_path": str(task_path),
        "run_dir_path": str(run_dir),
        "worktree_path": str(worktree_path),
        "branch_name": branch_name,
    }

    task_path.unlink()
    summary["removed_task_doc"] = True

    if run_dir.exists():
        shutil.rmtree(run_dir)
        summary["removed_run_dir"] = True

    if worktree_path.exists():
        if is_registered_worktree(paths.repo_root, worktree_path):
            run_git(paths.repo_root, ["worktree", "remove", "--force", str(worktree_path)])
        else:
            shutil.rmtree(worktree_path)
        summary["removed_worktree"] = True

    branch_lookup = run_git(paths.repo_root, ["branch", "--list", branch_name]).stdout.strip()
    if branch_lookup:
        run_git(paths.repo_root, ["branch", "-D", branch_name])
        summary["removed_branch"] = True

    return summary


def format_clean_summary(summary: dict[str, Any]) -> str:
    """Render a human-readable cleanup summary."""
    lines = [f"task: {summary['task_slug']}"]
    if summary["removed_task_doc"]:
        lines.append(f"removed task doc: {summary['task_doc_path']}")
    else:
        lines.append("removed task doc: no")
    if summary["removed_run_dir"]:
        lines.append(f"removed run dir: {summary['run_dir_path']}")
    else:
        lines.append("removed run dir: no")
    if summary["removed_worktree"]:
        lines.append(f"removed worktree: {summary['worktree_path']}")
    else:
        lines.append("removed worktree: no")
    if summary["removed_branch"]:
        lines.append(f"removed branch: {summary['branch_name']}")
    else:
        lines.append("removed branch: no")
    return "\n".join(lines)


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

    clean = subparsers.add_parser("clean", help="Delete a task document, run artifacts, worktree, and branch")
    clean.add_argument("task_slug")

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
    args, extra_args = parser.parse_known_args(argv)
    paths = get_task_paths(repo_root)

    try:
        if extra_args and args.command not in {"plan", "start", "review"}:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
        if args.command == "create":
            task_path = create_task(paths, args.task_slug)
            print(task_path)
            return 0
        if args.command == "plan":
            user_prompt = collect_plan_prompt()
            task_path = plan_task(
                paths,
                args.task_slug,
                user_prompt=user_prompt,
                headless=args.headless,
                extra_args=extra_args,
            )
            print(task_path)
            return 0
        if args.command == "review":
            review_path = review_task(paths, args.task_slug, headless=args.headless, extra_args=extra_args)
            print(review_path)
            return 0
        if args.command == "start":
            manifest = start_task(paths, args.task_slug, headless=args.headless, extra_args=extra_args)
            return int(manifest.get("exit_code", 0))
        if args.command == "stop":
            manifest = stop_task(paths, args.task_slug)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.command == "archive":
            archive_path = archive_task(paths, args.task_slug)
            print(archive_path)
            return 0
        if args.command == "clean":
            confirm_clean(args.task_slug)
            summary = clean_task(paths, args.task_slug)
            print(format_clean_summary(summary))
            return 0
        if args.command == "status":
            print(format_status(task_runtime_status(paths, args.task_slug)))
            return 0
        if args.command == "list":
            print(f"{'TASK':15s} {'STATE'}")
            for item in list_tasks(paths):
                print(f"{item['task_slug']:15s} {item['task_state']}")
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
    "PHASE_NAMES",
    "TASK_STATES",
    "TaskPaths",
    "TaskWorkflowError",
    "agent_environment",
    "archive_task",
    "build_agent_prompt",
    "build_phase_prompt",
    "cli",
    "clean_task",
    "confirm_clean",
    "create_task",
    "collect_plan_prompt",
    "current_git_branch",
    "ensure_task_layout",
    "ensure_task_ready",
    "find_task_document",
    "format_clean_summary",
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
    "review_task",
    "run_implementation_agent",
    "run_planning_agent",
    "run_review_agent",
    "shared_venv_bin_dir",
    "shared_venv_path",
    "start_task",
    "stop_task",
    "task_has_running_phase",
    "task_runtime_status",
    "terminate_process",
    "validate_task_slug",
    "write_manifest",
]
