"""Validator-friendly task discovery surface.

This module exposes the task suite and scoring helpers in a simple, conventional
shape so external tooling can discover tasks without needing to inspect the
server implementation or parse human-readable CLI output.
"""

from __future__ import annotations

from grader import TASKS as GRADER_TASKS, list_task_metadata, run_task_score


# Expose a raw task list at module scope for validators that look for a
# conventional `TASKS` constant inside `tasks.py`.
TASKS = list_task_metadata()


def list_tasks() -> list[dict]:
    """Return the 3 task definitions as a raw list."""
    return [dict(task) for task in TASKS]


def score_task(task_id: str, seed: int = 42) -> float:
    """Return a single validator-safe task score."""
    config = GRADER_TASKS[task_id]
    return run_task_score(config, base_seed=seed)


def score_all_tasks(seed: int = 42) -> dict[str, float]:
    """Return a flat mapping from task id to validator-safe score."""
    return {
        task["id"]: run_task_score(GRADER_TASKS[task["id"]], base_seed=seed)
        for task in TASKS
    }


if __name__ == "__main__":
    import json

    payload = {
        "tasks_discovered": len(TASKS),
        "all_tasks_have_graders": all(task.get("has_grader") for task in TASKS),
        "all_scores_strictly_between_zero_and_one": all(
            0.0 < score < 1.0 for score in score_all_tasks().values()
        ),
        "tasks": list_tasks(),
        "scores": score_all_tasks(),
    }
    print(json.dumps(payload, indent=2))
