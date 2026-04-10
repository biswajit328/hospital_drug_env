"""Public benchmark task surface.

This module exposes task discovery and score lookup in a simple scriptable form
so external tooling, demos, and reviewers can inspect the benchmark without
depending on server internals.
"""

from __future__ import annotations

from benchmark_registry import TASKS as REGISTRY_TASKS, list_task_metadata
from grader import run_task_score


# Expose a raw task list at module scope for validators that look for a
# conventional `TASKS` constant inside `tasks.py`.
TASKS = list_task_metadata()


def list_tasks() -> list[dict]:
    """Return the task definitions as a raw list."""
    return [dict(task) for task in TASKS]


def score_task(task_id: str, seed: int = 42) -> float:
    """Return a single validator-safe task score."""
    config = REGISTRY_TASKS[task_id]
    return run_task_score(config, base_seed=seed)


def score_all_tasks(seed: int = 42) -> dict[str, float]:
    """Return a flat mapping from task id to validator-safe score."""
    return {
        task["id"]: run_task_score(REGISTRY_TASKS[task["id"]], base_seed=seed)
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
