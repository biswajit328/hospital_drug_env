"""Compatibility shim for validators that look for `graders.py`.

This module mirrors the task-scoring surface already implemented in
`grader.py`, but exposes it under the plural filename many simple validators
expect when scanning a submission repo.
"""

from __future__ import annotations

import json

from grader import (
    TASKS as GRADER_TASKS,
    grade_easy,
    grade_hard,
    grade_medium,
    list_task_metadata,
    run_all_graders,
    run_task_score,
)

# Expose a raw task list at module scope for simple validators.
TASKS = list_task_metadata()


def list_tasks() -> list[dict]:
    return [dict(task) for task in TASKS]


def grade_task(task_id: str, seed: int = 42) -> float:
    return run_task_score(GRADER_TASKS[task_id], base_seed=seed)


def score_all_tasks(seed: int = 42) -> dict[str, float]:
    return run_all_graders(seed=seed, verbose=False)


if __name__ == "__main__":
    scores = score_all_tasks(seed=42)
    payload = {
        "tasks_discovered": len(TASKS),
        "all_tasks_have_graders": all(task.get("has_grader") for task in TASKS),
        "all_scores_strictly_between_zero_and_one": all(
            0.0 < score < 1.0 for score in scores.values()
        ),
        "tasks": list_tasks(),
        "scores": scores,
    }
    print(json.dumps(payload, indent=2))
