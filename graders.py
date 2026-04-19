"""Public benchmark scoring surface.

This module mirrors the task-scoring surface already implemented in
`grader.py`, but exposes it under the plural filename many simple tools and
reviewers expect when scanning a benchmark repo.
"""

from __future__ import annotations

import json

from benchmark_registry import list_task_metadata
from grader import (
    grade_clinical,
    grade_easy,
    grade_forecast,
    grade_hard,
    grade_logistics,
    grade_medium,
    run_all_graders,
)

# Expose a raw task list at module scope for simple validators.
TASKS = list_task_metadata()
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "clinical": grade_clinical,
    "forecast": grade_forecast,
    "logistics": grade_logistics,
}

# Extra conventional aliases for simplistic validator lookups.
grade_task_easy = grade_easy
grade_task_medium = grade_medium
grade_task_hard = grade_hard
grade_task_clinical = grade_clinical
grade_task_forecast = grade_forecast
grade_task_logistics = grade_logistics


def list_tasks() -> list[dict]:
    return [dict(task) for task in TASKS]


def grade_task(task_id: str, seed: int = 42) -> float:
    return GRADERS[task_id](seed=seed)


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
