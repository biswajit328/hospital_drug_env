"""Validator-friendly task discovery surface.

This module exposes the task suite and scoring helpers in a very simple shape
so external tooling can discover tasks without needing to inspect the server
implementation or parse human-readable CLI output.
"""

from __future__ import annotations

from grader import TASKS, list_task_metadata, run_task_score


def list_tasks() -> list[dict]:
    """Return the 3 task definitions as a raw list."""
    return list_task_metadata()


def score_task(task_id: str, seed: int = 42) -> float:
    """Return a single validator-safe task score."""
    config = TASKS[task_id]
    return run_task_score(config, base_seed=seed)


def score_all_tasks(seed: int = 42) -> dict[str, float]:
    """Return a flat mapping from task id to validator-safe score."""
    return {
        task_id: run_task_score(config, base_seed=seed)
        for task_id, config in TASKS.items()
    }


if __name__ == "__main__":
    import json

    print(json.dumps({"tasks": list_tasks(), "scores": score_all_tasks()}, indent=2))
