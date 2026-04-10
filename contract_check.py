from __future__ import annotations

import json
from pathlib import Path

import yaml

from benchmark_registry import TASKS, list_task_metadata
from grader import run_all_graders


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    openenv_path = repo_root / "openenv.yaml"
    openenv_data = yaml.safe_load(openenv_path.read_text(encoding="utf-8"))

    registry_tasks = list_task_metadata()
    registry_by_id = {task["id"]: task for task in registry_tasks}
    yaml_tasks = openenv_data.get("tasks", [])
    yaml_by_id = {task["id"]: task for task in yaml_tasks}
    scores = run_all_graders(seed=42, verbose=False)

    mismatches = []
    if set(registry_by_id) != set(yaml_by_id):
        mismatches.append(
            {
                "kind": "task_id_set",
                "registry_only": sorted(set(registry_by_id) - set(yaml_by_id)),
                "yaml_only": sorted(set(yaml_by_id) - set(registry_by_id)),
            }
        )

    for task_id, task in registry_by_id.items():
        yaml_task = yaml_by_id.get(task_id)
        if not yaml_task:
            continue
        for field in ("name", "difficulty", "max_steps", "success_threshold", "grader", "has_grader"):
            if task.get(field) != yaml_task.get(field):
                mismatches.append(
                    {
                        "kind": "field_mismatch",
                        "task_id": task_id,
                        "field": field,
                        "registry": task.get(field),
                        "openenv_yaml": yaml_task.get(field),
                    }
                )

    payload = {
        "tasks_discovered": len(registry_tasks),
        "task_ids": sorted(registry_by_id),
        "all_tasks_have_graders": all(task.get("has_grader") for task in registry_tasks),
        "all_scores_strictly_between_zero_and_one": all(0.0 < score < 1.0 for score in scores.values()),
        "scores": scores,
        "registry_matches_openenv_yaml": not mismatches,
        "mismatches": mismatches,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
