from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Callable, Dict, List

from benchmark_registry import TASKS, TaskConfig
from grader import run_task_score


VariantBuilder = Callable[[TaskConfig], TaskConfig]


VARIANTS: Dict[str, tuple[str, VariantBuilder]] = {
    "full_heuristic": (
        "Current hand-coded heuristic baseline with all supported reasoning features enabled.",
        lambda config: config,
    ),
    "no_substitutions": (
        "Same policy, but never uses substitute drugs even when they are clinically valid.",
        lambda config: replace(config, use_substitutions=False),
    ),
    "no_emergency_orders": (
        "Same policy, but never places emergency orders.",
        lambda config: replace(
            config,
            allow_emergency_orders=False,
            max_emergency_orders_per_day=0,
        ),
    ),
    "no_forecast_reserve": (
        "Same policy, but ignores forecast-based reserve planning. Mostly affects the forecast task.",
        lambda config: replace(config, use_forecast_reserve=False),
    ),
}


def parse_base_seeds(seed_text: str) -> List[int]:
    return [int(chunk.strip()) for chunk in seed_text.split(",") if chunk.strip()]


def selected_tasks(choice: str) -> List[str]:
    if choice == "all":
        return list(TASKS.keys())
    return [choice]


def run_variant_scores(task_ids: List[str], base_seeds: List[int]) -> dict:
    variant_payload: dict = {}
    full_scores: Dict[str, float] | None = None

    for variant_name, (description, builder) in VARIANTS.items():
        task_scores: Dict[str, float] = {}
        per_seed_scores: Dict[str, List[float]] = {}

        for task_id in task_ids:
            config = builder(TASKS[task_id])
            seed_scores = [run_task_score(config, base_seed=seed, task_id=task_id) for seed in base_seeds]
            per_seed_scores[task_id] = [round(score, 3) for score in seed_scores]
            task_scores[task_id] = round(sum(seed_scores) / len(seed_scores), 3)

        variant_payload[variant_name] = {
            "description": description,
            "scores": task_scores,
            "per_seed_scores": per_seed_scores,
            "average_score": round(sum(task_scores.values()) / len(task_scores), 3),
        }

        if variant_name == "full_heuristic":
            full_scores = task_scores

    deltas_vs_full: Dict[str, Dict[str, float]] = {}
    if full_scores is not None:
        for variant_name, payload in variant_payload.items():
            if variant_name == "full_heuristic":
                continue
            deltas_vs_full[variant_name] = {
                task_id: round(payload["scores"][task_id] - full_scores[task_id], 3)
                for task_id in task_ids
            }

    per_task_ordering = {
        task_id: sorted(
            (
                {
                    "variant": variant_name,
                    "score": payload["scores"][task_id],
                }
                for variant_name, payload in variant_payload.items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        for task_id in task_ids
    }

    return {
        "tasks": task_ids,
        "base_seeds": base_seeds,
        "variants": variant_payload,
        "deltas_vs_full": deltas_vs_full,
        "per_task_ordering": per_task_ordering,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run low-risk heuristic ablations to show benchmark sensitivity."
    )
    parser.add_argument(
        "--task",
        choices=[*TASKS.keys(), "all"],
        default="all",
        help="Run one task or the full task suite. Default: all.",
    )
    parser.add_argument(
        "--base-seeds",
        default="42,44,46",
        help="Comma-separated base seeds. Each base seed internally evaluates seed and seed+1. Default: 42,44,46",
    )
    args = parser.parse_args()

    task_ids = selected_tasks(args.task)
    base_seeds = parse_base_seeds(args.base_seeds)
    payload = run_variant_scores(task_ids, base_seeds)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
