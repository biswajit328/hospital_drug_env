"""Reviewer-friendly end-to-end demo for the hospital drug shortage environment.

Run:
    python demo.py

This script:
1. Resets the environment
2. Runs a small deterministic policy through the task
3. Prints observations, actions, rewards, done flags, and final scores
4. Exits cleanly
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

try:
    from hospital_drug_env.server.environment import HospitalDrugEnvironment
except ModuleNotFoundError:
    from server.environment import HospitalDrugEnvironment

from grader import (
    MAX_VALID_SCORE,
    MIN_VALID_SCORE,
    TASKS,
    TaskConfig,
    build_action,
    list_task_metadata as grader_task_metadata,
    run_task_score,
)
from score_utils import clamp_validator_safe_score


def inventory_snapshot(inventory: dict[str, int]) -> str:
    ordered = sorted(inventory.items(), key=lambda item: item[0])
    return ", ".join(f"{drug}={qty}" for drug, qty in ordered)


def ward_snapshot(wards: list[dict], limit: int = 3) -> str:
    snippets: list[str] = []
    for ward in wards[:limit]:
        severities = ward.get("severity_scores", [])
        max_severity = max(severities) if severities else 0.0
        snippets.append(
            f"{ward['ward_id']}[{ward.get('specialty', 'unknown')}] "
            f"patients={ward.get('patient_count', 0)} max_severity={max_severity:.2f}"
        )
    return " | ".join(snippets)


def action_snapshot(action) -> str:
    total_allocated = sum(
        qty
        for ward_alloc in action.allocations.values()
        for qty in ward_alloc.values()
    )
    wards_touched = len(action.allocations)
    emergency = action.emergency_orders or []
    substitutions = action.substitutions or {}
    return (
        f"allocated_doses={total_allocated}, "
        f"wards_touched={wards_touched}, "
        f"emergency_orders={emergency}, "
        f"substitutions={substitutions}"
    )


def print_header(title: str, *, stream=None) -> None:
    stream = stream or sys.stderr
    print("-" * 80, file=stream)
    print(title, file=stream)
    print("-" * 80, file=stream)


def clamp_task_score(score: float | None) -> float | None:
    return clamp_validator_safe_score(score)


def list_task_metadata() -> list[dict]:
    return grader_task_metadata()


def run_demo_episode(
    task_id: str,
    config: TaskConfig,
    seed: int,
    max_steps: int | None,
    *,
    verbose: bool,
    stream=None,
) -> float | None:
    stream = stream or sys.stderr
    env = HospitalDrugEnvironment()
    observation = env.reset(task_id=task_id, difficulty=config.difficulty, seed=seed)

    if verbose:
        print_header(f"Task: {config.name} ({config.difficulty})", stream=stream)
        print(f"Objective: {config.objective}", file=stream)
        print(f"Policy style: {config.policy_style}", file=stream)
        print(
            f"Reset complete -> day={observation.day}, "
            f"budget={observation.budget_remaining}, "
            f"total_score={observation.total_score:.3f}",
            file=stream,
        )
        print(f"Initial inventory: {inventory_snapshot(observation.inventory)}", file=stream)
        print(f"Sample wards: {ward_snapshot(observation.wards)}", file=stream)
        print(f"Initial message: {observation.message}", file=stream)

    step_num = 0
    while not observation.done:
        if max_steps is not None and step_num >= max_steps:
            if verbose:
                print(f"Stopped early after {step_num} steps (demo limit).", file=stream)
            return None

        action = build_action(observation, config)
        next_observation = env.step(action)
        step_num += 1

        if verbose:
            print(f"\nStep {step_num}", file=stream)
            print(f"Action -> {action_snapshot(action)}", file=stream)
            print(
                f"Reward={next_observation.reward} | "
                f"Done={next_observation.done} | "
                f"Running total={next_observation.total_score:.3f} | "
                f"Budget={next_observation.budget_remaining}",
                file=stream,
            )
            print(f"Message -> {next_observation.message}", file=stream)

            if next_observation.patient_outcomes:
                top_outcomes = ", ".join(
                    f"{ward_id}={score:.3f}"
                    for ward_id, score in sorted(next_observation.patient_outcomes.items())
                )
                print(f"Ward outcomes -> {top_outcomes}", file=stream)

        observation = next_observation

    episode_score = clamp_task_score(observation.reward)
    official_score = run_task_score(config, base_seed=seed)
    if verbose:
        print(f"Single-seed demo score: {episode_score:.3f}", file=stream)
        print(f"Official multi-seed task score: {official_score:.3f}", file=stream)
    return official_score


def difficulties_to_run(choice: str) -> Iterable[str]:
    if choice == "all":
        return tuple(TASKS.keys())
    return (choice,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end demo of the hospital drug shortage environment."
    )
    parser.add_argument(
        "--difficulty",
        choices=[*TASKS.keys(), "all"],
        default="all",
        help="Which task difficulty to demo. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic environment seed. Default: 42.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on the number of steps per episode for shorter demos.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full step-by-step walkthrough instead of concise task summaries.",
    )
    args = parser.parse_args()

    if args.verbose:
        print_header("Hospital Drug Shortage OpenEnv Demo")
        print("This demo exposes the task suite and prints normalized task scores.", file=sys.stderr)
        print("Verbose mode is enabled: step-by-step episode logs will be shown.", file=sys.stderr)
        print(f"Seed: {args.seed}", file=sys.stderr)
        print(f"Difficulties: {', '.join(difficulties_to_run(args.difficulty))}", file=sys.stderr)
        print(f"Tasks discovered: {len(list_task_metadata())}", file=sys.stderr)

    final_scores: dict[str, float] = {}
    for difficulty in difficulties_to_run(args.difficulty):
        score = run_demo_episode(
            difficulty,
            TASKS[difficulty],
            seed=args.seed,
            max_steps=args.max_steps,
            verbose=args.verbose,
            stream=sys.stderr,
        )
        if score is not None:
            final_scores[difficulty] = score
        if args.verbose:
            print(file=sys.stderr)

    payload = {
        "tasks_discovered": len(list_task_metadata()),
        "scores": final_scores,
        "all_scores_strictly_between_zero_and_one": all(
            0.0 < score < 1.0 for score in final_scores.values()
        ),
        "tasks": list_task_metadata(),
    }

    if args.verbose:
        print_header("Demo Summary", stream=sys.stderr)
        if final_scores:
            for difficulty, score in final_scores.items():
                print(f"{difficulty}: {score:.3f}", file=sys.stderr)
            print(
                f"\nAll task scores strictly in (0.0, 1.0): "
                f"{payload['all_scores_strictly_between_zero_and_one']}",
                file=sys.stderr,
            )
            print("\nTask metadata:", file=sys.stderr)
            for task in list_task_metadata():
                print(
                    f"- {task['id']}: grader={task['has_grader']} max_steps={task['max_steps']} "
                    f"success_threshold={task['success_threshold']}",
                    file=sys.stderr,
                )
        else:
            print("No episode reached termination because a max-step limit was used.", file=sys.stderr)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
