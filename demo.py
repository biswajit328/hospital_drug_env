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
from typing import Iterable

try:
    from hospital_drug_env.server.environment import HospitalDrugEnvironment
except ModuleNotFoundError:
    from server.environment import HospitalDrugEnvironment

from grader import TASKS, TaskConfig, build_action


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


def print_header(title: str) -> None:
    print("-" * 80)
    print(title)
    print("-" * 80)


def run_demo_episode(config: TaskConfig, seed: int, max_steps: int | None) -> float | None:
    env = HospitalDrugEnvironment()
    observation = env.reset(difficulty=config.difficulty, seed=seed)

    print_header(f"Task: {config.name} ({config.difficulty})")
    print(f"Objective: {config.objective}")
    print(f"Policy style: {config.policy_style}")
    print(
        f"Reset complete -> day={observation.day}, "
        f"budget={observation.budget_remaining}, "
        f"total_score={observation.total_score:.3f}"
    )
    print(f"Initial inventory: {inventory_snapshot(observation.inventory)}")
    print(f"Sample wards: {ward_snapshot(observation.wards)}")
    print(f"Initial message: {observation.message}")

    step_num = 0
    while not observation.done:
        if max_steps is not None and step_num >= max_steps:
            print(f"Stopped early after {step_num} steps (demo limit).")
            return None

        action = build_action(observation, config)
        next_observation = env.step(action)
        step_num += 1

        print(f"\nStep {step_num}")
        print(f"Action -> {action_snapshot(action)}")
        print(
            f"Reward={next_observation.reward} | "
            f"Done={next_observation.done} | "
            f"Running total={next_observation.total_score:.3f} | "
            f"Budget={next_observation.budget_remaining}"
        )
        print(f"Message -> {next_observation.message}")

        if next_observation.patient_outcomes:
            top_outcomes = ", ".join(
                f"{ward_id}={score:.3f}"
                for ward_id, score in sorted(next_observation.patient_outcomes.items())
            )
            print(f"Ward outcomes -> {top_outcomes}")

        observation = next_observation

    print(f"\nFinal normalized score for {config.difficulty}: {observation.reward:.3f}")
    return float(observation.reward) if observation.reward is not None else None


def difficulties_to_run(choice: str) -> Iterable[str]:
    if choice == "all":
        return ("easy", "medium", "hard")
    return (choice,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end demo of the hospital drug shortage environment."
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
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
    args = parser.parse_args()

    print_header("Hospital Drug Shortage OpenEnv Demo")
    print(
        "This demo resets the environment, runs sample actions, prints rewards/done flags, "
        "and shows final scores."
    )
    print("Note: demo scores are single-seed illustrative runs; use grader.py for official benchmark scores.")
    print(f"Seed: {args.seed}")
    print(f"Difficulties: {', '.join(difficulties_to_run(args.difficulty))}")

    final_scores: dict[str, float] = {}
    for difficulty in difficulties_to_run(args.difficulty):
        score = run_demo_episode(TASKS[difficulty], seed=args.seed, max_steps=args.max_steps)
        if score is not None:
            final_scores[difficulty] = score
        print()

    print_header("Demo Summary")
    if final_scores:
        for difficulty, score in final_scores.items():
            print(f"{difficulty}: {score:.3f}")
        print("\nUse `python grader.py --difficulty all --seed 42` for the official multi-seed reference benchmark.")
    else:
        print("No episode reached termination because a max-step limit was used.")


if __name__ == "__main__":
    main()
