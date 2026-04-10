# grader.py
import argparse
import json
import sys
from typing import Dict, List, Tuple

try:
    from hospital_drug_env.benchmark_registry import (
        TASKS,
        TASK_MAX_STEPS,
        TASK_SUCCESS_THRESHOLDS,
        TaskConfig,
        list_task_metadata,
    )
    from hospital_drug_env.score_utils import (
        MAX_VALID_SCORE,
        MIN_VALID_SCORE,
        clamp_validator_safe_score,
    )
    from hospital_drug_env.server.environment import (
        DRUG_COSTS,
        SUBSTITUTE_MAP,
        HospitalDrugEnvironment,
    )
    from hospital_drug_env.models import DrugShortageAction
except ModuleNotFoundError:
    from benchmark_registry import TASKS, TASK_MAX_STEPS, TASK_SUCCESS_THRESHOLDS, TaskConfig, list_task_metadata
    from score_utils import MAX_VALID_SCORE, MIN_VALID_SCORE, clamp_validator_safe_score
    from server.environment import DRUG_COSTS, SUBSTITUTE_MAP, HospitalDrugEnvironment
    from models import DrugShortageAction


def print_task_header(config: TaskConfig, seed: int, *, stream=None) -> None:
    stream = stream or sys.stderr
    print(f"Task: {config.name}", file=stream)
    print(f"Difficulty: {config.difficulty}", file=stream)
    print(f"Objective: {config.objective}", file=stream)
    print(f"Policy style: {config.policy_style}", file=stream)
    print(f"Deterministic seeds: [{seed}, {seed + 1}]", file=stream)


def collect_patient_needs(observation) -> List[dict]:
    patient_needs: List[dict] = []
    for ward in observation.wards:
        for patient in ward.get("patients", []):
            remaining_need = max(
                0,
                int(patient.get("doses_needed", 0)) - int(patient.get("doses_received", 0)),
            )
            if remaining_need <= 0:
                continue
            patient_needs.append(
                {
                    "ward_id": ward["ward_id"],
                    "drug": patient["drug"],
                    "severity": float(patient.get("severity", 0.0)),
                    "remaining_need": remaining_need,
                    "requires_primary_drug": bool(patient.get("requires_primary_drug", False)),
                }
            )

    patient_needs.sort(
        key=lambda need: (
            need["requires_primary_drug"],
            need["severity"],
            need["remaining_need"],
        ),
        reverse=True,
    )
    return patient_needs


def allocate_patients(
    patient_needs: List[dict],
    inventory: Dict[str, int],
    *,
    use_substitutions: bool,
    respect_direct_only_constraints: bool,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str], Dict[str, dict]]:
    allocations: Dict[str, Dict[str, int]] = {}
    substitutions: Dict[str, str] = {}
    shortages: Dict[str, dict] = {}
    remaining_inventory = {drug: max(0, int(qty)) for drug, qty in inventory.items()}

    for need in patient_needs:
        ward_alloc = allocations.setdefault(need["ward_id"], {})
        drug = need["drug"]
        remaining_need = need["remaining_need"]

        direct_given = min(remaining_need, remaining_inventory.get(drug, 0))
        if direct_given > 0:
            ward_alloc[drug] = ward_alloc.get(drug, 0) + direct_given
            remaining_inventory[drug] = remaining_inventory.get(drug, 0) - direct_given
            remaining_need -= direct_given

        if (
            remaining_need > 0
            and use_substitutions
            and not (respect_direct_only_constraints and need["requires_primary_drug"])
        ):
            substitute, _ = SUBSTITUTE_MAP.get(drug, (None, 0))
            if substitute:
                substitute_given = min(remaining_need, remaining_inventory.get(substitute, 0))
                if substitute_given > 0:
                    ward_alloc[substitute] = ward_alloc.get(substitute, 0) + substitute_given
                    remaining_inventory[substitute] = (
                        remaining_inventory.get(substitute, 0) - substitute_given
                    )
                    remaining_need -= substitute_given
                    substitutions[drug] = substitute

        if remaining_need > 0:
            shortage = shortages.setdefault(
                drug,
                {"max_severity": 0.0, "total_unmet": 0, "direct_only_unmet": 0},
            )
            shortage["max_severity"] = max(shortage["max_severity"], need["severity"])
            shortage["total_unmet"] += remaining_need
            if need["requires_primary_drug"]:
                shortage["direct_only_unmet"] += remaining_need

    allocations = {ward_id: ward_alloc for ward_id, ward_alloc in allocations.items() if ward_alloc}
    return allocations, substitutions, shortages


def allocate_wards(
    observation,
    inventory: Dict[str, int],
    *,
    use_substitutions: bool,
    respect_direct_only_constraints: bool,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str], Dict[str, dict]]:
    allocations: Dict[str, Dict[str, int]] = {}
    substitutions: Dict[str, str] = {}
    shortages: Dict[str, dict] = {}
    remaining_inventory = {drug: max(0, int(qty)) for drug, qty in inventory.items()}

    ranked_wards = sorted(
        observation.wards,
        key=lambda ward: max(ward.get("severity_scores", [0.0])),
        reverse=True,
    )

    for ward in ranked_wards:
        ward_alloc = allocations.setdefault(ward["ward_id"], {})
        ward_severity = max(ward.get("severity_scores", [0.0]))

        for drug, needed in ward.get("drug_needs", {}).items():
            remaining_need = max(0, int(needed))
            direct_given = min(remaining_need, remaining_inventory.get(drug, 0))
            if direct_given > 0:
                ward_alloc[drug] = ward_alloc.get(drug, 0) + direct_given
                remaining_inventory[drug] = remaining_inventory.get(drug, 0) - direct_given
                remaining_need -= direct_given

            if remaining_need > 0 and use_substitutions:
                substitute, _ = SUBSTITUTE_MAP.get(drug, (None, 0))
                direct_only_in_ward = int(ward.get("direct_only_counts", {}).get(drug, 0))
                if substitute and not (respect_direct_only_constraints and direct_only_in_ward > 0):
                    substitute_given = min(remaining_need, remaining_inventory.get(substitute, 0))
                    if substitute_given > 0:
                        ward_alloc[substitute] = ward_alloc.get(substitute, 0) + substitute_given
                        remaining_inventory[substitute] = (
                            remaining_inventory.get(substitute, 0) - substitute_given
                        )
                        remaining_need -= substitute_given
                        substitutions[drug] = substitute

            if remaining_need > 0:
                shortage = shortages.setdefault(
                    drug,
                    {"max_severity": 0.0, "total_unmet": 0, "direct_only_unmet": 0},
                )
                shortage["max_severity"] = max(shortage["max_severity"], ward_severity)
                shortage["total_unmet"] += remaining_need
                shortage["direct_only_unmet"] += int(ward.get("direct_only_counts", {}).get(drug, 0))

        if not ward_alloc:
            allocations.pop(ward["ward_id"], None)

    return allocations, substitutions, shortages


def choose_emergency_orders(
    shortages: Dict[str, dict],
    budget_remaining: float,
    config: TaskConfig,
) -> List[str]:
    emergency_orders: List[str] = []
    remaining_budget = budget_remaining

    ranked_shortages = sorted(
        shortages.items(),
        key=lambda item: (
            item[1].get("direct_only_unmet", 0),
            item[1]["max_severity"],
            item[1]["total_unmet"],
        ),
        reverse=True,
    )

    for drug, shortage in ranked_shortages:
        if shortage["max_severity"] < config.critical_threshold:
            continue

        emergency_cost = DRUG_COSTS.get(drug, 500) * 10
        if remaining_budget < emergency_cost:
            continue

        emergency_orders.append(drug)
        remaining_budget -= emergency_cost

        if len(emergency_orders) >= config.max_emergency_orders_per_day:
            break

    return emergency_orders


def build_forecast_reserve(observation) -> Dict[str, int]:
    reserve: Dict[str, int] = {}
    for ward in observation.wards:
        forecast = ward.get("demand_forecast") or {}
        band = forecast.get("expected_new_patients_band") or {}
        upper_bound = int(band.get("max", 0))
        if upper_bound <= 0:
            continue

        risk_band = forecast.get("risk_band", "low")
        if risk_band == "high":
            reserve_units = min(6, max(2, upper_bound * 2))
        elif risk_band == "medium":
            reserve_units = min(4, max(1, upper_bound + 1))
        else:
            reserve_units = min(2, upper_bound)

        for drug in forecast.get("priority_drugs", [])[:2]:
            reserve[drug] = reserve.get(drug, 0) + reserve_units

    return reserve


def apply_reserve_to_inventory(inventory: Dict[str, int], reserve: Dict[str, int]) -> Dict[str, int]:
    adjusted = {drug: max(0, int(qty)) for drug, qty in inventory.items()}
    for drug, qty in reserve.items():
        adjusted[drug] = max(0, adjusted.get(drug, 0) - qty)
    return adjusted


def build_action(observation, config: TaskConfig) -> DrugShortageAction:
    patient_needs = collect_patient_needs(observation)
    reserve_by_drug = build_forecast_reserve(observation) if config.use_forecast_reserve else {}
    working_inventory = (
        apply_reserve_to_inventory(observation.inventory, reserve_by_drug)
        if reserve_by_drug
        else observation.inventory
    )

    if config.allocation_mode == "severity_first":
        allocations, substitutions, shortages = allocate_patients(
            patient_needs,
            working_inventory,
            use_substitutions=config.use_substitutions,
            respect_direct_only_constraints=config.respect_direct_only_constraints,
        )
    else:
        allocations, substitutions, shortages = allocate_wards(
            observation,
            working_inventory,
            use_substitutions=config.use_substitutions,
            respect_direct_only_constraints=config.respect_direct_only_constraints,
        )

    emergency_orders: List[str] = []
    if config.allow_emergency_orders and shortages:
        emergency_orders = choose_emergency_orders(
            shortages,
            observation.budget_remaining,
            config,
        )
        if emergency_orders:
            augmented_inventory = dict(observation.inventory)
            for drug in emergency_orders:
                augmented_inventory[drug] = augmented_inventory.get(drug, 0) + 20
            if reserve_by_drug:
                augmented_inventory = apply_reserve_to_inventory(augmented_inventory, reserve_by_drug)

            if config.allocation_mode == "severity_first":
                allocations, substitutions, _ = allocate_patients(
                    patient_needs,
                    augmented_inventory,
                    use_substitutions=config.use_substitutions,
                    respect_direct_only_constraints=config.respect_direct_only_constraints,
                )
            else:
                allocations, substitutions, _ = allocate_wards(
                    observation,
                    augmented_inventory,
                    use_substitutions=config.use_substitutions,
                    respect_direct_only_constraints=config.respect_direct_only_constraints,
                )

    return DrugShortageAction(
        allocations=allocations,
        emergency_orders=emergency_orders,
        substitutions=substitutions,
    )


def run_episode(config: TaskConfig, seed: int = 42) -> float:
    """
    Run one full episode with a deterministic task-specific policy.
    Returns a normalized score in the 0.0-1.0 range.
    """
    env = HospitalDrugEnvironment()
    observation = env.reset(difficulty=config.difficulty, seed=seed)

    done = False
    while not done:
        action = build_action(observation, config)
        observation = env.step(action)
        done = observation.done

    raw_score = float(observation.reward) if observation.reward is not None else 0.0
    return clamp_validator_safe_score(raw_score)


def run_task_score(config: TaskConfig, base_seed: int = 42) -> float:
    seeds = [base_seed, base_seed + 1]
    scores = [run_episode(config, seed=seed) for seed in seeds]
    score = sum(scores) / len(scores)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Task score out of range for {config.name}: {score}")
    return clamp_validator_safe_score(score)


def grade_easy(seed: int = 42, *, verbose: bool = False, stream=None) -> float:
    """Task 1 - Critical Care Stabilization."""
    config = TASKS["easy"]
    score = run_task_score(config, base_seed=seed)
    if verbose:
        stream = stream or sys.stderr
        print_task_header(config, seed, stream=stream)
        print(f"Score: {score:.3f}", file=stream)
    return score


def grade_medium(seed: int = 42, *, verbose: bool = False, stream=None) -> float:
    """Task 2 - Budget-Constrained Ward Balancing."""
    config = TASKS["medium"]
    score = run_task_score(config, base_seed=seed)
    if verbose:
        stream = stream or sys.stderr
        print_task_header(config, seed, stream=stream)
        print(f"Score: {score:.3f}", file=stream)
    return score


def grade_hard(seed: int = 42, *, verbose: bool = False, stream=None) -> float:
    """Task 3 - Substitution-Aware Surge Response."""
    config = TASKS["hard"]
    score = run_task_score(config, base_seed=seed)
    if verbose:
        stream = stream or sys.stderr
        print_task_header(config, seed, stream=stream)
        print(f"Score: {score:.3f}", file=stream)
    return score


def grade_clinical(seed: int = 42, *, verbose: bool = False, stream=None) -> float:
    """Task 4 - Clinical Override Triage."""
    config = TASKS["clinical"]
    score = run_task_score(config, base_seed=seed)
    if verbose:
        stream = stream or sys.stderr
        print_task_header(config, seed, stream=stream)
        print(f"Score: {score:.3f}", file=stream)
    return score


def grade_forecast(seed: int = 42, *, verbose: bool = False, stream=None) -> float:
    """Task 5 - Forecast-Aware Reserve Planning."""
    config = TASKS["forecast"]
    score = run_task_score(config, base_seed=seed)
    if verbose:
        stream = stream or sys.stderr
        print_task_header(config, seed, stream=stream)
        print(f"Score: {score:.3f}", file=stream)
    return score


def run_all_graders(seed: int = 42, *, verbose: bool = False, stream=None) -> dict:
    """Run the full task suite and return named task scores."""
    stream = stream or sys.stderr
    if verbose:
        print("Running all graders...", file=stream)
        print("-" * 60, file=stream)

    results = {
        "easy": grade_easy(seed=seed, verbose=verbose, stream=stream),
        "medium": grade_medium(seed=seed, verbose=verbose, stream=stream),
        "hard": grade_hard(seed=seed, verbose=verbose, stream=stream),
        "clinical": grade_clinical(seed=seed, verbose=verbose, stream=stream),
        "forecast": grade_forecast(seed=seed, verbose=verbose, stream=stream),
    }

    if verbose:
        print("-" * 60, file=stream)
        print(f"Average score: {sum(results.values()) / len(results):.3f}", file=stream)
        print(
            "All scores strictly in (0.0, 1.0):",
            all(0.0 < v < 1.0 for v in results.values()),
            file=stream,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "clinical", "forecast", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print human-readable grader commentary to stderr.",
    )
    args = parser.parse_args()

    if args.difficulty == "all":
        results = run_all_graders(seed=args.seed, verbose=args.verbose)
    else:
        config = TASKS[args.difficulty]
        results = {args.difficulty: run_task_score(config, base_seed=args.seed)}
        if args.verbose:
            print_task_header(config, args.seed)
            print(f"Score: {results[args.difficulty]:.3f}", file=sys.stderr)

    print(json.dumps(results, indent=2, sort_keys=True))
