# grader.py
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
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
    from score_utils import MAX_VALID_SCORE, MIN_VALID_SCORE, clamp_validator_safe_score
    from server.environment import DRUG_COSTS, SUBSTITUTE_MAP, HospitalDrugEnvironment
    from models import DrugShortageAction


@dataclass(frozen=True)
class TaskConfig:
    name: str
    difficulty: str
    objective: str
    policy_style: str
    allocation_mode: str
    use_substitutions: bool
    allow_emergency_orders: bool
    critical_threshold: float
    max_emergency_orders_per_day: int


TASKS = {
    "easy": TaskConfig(
        name="Critical Care Stabilization",
        difficulty="easy",
        objective=(
            "Maintain high service levels when supplies are mostly reliable by "
            "routing inventory to the highest-severity patients first."
        ),
        policy_style="Severity-aware ward balancing without emergency procurement",
        allocation_mode="ward_balanced",
        use_substitutions=False,
        allow_emergency_orders=False,
        critical_threshold=0.65,
        max_emergency_orders_per_day=0,
    ),
    "medium": TaskConfig(
        name="Budget-Constrained Ward Balancing",
        difficulty="medium",
        objective=(
            "Balance medium-term patient outcomes with budget discipline, using "
            "targeted emergency orders only when high-severity shortages remain."
        ),
        policy_style="Budget-aware balancing with selective emergency ordering",
        allocation_mode="ward_balanced",
        use_substitutions=False,
        allow_emergency_orders=True,
        critical_threshold=0.78,
        max_emergency_orders_per_day=1,
    ),
    "hard": TaskConfig(
        name="Substitution-Aware Surge Response",
        difficulty="hard",
        objective=(
            "Handle prolonged scarcity by combining severity-first allocation, "
            "selective emergency procurement, and clinically valid substitutions."
        ),
        policy_style="Severity-first triage with substitutions and disruption recovery",
        allocation_mode="severity_first",
        use_substitutions=True,
        allow_emergency_orders=True,
        critical_threshold=0.82,
        max_emergency_orders_per_day=2,
    ),
}

TASK_MAX_STEPS = {
    "easy": 5,
    "medium": 7,
    "hard": 10,
}
TASK_SUCCESS_THRESHOLDS = {
    "easy": 0.85,
    "medium": 0.70,
    "hard": 0.45,
}


def list_task_metadata() -> List[dict]:
    tasks: List[dict] = []
    for task_id, config in TASKS.items():
        tasks.append(
            {
                "id": task_id,
                "name": config.name,
                "difficulty": config.difficulty,
                "description": config.objective,
                "policy_style": config.policy_style,
                "max_steps": TASK_MAX_STEPS[task_id],
                "success_threshold": TASK_SUCCESS_THRESHOLDS[task_id],
                "grader": True,
                "has_grader": True,
            }
        )
    return tasks


def print_task_header(config: TaskConfig, seed: int) -> None:
    print(f"Task: {config.name}")
    print(f"Difficulty: {config.difficulty}")
    print(f"Objective: {config.objective}")
    print(f"Policy style: {config.policy_style}")
    print(f"Deterministic seeds: [{seed}, {seed + 1}]")


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
                }
            )

    patient_needs.sort(
        key=lambda need: (need["severity"], need["remaining_need"]),
        reverse=True,
    )
    return patient_needs


def allocate_patients(
    patient_needs: List[dict],
    inventory: Dict[str, int],
    *,
    use_substitutions: bool,
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

        if remaining_need > 0 and use_substitutions:
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
                {"max_severity": 0.0, "total_unmet": 0},
            )
            shortage["max_severity"] = max(shortage["max_severity"], need["severity"])
            shortage["total_unmet"] += remaining_need

    allocations = {ward_id: ward_alloc for ward_id, ward_alloc in allocations.items() if ward_alloc}
    return allocations, substitutions, shortages


def allocate_wards(
    observation,
    inventory: Dict[str, int],
    *,
    use_substitutions: bool,
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
                    {"max_severity": 0.0, "total_unmet": 0},
                )
                shortage["max_severity"] = max(shortage["max_severity"], ward_severity)
                shortage["total_unmet"] += remaining_need

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
        key=lambda item: (item[1]["max_severity"], item[1]["total_unmet"]),
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


def build_action(observation, config: TaskConfig) -> DrugShortageAction:
    patient_needs = collect_patient_needs(observation)

    if config.allocation_mode == "severity_first":
        allocations, substitutions, shortages = allocate_patients(
            patient_needs,
            observation.inventory,
            use_substitutions=config.use_substitutions,
        )
    else:
        allocations, substitutions, shortages = allocate_wards(
            observation,
            observation.inventory,
            use_substitutions=config.use_substitutions,
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

            if config.allocation_mode == "severity_first":
                allocations, substitutions, _ = allocate_patients(
                    patient_needs,
                    augmented_inventory,
                    use_substitutions=config.use_substitutions,
                )
            else:
                allocations, substitutions, _ = allocate_wards(
                    observation,
                    augmented_inventory,
                    use_substitutions=config.use_substitutions,
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


def grade_easy(seed: int = 42) -> float:
    """Task 1 - Critical Care Stabilization."""
    config = TASKS["easy"]
    score = run_task_score(config, base_seed=seed)
    print_task_header(config, seed)
    print(f"Score: {score:.3f}")
    return score


def grade_medium(seed: int = 42) -> float:
    """Task 2 - Budget-Constrained Ward Balancing."""
    config = TASKS["medium"]
    score = run_task_score(config, base_seed=seed)
    print_task_header(config, seed)
    print(f"Score: {score:.3f}")
    return score


def grade_hard(seed: int = 42) -> float:
    """Task 3 - Substitution-Aware Surge Response."""
    config = TASKS["hard"]
    score = run_task_score(config, base_seed=seed)
    print_task_header(config, seed)
    print(f"Score: {score:.3f}")
    return score


def run_all_graders(seed: int = 42) -> dict:
    """Run all 3 graders and return named task scores."""
    print("Running all graders...")
    print("-" * 60)

    results = {
        "easy": grade_easy(seed=seed),
        "medium": grade_medium(seed=seed),
        "hard": grade_hard(seed=seed),
    }

    print("-" * 60)
    print(f"Average score: {sum(results.values()) / len(results):.3f}")
    print("All scores strictly in (0.0, 1.0):", all(0.0 < v < 1.0 for v in results.values()))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.difficulty == "all":
        results = run_all_graders(seed=args.seed)
    else:
        config = TASKS[args.difficulty]
        results = {args.difficulty: run_task_score(config, base_seed=args.seed)}
        print_task_header(config, args.seed)
        print(f"Score: {results[args.difficulty]:.3f}")

    print(json.dumps(results, indent=2, sort_keys=True))
