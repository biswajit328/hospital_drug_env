from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


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
    respect_direct_only_constraints: bool
    use_forecast_reserve: bool
    family: str
    reasoning_mode: str
    observation_regime: str


TASKS: Dict[str, TaskConfig] = {
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
        respect_direct_only_constraints=False,
        use_forecast_reserve=False,
        family="stabilization",
        reasoning_mode="reactive triage",
        observation_regime="fully observed",
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
        respect_direct_only_constraints=False,
        use_forecast_reserve=False,
        family="resource balancing",
        reasoning_mode="budget-constrained planning",
        observation_regime="fully observed",
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
        respect_direct_only_constraints=False,
        use_forecast_reserve=False,
        family="scarcity adaptation",
        reasoning_mode="long-horizon scarcity planning",
        observation_regime="fully observed",
    ),
    "clinical": TaskConfig(
        name="Clinical Override Triage",
        difficulty="hard",
        objective=(
            "Protect patients who require primary drugs only by reserving direct stock "
            "for contraindicated cases and using substitutes only where clinically valid."
        ),
        policy_style="Constraint-aware triage with selective substitute deferral",
        allocation_mode="severity_first",
        use_substitutions=True,
        allow_emergency_orders=True,
        critical_threshold=0.80,
        max_emergency_orders_per_day=2,
        respect_direct_only_constraints=True,
        use_forecast_reserve=False,
        family="clinical constraints",
        reasoning_mode="constraint-aware triage",
        observation_regime="fully observed",
    ),
    "forecast": TaskConfig(
        name="Forecast-Aware Reserve Planning",
        difficulty="hard",
        objective=(
            "Maintain hospital resilience under uncertain next-day demand by reserving "
            "stock for high-risk wards instead of fully optimizing only the current day."
        ),
        policy_style="Forecast-aware reserve planning under partial observability",
        allocation_mode="severity_first",
        use_substitutions=True,
        allow_emergency_orders=True,
        critical_threshold=0.79,
        max_emergency_orders_per_day=2,
        respect_direct_only_constraints=False,
        use_forecast_reserve=True,
        family="partial observability",
        reasoning_mode="uncertainty-aware reserve planning",
        observation_regime="partially observed",
    ),
}

TASK_MAX_STEPS: Dict[str, int] = {
    "easy": 5,
    "medium": 7,
    "hard": 10,
    "clinical": 10,
    "forecast": 10,
}

TASK_SUCCESS_THRESHOLDS: Dict[str, float] = {
    "easy": 0.85,
    "medium": 0.70,
    "hard": 0.45,
    "clinical": 0.46,
    "forecast": 0.46,
}

TASK_ID_TO_DIFFICULTY: Dict[str, str] = {
    task_id: config.difficulty
    for task_id, config in TASKS.items()
}

TASK_RUNTIME_CONFIGS: Dict[str, Dict[str, int | str]] = {
    task_id: {
        "difficulty": config.difficulty,
        "max_steps": TASK_MAX_STEPS[task_id],
    }
    for task_id, config in TASKS.items()
}

DEFAULT_INFERENCE_TASKS: Tuple[str, ...] = ("easy", "medium", "hard")
CLINICAL_CONSTRAINT_TASKS = {
    task_id
    for task_id, config in TASKS.items()
    if config.respect_direct_only_constraints
}
UNCERTAIN_DEMAND_TASKS = {
    task_id
    for task_id, config in TASKS.items()
    if config.observation_regime == "partially observed"
}


def list_task_metadata() -> List[dict]:
    return [
        {
            "id": task_id,
            "name": config.name,
            "difficulty": config.difficulty,
            "description": config.objective,
            "policy_style": config.policy_style,
            "max_steps": TASK_MAX_STEPS[task_id],
            "success_threshold": TASK_SUCCESS_THRESHOLDS[task_id],
            "family": config.family,
            "reasoning_mode": config.reasoning_mode,
            "observation_regime": config.observation_regime,
            "grader": True,
            "has_grader": True,
        }
        for task_id, config in TASKS.items()
    ]
