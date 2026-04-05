# client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from hospital_drug_env.models import (
        DrugShortageAction,
        DrugShortageObservation,
        DrugShortageState,
    )
except ModuleNotFoundError:
    from models import DrugShortageAction, DrugShortageObservation, DrugShortageState

class HospitalDrugEnv(EnvClient[DrugShortageAction, DrugShortageObservation, DrugShortageState]):

    def _step_payload(self, action: DrugShortageAction) -> dict:
        return {
            "allocations": action.allocations,
            "emergency_orders": action.emergency_orders,
            "substitutions": action.substitutions,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=DrugShortageObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                day=obs_data.get("day", 1),
                budget_remaining=obs_data.get("budget_remaining", 0.0),
                inventory=obs_data.get("inventory", {}),
                incoming_shipments=obs_data.get("incoming_shipments", {}),
                wards=obs_data.get("wards", []),
                patient_outcomes=obs_data.get("patient_outcomes", {}),
                total_score=obs_data.get("total_score", 0.0),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DrugShortageState:
        return DrugShortageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "medium"),
            total_days=payload.get("total_days", 7),
            num_wards=payload.get("num_wards", 5),
            drugs=payload.get("drugs", []),
            initial_budget=payload.get("initial_budget", 10000.0),
        )
