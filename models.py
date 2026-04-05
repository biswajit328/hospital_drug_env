import json
from typing import Any, Dict, List

from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from pydantic import field_validator

class DrugShortageAction(Action):
    allocations: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    emergency_orders: List[str] = Field(default_factory=list)
    substitutions: Dict[str, str] = Field(default_factory=dict)

    @field_validator("allocations", mode="before")
    @classmethod
    def parse_allocations(cls, value: Any) -> Dict[str, Dict[str, int]]:
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            value = json.loads(value)
        return value

    @field_validator("emergency_orders", mode="before")
    @classmethod
    def parse_emergency_orders(cls, value: Any) -> List[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            value = json.loads(value)
        return value

    @field_validator("substitutions", mode="before")
    @classmethod
    def parse_substitutions(cls, value: Any) -> Dict[str, str]:
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            value = json.loads(value)
        return value

class DrugShortageObservation(Observation):
    day: int
    budget_remaining: float
    inventory: Dict[str, int]
    incoming_shipments: Dict[str, int]
    wards: List[dict]
    patient_outcomes: Dict[str, float]
    total_score: float
    message: str

class DrugShortageState(State):
    difficulty: str = "easy"
    total_days: int = 7
    num_wards: int = 5
    drugs: List[str] = Field(default_factory=list)
    initial_budget: float = 10000.0
