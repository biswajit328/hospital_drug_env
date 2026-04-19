# server/environment.py
import random
import uuid
from typing import Dict, List

from openenv.core.env_server import Environment

try:
    from hospital_drug_env.benchmark_registry import (
        CLINICAL_CONSTRAINT_TASKS,
        LOGISTICS_REALISM_TASKS,
        TASK_ID_TO_DIFFICULTY,
        UNCERTAIN_DEMAND_TASKS,
    )
    from hospital_drug_env.score_utils import clamp_validator_safe_score
    from hospital_drug_env.models import (
        DrugShortageAction,
        DrugShortageObservation,
        DrugShortageState,
    )
except ModuleNotFoundError:
    from benchmark_registry import (
        CLINICAL_CONSTRAINT_TASKS,
        LOGISTICS_REALISM_TASKS,
        TASK_ID_TO_DIFFICULTY,
        UNCERTAIN_DEMAND_TASKS,
    )
    from score_utils import clamp_validator_safe_score
    from models import (
        DrugShortageAction,
        DrugShortageObservation,
        DrugShortageState,
    )

PRIMARY_DRUGS = ["amoxicillin", "remdesivir", "paracetamol", "insulin", "morphine"]

DRUG_COSTS = {
    "amoxicillin": 50,
    "remdesivir": 800,
    "paracetamol": 10,
    "insulin": 200,
    "morphine": 150,
    "azithromycin": 40,
    "ibuprofen": 12,
    "tramadol": 90,
}

SUBSTITUTE_MAP = {
    # drug -> substitute, but substitute has side_effect_penalty
    "amoxicillin": ("azithromycin", 0.2),
    "remdesivir": (None, 0),   # no substitute
    "paracetamol": ("ibuprofen", 0.05),
    "insulin": (None, 0),
    "morphine": ("tramadol", 0.15),
}

SUBSTITUTE_DRUGS = sorted(
    substitute for substitute, _ in SUBSTITUTE_MAP.values() if substitute is not None
)
ALL_DRUGS = PRIMARY_DRUGS + SUBSTITUTE_DRUGS
COLD_CHAIN_DRUGS = {"remdesivir", "insulin"}
DRUG_SHELF_LIFE = {
    "amoxicillin": 4,
    "remdesivir": 2,
    "paracetamol": 5,
    "insulin": 3,
    "morphine": 4,
    "azithromycin": 4,
    "ibuprofen": 5,
    "tramadol": 4,
}
COLD_CHAIN_CAPACITY_BY_DIFFICULTY = {
    "easy": 38,
    "medium": 28,
    "hard": 22,
}
SUPPLIER_BASE_PROFILES = {
    "amoxicillin": {"reliability": 0.82, "lead_time_band": (1, 2)},
    "remdesivir": {"reliability": 0.55, "lead_time_band": (2, 3)},
    "paracetamol": {"reliability": 0.9, "lead_time_band": (1, 2)},
    "insulin": {"reliability": 0.62, "lead_time_band": (2, 3)},
    "morphine": {"reliability": 0.68, "lead_time_band": (1, 2)},
    "azithromycin": {"reliability": 0.76, "lead_time_band": (1, 2)},
    "ibuprofen": {"reliability": 0.88, "lead_time_band": (1, 2)},
    "tramadol": {"reliability": 0.72, "lead_time_band": (1, 2)},
}

WARD_SPECIALTY_PROFILES = {
    "general_medicine": [3, 1, 4, 2, 1],
    "infectious_disease": [2, 5, 2, 1, 1],
    "endocrine": [1, 1, 2, 5, 1],
    "pain_management": [1, 1, 2, 1, 5],
    "respiratory": [2, 4, 3, 1, 1],
}

WARD_PRIORITY_MULTIPLIERS = {
    "general_medicine": 1.0,
    "infectious_disease": 1.15,
    "endocrine": 1.05,
    "pain_management": 0.95,
    "respiratory": 1.1,
}

DIFFICULTY_SETTINGS = {
    "easy": {
        "num_wards": 3,
        "total_days": 5,
        "initial_inventory_multiplier": 2.0,
        "shipment_reliability": 0.9,
        "budget": 15000,
        "patient_arrival_rate": 0.1,
    },
    "medium": {
        "num_wards": 5,
        "total_days": 7,
        "initial_inventory_multiplier": 1.0,
        "shipment_reliability": 0.6,
        "budget": 10000,
        "patient_arrival_rate": 0.25,
    },
    "hard": {
        "num_wards": 5,
        "total_days": 10,
        "initial_inventory_multiplier": 0.5,
        "shipment_reliability": 0.3,
        "budget": 6000,
        "patient_arrival_rate": 0.4,
    },
}

class HospitalWard:
    def __init__(
        self,
        ward_id: str,
        difficulty: str,
        rng: random.Random,
        specialty: str,
        *,
        clinical_mode: bool = False,
    ):
        self.ward_id = ward_id
        self.difficulty = difficulty
        self._rng = rng
        self.specialty = specialty
        self._clinical_mode = clinical_mode
        self.patients = self._generate_initial_patients()

    def _sample_drug(self) -> str:
        weights = WARD_SPECIALTY_PROFILES.get(self.specialty, [1] * len(PRIMARY_DRUGS))
        return self._rng.choices(PRIMARY_DRUGS, weights=weights, k=1)[0]

    def _sample_from_preferred(self, preferred_drugs: List[str] | None = None) -> str:
        if not preferred_drugs:
            return self._sample_drug()

        base_weights = WARD_SPECIALTY_PROFILES.get(self.specialty, [1] * len(PRIMARY_DRUGS))
        weighted_preferences = []
        for drug in preferred_drugs:
            if drug in PRIMARY_DRUGS:
                weighted_preferences.append(
                    base_weights[PRIMARY_DRUGS.index(drug)] + 2
                )
            else:
                weighted_preferences.append(1)
        return self._rng.choices(preferred_drugs, weights=weighted_preferences, k=1)[0]

    def _create_patient(
        self,
        patient_id: str,
        *,
        severity_range: tuple[float, float] = (0.2, 1.0),
        preferred_drugs: List[str] | None = None,
    ) -> dict:
        severity = self._rng.uniform(*severity_range)
        needed_drug = self._sample_from_preferred(preferred_drugs)
        doses_needed = max(1, int(severity * 3))
        allowed_substitute, _ = SUBSTITUTE_MAP.get(needed_drug, (None, 0))
        requires_primary_drug = False
        clinical_reason = None
        if self._clinical_mode and allowed_substitute:
            direct_only_probability = min(0.7, 0.15 + (severity * 0.45))
            requires_primary_drug = self._rng.random() < direct_only_probability
            if requires_primary_drug:
                clinical_reason = self._rng.choice(["contraindication", "clinician_override"])
        return {
            "id": patient_id,
            "severity": round(severity, 2),
            "drug": needed_drug,
            "doses_needed": doses_needed,
            "doses_received": 0,
            "stable_days": 0,
            "requires_primary_drug": requires_primary_drug,
            "contraindicated_substitute": allowed_substitute if requires_primary_drug else None,
            "clinical_reason": clinical_reason,
        }

    def _generate_initial_patients(self) -> List[dict]:
        n = self._rng.randint(3, 8)
        return [self._create_patient(f"P{i}") for i in range(n)]

    def new_arrivals(
        self,
        arrival_rate: float,
        *,
        forced_count: int | None = None,
        severity_range: tuple[float, float] = (0.3, 1.0),
        preferred_drugs: List[str] | None = None,
    ) -> int:
        if forced_count is None:
            if self._rng.random() >= arrival_rate:
                return 0
            n_new = self._rng.randint(1, 3)
        else:
            n_new = max(0, int(forced_count))

        for _ in range(n_new):
            self.patients.append(
                self._create_patient(
                    f"NEW_{uuid.uuid4().hex[:4]}",
                    severity_range=severity_range,
                    preferred_drugs=preferred_drugs,
                )
            )
        return n_new

    def add_surge_patients(self, count: int):
        for _ in range(count):
            self.patients.append(
                self._create_patient(
                    f"SURGE_{uuid.uuid4().hex[:4]}",
                    severity_range=(0.75, 1.0),
                )
            )

    @property
    def priority_weight(self) -> float:
        return WARD_PRIORITY_MULTIPLIERS.get(self.specialty, 1.0)

    def compute_direct_only_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for patient in self.patients:
            if patient.get("requires_primary_drug"):
                drug = patient["drug"]
                counts[drug] = counts.get(drug, 0) + 1
        return counts

    def compute_drug_needs(self) -> Dict[str, int]:
        needs = {}
        for p in self.patients:
            drug = p["drug"]
            needs[drug] = needs.get(drug, 0) + p["doses_needed"]
        return needs

    def apply_allocation(self, allocation: Dict[str, int], substitutions: Dict[str, str]) -> float:
        """
        Apply doses to patients. Returns outcome score for this ward today.
        Score = weighted average of (doses_received / doses_needed) per patient,
        weighted by severity. Substitutions apply with penalty.
        """
        total_weight = sum(p["severity"] for p in self.patients)
        weighted_score = 0.0
        remaining_doses = {
            drug: max(0, int(doses))
            for drug, doses in allocation.items()
        }

        for p in sorted(self.patients, key=lambda patient: patient["severity"], reverse=True):
            drug = p["drug"]
            needed = p["doses_needed"]

            # Check direct allocation
            given = remaining_doses.get(drug, 0)
            actual_given = min(given, needed)
            if actual_given:
                remaining_doses[drug] = given - actual_given

            penalty = 0.0
            clinical_gap_penalty = 0.0
            # Check if substitute was used
            allowed_substitute, side_effect = SUBSTITUTE_MAP.get(drug, (None, 0))
            requested_substitute = substitutions.get(drug)
            if (
                actual_given < needed
                and allowed_substitute
                and requested_substitute == allowed_substitute
            ):
                if not p.get("requires_primary_drug"):
                    sub_given = remaining_doses.get(allowed_substitute, 0)
                    actual_sub = min(sub_given, needed - actual_given)
                    if actual_sub:
                        remaining_doses[allowed_substitute] = sub_given - actual_sub
                    actual_given += actual_sub
                    penalty = side_effect * (actual_sub / needed) if needed > 0 else 0
                else:
                    clinical_gap_penalty += 0.08

            p["doses_received"] = actual_given
            ratio = (actual_given / needed) if needed > 0 else 1.0
            ratio = min(ratio, 1.0)
            if p.get("requires_primary_drug") and ratio < 1.0:
                clinical_gap_penalty += 0.12 * (1.0 - ratio)
            score_contribution = (ratio - penalty - clinical_gap_penalty) * p["severity"]
            weighted_score += score_contribution

        ward_score = weighted_score / total_weight if total_weight > 0 else 0.0
        return max(0.0, min(1.0, ward_score))

    def advance_day(self) -> int:
        discharged = 0
        remaining_patients = []

        for patient in self.patients:
            needed = max(1, int(patient["doses_needed"]))
            ratio = patient["doses_received"] / needed

            if ratio >= 1.0:
                patient["stable_days"] = patient.get("stable_days", 0) + 1
                patient["severity"] = round(max(0.2, patient["severity"] - 0.04), 2)
            elif ratio < 0.5:
                patient["stable_days"] = 0
                patient["severity"] = round(min(1.0, patient["severity"] + 0.06), 2)
            else:
                patient["stable_days"] = 0
                patient["severity"] = round(min(1.0, patient["severity"] + 0.02), 2)

            patient["doses_needed"] = max(1, int(patient["severity"] * 3))
            patient["doses_received"] = 0

            if patient["stable_days"] >= 2 and patient["severity"] <= 0.35:
                discharged += 1
                continue

            remaining_patients.append(patient)

        self.patients = remaining_patients
        return discharged

    def to_dict(self) -> dict:
        drug_received = {}
        for patient in self.patients:
            drug_received[patient["drug"]] = (
                drug_received.get(patient["drug"], 0) + patient["doses_received"]
            )

        return {
            "ward_id": self.ward_id,
            "specialty": self.specialty,
            "priority_weight": round(self.priority_weight, 2),
            "patient_count": len(self.patients),
            "severity_scores": [p["severity"] for p in self.patients],
            "drug_needs": self.compute_drug_needs(),
            "direct_only_counts": self.compute_direct_only_counts(),
            "drug_received": drug_received,
            "patients": [
                {
                    "id": p["id"],
                    "severity": p["severity"],
                    "drug": p["drug"],
                    "doses_needed": p["doses_needed"],
                    "doses_received": p["doses_received"],
                    "requires_primary_drug": p.get("requires_primary_drug", False),
                    "contraindicated_substitute": p.get("contraindicated_substitute"),
                    "clinical_reason": p.get("clinical_reason"),
                }
                for p in sorted(
                    self.patients,
                    key=lambda patient: patient["severity"],
                    reverse=True,
                )
            ],
        }


class HospitalDrugEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = DrugShortageState()
        self._wards: List[HospitalWard] = []
        self._inventory: Dict[str, int] = {}
        self._budget: float = 0.0
        self._day: int = 0
        self._total_score: float = 0.0
        self._incoming_shipments: Dict[str, int] = {}
        self._difficulty: str = "medium"
        self._settings: dict = {}
        self._rng = random.Random()
        self._scheduled_event_day: int | None = None
        self._event_type: str | None = None
        self._event_applied = False
        self._task_id = "medium"
        self._clinical_mode = False
        self._forecast_mode = False
        self._logistics_mode = False
        self._latent_demand_plan: Dict[str, dict] = {}
        self._forecast_summary: Dict[str, dict] = {}
        self._inventory_lots: Dict[str, List[dict]] = {}
        self._pending_shipments: List[dict] = []
        self._supplier_status: Dict[str, dict] = {}
        self._storage_status: Dict[str, int | float | str] = {}

    def reset(
        self,
        seed=None,
        episode_id=None,
        difficulty="medium",
        task_id=None,
        **kwargs,
    ) -> DrugShortageObservation:
        self._rng = random.Random(seed)
        self._task_id = task_id or difficulty
        self._clinical_mode = self._task_id in CLINICAL_CONSTRAINT_TASKS
        self._forecast_mode = self._task_id in UNCERTAIN_DEMAND_TASKS
        self._logistics_mode = self._task_id in LOGISTICS_REALISM_TASKS

        if task_id in TASK_ID_TO_DIFFICULTY:
            difficulty = TASK_ID_TO_DIFFICULTY[task_id]

        self._difficulty = difficulty if difficulty in DIFFICULTY_SETTINGS else "medium"
        self._settings = DIFFICULTY_SETTINGS[self._difficulty]
        self._day = 1
        self._total_score = 0.0
        self._budget = self._settings["budget"]
        self._event_applied = False
        self._scheduled_event_day = None
        self._event_type = None
        self._inventory_lots = {drug: [] for drug in ALL_DRUGS}
        self._pending_shipments = []
        self._supplier_status = {}
        self._storage_status = {}
        if self._difficulty == "hard":
            self._scheduled_event_day = self._rng.randint(3, self._settings["total_days"] - 1)
            self._event_type = self._rng.choice(
                ["shipment_disruption", "patient_surge", "cold_chain_loss"]
            )

        # Initialize inventory
        multiplier = self._settings["initial_inventory_multiplier"]
        self._inventory = {
            drug: int(self._rng.randint(10, 30) * multiplier)
            for drug in PRIMARY_DRUGS
        }
        for drug in SUBSTITUTE_DRUGS:
            self._inventory[drug] = int(self._rng.randint(3, 10) * max(0.5, multiplier))

        if self._logistics_mode:
            seeded_inventory = dict(self._inventory)
            self._inventory = {drug: 0 for drug in ALL_DRUGS}
            self._roll_supplier_status()
            for drug, qty in seeded_inventory.items():
                self._receive_inventory(drug, qty, source="initial")
            self._refresh_inventory_totals()

        # Initialize wards
        num_wards = self._settings["num_wards"]
        specialties = list(WARD_SPECIALTY_PROFILES.keys())
        self._wards = [
            HospitalWard(
                f"ward_{i}",
                self._difficulty,
                self._rng,
                specialties[i % len(specialties)],
                clinical_mode=self._clinical_mode,
            )
            for i in range(num_wards)
        ]

        # Generate first incoming shipment
        if self._logistics_mode:
            self._schedule_base_shipments()
            self._incoming_shipments = self._summarize_incoming_shipments()
            self._storage_status = self._build_storage_status(overflow_units=0)
        else:
            self._incoming_shipments = self._generate_shipment()
        self._generate_next_day_demand_plan()

        self._state = DrugShortageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            difficulty=self._difficulty,
            total_days=self._settings["total_days"],
            num_wards=num_wards,
            drugs=ALL_DRUGS,
            initial_budget=self._settings["budget"],
        )

        return DrugShortageObservation(
            done=False,
            reward=None,
            day=self._day,
            budget_remaining=self._budget,
            inventory=dict(self._inventory),
            incoming_shipments=dict(self._incoming_shipments),
            wards=self._build_ward_observations(),
            patient_outcomes={},
            total_score=self._total_score,
            inventory_risk=self._build_inventory_risk(),
            supplier_status=dict(self._supplier_status),
            pending_orders=self._build_pending_orders_view(),
            storage_status=dict(self._storage_status),
            message=(
                f"Day 1 of {self._settings['total_days']}. Allocate drugs to wards."
                + (
                    " Some patients require the primary drug only; substitutes may be contraindicated."
                    if self._clinical_mode
                    else ""
                )
                + (
                    " Tomorrow's demand is uncertain; use ward forecast bands instead of assuming exact arrivals."
                    if self._forecast_mode
                    else ""
                )
                + (
                    " Logistics mode is active: shelf life, cold-chain capacity, and variable supplier lead times now matter."
                    if self._logistics_mode
                    else ""
                )
            ),
        )

    def step(self, action: DrugShortageAction, timeout_s=None, **kwargs) -> DrugShortageObservation:
        self._state.step_count += 1

        # 1. Schedule emergency orders (costly, guaranteed, but delivered next day)
        emergency_spend = 0.0
        scheduled_emergency_arrivals: Dict[str, int] = {}
        for drug in dict.fromkeys(action.emergency_orders):
            cost = DRUG_COSTS.get(drug, 500) * 10  # emergency premium
            if self._budget >= cost:
                self._budget -= cost
                emergency_spend += cost
                scheduled_emergency_arrivals[drug] = (
                    scheduled_emergency_arrivals.get(drug, 0) + 20
                )

        # 2. Apply allocations per ward
        patient_outcomes = {}
        remaining_inventory = dict(self._inventory)
        total_requested_doses = 0
        rejected_doses = 0
        wasted_doses = 0
        total_weight = 0.0
        weighted_outcome_sum = 0.0
        highest_risk_ward = None
        highest_risk_gap = -1.0
        consumed_by_drug: Dict[str, int] = {}

        for ward in self._wards:
            ward_alloc = action.allocations.get(ward.ward_id, {})
            # Deduct from inventory
            final_alloc = {}
            for drug, doses in ward_alloc.items():
                requested_doses = max(0, int(doses))
                total_requested_doses += requested_doses
                available = remaining_inventory.get(drug, 0)
                actual = min(requested_doses, available)
                final_alloc[drug] = actual
                consumed_by_drug[drug] = consumed_by_drug.get(drug, 0) + actual
                remaining_inventory[drug] = available - actual
                rejected_doses += requested_doses - actual

            score = ward.apply_allocation(final_alloc, action.substitutions)
            patient_outcomes[ward.ward_id] = round(score, 3)
            ward_severity = sum(patient["severity"] for patient in ward.patients)
            ward_weight = ward_severity * ward.priority_weight
            total_weight += ward_weight
            weighted_outcome_sum += score * ward_weight
            delivered_doses = sum(patient["doses_received"] for patient in ward.patients)
            wasted_doses += max(0, sum(final_alloc.values()) - delivered_doses)
            ward_risk_gap = ward.priority_weight * (1.0 - score)
            if ward_risk_gap > highest_risk_gap:
                highest_risk_gap = ward_risk_gap
                highest_risk_ward = ward.ward_id

        if self._logistics_mode:
            for drug, qty in consumed_by_drug.items():
                self._consume_inventory(drug, qty)
            self._refresh_inventory_totals()
        else:
            self._inventory = remaining_inventory

        # 3. Compute day reward (severity-weighted hospital score with small operational penalties)
        weighted_outcome = (
            weighted_outcome_sum / total_weight if total_weight > 0 else 0.0
        )
        inefficiency_penalty = min(
            0.03,
            rejected_doses / max(total_requested_doses, 1) * 0.03,
        )
        waste_penalty = min(
            0.02,
            wasted_doses / max(total_requested_doses, 1) * 0.02,
        )
        emergency_penalty = min(
            0.05,
            emergency_spend / max(self._settings["budget"], 1) * 0.05,
        )
        base_day_score = max(
            0.0,
            min(1.0, weighted_outcome - inefficiency_penalty - waste_penalty - emergency_penalty),
        )

        # 4. Receive normal and emergency shipments for the next day
        delivered_shipments: Dict[str, int] = {}
        spoilage_report: Dict[str, int] = {}
        overflow_report: Dict[str, int] = {}
        logistics_penalty = 0.0
        if self._logistics_mode:
            if scheduled_emergency_arrivals:
                self._queue_emergency_orders(scheduled_emergency_arrivals)
            delivered_shipments = self._advance_pending_shipments()
            spoilage_report = self._age_inventory()
            overflow_report = self._apply_cold_chain_overflow()
            spoiled_units = sum(spoilage_report.values()) + sum(overflow_report.values())
            logistics_penalty = min(
                0.06,
                spoiled_units / max(total_requested_doses + sum(self._inventory.values()), 1) * 0.08,
            )
            self._roll_supplier_status()
            self._schedule_base_shipments()
            self._incoming_shipments = self._summarize_incoming_shipments()
            self._storage_status = self._build_storage_status(
                overflow_units=sum(overflow_report.values())
            )
        else:
            for drug, qty in self._incoming_shipments.items():
                self._inventory[drug] = self._inventory.get(drug, 0) + qty
            for drug, qty in scheduled_emergency_arrivals.items():
                self._inventory[drug] = self._inventory.get(drug, 0) + qty

        day_score = max(0.0, min(1.0, base_day_score - logistics_penalty))
        self._total_score += day_score

        # 5. Advance patient health state
        discharged_patients = 0
        for ward in self._wards:
            discharged_patients += ward.advance_day()

        # 6. New patient arrivals
        if self._forecast_mode:
            for ward in self._wards:
                plan = self._latent_demand_plan.get(
                    ward.ward_id,
                    {
                        "arrival_count": 0,
                        "severity_range": (0.3, 0.8),
                        "preferred_drugs": None,
                    },
                )
                ward.new_arrivals(
                    self._settings["patient_arrival_rate"],
                    forced_count=plan["arrival_count"],
                    severity_range=plan["severity_range"],
                    preferred_drugs=plan["preferred_drugs"],
                )
        else:
            for ward in self._wards:
                ward.new_arrivals(self._settings["patient_arrival_rate"])

        # 7. Next shipment (stochastic)
        if not self._logistics_mode:
            self._incoming_shipments = self._generate_shipment()
        self._generate_next_day_demand_plan()

        # 8. Optional hard-mode disruption event
        event_message = self._apply_scheduled_event()
        pressure_summary = self._summarize_operational_pressure()

        self._day += 1
        done = self._day > self._settings["total_days"]

        final_reward = None
        if done:
            final_reward = clamp_validator_safe_score(
                round(self._total_score / self._settings["total_days"], 3)
            )
            message = f"Episode complete. Final score: {final_reward:.3f}"
        else:
            top_ward = pressure_summary["top_wards"][0] if pressure_summary["top_wards"] else None
            top_shortage = (
                pressure_summary["top_shortages"][0]
                if pressure_summary["top_shortages"]
                else None
            )
            message = (
                f"Day {self._day}/{self._settings['total_days']}. "
                f"Day score: {day_score:.2f}. Budget: {self._budget:.0f}. "
                f"Rejected doses: {rejected_doses}. Discharged: {discharged_patients}. "
                f"Highest-risk ward: {highest_risk_ward or 'none'}."
            )
            if top_ward:
                message += (
                    f" Pressure hotspot: {top_ward['ward_id']} ({top_ward['specialty']}) "
                    f"score={top_ward['pressure_score']:.2f}."
                )
                if top_ward.get("direct_only_count", 0) > 0:
                    message += (
                        f" Direct-only patients there: {top_ward['direct_only_count']}."
                    )
            if top_shortage:
                message += (
                    f" Largest projected shortfall: {top_shortage['drug']} "
                    f"missing {top_shortage['shortfall']}."
                )
            if pressure_summary.get("direct_only_total", 0) > 0:
                message += (
                    f" Hospital direct-only patients: {pressure_summary['direct_only_total']}."
                )
            if self._forecast_mode and pressure_summary.get("forecast_hotspots"):
                hotspot = pressure_summary["forecast_hotspots"][0]
                message += (
                    f" Forecast hotspot: {hotspot['ward_id']} "
                    f"risk={hotspot['risk_band']} drugs={hotspot['priority_drugs']}."
                )
            if scheduled_emergency_arrivals:
                if self._logistics_mode:
                    message += (
                        f" Emergency orders queued with variable lead times: "
                        f"{scheduled_emergency_arrivals}."
                    )
                else:
                    message += (
                        f" Emergency arrivals queued for next day: "
                        f"{scheduled_emergency_arrivals}."
                    )
            if self._logistics_mode and delivered_shipments:
                message += f" Delivered overnight: {delivered_shipments}."
            if self._logistics_mode and spoilage_report:
                message += f" Expired overnight: {spoilage_report}."
            if self._logistics_mode and overflow_report:
                message += f" Cold-chain overflow loss: {overflow_report}."
            if self._logistics_mode and self._storage_status.get("overflow_risk") == "high":
                message += (
                    " Cold-chain storage is near capacity; remdesivir and insulin orders must be timed carefully."
                )
            if event_message:
                message += f" Event: {event_message}"

        step_reward = (
            final_reward if done else clamp_validator_safe_score(round(day_score, 3))
        )

        return DrugShortageObservation(
            done=done,
            reward=step_reward,
            day=self._day,
            budget_remaining=round(self._budget, 2),
            inventory=dict(self._inventory),
            incoming_shipments=dict(self._incoming_shipments),
            wards=self._build_ward_observations(),
            patient_outcomes=patient_outcomes,
            total_score=round(self._total_score, 3),
            inventory_risk=self._build_inventory_risk(),
            supplier_status=dict(self._supplier_status),
            pending_orders=self._build_pending_orders_view(),
            storage_status=dict(self._storage_status),
            message=message,
        )

    @property
    def state(self) -> DrugShortageState:
        return self._state

    def _generate_shipment(self) -> Dict[str, int]:
        shipment = {}
        reliability = self._settings["shipment_reliability"]
        for drug in PRIMARY_DRUGS:
            if self._rng.random() < reliability:
                shipment[drug] = self._rng.randint(5, 20)
        for drug in SUBSTITUTE_DRUGS:
            if self._rng.random() < reliability * 0.7:
                shipment[drug] = self._rng.randint(2, 8)
        return shipment

    def _receive_inventory(
        self,
        drug: str,
        qty: int,
        *,
        source: str,
        days_to_expiry: int | None = None,
    ) -> None:
        qty = max(0, int(qty))
        if qty <= 0:
            return

        if not self._logistics_mode:
            self._inventory[drug] = self._inventory.get(drug, 0) + qty
            return

        shelf_life = max(1, int(days_to_expiry or DRUG_SHELF_LIFE.get(drug, 4)))
        self._inventory_lots.setdefault(drug, []).append(
            {
                "qty": qty,
                "days_to_expiry": shelf_life,
                "source": source,
            }
        )
        self._refresh_inventory_totals()

    def _refresh_inventory_totals(self) -> None:
        if not self._logistics_mode:
            return
        self._inventory = {
            drug: sum(lot["qty"] for lot in self._inventory_lots.get(drug, []))
            for drug in ALL_DRUGS
        }

    def _consume_inventory(self, drug: str, qty: int) -> None:
        if not self._logistics_mode:
            self._inventory[drug] = max(0, self._inventory.get(drug, 0) - max(0, int(qty)))
            return

        remaining = max(0, int(qty))
        lots = sorted(
            self._inventory_lots.get(drug, []),
            key=lambda lot: lot["days_to_expiry"],
        )
        updated_lots = []
        for lot in lots:
            if remaining <= 0:
                updated_lots.append(lot)
                continue
            take = min(remaining, lot["qty"])
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] > 0:
                updated_lots.append(lot)
        self._inventory_lots[drug] = updated_lots
        self._refresh_inventory_totals()

    def _build_inventory_risk(self) -> Dict[str, dict]:
        if not self._logistics_mode:
            return {}

        inventory_risk: Dict[str, dict] = {}
        for drug in ALL_DRUGS:
            lots = self._inventory_lots.get(drug, [])
            expiring_1 = sum(lot["qty"] for lot in lots if lot["days_to_expiry"] <= 1)
            expiring_2 = sum(lot["qty"] for lot in lots if lot["days_to_expiry"] <= 2)
            if expiring_1 or expiring_2 or self._inventory.get(drug, 0) > 0:
                inventory_risk[drug] = {
                    "on_hand": self._inventory.get(drug, 0),
                    "expiring_within_1_day": expiring_1,
                    "expiring_within_2_days": expiring_2,
                    "storage_class": "cold_chain" if drug in COLD_CHAIN_DRUGS else "standard",
                }
        return inventory_risk

    def _roll_supplier_status(self) -> None:
        if not self._logistics_mode:
            self._supplier_status = {}
            return

        difficulty_penalty = {"easy": 0.0, "medium": 0.04, "hard": 0.08}[self._difficulty]
        supplier_status: Dict[str, dict] = {}
        for drug in ALL_DRUGS:
            base_profile = SUPPLIER_BASE_PROFILES.get(
                drug,
                {"reliability": 0.75, "lead_time_band": (1, 2)},
            )
            reliability_score = min(
                0.95,
                max(
                    0.35,
                    base_profile["reliability"] - difficulty_penalty + self._rng.uniform(-0.06, 0.06),
                ),
            )
            reliability_band = (
                "high"
                if reliability_score >= 0.78
                else "medium"
                if reliability_score >= 0.6
                else "low"
            )
            lead_min, lead_max = base_profile["lead_time_band"]
            if reliability_band == "low":
                lead_max += 1
            elif reliability_band == "high" and lead_min > 1:
                lead_min -= 1
            supplier_status[drug] = {
                "reliability_band": reliability_band,
                "reliability_score": round(reliability_score, 2),
                "lead_time_band": {
                    "min_days": lead_min,
                    "max_days": lead_max,
                },
                "storage_class": "cold_chain" if drug in COLD_CHAIN_DRUGS else "standard",
            }
        self._supplier_status = supplier_status

    def _schedule_shipment(self, drug: str, qty: int, *, eta: int, source: str) -> None:
        qty = max(0, int(qty))
        eta = max(1, int(eta))
        if qty <= 0:
            return

        status = self._supplier_status.get(drug) or {
            "reliability_band": "medium",
            "reliability_score": 0.75,
            "lead_time_band": {"min_days": 1, "max_days": 2},
            "storage_class": "standard",
        }
        shelf_life = DRUG_SHELF_LIFE.get(drug, 4)
        if drug in COLD_CHAIN_DRUGS and source != "initial" and self._rng.random() < 0.4:
            shelf_life = max(1, shelf_life - 1)

        self._pending_shipments.append(
            {
                "drug": drug,
                "requested_qty": qty,
                "eta": eta,
                "source": source,
                "reliability_band": status["reliability_band"],
                "reliability_score": float(status["reliability_score"]),
                "lead_time_band": dict(status["lead_time_band"]),
                "days_to_expiry": shelf_life,
            }
        )

    def _schedule_base_shipments(self) -> None:
        if not self._logistics_mode:
            return

        for drug in ALL_DRUGS:
            on_hand = self._inventory.get(drug, 0)
            target_inventory = 12 if drug in PRIMARY_DRUGS else 6
            if on_hand >= target_inventory and self._rng.random() < 0.5:
                continue

            status = self._supplier_status[drug]
            reliability_score = float(status["reliability_score"])
            plan_probability = min(0.95, 0.25 + reliability_score * 0.75)
            if self._rng.random() > plan_probability:
                continue

            qty_low, qty_high = ((5, 16) if drug in PRIMARY_DRUGS else (2, 7))
            planned_qty = self._rng.randint(qty_low, qty_high)
            lead_min = status["lead_time_band"]["min_days"]
            lead_max = status["lead_time_band"]["max_days"]
            eta = self._rng.randint(lead_min, lead_max)
            self._schedule_shipment(drug, planned_qty, eta=eta, source="routine")

    def _summarize_incoming_shipments(self) -> Dict[str, int]:
        if not self._logistics_mode:
            return dict(self._incoming_shipments)

        summary: Dict[str, int] = {}
        for shipment in self._pending_shipments:
            if shipment["eta"] == 1:
                drug = shipment["drug"]
                summary[drug] = summary.get(drug, 0) + shipment["requested_qty"]
        return summary

    def _build_pending_orders_view(self) -> List[dict]:
        if not self._logistics_mode:
            return []

        return sorted(
            [
                {
                    "drug": shipment["drug"],
                    "qty": shipment["requested_qty"],
                    "eta": shipment["eta"],
                    "source": shipment["source"],
                    "reliability_band": shipment["reliability_band"],
                    "lead_time_band": shipment["lead_time_band"],
                }
                for shipment in self._pending_shipments
            ],
            key=lambda shipment: (shipment["eta"], shipment["drug"], shipment["source"]),
        )

    def _queue_emergency_orders(self, emergency_orders: Dict[str, int]) -> None:
        if not self._logistics_mode:
            return

        for drug, qty in emergency_orders.items():
            status = self._supplier_status.get(drug)
            if not status:
                continue
            eta = self._rng.randint(
                status["lead_time_band"]["min_days"],
                status["lead_time_band"]["max_days"],
            )
            self._schedule_shipment(drug, qty, eta=eta, source="emergency")

    def _sample_delivery_quantity(self, shipment: dict) -> int:
        requested_qty = int(shipment["requested_qty"])
        reliability_score = float(shipment["reliability_score"])
        lower = max(0.35, reliability_score - 0.25)
        upper = min(1.0, reliability_score + 0.08)
        fill_ratio = self._rng.uniform(lower, upper)
        delivered_qty = max(0, int(round(requested_qty * fill_ratio)))
        if shipment["reliability_band"] == "low" and self._rng.random() < 0.2:
            delivered_qty = max(0, delivered_qty - self._rng.randint(1, max(1, requested_qty // 3)))
        return delivered_qty

    def _advance_pending_shipments(self) -> Dict[str, int]:
        if not self._logistics_mode:
            return {}

        delivered: Dict[str, int] = {}
        remaining_shipments: List[dict] = []
        for shipment in self._pending_shipments:
            shipment["eta"] -= 1
            if shipment["eta"] <= 0:
                delivered_qty = self._sample_delivery_quantity(shipment)
                if delivered_qty > 0:
                    self._receive_inventory(
                        shipment["drug"],
                        delivered_qty,
                        source=shipment["source"],
                        days_to_expiry=shipment["days_to_expiry"],
                    )
                    delivered[shipment["drug"]] = delivered.get(shipment["drug"], 0) + delivered_qty
                continue
            remaining_shipments.append(shipment)

        self._pending_shipments = remaining_shipments
        self._refresh_inventory_totals()
        return delivered

    def _age_inventory(self) -> Dict[str, int]:
        if not self._logistics_mode:
            return {}

        spoiled: Dict[str, int] = {}
        for drug in ALL_DRUGS:
            updated_lots = []
            for lot in self._inventory_lots.get(drug, []):
                lot["days_to_expiry"] -= 1
                if lot["days_to_expiry"] <= 0:
                    spoiled[drug] = spoiled.get(drug, 0) + lot["qty"]
                    continue
                updated_lots.append(lot)
            self._inventory_lots[drug] = updated_lots
        self._refresh_inventory_totals()
        return spoiled

    def _apply_cold_chain_overflow(self) -> Dict[str, int]:
        if not self._logistics_mode:
            return {}

        capacity = COLD_CHAIN_CAPACITY_BY_DIFFICULTY[self._difficulty]
        current_load = sum(self._inventory.get(drug, 0) for drug in COLD_CHAIN_DRUGS)
        overflow = max(0, current_load - capacity)
        if overflow <= 0:
            return {}

        spoiled: Dict[str, int] = {}
        cold_chain_lots = []
        for drug in COLD_CHAIN_DRUGS:
            for lot in self._inventory_lots.get(drug, []):
                cold_chain_lots.append((drug, lot))
        cold_chain_lots.sort(key=lambda item: item[1]["days_to_expiry"])

        remaining_overflow = overflow
        for drug, lot in cold_chain_lots:
            if remaining_overflow <= 0:
                break
            lost = min(remaining_overflow, lot["qty"])
            lot["qty"] -= lost
            spoiled[drug] = spoiled.get(drug, 0) + lost
            remaining_overflow -= lost

        for drug in COLD_CHAIN_DRUGS:
            self._inventory_lots[drug] = [
                lot for lot in self._inventory_lots.get(drug, []) if lot["qty"] > 0
            ]
        self._refresh_inventory_totals()
        return spoiled

    def _build_storage_status(self, *, overflow_units: int) -> Dict[str, int | float | str]:
        if not self._logistics_mode:
            return {}

        capacity = COLD_CHAIN_CAPACITY_BY_DIFFICULTY[self._difficulty]
        load = sum(self._inventory.get(drug, 0) for drug in COLD_CHAIN_DRUGS)
        utilization = (load / capacity) if capacity > 0 else 0.0
        overflow_risk = (
            "high"
            if utilization >= 0.9
            else "medium"
            if utilization >= 0.7
            else "low"
        )
        return {
            "cold_chain_load": load,
            "cold_chain_capacity": capacity,
            "utilization": round(utilization, 2),
            "overflow_units": overflow_units,
            "overflow_risk": overflow_risk,
        }

    def _summarize_operational_pressure(self) -> dict:
        ward_pressure = []
        hospital_needs: Dict[str, int] = {}

        for ward in self._wards:
            needs = ward.compute_drug_needs()
            ward_need_total = sum(needs.values())
            severity_total = sum(patient["severity"] for patient in ward.patients)
            pressure_score = round(ward.priority_weight * severity_total, 3)
            direct_only_count = sum(
                1 for patient in ward.patients if patient.get("requires_primary_drug")
            )
            ward_pressure.append(
                {
                    "ward_id": ward.ward_id,
                    "specialty": ward.specialty,
                    "pressure_score": pressure_score,
                    "need_total": ward_need_total,
                    "direct_only_count": direct_only_count,
                }
            )
            for drug, qty in needs.items():
                hospital_needs[drug] = hospital_needs.get(drug, 0) + qty

        ward_pressure.sort(
            key=lambda item: (item["pressure_score"], item["need_total"]),
            reverse=True,
        )

        shortage_alerts = []
        for drug, needed in sorted(hospital_needs.items(), key=lambda item: item[1], reverse=True):
            available = self._inventory.get(drug, 0)
            if needed > available:
                shortage_alerts.append(
                    {
                        "drug": drug,
                        "shortfall": needed - available,
                        "available": available,
                    }
                )

        return {
            "top_wards": ward_pressure[:2],
            "top_shortages": shortage_alerts[:3],
            "direct_only_total": sum(
                1 for ward in self._wards for patient in ward.patients if patient.get("requires_primary_drug")
            ),
            "forecast_hotspots": sorted(
                self._forecast_summary.values(),
                key=lambda item: (item.get("risk_rank", 0), item.get("upper_bound", 0)),
                reverse=True,
            )[:2]
            if self._forecast_mode
            else [],
        }

    def _top_specialty_drugs(self, specialty: str, limit: int = 2) -> List[str]:
        weights = WARD_SPECIALTY_PROFILES.get(specialty, [1] * len(PRIMARY_DRUGS))
        ranked = sorted(
            zip(PRIMARY_DRUGS, weights),
            key=lambda item: item[1],
            reverse=True,
        )
        return [drug for drug, _ in ranked[:limit]]

    def _generate_next_day_demand_plan(self) -> None:
        self._latent_demand_plan = {}
        self._forecast_summary = {}
        if not self._forecast_mode:
            return

        for ward in self._wards:
            severity_total = sum(patient["severity"] for patient in ward.patients)
            pressure_factor = min(0.25, severity_total / max(len(ward.patients), 1) * 0.15)
            risk_roll = self._rng.random() + pressure_factor + ((ward.priority_weight - 1.0) * 0.5)

            if risk_roll > 1.0:
                risk_band = "high"
                lower_bound, upper_bound = 2, 4
                severity_range = (0.55, 1.0)
                risk_rank = 3
            elif risk_roll > 0.62:
                risk_band = "medium"
                lower_bound, upper_bound = 1, 3
                severity_range = (0.35, 0.9)
                risk_rank = 2
            else:
                risk_band = "low"
                lower_bound, upper_bound = 0, 1
                severity_range = (0.25, 0.75)
                risk_rank = 1

            arrival_count = self._rng.randint(lower_bound, upper_bound)
            priority_drugs = self._top_specialty_drugs(ward.specialty)

            self._latent_demand_plan[ward.ward_id] = {
                "arrival_count": arrival_count,
                "severity_range": severity_range,
                "preferred_drugs": priority_drugs,
            }
            self._forecast_summary[ward.ward_id] = {
                "ward_id": ward.ward_id,
                "risk_band": risk_band,
                "risk_rank": risk_rank,
                "expected_new_patients_band": {
                    "min": lower_bound,
                    "max": upper_bound,
                },
                "upper_bound": upper_bound,
                "priority_drugs": priority_drugs,
            }

    def _build_ward_observations(self) -> List[dict]:
        payloads = []
        for ward in self._wards:
            ward_payload = ward.to_dict()
            if self._forecast_mode:
                ward_payload["demand_forecast"] = self._forecast_summary.get(
                    ward.ward_id,
                    {
                        "risk_band": "low",
                        "expected_new_patients_band": {"min": 0, "max": 1},
                        "priority_drugs": self._top_specialty_drugs(ward.specialty),
                    },
                )
            payloads.append(ward_payload)
        return payloads

    def _apply_scheduled_event(self) -> str | None:
        if (
            self._difficulty != "hard"
            or self._event_applied
            or self._scheduled_event_day is None
            or self._event_type is None
            or self._day != self._scheduled_event_day
        ):
            return None

        self._event_applied = True

        if self._event_type == "shipment_disruption":
            if self._incoming_shipments:
                affected_drug, original_qty = max(
                    self._incoming_shipments.items(),
                    key=lambda item: item[1],
                )
                reduced_qty = max(0, int(original_qty * 0.25))
                self._incoming_shipments[affected_drug] = reduced_qty
                return (
                    f"Supplier disruption cut the next {affected_drug} shipment "
                    f"from {original_qty} to {reduced_qty}."
                )

            affected_drug = self._rng.choice(PRIMARY_DRUGS)
            return f"Supplier disruption affected {affected_drug}, but no shipment was scheduled."

        if self._event_type == "cold_chain_loss":
            stocked_primary_drugs = [
                (drug, qty)
                for drug, qty in self._inventory.items()
                if drug in PRIMARY_DRUGS and qty > 0
            ]
            if stocked_primary_drugs:
                affected_drug, current_qty = max(
                    stocked_primary_drugs,
                    key=lambda item: item[1],
                )
                lost_units = max(1, int(current_qty * 0.35))
                self._inventory[affected_drug] = max(0, current_qty - lost_units)
                return (
                    f"Cold-chain failure spoiled {lost_units} units of {affected_drug} "
                    f"from on-hand inventory."
                )
            return "Cold-chain failure occurred, but no primary-drug inventory was available to spoil."

        surge_ward = max(
            self._wards,
            key=lambda ward: ward.priority_weight * max(ward.compute_drug_needs().values(), default=0),
        )
        surge_count = self._rng.randint(2, 4)
        surge_ward.add_surge_patients(surge_count)
        return (
            f"Mass-casualty surge added {surge_count} critical patients to "
            f"{surge_ward.ward_id} ({surge_ward.specialty})."
        )
