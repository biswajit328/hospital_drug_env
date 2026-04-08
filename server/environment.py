# server/environment.py
import random
import uuid
from typing import Dict, List

from openenv.core.env_server import Environment

try:
    from hospital_drug_env.score_utils import clamp_validator_safe_score
    from hospital_drug_env.models import (
        DrugShortageAction,
        DrugShortageObservation,
        DrugShortageState,
    )
except ModuleNotFoundError:
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
    def __init__(self, ward_id: str, difficulty: str, rng: random.Random, specialty: str):
        self.ward_id = ward_id
        self.difficulty = difficulty
        self._rng = rng
        self.specialty = specialty
        self.patients = self._generate_initial_patients()

    def _sample_drug(self) -> str:
        weights = WARD_SPECIALTY_PROFILES.get(self.specialty, [1] * len(PRIMARY_DRUGS))
        return self._rng.choices(PRIMARY_DRUGS, weights=weights, k=1)[0]

    def _create_patient(
        self,
        patient_id: str,
        *,
        severity_range: tuple[float, float] = (0.2, 1.0),
    ) -> dict:
        severity = self._rng.uniform(*severity_range)
        needed_drug = self._sample_drug()
        doses_needed = max(1, int(severity * 3))
        return {
            "id": patient_id,
            "severity": round(severity, 2),
            "drug": needed_drug,
            "doses_needed": doses_needed,
            "doses_received": 0,
            "stable_days": 0,
        }

    def _generate_initial_patients(self) -> List[dict]:
        n = self._rng.randint(3, 8)
        return [self._create_patient(f"P{i}") for i in range(n)]

    def new_arrivals(self, arrival_rate: float):
        if self._rng.random() < arrival_rate:
            n_new = self._rng.randint(1, 3)
            for i in range(n_new):
                self.patients.append(
                    self._create_patient(
                        f"NEW_{uuid.uuid4().hex[:4]}",
                        severity_range=(0.3, 1.0),
                    )
                )

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
            # Check if substitute was used
            allowed_substitute, side_effect = SUBSTITUTE_MAP.get(drug, (None, 0))
            requested_substitute = substitutions.get(drug)
            if (
                actual_given < needed
                and allowed_substitute
                and requested_substitute == allowed_substitute
            ):
                sub_given = remaining_doses.get(allowed_substitute, 0)
                actual_sub = min(sub_given, needed - actual_given)
                if actual_sub:
                    remaining_doses[allowed_substitute] = sub_given - actual_sub
                actual_given += actual_sub
                penalty = side_effect * (actual_sub / needed) if needed > 0 else 0

            p["doses_received"] = actual_given
            ratio = (actual_given / needed) if needed > 0 else 1.0
            ratio = min(ratio, 1.0)
            score_contribution = (ratio - penalty) * p["severity"]
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
            "drug_received": drug_received,
            "patients": [
                {
                    "id": p["id"],
                    "severity": p["severity"],
                    "drug": p["drug"],
                    "doses_needed": p["doses_needed"],
                    "doses_received": p["doses_received"],
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

    def reset(
        self,
        seed=None,
        episode_id=None,
        difficulty="medium",
        task_id=None,
        **kwargs,
    ) -> DrugShortageObservation:
        self._rng = random.Random(seed)

        if task_id in DIFFICULTY_SETTINGS:
            difficulty = task_id

        self._difficulty = difficulty if difficulty in DIFFICULTY_SETTINGS else "medium"
        self._settings = DIFFICULTY_SETTINGS[self._difficulty]
        self._day = 1
        self._total_score = 0.0
        self._budget = self._settings["budget"]
        self._event_applied = False
        self._scheduled_event_day = None
        self._event_type = None
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

        # Initialize wards
        num_wards = self._settings["num_wards"]
        specialties = list(WARD_SPECIALTY_PROFILES.keys())
        self._wards = [
            HospitalWard(
                f"ward_{i}",
                self._difficulty,
                self._rng,
                specialties[i % len(specialties)],
            )
            for i in range(num_wards)
        ]

        # Generate first incoming shipment
        self._incoming_shipments = self._generate_shipment()

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
            wards=[w.to_dict() for w in self._wards],
            patient_outcomes={},
            total_score=self._total_score,
            message=f"Day 1 of {self._settings['total_days']}. Allocate drugs to wards.",
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
        day_score = max(
            0.0,
            min(1.0, weighted_outcome - inefficiency_penalty - waste_penalty - emergency_penalty),
        )
        self._total_score += day_score

        # 4. Receive normal and emergency shipments for the next day
        for drug, qty in self._incoming_shipments.items():
            self._inventory[drug] = self._inventory.get(drug, 0) + qty
        for drug, qty in scheduled_emergency_arrivals.items():
            self._inventory[drug] = self._inventory.get(drug, 0) + qty

        # 5. Advance patient health state
        discharged_patients = 0
        for ward in self._wards:
            discharged_patients += ward.advance_day()

        # 6. New patient arrivals
        for ward in self._wards:
            ward.new_arrivals(self._settings["patient_arrival_rate"])

        # 7. Next shipment (stochastic)
        self._incoming_shipments = self._generate_shipment()

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
            if top_shortage:
                message += (
                    f" Largest projected shortfall: {top_shortage['drug']} "
                    f"missing {top_shortage['shortfall']}."
                )
            if scheduled_emergency_arrivals:
                message += (
                    f" Emergency arrivals queued for next day: "
                    f"{scheduled_emergency_arrivals}."
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
            wards=[w.to_dict() for w in self._wards],
            patient_outcomes=patient_outcomes,
            total_score=round(self._total_score, 3),
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

    def _summarize_operational_pressure(self) -> dict:
        ward_pressure = []
        hospital_needs: Dict[str, int] = {}

        for ward in self._wards:
            needs = ward.compute_drug_needs()
            ward_need_total = sum(needs.values())
            severity_total = sum(patient["severity"] for patient in ward.patients)
            pressure_score = round(ward.priority_weight * severity_total, 3)
            ward_pressure.append(
                {
                    "ward_id": ward.ward_id,
                    "specialty": ward.specialty,
                    "pressure_score": pressure_score,
                    "need_total": ward_need_total,
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
        }

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
