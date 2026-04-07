import asyncio
import json
import os
import re
import sys
import time
from typing import List, Optional

from openai import OpenAI

try:
    from hospital_drug_env.client import HospitalDrugEnv
    from hospital_drug_env.models import DrugShortageAction
except ModuleNotFoundError:
    from client import HospitalDrugEnv
    from models import DrugShortageAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
SPACE_URL = os.getenv("SPACE_URL")
DEFAULT_SPACE_URL = "https://biswajit328-hospital-drug-env.hf.space"
DIFFICULTY = os.getenv("DIFFICULTY", "medium")
TASK_NAME = os.getenv("TASK_NAME", DIFFICULTY)
BENCHMARK = os.getenv("BENCHMARK", "hospital_drug_env")
MAX_STEPS = max(1, int(os.getenv("MAX_STEPS", "10")))
TEMPERATURE = 0.2
MAX_TOKENS = 450
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.10"))
ENV_CALL_RETRIES = int(os.getenv("ENV_CALL_RETRIES", "1"))

SUBSTITUTE_OPTIONS = {
    "amoxicillin": "azithromycin",
    "paracetamol": "ibuprofen",
    "morphine": "tramadol",
}

EMERGENCY_ORDER_COSTS = {
    "amoxicillin": 500,
    "remdesivir": 8000,
    "paracetamol": 100,
    "insulin": 2000,
    "morphine": 1500,
}

SYSTEM_PROMPT = """
You are a hospital resource manager during a drug shortage crisis.
Each day you must allocate scarce medicines across hospital wards.

You will receive:
- Current drug inventory
- Each ward's patients and their drug needs
- Budget remaining
- Expected incoming shipments

Respond ONLY with a valid JSON object in this exact format:
{
  "allocations": {
    "ward_0": {"amoxicillin": 5, "paracetamol": 10},
    "ward_1": {"remdesivir": 2}
  },
  "emergency_orders": [],
  "substitutions": {}
}

Strategy:
- Prioritize critical patients (high severity score)
- Don't over-allocate - check inventory
- Emergency orders cost 10x normal price and arrive the next day - use sparingly and plan ahead
- Valid substitutions:
  - amoxicillin -> azithromycin
  - paracetamol -> ibuprofen
  - morphine -> tramadol
- Substitutions reduce side effects but have a penalty
"""

PROXY_HEALTHCHECK_PROMPT = "Reply with OK."


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_val = action.replace("\n", " ").strip()
    error_val = error.replace("\n", " ").strip() if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def format_action(action: DrugShortageAction) -> str:
    payload = {
        "allocations": action.allocations,
        "emergency_orders": action.emergency_orders,
        "substitutions": action.substitutions,
    }
    return json.dumps(payload, separators=(",", ":"))


def format_observation(obs, step) -> str:
    lines = [
        f"Day: {obs.day}",
        f"Budget remaining: {obs.budget_remaining}",
        f"Inventory: {json.dumps(obs.inventory)}",
        f"Incoming tomorrow: {json.dumps(obs.incoming_shipments)}",
        "Emergency orders placed today arrive tomorrow.",
        f"Valid substitutions: {json.dumps(SUBSTITUTE_OPTIONS)}",
        "Wards:",
    ]
    for ward in obs.wards:
        top_patients = sorted(
            ward.get("patients", []),
            key=lambda patient: patient.get("severity", 0.0),
            reverse=True,
        )[:3]
        lines.append(
            f"  {ward['ward_id']}: {ward['patient_count']} patients, "
            f"needs={json.dumps(ward['drug_needs'])}"
        )
        if top_patients:
            lines.append(
                f"    critical={json.dumps(top_patients)}"
            )
    lines.append(f"Running score: {obs.total_score}")
    lines.append(f"Message: {obs.message}")
    lines.append(f"\nStep: {step}")
    return "\n".join(lines)


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


def allocate_greedily(patient_needs: List[dict], inventory: dict) -> tuple[dict, dict, dict]:
    allocations: dict = {}
    substitutions: dict = {}
    unmet_by_drug: dict = {}
    remaining_inventory = {drug: max(0, int(qty)) for drug, qty in inventory.items()}

    for need in patient_needs:
        ward_alloc = allocations.setdefault(need["ward_id"], {})
        drug = need["drug"]
        remaining_need = need["remaining_need"]

        direct_given = min(remaining_need, remaining_inventory.get(drug, 0))
        if direct_given > 0:
            ward_alloc[drug] = ward_alloc.get(drug, 0) + direct_given
            remaining_inventory[drug] -= direct_given
            remaining_need -= direct_given

        substitute = SUBSTITUTE_OPTIONS.get(drug)
        if remaining_need > 0 and substitute:
            substitute_given = min(remaining_need, remaining_inventory.get(substitute, 0))
            if substitute_given > 0:
                ward_alloc[substitute] = ward_alloc.get(substitute, 0) + substitute_given
                remaining_inventory[substitute] -= substitute_given
                remaining_need -= substitute_given
                substitutions[drug] = substitute

        if remaining_need > 0:
            shortage = unmet_by_drug.setdefault(
                drug,
                {"max_severity": 0.0, "total_unmet": 0},
            )
            shortage["max_severity"] = max(shortage["max_severity"], need["severity"])
            shortage["total_unmet"] += remaining_need

    allocations = {ward_id: ward_alloc for ward_id, ward_alloc in allocations.items() if ward_alloc}
    return allocations, substitutions, unmet_by_drug


def plan_emergency_orders(
    unmet_by_drug: dict,
    budget_remaining: float,
    max_orders: int = 2,
) -> List[str]:
    emergency_orders: List[str] = []
    remaining_budget = budget_remaining

    for drug, shortage in sorted(
        unmet_by_drug.items(),
        key=lambda item: (item[1]["max_severity"], item[1]["total_unmet"]),
        reverse=True,
    ):
        cost = EMERGENCY_ORDER_COSTS.get(drug, 5000)
        if shortage["max_severity"] < 0.9 or remaining_budget < cost:
            continue
        emergency_orders.append(drug)
        remaining_budget -= cost
        if len(emergency_orders) >= max_orders:
            break

    return emergency_orders


def fallback_action(observation) -> DrugShortageAction:
    patient_needs = collect_patient_needs(observation)
    working_inventory = dict(observation.inventory)

    allocations, substitutions, unmet_by_drug = allocate_greedily(
        patient_needs,
        working_inventory,
    )

    emergency_orders = plan_emergency_orders(
        unmet_by_drug,
        observation.budget_remaining,
    )
    if emergency_orders:
        for drug in emergency_orders:
            working_inventory[drug] = working_inventory.get(drug, 0) + 20
        allocations, substitutions, _ = allocate_greedily(patient_needs, working_inventory)

    return DrugShortageAction(
        allocations=allocations,
        emergency_orders=emergency_orders,
        substitutions=substitutions,
    )


def parse_action(response_text: str, observation) -> DrugShortageAction:
    try:
        clean = re.sub(r"```json|```", "", response_text).strip()
        data = json.loads(clean)
        action = DrugShortageAction(
            allocations=data.get("allocations", {}),
            emergency_orders=data.get("emergency_orders", []),
            substitutions=data.get("substitutions", {}),
        )
        if action.allocations or action.emergency_orders or action.substitutions:
            return action
    except Exception:
        pass

    return fallback_action(observation)


def request_model_response(
    client: OpenAI,
    user_content: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def build_sync_env():
    if LOCAL_IMAGE_NAME:
        async_client = asyncio.run(HospitalDrugEnv.from_docker_image(LOCAL_IMAGE_NAME))
        return async_client.sync()

    if SPACE_URL:
        return HospitalDrugEnv(base_url=SPACE_URL).sync()

    return HospitalDrugEnv(base_url=DEFAULT_SPACE_URL).sync()


def call_with_retry(func, *, label: str):
    last_error = None
    for attempt in range(ENV_CALL_RETRIES + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= ENV_CALL_RETRIES:
                break
            print(
                f"[inference] Retrying {label} after {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.5 * (attempt + 1))
    raise last_error

def main():
    rewards: List[float] = []
    steps_taken = 0
    success = False
    stage = "startup"

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    try:
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN or API_KEY environment variable is required")
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        request_model_response(
            client,
            PROXY_HEALTHCHECK_PROMPT,
            system_prompt="You are validating connectivity. Reply with OK only.",
            temperature=0.0,
            max_tokens=8,
        )
        env = build_sync_env()
        with env as e:
            stage = "reset"
            result = call_with_retry(
                lambda: e.reset(difficulty=DIFFICULTY),
                label="reset",
            )
            observation = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                stage = f"model step {step}"
                user_content = format_observation(observation, step)
                response_text = request_model_response(client, user_content)

                action = parse_action(response_text, observation)
                stage = f"env step {step}"
                result = call_with_retry(
                    lambda: e.step(action),
                    label=f"step {step}",
                )
                observation = result.observation
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken = step
                log_step(
                    step=step,
                    action=format_action(action),
                    reward=reward,
                    done=result.done,
                    error=None,
                )

                if result.done:
                    break

            if rewards:
                final_score = float(result.reward or 0.0)
                success = final_score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(
            f"[inference] Failed during {stage}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        success = False
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    main()
