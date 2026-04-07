---
title: Hospital Drug Env Environment Server
emoji: "💊"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Hospital Drug Shortage Environment

A reinforcement learning environment simulating a real-world hospital drug shortage crisis.
Built for the Meta x Hugging Face OpenEnv Hackathon.

## Problem Statement

During COVID-19, hospitals across India ran out of critical drugs like Remdesivir and oxygen.
WHO reports that 50% of the world's hospitals face critical drug shortages every year.
No AI system exists today that can train agents to handle this crisis effectively.

This environment simulates exactly that - an AI agent must manage scarce medicines
across multiple hospital wards, keeping the most critical patients alive first.

## Environment Overview

- **5 primary drugs**: amoxicillin, remdesivir, paracetamol, insulin, morphine
- **3 substitute drugs**: azithromycin, ibuprofen, tramadol
- **Up to 5 wards** with patients of varying severity (0.0 = mild, 1.0 = critical)
- **Ward specialties** such as infectious disease, endocrine, and pain management
- **Operational priority weighting** - infectious disease and respiratory wards carry slightly higher service-critical weight
- **Stochastic supply** - shipments may or may not arrive each day
- **Budget constraints** - emergency orders cost 10x normal price
- **Procurement delay** - emergency orders are guaranteed but only arrive for the next day
- **Substitute drugs** - available but with side effect penalties
- **New patients arrive** each day - resource pressure increases over time
- **Patient progression** - untreated patients deteriorate, stable patients can recover and be discharged
- **Rare hard-mode disruptions** - shipment failures or sudden patient surges increase decision difficulty
- **Operational summaries** - each day highlights the pressure hotspot and largest projected shortfall for more realistic coordination signals

## Why This Is A Real OpenEnv Task

This environment models a genuine hospital operations problem: pharmacy and operations
staff must decide which wards receive scarce medication first, when to spend limited
budget on emergency procurement, and when clinically valid substitutions are acceptable.

That makes it useful for:

- training triage and resource-allocation agents
- evaluating long-horizon planning under uncertainty
- stress-testing agent behavior in safety-critical logistics settings

## Benchmark Flow

```text
Reset episode
  -> inspect hospital state
  -> allocate drugs / order emergency stock / choose substitutions
  -> receive reward and updated ward outcomes
  -> patients deteriorate or recover
  -> daily operational summary highlights pressure hotspot and shortage risk
  -> new arrivals and shipment uncertainty change tomorrow's state
Repeat until end of horizon
```

This makes the benchmark sequential rather than one-shot: every daily decision changes
 the state that the agent must handle next.

## Action Space
```json
{
  "allocations": {
    "ward_0": {"amoxicillin": 5, "paracetamol": 10},
    "ward_1": {"remdesivir": 2}
  },
  "emergency_orders": [],
  "substitutions": {}
}
```

## Observation Space
```json
{
  "day": 3,
  "budget_remaining": 8500.0,
  "inventory": {"amoxicillin": 12, "remdesivir": 3},
  "incoming_shipments": {"paracetamol": 10},
  "wards": [
    {
      "ward_id": "ward_0",
      "patient_count": 5,
      "severity_scores": [0.8, 0.6, 0.9, 0.3, 0.7],
      "drug_needs": {"amoxicillin": 8, "paracetamol": 5}
    }
  ],
  "patient_outcomes": {"ward_0": 0.85},
  "total_score": 2.45,
  "message": "Day 3/7. Day score: 0.82. Budget: 8500"
}
```

## Reward Function

- **Daily reward** = severity-weighted hospital-wide outcome across all wards
- **Patient weight** = severity score (critical patients count more)
- **Ward weight** = service-critical specialty multiplier x total ward severity
- **Ward allocation is distributed to the most severe patients first**
- **Partial credit** = doses_received / doses_needed per patient
- **Substitute penalty** = side effect deduction applied per substitute dose
- **Inefficiency penalty** = small penalty for rejected or over-requested allocations
- **Emergency spend penalty** = small penalty for expensive emergency procurement
- **Emergency timing** = emergency procurement helps future state, not the current day
- **Final reward** = average daily score over entire episode (0.0 to 1.0)

## Task Suite

The environment exposes three deterministic grader-backed tasks. They are not just
"the same task with three labels"; each task emphasizes a different hospital
operations objective and uses a different policy/grader setup.

| Task | Difficulty | Objective | Key Challenge |
|------|------------|-----------|---------------|
| Critical Care Stabilization | Easy | Keep service levels high with reliable supply and prioritize the sickest wards first | Strong basic triage and clean state handling |
| Budget-Constrained Ward Balancing | Medium | Preserve outcomes while limiting overspend on emergency procurement | Trade off scarcity against budget discipline |
| Substitution-Aware Surge Response | Hard | Maintain coverage during prolonged shortages with limited budget and substitute drugs | Long-horizon scarcity, substitutions, and surge arrivals |

### Grader Policy Styles

- **Critical Care Stabilization**: severity-aware ward balancing without emergency procurement
- **Budget-Constrained Ward Balancing**: budget-aware balancing with selective emergency ordering
- **Substitution-Aware Surge Response**: severity-first triage with substitutions and disruption recovery

## Why Weak Agents Fail

- They greedily over-request inventory and let the environment clamp allocations, which now incurs small efficiency penalties.
- They ignore patient severity and spread stock evenly, causing critical-patient outcomes to collapse.
- They ignore ward specialty pressure, which now means under-serving high-priority wards hurts more than under-serving low-pressure wards.
- They overuse emergency orders and protect short-term reward while damaging budget efficiency.
- They treat emergency procurement as an instant rescue tool instead of a next-day planning tool.
- They fail to plan for next-day consequences such as patient deterioration, discharge dynamics, and disruption events.
- They ignore valid substitutions in hard mode and underperform during prolonged scarcity.

## Why Hard Mode Is Actually Hard

- Supply is structurally scarce, so direct fulfillment is often impossible.
- High-priority specialties such as infectious disease and respiratory care are harder to neglect without taking a score hit.
- Patient deterioration creates delayed consequences for poor early decisions.
- A seeded disruption event forces adaptation instead of repeating the same allocation pattern.
- Substitutions help, but only when used selectively because they carry clinical penalties.
- Emergency procurement can rescue critical cases, but overspending is explicitly punished.
- Emergency procurement is delayed, so strong agents must anticipate tomorrow's shortages instead of reacting too late.

## Difficulty Settings

| Difficulty | Wards | Days | Inventory | Shipment Reliability | Budget |
|------------|-------|------|-----------|----------------------|--------|
| Easy | 3 | 5 | 2x normal | 90% | 15,000 |
| Medium | 5 | 7 | 1x normal | 60% | 10,000 |
| Hard | 5 | 10 | 0.5x normal | 30% | 6,000 |

## Substitution Rules

| Primary Drug | Substitute | Penalty |
|--------------|------------|---------|
| amoxicillin | azithromycin | 0.20 |
| paracetamol | ibuprofen | 0.05 |
| morphine | tramadol | 0.15 |

## Baseline Evaluation

Run the built-in graders locally:

```bash
python grader.py --difficulty all --seed 42
```

All task scores are deterministic for a fixed seed and normalized to the `0.0-1.0` range.
Each task score is averaged over two fixed seeds (`seed` and `seed + 1`) to reduce overfitting.

### Reproducible Baseline Scores

Reference run:

```bash
python grader.py --difficulty all --seed 42
```

Expected output:

```json
{
  "easy": 1.0,
  "hard": 0.576,
  "medium": 0.972
}
```

Interpretation:

- **Easy / Critical Care Stabilization**: near-perfect score under favorable supply
- **Medium / Budget-Constrained Ward Balancing**: strong performance with selective emergency ordering and budget discipline
- **Hard / Substitution-Aware Surge Response**: clearly harder because it mixes scarcity, substitutions, patient deterioration, and rare disruption events

## What This Benchmark Evaluates Well

- Severity-aware prioritization rather than naive fairness
- Operational prioritization across different hospital service lines
- Multi-day planning under stochastic supply
- Budget-aware intervention timing
- Delayed procurement planning under uncertainty
- Safe substitute usage under scarcity
- Robustness to sudden operational shocks

## Why This Is More Than A Generic Allocation Simulator

This benchmark is not just "divide scarce resources across buckets."
It combines several hospital-specific pressures that interact over time:

- **Clinical triage pressure**: high-severity patients should dominate decisions
- **Service-line priority**: some ward specialties are more operationally sensitive than others
- **Supply uncertainty**: tomorrow's inventory is not guaranteed
- **Budget pressure**: emergency rescue options exist, but overspending is punished
- **Clinical substitutes**: alternative drugs can help, but only with explicit penalties
- **Stateful patient progression**: today's under-treatment worsens tomorrow's hospital state

That combination is what makes the environment useful for evaluating real planning quality rather than one-step greedy heuristics.

## Weak vs Strong Agent Behavior

### Weak agents typically

- spread inventory too evenly instead of focusing on the highest-risk patients
- ignore ward specialty pressure and under-serve high-priority service lines
- over-request inventory and rely on clamping rather than planning with stock constraints
- delay use of substitutions until shortages become catastrophic
- overspend on emergency procurement or never use it when it is actually needed

### Strong agents typically

- route scarce stock toward the sickest patients first
- recognize when infectious-disease or respiratory pressure should dominate the plan
- anticipate future deterioration instead of optimizing only the current day
- use substitutions selectively when they improve net hospital outcomes
- treat emergency orders as a last-resort rescue tool, not a default action

## API Usage
```python
from hospital_drug_env.client import HospitalDrugEnv
from hospital_drug_env.models import DrugShortageAction

env = HospitalDrugEnv(base_url="https://biswajit328-hospital-drug-env.hf.space")

with env.sync() as e:
    result = e.reset(difficulty="medium")
    observation = result.observation

    action = DrugShortageAction(
        allocations={"ward_0": {"amoxicillin": 5}},
        emergency_orders=[],
        substitutions={}
    )

    result = e.step(action)
    print(result.observation.message)
```

## Installation
```bash
pip install git+https://huggingface.co/spaces/biswajit328/hospital-drug-env
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /metadata | GET | Environment metadata |
| /reset | POST | Start new episode |
| /step | POST | Take action |
| /state | GET | Get current state |
| /schema | GET | Action/observation/state schema |
| /ws | WebSocket | Real-time interaction |
| /web | GET | OpenEnv web interface |
| /docs | GET | API documentation |

## Real World Impact

- Trains AI agents to handle life-or-death resource allocation
- Directly applicable to hospital administrators during crises
- Models real supply chain uncertainty and budget constraints
- Severity-weighted scoring ensures critical patients are prioritized
- Service-critical wards are weighted more heavily, reflecting operational triage pressure in real hospitals
- Captures the tradeoff between immediate rescue decisions and long-term hospital resilience

## What Judges Should Notice

- The environment is **stateful**, not one-shot: poor allocations create worse future hospital states.
- The reward is not binary; it gives useful partial credit while still punishing waste, overspend, and bad prioritization.
- Hard mode is not just "less inventory" - it mixes scarcity, patient progression, substitutions, and seeded disruptions.
- The benchmark differentiates weak and strong agents in clinically meaningful ways, especially around triage, substitutes, and budget timing.
- The system is deployable and reproducible end-to-end: Docker, Hugging Face Space, local validation, live validation, grader, and baseline inference all work.

## Suggested 60-Second Demo Flow

1. Open `/web` and call `reset()`.
2. Point out the ward specialties, patient severity differences, and uncertain incoming shipments.
3. Show that the action space combines **allocation**, **emergency procurement**, and **substitution policy** in a single daily decision.
4. Take one `step()` and highlight:
   - day reward
   - highest-risk ward
   - pressure hotspot
   - projected shortfall
5. Explain that the same environment supports three benchmark tasks:
   - stabilization
   - budget-constrained balancing
   - surge-response under disruption
6. Close with the baseline and validation story:
   - deterministic graders
   - normalized scores
   - successful local and live `openenv validate`
- Surfaces daily operational pressure so agents can reason about where the hospital is most likely to fail next

## Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN / API_KEY | LiteLLM proxy key used by `inference.py` |
| SPACE_URL | Running environment URL |
| LOCAL_IMAGE_NAME / IMAGE_NAME | Optional local Docker image for offline inference runs |
| DIFFICULTY | Task difficulty for `inference.py` |

## Running Grader
```bash
python grader.py --difficulty all --seed 42
```

## Running Demo
```bash
python demo.py
```

The demo script:

- resets the environment with deterministic seeds
- runs sample actions end-to-end across the task suite
- prints observations, rewards, done flags, and final normalized scores
- gives reviewers a one-command proof that the environment works

Note: `demo.py` is a reviewer-friendly proof run. For the official deterministic benchmark,
use `grader.py`, which averages fixed seeds and reports task scores in the `0.0-1.0` range.

## Running Inference
```bash
python inference.py
```

The baseline script:

- uses the OpenAI client with `API_BASE_URL`, `MODEL_NAME`, and the injected `HF_TOKEN` / `API_KEY`
- performs a startup connectivity check through the injected LiteLLM/OpenAI-compatible proxy before stepping the environment
- can target either a live Space via `SPACE_URL` or a local image via `LOCAL_IMAGE_NAME`
- emits structured stdout logs with `[START]`, `[STEP]`, and `[END]` markers for validator-friendly parsing
- falls back to a deterministic heuristic planner only when the model returns malformed action JSON

## Built With

- OpenEnv Core
- FastAPI
- Pydantic
- Docker
- HuggingFace Spaces
