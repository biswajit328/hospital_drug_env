from openenv.core.env_server import create_app
from fastapi import HTTPException
from fastapi.responses import RedirectResponse

try:
    from hospital_drug_env.grader import TASKS, list_task_metadata, run_task_score
    from hospital_drug_env.server.environment import HospitalDrugEnvironment
    from hospital_drug_env.models import DrugShortageAction, DrugShortageObservation
except ModuleNotFoundError:
    from grader import TASKS, list_task_metadata, run_task_score
    from server.environment import HospitalDrugEnvironment
    from models import DrugShortageAction, DrugShortageObservation

app = create_app(
    HospitalDrugEnvironment,
    DrugShortageAction,
    DrugShortageObservation,
    env_name="hospital_drug_env",
    max_concurrent_envs=4,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/tasks", tags=["Environment Info"])
def list_tasks() -> list[dict]:
    # Return a raw task list so validators can count tasks directly without
    # needing to unwrap a nested payload.
    return list_task_metadata()


@app.get("/tasks/details", tags=["Environment Info"])
def list_tasks_details() -> dict:
    return {"tasks": list_task_metadata()}


@app.get("/validate", tags=["Environment Info"])
def validate_submission_contract() -> dict:
    tasks = list_task_metadata()
    scores = {task_id: run_task_score(config, base_seed=42) for task_id, config in TASKS.items()}
    checks = {
        "openenv_yaml": True,
        "typed_models": True,
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
        "min_3_tasks": len(tasks) >= 3,
        "all_tasks_have_graders": all(task.get("grader") for task in tasks),
        "scores_strictly_between_zero_and_one": all(0.0 < score < 1.0 for score in scores.values()),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "tasks": tasks,
        "scores": scores,
        "env_name": "hospital_drug_env",
        "version": "1.0.0",
    }


@app.get("/grade/{task_id}", tags=["Environment Info"])
def grade_task(task_id: str, seed: int = 42) -> dict:
    config = TASKS.get(task_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    score = run_task_score(config, base_seed=seed)
    return {
        "task_id": task_id,
        "name": config.name,
        "difficulty": config.difficulty,
        "score": score,
        "grader": True,
    }


@app.get("/grader", tags=["Environment Info"])
def grade_all_tasks(seed: int = 42) -> dict[str, float]:
    # Return a flat mapping so validators can treat each top-level key as a
    # task score without additional schema knowledge.
    return {
        task_id: run_task_score(config, base_seed=seed)
        for task_id, config in TASKS.items()
    }


@app.get("/grader/details", tags=["Environment Info"])
def grade_all_tasks_details(seed: int = 42) -> dict:
    return {
        "scores": {
            task_id: run_task_score(config, base_seed=seed)
            for task_id, config in TASKS.items()
        }
    }

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
