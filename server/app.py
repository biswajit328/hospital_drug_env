from openenv.core.env_server import create_app
from fastapi.responses import RedirectResponse

try:
    from hospital_drug_env.server.environment import HospitalDrugEnvironment
    from hospital_drug_env.models import DrugShortageAction, DrugShortageObservation
except ModuleNotFoundError:
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

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
