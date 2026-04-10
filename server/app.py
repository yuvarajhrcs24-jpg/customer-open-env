from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, HTTPException

from customer_support_env.env import CustomerSupportEnv
from customer_support_env.models import Action

app = FastAPI(title="Customer Support OpenEnv Server")
env = CustomerSupportEnv()


def _read_task_id(payload: dict[str, Any]) -> str | None:
    task_id = payload.get("task_id")
    if task_id is None:
        return None
    if not isinstance(task_id, str):
        raise HTTPException(status_code=400, detail="task_id must be a string")
    return task_id


def _read_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("action") if isinstance(payload.get("action"), dict) else payload
    if not isinstance(candidate, dict) or "action_type" not in candidate:
        raise HTTPException(status_code=400, detail="Action payload must include action_type")
    return candidate


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
@app.post("/openenv/reset")
def reset_api(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    try:
        observation = env.reset(task_id=_read_task_id(payload or {}))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return observation.model_dump()


@app.post("/step")
@app.post("/openenv/step")
def step_api(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    try:
        if env.state().done:
            raise HTTPException(status_code=400, detail="Episode already completed. Call reset() first.")
        action_payload = _read_action_payload(payload or {})
        action = Action.model_validate(action_payload)
        obs, reward, done, info = env.step(action)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {exc}") from exc

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
@app.get("/openenv/state")
@app.post("/state")
@app.post("/openenv/state")
def state_api() -> dict[str, Any]:
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, log_level="info")


def server() -> None:
    main()


if __name__ == "__main__":
    main()
