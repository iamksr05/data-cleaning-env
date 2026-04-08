from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app import app
from environment import DataCleaningEnv
from environment import TASKS
from models import CleaningAction

app = FastAPI()

# Global environment instance
env = DataCleaningEnv()


# -------------------------
# REQUEST MODELS
# -------------------------
class StepRequest(BaseModel):
    action_type: str
    column: Optional[str] = None


# -------------------------
# RESET ENDPOINT
# -------------------------
@app.post("/reset")
def reset(task: str = "easy"):
    if task not in TASKS:
        return {
            "observation": None,
            "reward": 0.0,
            "done": True,
            "info": {"error": "invalid_task"}
        }
    observation = env.reset(task)
    return {
        "observation": observation.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {"task": task}
    }


# -------------------------
# STEP ENDPOINT
# -------------------------
@app.post("/step")
def step(request: StepRequest):
    from pydantic import ValidationError

    try:
        action = CleaningAction(
            action_type=request.action_type,
            column=request.column
        )
    except ValidationError:
        return {
            "observation": None,
            "reward": 0.0,
            "done": True,
            "info": {"error": "invalid_action"}
        }

    observation, reward, done, info = env.step(action)

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


# -------------------------
# STATE ENDPOINT (optional but useful)
# -------------------------
@app.get("/state")
def state():
    if env.current_state:
        return env.current_state.model_dump()
    return {"error": "state_not_initialized"}


# -------------------------
# ROOT ENDPOINT
# -------------------------
@app.get("/")
def home():
    return {"message": "Data Cleaning Environment is running"}


# -------------------------
# HEALTH ENDPOINT
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()