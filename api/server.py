from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import CustomerSupportEnv
from grader.grader import grade
from env.tasks import TASKS

app = FastAPI()

env = CustomerSupportEnv()
trajectory = []


# ✅ NEW HOME ROUTE (for browser test)
@app.get("/")
def home():
    return {"message": "Customer Support API is running"}


# Request model
class Action(BaseModel):
    action: str


# Get all tasks
@app.get("/tasks")
def tasks():
    return {"tasks": TASKS}


# Reset environment
@app.post("/reset")
def reset():
    global trajectory
    trajectory = []
    return {"observation": env.reset()}


# Take a step
@app.post("/step")
def step(action: Action):
    global trajectory
    obs, reward, done, info = env.step(action.action)

    trajectory.append({
        "action": action.action,
        "reward": reward,
        "info": info
    })

    return {
        "observation": obs,
        "reward": reward,
        "done": done
    }


# Get trajectory
@app.get("/state")
def state():
    return {"trajectory": trajectory}


# Get score
@app.get("/grader")
def grader():
    return {"score": grade(trajectory)}