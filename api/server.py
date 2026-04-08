from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import CustomerSupportEnv
from grader.grader import grade
from env.tasks import TASKS

app = FastAPI()
env = CustomerSupportEnv()
trajectory = []

class Action(BaseModel):
    action: str

@app.get("/tasks")
def tasks():
    return {"tasks": TASKS}

@app.post("/reset")
def reset():
    global trajectory
    trajectory = []
    return {"observation": env.reset()}

@app.post("/step")
def step(action: Action):
    global trajectory
    obs, reward, done, info = env.step(action.action)
    trajectory.append({"action": action.action, "reward": reward, "info": info})
    return {"observation": obs, "reward": reward, "done": done}

@app.get("/state")
def state():
    return {"trajectory": trajectory}

@app.get("/grader")
def grader():
    return {"score": grade(trajectory)}