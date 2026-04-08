import os
import requests
from openai import OpenAI

# ===== ENV =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# OpenAI client (uses OPENAI_API_KEY)
client = OpenAI()

ENV_URL = "http://127.0.0.1:8000"


# ===== ACTION SELECTION =====
def choose_action(obs):
    prompt = f"""
User: {obs['message']}
User type: {obs['user_type']}
History: {obs['history']}

Choose ONE action:
apologize, ask_details, check_order, refund, replace, escalate

Respond ONLY with the action name.
"""

    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=20
    )

    return res.choices[0].message.content.strip()


# ===== MAIN LOOP =====
def main():
    print("[START]")

    # Reset environment
    obs = requests.post(f"{ENV_URL}/reset").json()["observation"]

    done = False
    step = 0

    while not done and step < 8:
        step += 1

        action = choose_action(obs)

        res = requests.post(
            f"{ENV_URL}/step",
            json={"action": action}
        ).json()

        obs = res["observation"]
        reward = res["reward"]
        done = res["done"]

        print(f"[STEP] step={step} action={action} reward={reward}")

    # Get final score from grader
    score = requests.get(f"{ENV_URL}/grader").json()["score"]

    print(f"[END] score={score}")


# ===== FIXED ENTRY POINT =====
if __name__ == "__main__":
    main()