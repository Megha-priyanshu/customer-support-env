import os
import requests
from openai import OpenAI

# ===== ENV =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ENV_URL = "http://127.0.0.1:8000"


def _fallback_action(obs):
    """Rule-based fallback for environments without an API key."""
    msg = str(obs.get("message", "")).lower()
    history = " ".join(str(x).lower() for x in obs.get("history", []))
    text = f"{msg} {history}"

    if "wrong product" in text or "replacement" in text or "replace" in text:
        return "replace"
    if "refund" in text:
        return "refund"
    if "order" in text or "tracking" in text or "delivered" in text:
        return "check_order"
    if "angry" in str(obs.get("user_type", "")).lower():
        return "escalate"
    return "ask_details"


# ===== ACTION SELECTION =====
def choose_action(obs):
    if not OPENAI_API_KEY:
        return _fallback_action(obs)

    prompt = f"""
User: {obs['message']}
User type: {obs['user_type']}
History: {obs['history']}

Choose ONE action:
apologize, ask_details, check_order, refund, replace, escalate

Respond ONLY with the action name.
"""

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
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