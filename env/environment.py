import random

class CustomerSupportEnv:
    def __init__(self):
        self.max_steps = 4
        self.reset()

    def _sample_case(self):
        return random.choice([
            {
                "message": "My order is late and I want a refund",
                "solutions": ["check_order", "refund"]
            },
            {
                "message": "Wrong product delivered and I need replacement",
                "solutions": ["replace"]
            },
            {
                "message": "Payment done but no confirmation",
                "solutions": ["check_order"]
            }
        ])

    def reset(self, difficulty="hard"):
        self.case = self._sample_case()
        self.steps = 0
        self.done = False
        self.user_type = random.choice(["polite", "neutral", "angry"])
        self.history = []
        return self._get_obs()

    def _get_obs(self):
        return {
            "company": "ShopEase",
            "message": f"[ShopEase] {self.case['message']}",
            "user_type": self.user_type,
            "history": self.history,
            "step": self.steps
        }

    def step(self, action):
        self.steps += 1
        reward = 0

        if action == "apologize":
            reward += 0.3
            if self.user_type == "angry":
                reward += 0.2

        if action in self.case["solutions"]:
            reward += 0.6

        if action not in self.case["solutions"] and action not in ["apologize", "ask_details"]:
            reward -= 0.4

        self.history.append(action)

        if set(self.case["solutions"]).issubset(set(self.history)) or self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {"solutions": self.case["solutions"]}