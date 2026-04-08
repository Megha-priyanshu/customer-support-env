def grade(trajectory):
    if not trajectory:
        return 0

    score = 0
    solutions = trajectory[-1]["info"]["solutions"]
    actions = [t["action"] for t in trajectory]

    if all(sol in actions for sol in solutions):
        score += 0.6

    if actions and actions[0] == "apologize":
        score += 0.3

    if len(actions) <= 3:
        score += 0.1

    return min(score, 1.0)