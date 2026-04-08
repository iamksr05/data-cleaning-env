import requests

BASE_URL = "https://iamksr05-data-cleaning-env.hf.space"


def choose_action(observation):
    issues = observation["issues"]
    preview = observation["dataset_preview"]

    # check if text is inconsistent (uppercase vs lowercase)
    for row in preview:
        if isinstance(row.get("city"), str) and not row["city"].islower():
            return "normalize_text"

    if issues["duplicates"] > 0:
        return "remove_duplicates"

    if issues["missing"] > 0:
        return "fill_missing_mean"

    if issues["duplicates"] == 0 and issues["missing"] == 0:
        return "stop"

    return "normalize_text"

def run_episode():
    # reset
    response = requests.post(f"{BASE_URL}/reset")
    data = response.json()

    observation = data["observation"]
    done = False
    final_reward = 0.0

    while not done:
        action = choose_action(observation)

        response = requests.post(
            f"{BASE_URL}/step",
            json={"action_type": action}
        )

        data = response.json()

        observation = data["observation"]
        reward = data["reward"]
        done = data["done"]

        final_reward = reward

        print("Observation:", observation)
        print("Action:", action)
        print("Reward:", reward)
        print("Done:", done)
        print("------")
        
    return final_reward


if __name__ == "__main__":
    score = run_episode()
    print("Final Score:", score)
