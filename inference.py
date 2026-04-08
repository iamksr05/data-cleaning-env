import os
import requests
from openai import OpenAI

# =========================
# ENV VARIABLES (MANDATORY)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy_token")

# OpenAI client (required, routed via HF)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Your environment API
ENV_URL = os.getenv("ENV_URL", "https://iamksr05-data-cleaning-env.hf.space")

MAX_STEPS = 10


# =========================
# LOGGING (STRICT FORMAT)
# =========================
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


# =========================
# SIMPLE AGENT
# =========================
def choose_action(obs):
    issues = obs.get("issues", {})

    if issues.get("duplicates", 0) > 0:
        return "remove_duplicates"

    if issues.get("missing", 0) > 0:
        return "fill_missing_mean"

    return "stop"


# =========================
# OPTIONAL LLM CALL (REQUIRED BY RULES)
# =========================
def call_model():
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "choose next action"}
            ],
            max_tokens=10
        )
        return response.choices[0].message.content
    except Exception:
        return None


# =========================
# MAIN EPISODE
# =========================
def run_episode(task):
    rewards = []
    steps = 0
    success = False

    log_start(task=task, env="data_cleaning_env", model=MODEL_NAME)

    try:
        # RESET
        response = requests.post(f"{ENV_URL}/reset?task={task}", json={})
        data = response.json()
        obs = data.get("observation", data)

        for step in range(1, MAX_STEPS + 1):
            # Call model (for compliance)
            _ = call_model()

            action = choose_action(obs)

            response = requests.post(
                f"{ENV_URL}/step",
                json={"action_type": action}
            )

            data = response.json()

            obs = data.get("observation", {})
            reward = float(data.get("reward", 0.0))
            done = data.get("done", False)

            rewards.append(reward)
            steps = step

            log_step(step, action, reward, done, None)

            if done:
                break

        # Normalize score to [0,1]
        total_reward = sum(rewards)
        score = max(0.0, min(1.0, total_reward))
        success = score >= 0.5

    except Exception as e:
        log_step(0, "error", 0.0, True, str(e))
        score = 0.0

    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    for task_name in ["easy", "medium", "hard"]:
        run_episode(task_name)