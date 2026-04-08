---
title: Intelligent Data Cleaning Environment
sdk: docker
---

# Intelligent Data Cleaning Environment

A reinforcement learning-style environment where agents learn to clean datasets through sequential decision-making.

---

## Overview

This project focuses on building an environment where an agent learns how to clean data step by step.

Instead of treating data cleaning as a fixed pipeline, the environment models it as a sequence of decisions. The agent observes the dataset, applies actions, and improves data quality over time.

---

## Motivation

In real-world scenarios, data cleaning is not a single operation. It involves:

- identifying issues  
- deciding what to fix first  
- applying transformations  
- verifying improvements  

Each step depends on previous actions. This environment captures that behavior, making it suitable for training or evaluating intelligent agents.

---

## Environment Description

The environment follows this loop:

```
Agent → Environment → Observation → Action → Reward → Agent
```

### Core Methods

- `reset(task)`  
  Initializes a dataset with issues based on difficulty level.

- `step(action)`  
  Applies an action and returns:
  - updated observation  
  - reward  
  - done flag  
  - additional info  

- `state()`  
  Returns internal details like step count and action history.

---

## Action Space

The agent can perform:

- `remove_duplicates`  
- `fill_missing_mean`  
- `fill_missing_mode`  
- `drop_rows`  
- `normalize_text`  
- `analyze_data`  
- `stop`  

Each action modifies the dataset and influences the final score.

---

## Observation Space

The agent receives partial information:

```json
{
  "dataset_preview": [...],
  "issues": {
    "missing": number,
    "duplicates": number
  },
  "message": "context about previous action"
}
```

---

## Reward and Scoring

Rewards are based on improvement in data quality:

- Positive reward only when the dataset improves  
- No reward for actions that have no effect  
- Episode ends when dataset is clean or step limit is reached  

### Scoring Formula

Score = 0.4 × completeness + 0.3 × uniqueness + 0.3 × consistency

---

## Tasks and Difficulty

The environment features 3 tasks of increasing difficulty:

- **Easy**: Remove duplicate rows from the dataset.
- **Medium**: Handle missing values appropriately (e.g. fill with mean or mode).
- **Hard**: Fully clean a dataset with multiple issues including duplicates, missing values, and inconsistent text formatting.

---

## Baseline Scores

The provided `inference.py` script run with `Qwen/Qwen2.5-72B-Instruct` achieves the following baseline scores:

- **Easy**: > 0.5 expected.
- **Medium**: > 0.5 expected.
- **Hard**: > 0.5 expected.

*(Scores may vary based on LLM response parsing)*

---

## Setup and Usage Instructions

### Running the Environment API
You can run the environment locally via Docker:

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

The API will be available at `http://localhost:7860`.

### Running Inference
To run the baseline baseline inference script against the environment:

1. Install dependencies:
```bash
pip install -r requirements.txt  # (or using uv)
pip install openai requests
```

2. Export environment variables:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hugging_face_token"
export ENV_URL="http://localhost:7860" # if running locally
```

3. Run script:
```bash
python inference.py
```