from typing import Tuple, Literal
import random
import copy
from grader import compute_score
from models import CleaningAction, CleaningObservation, CleaningState

TASKS = ["easy", "medium", "hard"]

class DataCleaningEnv:

    def __init__(self):
        self.max_steps = 10
        self.state = None
        self.dataset = None
        self.last_action = None
        self.same_action_count = 0

    # -------------------------
    # RESET (start environment)
    # -------------------------
    def reset(self, task="easy"):
        self.task = task
        self.steps = 0
        if task not in TASKS:
            task = "easy"
            self.task = task

        if task == "easy":
            self.dataset = [
                {"name": "Tom", "age": 25, "city": "mumbai"},
                {"name": "Tom", "age": 25, "city": "mumbai"},  # duplicate
            ]

        elif task == "medium":
            self.dataset = [
                {"name": "Tom", "age": None, "city": "mumbai"},
                {"name": "Jerry", "age": None, "city": "mumbai"},
            ]

        elif task == "hard":
            self.dataset = [
                {"name": "Tom", "age": None, "city": "MUMBAI"},
                {"name": "Tom", "age": None, "city": "mumbai"},
                {"name": "Jerry", "age": None, "city": "MUMBAI"},
            ]

        # Initialize state and score tracking
        from models import CleaningState
        self.state = CleaningState(step_count=0, max_steps=self.max_steps, history=[])
        self.previous_score = compute_score(self.dataset)
        self.last_action = None
        self.same_action_count = 0

        return self._get_observation()

    # -------------------------
    # STEP (core logic)
    # -------------------------
    def step(self, action: CleaningAction) -> Tuple[CleaningObservation, float, bool, dict]:
        if self.dataset is None:
            return self._get_observation(), 0.0, True, {"error": "env_not_initialized"}
        if self.state is None:
            from models import CleaningState
            self.state = CleaningState(step_count=0, max_steps=self.max_steps, history=[])
        self.state.step_count += 1
        self.state.history.append(action.action_type)

        # step penalty removed for compliance

        # track repeated actions
        if self.last_action == action.action_type:
            self.same_action_count += 1
        else:
            self.same_action_count = 1
        self.last_action = action.action_type

        # prevent infinite loops (same action repeatedly)
        if self.same_action_count >= 3:
            # terminate with neutral reward
            return self._get_observation(), 0.0, True, {"error": "repeated_action_loop"}

        # snapshot before action
        before_dataset = copy.deepcopy(self.dataset)
        before_score = self.previous_score

        done = False

        if action.action_type == "remove_duplicates":
            self._remove_duplicates()

        elif action.action_type == "fill_missing_mean":
            self._fill_missing()

        elif action.action_type == "fill_missing_mode":
            self._fill_missing()

        elif action.action_type == "drop_rows":
            self._drop_rows()

        elif action.action_type == "normalize_text":
            self._normalize_text()

        elif action.action_type == "stop":
            done = True

        elif action.action_type == "analyze_data":
            pass

        else:
            # invalid action
            return self._get_observation(), 0.0, True, {"error": "invalid_action"}

        # detect no-op (action did not change dataset)
        if before_dataset == self.dataset and action.action_type not in ["stop", "analyze_data"]:
            # no-op penalty: neutral reward
            score = compute_score(self.dataset)
            reward = 0.0
            done = False
            obs = self._get_observation()
            return obs, reward, done, {"score": score, "warning": "no_effect_action"}

        # compute score using grader
        score = compute_score(self.dataset)

        # reward = improvement only, normalized to [0,1]
        improvement = score - self.previous_score
        reward = max(0.0, improvement)
        self.previous_score = score

        # termination conditions
        if score >= 0.95:
            done = True

        if self.state.step_count >= self.max_steps:
            done = True

        obs = self._get_observation()

        # normalize reward to [0,1] for compliance
        reward = max(0.0, min(1.0, reward))
        return obs, reward, done, {"score": score}
    
    # -------------------------
    # STATE PROPERTY
    # -------------------------
    @property
    def current_state(self) -> CleaningState:
        return self.state

    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------

    def _generate_dataset(self):
        samples = [
            [
                {"name": "John", "age": None, "city": "NY"},
                {"name": "John", "age": None, "city": "NY"},
                {"name": "Anna", "age": 25, "city": "ny"},
                {"name": "Mike", "age": None, "city": "LA"},
            ],
            [
                {"name": "Alice", "age": None, "city": "DELHI"},
                {"name": "Bob", "age": 30, "city": "delhi"},
                {"name": "Alice", "age": None, "city": "DELHI"},
            ],
            [
                {"name": "Tom", "age": None, "city": "MUMBAI"},
                {"name": "Jerry", "age": None, "city": "mumbai"},
                {"name": "Tom", "age": None, "city": "MUMBAI"},
            ]
    ]
        return random.choice(samples)

    def _get_observation(self) -> CleaningObservation:
        issues = {
            "missing": sum(1 for row in self.dataset if None in row.values()),
            "duplicates": len(self.dataset) - len({str(r) for r in self.dataset}),
        }

        message = "Clean the dataset"

        if hasattr(self, "last_action") and self.last_action:
            message = f"Last action: {self.last_action}"

        return CleaningObservation(
            dataset_preview=self.dataset[:3],
            issues=issues,
            message=message
        )

    def _remove_duplicates(self):
        before = len(self.dataset)
        unique = [dict(t) for t in {tuple(d.items()) for d in self.dataset}]
        self.dataset = unique
        after = len(self.dataset)

        return 0.2 if after < before else -0.1

    def _fill_missing(self):
        reward = 0.0
        for row in self.dataset:
            for k, v in row.items():
                if v is None:
                    row[k] = 0
                    reward += 0.05
        return reward

    def _drop_rows(self):
        before = len(self.dataset)
        self.dataset = [row for row in self.dataset if None not in row.values()]
        after = len(self.dataset)

        return 0.2 if after < before else -0.1

    def _normalize_text(self):
        for row in self.dataset:
            if "city" in row and isinstance(row["city"], str):
                row["city"] = row["city"].lower()
        return 0.1

    # _calculate_quality and _normalize_reward removed