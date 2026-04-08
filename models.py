from pydantic import BaseModel
from typing import List, Dict, Optional, Literal

# -------------------------
# ACTION (what agent does)
# -------------------------
class CleaningAction(BaseModel):
    action_type: Literal[
        "remove_duplicates",
        "fill_missing_mean",
        "fill_missing_mode",
        "drop_rows",
        "normalize_text",
        "analyze_data",
        "stop"
    ]
    column: Optional[str] = None  # optional: for column-specific actions


# -------------------------
# OBSERVATION (what agent sees)
# -------------------------
class CleaningObservation(BaseModel):
    dataset_preview: List[Dict]  # small sample of dataset rows
    issues: Dict[str, int]       # e.g. {"missing": 3, "duplicates": 2}
    message: Optional[str] = None  # extra info for agent


# -------------------------
# STATE (internal tracking)
# -------------------------
class CleaningState(BaseModel):
    step_count: int
    max_steps: int
    history: List[str]