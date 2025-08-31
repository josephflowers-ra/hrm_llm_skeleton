# reasoning_gym_wrapper.py
"""
Reasoning Gym adapter for HRM training (answer-only targets).

- Supports single-task (e.g., "toy_add") or multi-task mixes via comma-separated list.
- Emits items shaped like:
    {
      "prompt": <string>,           # full prompt text given to the tokenizer
      "target": <string>,           # gold final answer ONLY (no ####, no extra text)
      "answer": <string>,           # plain gold answer (for convenience/logging)
      "verify": <callable(str)->bool>,
      "metadata": {...},            # may include source task name, etc.
    }

Training target is the *answer only* (no '####'). At inference, generations may
include extra text—verifier robustly extracts the candidate answer and compares.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Sequence
import random
import math
import re
import unicodedata

try:
    from reasoning_gym import create_dataset, get_score_answer_fn
except Exception:
    # Graceful fallback if RG isn't installed at import-time
    create_dataset = None
    get_score_answer_fn = None


# ------------------------------
# Prompt / target formatting
# ------------------------------
SYSTEM_HEADER = (
    "Answer with the final result ONLY. Do not include any extra text."
)

USER_TEMPLATE = """Question:
{question}
"""

def format_prompt(question: str) -> str:
    return f"{SYSTEM_HEADER}\n\n{USER_TEMPLATE.format(question=question)}"


# ------------------------------
# Normalization / extraction
# ------------------------------
def _normalize_answer(s: str) -> str:
    """Unicode/whitespace normalization for stable comparisons."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Normalize common minus variants
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return s


def _extract_pred(pred: Optional[str]) -> str:
    """
    Extract a candidate answer from a generated string.
    - Prefer content after the FIRST '####' if present (UI convention).
    - Otherwise use the whole string.
    - Keep only the first line and normalize.
    """
    if not isinstance(pred, str):
        return ""
    s = pred.strip()
    if not s:
        return ""
    if "####" in s:
        s = s.split("####", 1)[1]
    first_line = s.splitlines()[0] if s else ""
    return _normalize_answer(first_line)


# ------------------------------
# Gold answer extraction
# ------------------------------
def extract_gold_answer(item: Dict[str, Any]) -> str:
    """
    Best-effort: RG datasets commonly put the canonical answer under item['answer'].
    Some tasks store answers in metadata. Adjust here if you encounter a variant.
    """
    # Primary
    if "answer" in item and isinstance(item["answer"], str):
        return item["answer"].strip()

    # Fallbacks: Some RG tasks keep answers in metadata
    meta = item.get("metadata", {})
    if isinstance(meta, dict):
        if "answer" in meta and isinstance(meta["answer"], str):
            return meta["answer"].strip()
        if "target" in meta and isinstance(meta["target"], str):
            return meta["target"].strip()

    # As a last resort, just stringify
    return str(item.get("answer", "")).strip()


# ------------------------------
# Verifier factory
# ------------------------------
def make_verifier(source_dataset: Optional[str], gold: str) -> Callable[[str], bool]:
    """
    Returns a function pred -> bool indicating correctness.
    Prefers RG's official scorer when available; otherwise exact match on
    normalized strings. Empty/None predictions are incorrect.
    """
    scorer = None
    if get_score_answer_fn and source_dataset:
        try:
            scorer = get_score_answer_fn(source_dataset)
        except Exception:
            scorer = None

    gold_norm = _normalize_answer(gold)

    if scorer is not None:
        # Use RG’s scorer with a robust fallback
        def verify_fn(pred: str) -> bool:
            try:
                pred_norm = _extract_pred(pred)
                if not pred_norm:
                    return False
                # Some scorers expect raw strings; try normalized then raw
                ok = scorer(pred_norm, {"answer": gold}) or scorer(pred, {"answer": gold})
                return bool(ok)
            except Exception:
                return _extract_pred(pred) == gold_norm
        return verify_fn

    # Simple fallback: normalized exact match
    def verify_fallback(pred: str) -> bool:
        return _extract_pred(pred) == gold_norm

    return verify_fallback


# ------------------------------
# Dataset containers / mixing
# ------------------------------
@dataclass
class SimpleDataset:
    data: List[Dict[str, Any]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _make_single_task_dataset(task: str, split: str, n: int, seed: int) -> SimpleDataset:
    """
    Builds a single-task RG dataset with our prompt/verify packaging.
    """
    if create_dataset is None:
        raise RuntimeError("reasoning_gym is not available. Please `pip install reasoning-gym`.")

    # Map split name to a seed range so eval isn't the same as train by accident
    base_seed = seed + (1337 if split == "eval" else 0)

    raw = create_dataset(task, seed=base_seed, size=n)
    packed = []
    for ex in raw:
        q = ex.get("question", "").strip()
        gold = extract_gold_answer(ex)
        src = None
        md = ex.get("metadata", {})
        if isinstance(md, dict):
            src = md.get("source_dataset", task)

        prompt = format_prompt(q)
        verify = make_verifier(src, gold)

        packed.append({
            "prompt": prompt,
            "target": gold,            # <-- answer-only target (no '####')
            "answer": gold,            # for convenience/logging
            "verify": verify,
            "metadata": {"task": task, "source_dataset": src or task},
        })

    return SimpleDataset(packed)


def _round_robin_mix(datasets: Sequence[SimpleDataset], total_n: int) -> SimpleDataset:
    """
    Interleave examples from multiple datasets to produce a balanced mixed dataset.
    """
    if not datasets:
        return SimpleDataset([])
    idxs = [0] * len(datasets)
    data = []
    while len(data) < total_n:
        for k, ds in enumerate(datasets):
            if len(data) >= total_n:
                break
            if idxs[k] < len(ds):
                data.append(ds[idxs[k]])
                idxs[k] += 1
    return SimpleDataset(data)


def build_reasoning_dataset(task_or_tasks: str, split: str, n: int, seed: int = 1234):
    """
    Public entry used by train.py

    Args
    ----
    task_or_tasks: str
        - Single task name (e.g., "toy_add")
        - OR comma-separated list of tasks for a mixed dataset
          e.g., "basic_arithmetic,gsm_symbolic,chain_sum,simple_equations,propositional_logic"
    split: "train" | "eval"
    n: size of the dataset returned
    seed: RNG seed (deterministic unless you jitter it externally)

    Returns
    -------
    SimpleDataset
    """
    task_or_tasks = (task_or_tasks or "").strip()
    if "," not in task_or_tasks:
        return _make_single_task_dataset(task_or_tasks, split, n, seed)

    # Multi-task mix
    tasks = [t.strip() for t in task_or_tasks.split(",") if t.strip()]
    if not tasks:
        # Fallback if string was e.g. ",,,"
        return _make_single_task_dataset("toy_add", split, n, seed)

    # Build each sub-dataset with a proportional share (rounded)
    per = max(1, math.floor(n / len(tasks)))
    subs = []
    for i, t in enumerate(tasks):
        # vary seed per sub-dataset so they differ
        subs.append(_make_single_task_dataset(t, split, per, seed + 10 * (i + 1)))

    # If per*len(tasks) < n, top-up with the first dataset
    total = per * len(tasks)
    if total < n and subs:
        extra = _make_single_task_dataset(tasks[0], split, n - total, seed + 777)
        subs[0] = SimpleDataset(subs[0].data + extra.data)

    return _round_robin_mix(subs, n)
