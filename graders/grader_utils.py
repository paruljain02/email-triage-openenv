"""
Shared grading utilities used across all three task graders.
"""

from models import Action, Reward, RewardBreakdown


# ── Category Similarity ───────────────────────────────────────────────────
# Some misclassifications are less severe than others (partial credit)
CATEGORY_SIMILARITY = {
    ("urgent", "support"): 0.4,
    ("support", "urgent"): 0.4,
    ("sales", "partnership"): 0.5,
    ("partnership", "sales"): 0.5,
    ("spam", "sales"): 0.1,
    ("hr", "internal"): 0.3,
    ("internal", "hr"): 0.3,
    ("billing", "sales"): 0.3,
    ("billing", "support"): 0.2,
}

# ── Priority Distance ─────────────────────────────────────────────────────
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def score_category(predicted: str, actual: str) -> float:
    """Score category prediction with partial credit for related categories."""
    if predicted == actual:
        return 1.0
    return CATEGORY_SIMILARITY.get((predicted, actual), 0.0)


def score_sentiment(predicted: str, actual: str) -> float:
    """Score sentiment with partial credit for adjacent sentiments."""
    if predicted is None:
        return 0.0
    if predicted == actual:
        return 1.0
    # Adjacent sentiments get partial credit
    sentiment_distance = {
        ("positive", "neutral"): 0.5,
        ("neutral", "positive"): 0.5,
        ("neutral", "negative"): 0.5,
        ("negative", "neutral"): 0.5,
        ("negative", "angry"): 0.6,
        ("angry", "negative"): 0.6,
        ("positive", "negative"): 0.0,
        ("positive", "angry"): 0.0,
        ("angry", "positive"): 0.0,
    }
    return sentiment_distance.get((predicted, actual), 0.1)


def score_priority(predicted: str, actual: str) -> float:
    """Score priority with distance-based partial credit."""
    if predicted is None:
        return 0.0
    if predicted == actual:
        return 1.0

    pred_idx = PRIORITY_ORDER.get(predicted)
    actual_idx = PRIORITY_ORDER.get(actual)
    if pred_idx is None or actual_idx is None:
        return 0.0

    distance = abs(pred_idx - actual_idx)
    # 1 level off = 0.5, 2 levels = 0.2, 3 levels = 0.0
    scores = {1: 0.5, 2: 0.2, 3: 0.0}
    base = scores.get(distance, 0.0)

    # Extra penalty for under-prioritizing critical items
    if actual == "P0" and predicted in ("P2", "P3"):
        base *= 0.5  # Severe penalty for missing critical items

    return base


def score_department(predicted: str, actual: str) -> float:
    """Score department routing — exact match or 0."""
    if predicted is None:
        return 0.0
    if predicted == actual:
        return 1.0

    # Some partial credit for reasonable alternatives
    dept_similarity = {
        ("engineering", "support"): 0.3,
        ("support", "engineering"): 0.3,
        ("executive", "sales"): 0.2,
        ("finance", "sales"): 0.2,
        ("executive", "legal"): 0.3,
        ("legal", "executive"): 0.3,
    }
    return dept_similarity.get((predicted, actual), 0.0)


def score_action_items(predicted_items: list, actual_items: list) -> float:
    """
    Score action items based on coverage and relevance.
    Uses keyword overlap to measure if key actions were identified.
    """
    if not actual_items:
        return 1.0 if not predicted_items else 0.8  # Minor penalty for unnecessary items

    if not predicted_items:
        return 0.0

    actual_keywords = set()
    for item in actual_items:
        words = item.lower().split()
        actual_keywords.update(w for w in words if len(w) > 3)

    predicted_text = " ".join(
        item.description.lower() if hasattr(item, "description") else str(item).lower()
        for item in predicted_items
    )

    # What fraction of key concepts were captured?
    matches = sum(1 for kw in actual_keywords if kw in predicted_text)
    coverage = matches / len(actual_keywords) if actual_keywords else 0

    # Penalty for too many or too few items
    ratio = len(predicted_items) / len(actual_items)
    length_penalty = 1.0
    if ratio > 2.0:
        length_penalty = 0.8  # Too many items
    elif ratio < 0.3:
        length_penalty = 0.7  # Too few items

    return min(1.0, coverage * length_penalty)


def score_draft_reply(draft: str, ground_truth: dict) -> float:
    """
    Score draft reply quality based on tone alignment and content relevance.
    Uses heuristic checks — can be replaced with LLM grading.
    """
    if not draft or draft.strip() == "":
        # No reply when one was expected
        if ground_truth.get("requires_follow_up", False):
            return 0.0
        return 0.8  # Acceptable if no follow-up needed

    if ground_truth.get("reply_tone") in ("none", "none — do not reply"):
        # Replied to spam — penalty
        return 0.2

    reply_lower = draft.lower()
    score = 0.3  # Base score for attempting a reply

    # Check for basic structure
    if len(draft.split()) >= 20:
        score += 0.1  # Substantive response
    if any(g in reply_lower for g in ["hi", "hello", "dear", "thank"]):
        score += 0.1  # Has greeting
    if any(s in reply_lower[-100:] for s in ["regards", "best", "thank", "sincerely"]):
        score += 0.1  # Has sign-off

    # Tone-specific checks
    tone = ground_truth.get("reply_tone", "").lower()
    if "empathetic" in tone or "apologetic" in tone:
        if any(w in reply_lower for w in ["sorry", "understand", "apologize", "frustrat"]):
            score += 0.15
    if "urgent" in tone:
        if any(w in reply_lower for w in ["immediately", "right away", "priority", "asap", "now"]):
            score += 0.15
    if "professional" in tone:
        if len(draft.split()) > 30 and "\n" in draft:
            score += 0.1

    return min(1.0, score)


def apply_penalties(reward: float, action: Action, ground_truth: dict) -> tuple[float, float]:
    """
    Apply penalties for clearly undesirable behaviors.
    Returns (adjusted_reward, penalty_amount).
    """
    penalties = 0.0

    # Penalty: Replying to spam
    if ground_truth.get("category") == "spam" and action.draft_reply and len(action.draft_reply.strip()) > 10:
        penalties += 0.1

    # Penalty: Missing follow-up flag on items that need it
    if ground_truth.get("requires_follow_up") and not action.requires_follow_up:
        penalties += 0.05

    # Penalty: Flagging follow-up on items that don't need it
    if not ground_truth.get("requires_follow_up") and action.requires_follow_up:
        penalties += 0.02

    adjusted = max(0.0, reward - penalties)
    return adjusted, penalties
