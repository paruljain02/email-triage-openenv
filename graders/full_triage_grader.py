"""
Task 3 Grader (Hard): Full Email Triage with Draft Response
Grades all fields: category, sentiment, priority, department, action items, draft reply.
"""

from models import Action, Reward, RewardBreakdown
from graders.grader_utils import (
    score_category, score_sentiment, score_priority,
    score_department, score_action_items, score_draft_reply,
    apply_penalties
)


class FullTriageGrader:
    """Grader for the hard full-triage task."""

    WEIGHTS = {
        "category": 0.15,
        "sentiment": 0.10,
        "priority": 0.20,
        "department": 0.15,
        "action_items": 0.20,
        "draft_reply": 0.20,
    }

    def grade(self, action: Action, ground_truth: dict) -> Reward:
        cat_score = score_category(action.category.value, ground_truth["category"])
        sent_score = score_sentiment(
            action.sentiment.value if action.sentiment else None,
            ground_truth["sentiment"]
        )
        pri_score = score_priority(
            action.priority.value if action.priority else None,
            ground_truth["priority"]
        )
        dept_score = score_department(
            action.department.value if action.department else None,
            ground_truth["department"]
        )
        action_score = score_action_items(
            action.action_items,
            ground_truth.get("action_items", [])
        )
        reply_score = score_draft_reply(
            action.draft_reply or "",
            ground_truth
        )

        weighted = (
            cat_score * self.WEIGHTS["category"] +
            sent_score * self.WEIGHTS["sentiment"] +
            pri_score * self.WEIGHTS["priority"] +
            dept_score * self.WEIGHTS["department"] +
            action_score * self.WEIGHTS["action_items"] +
            reply_score * self.WEIGHTS["draft_reply"]
        )

        adjusted, penalties = apply_penalties(weighted, action, ground_truth)

        # Bonus for exceptional full triage
        bonus = 0.0
        all_scores = [cat_score, sent_score, pri_score, dept_score, action_score, reply_score]
        if all(s >= 0.8 for s in all_scores):
            bonus = 0.05
            adjusted = min(1.0, adjusted + bonus)

        # Build detailed feedback
        feedback_parts = []
        score_map = {
            "Category": cat_score, "Sentiment": sent_score,
            "Priority": pri_score, "Department": dept_score,
            "Actions": action_score, "Reply": reply_score
        }
        for name, score in score_map.items():
            if score >= 0.8:
                feedback_parts.append(f"{name}: ✓ ({score:.2f})")
            elif score >= 0.4:
                feedback_parts.append(f"{name}: ~ ({score:.2f})")
            else:
                feedback_parts.append(f"{name}: ✗ ({score:.2f})")

        return Reward(
            total=round(adjusted, 4),
            breakdown=RewardBreakdown(
                category_score=round(cat_score, 4),
                sentiment_score=round(sent_score, 4),
                priority_score=round(pri_score, 4),
                routing_score=round(dept_score, 4),
                action_items_score=round(action_score, 4),
                draft_reply_score=round(reply_score, 4),
            ),
            penalties=round(penalties, 4),
            bonus=round(bonus, 4),
            feedback="; ".join(feedback_parts)
        )
