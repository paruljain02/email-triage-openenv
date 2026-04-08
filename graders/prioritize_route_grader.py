"""
Task 2 Grader (Medium): Priority Assignment & Routing
Grades: category + sentiment + priority + department.
"""

from models import Action, Reward, RewardBreakdown
from graders.grader_utils import (
    score_category, score_sentiment, score_priority,
    score_department, apply_penalties
)


class PrioritizeRouteGrader:
    """Grader for the medium priority/routing task."""

    WEIGHTS = {
        "category": 0.25,
        "sentiment": 0.15,
        "priority": 0.30,
        "department": 0.30,
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

        weighted = (
            cat_score * self.WEIGHTS["category"] +
            sent_score * self.WEIGHTS["sentiment"] +
            pri_score * self.WEIGHTS["priority"] +
            dept_score * self.WEIGHTS["department"]
        )

        adjusted, penalties = apply_penalties(weighted, action, ground_truth)

        # Bonus for getting all four correct
        bonus = 0.0
        if cat_score == 1.0 and sent_score == 1.0 and pri_score == 1.0 and dept_score == 1.0:
            bonus = 0.05
            adjusted = min(1.0, adjusted + bonus)

        feedback_parts = []
        for name, score in [("Category", cat_score), ("Sentiment", sent_score),
                           ("Priority", pri_score), ("Department", dept_score)]:
            if score == 1.0:
                feedback_parts.append(f"{name}: ✓")
            elif score > 0:
                feedback_parts.append(f"{name}: partial ({score:.2f})")
            else:
                feedback_parts.append(f"{name}: ✗")

        return Reward(
            total=round(adjusted, 4),
            breakdown=RewardBreakdown(
                category_score=round(cat_score, 4),
                sentiment_score=round(sent_score, 4),
                priority_score=round(pri_score, 4),
                routing_score=round(dept_score, 4),
            ),
            penalties=round(penalties, 4),
            bonus=round(bonus, 4),
            feedback="; ".join(feedback_parts)
        )
