"""
Task 1 Grader (Easy): Email Classification
Grades: category + sentiment only.
"""

from models import Action, Reward, RewardBreakdown
from graders.grader_utils import score_category, score_sentiment, apply_penalties


class ClassifyGrader:
    """Grader for the easy classification task."""

    WEIGHTS = {
        "category": 0.65,
        "sentiment": 0.35,
    }

    def grade(self, action: Action, ground_truth: dict) -> Reward:
        cat_score = score_category(action.category.value, ground_truth["category"])
        sent_score = score_sentiment(
            action.sentiment.value if action.sentiment else None,
            ground_truth["sentiment"]
        )

        weighted = (
            cat_score * self.WEIGHTS["category"] +
            sent_score * self.WEIGHTS["sentiment"]
        )

        adjusted, penalties = apply_penalties(weighted, action, ground_truth)

        # Build feedback
        feedback_parts = []
        if cat_score == 1.0:
            feedback_parts.append("Category: correct")
        elif cat_score > 0:
            feedback_parts.append(f"Category: partial ({action.category.value} vs {ground_truth['category']})")
        else:
            feedback_parts.append(f"Category: wrong ({action.category.value} vs {ground_truth['category']})")

        if sent_score == 1.0:
            feedback_parts.append("Sentiment: correct")
        elif sent_score > 0:
            feedback_parts.append(f"Sentiment: partial")
        else:
            feedback_parts.append(f"Sentiment: wrong")

        return Reward(
            total=round(adjusted, 4),
            breakdown=RewardBreakdown(
                category_score=round(cat_score, 4),
                sentiment_score=round(sent_score, 4),
            ),
            penalties=round(penalties, 4),
            feedback="; ".join(feedback_parts)
        )
