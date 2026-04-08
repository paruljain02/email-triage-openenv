"""
EmailTriageEnv — Core OpenEnv environment implementation.
Implements the full step() / reset() / state() API.
"""

import json
import copy
from typing import Optional

from models import (
    Email, Observation, Action, Reward, RewardBreakdown,
    StepResult, EnvState, EmailCategory, Priority, Sentiment, Department
)
from data.email_generator import generate_email_batch


# ── Task Definitions ───────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task_classify": {
        "name": "Email Classification",
        "difficulty": "easy",
        "description": (
            "Classify each email by category (support, sales, internal, spam, urgent, hr, "
            "billing, partnership) and detect the sender's sentiment (positive, neutral, "
            "negative, angry). Only category and sentiment fields are graded."
        ),
        "email_count": 10,
        "graded_fields": ["category", "sentiment"],
        "seed": 42,
    },
    "task_prioritize_route": {
        "name": "Priority Assignment & Routing",
        "difficulty": "medium",
        "description": (
            "Classify each email, assign a priority level (P0=critical, P1=high, P2=medium, "
            "P3=low), and route it to the correct department (engineering, sales, support, hr, "
            "finance, legal, executive, marketing, spam_filter). Category, sentiment, priority, "
            "and department are all graded."
        ),
        "email_count": 10,
        "graded_fields": ["category", "sentiment", "priority", "department"],
        "seed": 42,
    },
    "task_full_triage": {
        "name": "Full Email Triage with Draft Response",
        "difficulty": "hard",
        "description": (
            "Perform complete email triage: classify, assign priority, route to department, "
            "extract concrete action items, determine if follow-up is needed, and draft an "
            "appropriate reply. All fields are graded including action item quality and "
            "draft reply appropriateness."
        ),
        "email_count": 12,
        "graded_fields": ["category", "sentiment", "priority", "department", "action_items", "draft_reply"],
        "seed": 42,
    },
}


class EmailTriageEnv:
    """
    Email Triage OpenEnv Environment.

    The agent receives emails one at a time and must triage them:
    classify, prioritize, route, extract action items, and optionally draft replies.

    Three difficulty levels test progressively more capabilities.
    """

    def __init__(self):
        self.task_id: Optional[str] = None
        self.task_config: Optional[dict] = None
        self.inbox: list[dict] = []  # list of {email, ground_truth}
        self.current_index: int = 0
        self.actions_taken: list[Action] = []
        self.rewards_history: list[Reward] = []
        self.cumulative_reward: float = 0.0
        self.done: bool = True
        self._grader = None

    def reset(self, task_id: str = "task_classify") -> StepResult:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'task_classify', 'task_prioritize_route', 'task_full_triage'

        Returns:
            StepResult with initial observation, zero reward, done=False
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_id}. Must be one of {list(TASK_CONFIGS.keys())}")

        self.task_id = task_id
        self.task_config = TASK_CONFIGS[task_id]
        self.inbox = generate_email_batch(
            count=self.task_config["email_count"],
            seed=self.task_config["seed"]
        )
        self.current_index = 0
        self.actions_taken = []
        self.rewards_history = []
        self.cumulative_reward = 0.0
        self.done = False

        # Load the appropriate grader
        self._grader = self._load_grader(task_id)

        return StepResult(
            observation=self._make_observation(),
            reward=Reward(total=0.0, feedback="Episode started"),
            done=False,
            info={"task_id": task_id, "task_name": self.task_config["name"]}
        )

    def close(self):
        """Clean up environment resources."""
        self.inbox = []
        self.actions_taken = []
        self.rewards_history = []
        self.done = True

    def step(self, action: Action) -> StepResult:
        """
        Take an action (triage decision) for the current email.

        Args:
            action: The agent's triage decision

        Returns:
            StepResult with next observation, reward, done flag, and info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if action.email_id != self.inbox[self.current_index]["email"].id:
            # Wrong email ID — penalty
            reward = Reward(
                total=0.0,
                breakdown=RewardBreakdown(),
                penalties=0.2,
                feedback=f"Action email_id '{action.email_id}' doesn't match current email '{self.inbox[self.current_index]['email'].id}'"
            )
        else:
            # Grade the action
            ground_truth = self.inbox[self.current_index]["ground_truth"]
            reward = self._grader.grade(action, ground_truth)

        self.actions_taken.append(action)
        self.rewards_history.append(reward)
        self.cumulative_reward += reward.total

        # Advance to next email
        self.current_index += 1

        # Check if episode is done
        if self.current_index >= len(self.inbox):
            self.done = True
            observation = None
            info = self._make_summary()
        else:
            observation = self._make_observation()
            info = {
                "step": self.current_index,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "last_reward": round(reward.total, 4)
            }

        return StepResult(
            observation=observation,
            reward=reward,
            done=self.done,
            info=info
        )

    def state(self) -> EnvState:
        """
        Return the full serializable state of the environment.
        """
        return EnvState(
            task_id=self.task_id or "",
            step_number=self.current_index,
            max_steps=len(self.inbox) if self.inbox else 0,
            inbox=[item["email"] for item in self.inbox],
            processed_emails=[a.email_id for a in self.actions_taken],
            actions_taken=self.actions_taken,
            rewards_history=self.rewards_history,
            cumulative_reward=round(self.cumulative_reward, 4),
            done=self.done
        )

    def get_task_ids(self) -> list[str]:
        """Return available task IDs."""
        return list(TASK_CONFIGS.keys())

    def get_task_info(self, task_id: str) -> dict:
        """Return info about a specific task."""
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_id}")
        config = TASK_CONFIGS[task_id]
        return {
            "task_id": task_id,
            "name": config["name"],
            "difficulty": config["difficulty"],
            "description": config["description"],
            "email_count": config["email_count"],
            "graded_fields": config["graded_fields"]
        }

    # ── Private Helpers ────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        """Build observation for the current email."""
        current = self.inbox[self.current_index]
        return Observation(
            current_email=current["email"],
            inbox_size=len(self.inbox),
            emails_processed=self.current_index,
            emails_remaining=len(self.inbox) - self.current_index,
            task_id=self.task_id,
            task_description=self.task_config["description"],
            step_number=self.current_index,
            max_steps=len(self.inbox)
        )

    def _make_summary(self) -> dict:
        """Build end-of-episode summary."""
        avg_reward = self.cumulative_reward / len(self.inbox) if self.inbox else 0
        return {
            "episode_complete": True,
            "total_emails": len(self.inbox),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "average_reward": round(avg_reward, 4),
            "task_id": self.task_id,
            "task_name": self.task_config["name"],
            "difficulty": self.task_config["difficulty"]
        }

    def _load_grader(self, task_id: str):
        """Dynamically load the appropriate grader for the task."""
        if task_id == "task_classify":
            from graders.classify_grader import ClassifyGrader
            return ClassifyGrader()
        elif task_id == "task_prioritize_route":
            from graders.prioritize_route_grader import PrioritizeRouteGrader
            return PrioritizeRouteGrader()
        elif task_id == "task_full_triage":
            from graders.full_triage_grader import FullTriageGrader
            return FullTriageGrader()
        else:
            raise ValueError(f"No grader for task: {task_id}")
