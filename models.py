"""
Typed models for the Email Triage OpenEnv environment.
Uses dataclasses for zero-dependency compatibility.
When deploying with pip install, swap to Pydantic for full validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
import json


# ── Enums ──────────────────────────────────────────────────────────────────

class EmailCategory(str, Enum):
    SUPPORT = "support"
    SALES = "sales"
    INTERNAL = "internal"
    SPAM = "spam"
    URGENT = "urgent"
    HR = "hr"
    BILLING = "billing"
    PARTNERSHIP = "partnership"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class Department(str, Enum):
    ENGINEERING = "engineering"
    SALES = "sales"
    SUPPORT = "support"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    EXECUTIVE = "executive"
    MARKETING = "marketing"
    SPAM_FILTER = "spam_filter"


# ── Helper mixin ───────────────────────────────────────────────────────────

class Serializable:
    """Mixin for JSON serialization of dataclasses."""
    def model_dump(self):
        """Serialize to dict (Pydantic-compatible name)."""
        def _convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        d = {}
        for k, v in self.__dict__.items():
            d[k] = _convert(v)
        return d

    def to_json(self):
        return json.dumps(self.model_dump(), indent=2, default=str)


# ── Observation Models ─────────────────────────────────────────────────────

@dataclass
class EmailAttachment(Serializable):
    filename: str = ""
    file_type: str = ""
    size_kb: int = 0


@dataclass
class ThreadMessage(Serializable):
    sender: str = ""
    body: str = ""
    timestamp: str = ""


@dataclass
class Email(Serializable):
    id: str = ""
    sender: str = ""
    sender_domain: str = ""
    recipient: str = ""
    subject: str = ""
    body: str = ""
    timestamp: str = ""
    has_attachments: bool = False
    attachments: list = field(default_factory=list)
    thread_history: list = field(default_factory=list)
    is_reply: bool = False
    cc: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Observation(Serializable):
    current_email: Optional[Email] = None
    inbox_size: int = 0
    emails_processed: int = 0
    emails_remaining: int = 0
    task_id: str = ""
    task_description: str = ""
    step_number: int = 0
    max_steps: int = 0


# ── Action Models ──────────────────────────────────────────────────────────

@dataclass
class ActionItem(Serializable):
    description: str = ""
    assignee: Optional[str] = None
    deadline: Optional[str] = None


@dataclass
class Action(Serializable):
    email_id: str = ""
    category: EmailCategory = EmailCategory.SUPPORT
    sentiment: Optional[Sentiment] = None
    priority: Optional[Priority] = None
    department: Optional[Department] = None
    action_items: list = field(default_factory=list)
    draft_reply: Optional[str] = None
    requires_follow_up: bool = False
    notes: Optional[str] = None


# ── Reward Models ──────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown(Serializable):
    category_score: float = 0.0
    sentiment_score: float = 0.0
    priority_score: float = 0.0
    routing_score: float = 0.0
    action_items_score: float = 0.0
    draft_reply_score: float = 0.0


@dataclass
class Reward(Serializable):
    total: float = 0.0
    breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    penalties: float = 0.0
    bonus: float = 0.0
    feedback: str = ""


# ── State Model ────────────────────────────────────────────────────────────

@dataclass
class EnvState(Serializable):
    task_id: str = ""
    step_number: int = 0
    max_steps: int = 0
    inbox: list = field(default_factory=list)
    processed_emails: list = field(default_factory=list)
    actions_taken: list = field(default_factory=list)
    rewards_history: list = field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False


# ── Step Result ────────────────────────────────────────────────────────────

@dataclass
class StepResult(Serializable):
    observation: Optional[Observation] = None
    reward: Reward = field(default_factory=Reward)
    done: bool = False
    info: dict = field(default_factory=dict)
