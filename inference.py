"""
Inference Script — Email Triage OpenEnv
=======================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Will use mock agent if openai not installed

# ── Add project root to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from email_triage_env import EmailTriageEnv
from models import (
    Action, ActionItem, EmailCategory, Priority, Sentiment, Department
)

# ── Environment Variables (MANDATORY) ─────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "email-triage"
MAX_STEPS_PER_TASK = 15  # safety cap (tasks have 10-12 emails)
TEMPERATURE = 0.1
MAX_TOKENS = 600

# ── Logging Functions (MANDATORY FORMAT) ──────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant. You receive a corporate email and must analyze it.

Respond with ONLY a valid JSON object (no markdown, no extra text):

{
  "email_id": "<exact email ID from the input>",
  "category": "<support|sales|internal|spam|urgent|hr|billing|partnership>",
  "sentiment": "<positive|neutral|negative|angry>",
  "priority": "<P0|P1|P2|P3>",
  "department": "<engineering|sales|support|hr|finance|legal|executive|marketing|spam_filter>",
  "action_items": [{"description": "action text"}],
  "draft_reply": "your draft reply here",
  "requires_follow_up": true
}

Priority guide:
- P0: Critical (outages, security, at-risk accounts)
- P1: High (upset customers, time-sensitive)
- P2: Medium (standard business, meetings)
- P3: Low (spam, newsletters)

IMPORTANT: respond with ONLY the JSON object. No markdown fences, no explanation.
""").strip()


# ── Build prompt from observation ─────────────────────────────────────────

def build_user_prompt(observation) -> str:
    email = observation.current_email
    msg = f"""Email {observation.emails_processed + 1} of {observation.max_steps}
Task: {observation.task_description}

--- EMAIL ---
ID: {email.id}
From: {email.sender} ({email.sender_domain})
To: {email.recipient}
Subject: {email.subject}
Date: {email.timestamp}
Has Attachments: {email.has_attachments}
Is Reply: {email.is_reply}
CC: {', '.join(email.cc) if email.cc else 'none'}

Body:
{email.body}"""

    if email.attachments:
        msg += "\n\nAttachments:"
        for att in email.attachments:
            msg += f"\n  - {att.filename} ({att.file_type}, {att.size_kb}KB)"

    if email.thread_history:
        msg += "\n\nThread History:"
        for th in email.thread_history:
            msg += f"\n  [{th.timestamp}] {th.sender}: {th.body[:200]}"

    return msg


# ── Parse LLM response into Action ───────────────────────────────────────

def parse_llm_response(response_text: str, email_id: str) -> Action:
    """Parse LLM JSON response into a typed Action."""
    try:
        text = response_text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        action_items = []
        for item in data.get("action_items", []):
            if isinstance(item, dict):
                action_items.append(ActionItem(
                    description=item.get("description", ""),
                    assignee=item.get("assignee"),
                    deadline=item.get("deadline")
                ))
            elif isinstance(item, str):
                action_items.append(ActionItem(description=item))

        return Action(
            email_id=data.get("email_id", email_id),
            category=EmailCategory(data.get("category", "support")),
            sentiment=Sentiment(data.get("sentiment", "neutral")) if data.get("sentiment") else Sentiment.NEUTRAL,
            priority=Priority(data.get("priority", "P2")) if data.get("priority") else Priority.P2,
            department=Department(data.get("department", "support")) if data.get("department") else Department.SUPPORT,
            action_items=action_items,
            draft_reply=data.get("draft_reply"),
            requires_follow_up=data.get("requires_follow_up", False),
            notes=data.get("notes"),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fallback: return a default action
        return Action(
            email_id=email_id,
            category=EmailCategory.SUPPORT,
            sentiment=Sentiment.NEUTRAL,
            priority=Priority.P2,
            department=Department.SUPPORT,
        )


# ── LLM Call ──────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, observation) -> Action:
    """Call the LLM to get an action for the current observation."""
    user_prompt = build_user_prompt(observation)
    email_id = observation.current_email.id

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text, email_id)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        # Return a reasonable default
        return Action(
            email_id=email_id,
            category=EmailCategory.SUPPORT,
            sentiment=Sentiment.NEUTRAL,
            priority=Priority.P2,
            department=Department.SUPPORT,
        )


# ── Mock Agent (for testing without API) ──────────────────────────────────

def get_mock_action(observation) -> Action:
    """Simple rule-based agent for testing without an API key."""
    email = observation.current_email
    body_lower = email.body.lower()
    subject_lower = email.subject.lower()
    sender = email.sender.lower()

    # Category
    category = EmailCategory.SUPPORT
    if any(w in subject_lower + body_lower for w in ["congratulations", "winner", "prize", "click now", "unsubscribe"]):
        category = EmailCategory.SPAM
    elif any(w in subject_lower for w in ["urgent", "critical", "down", "breach", "security"]):
        category = EmailCategory.URGENT
    elif any(w in body_lower for w in ["invoice", "payment", "billing"]):
        category = EmailCategory.BILLING
    elif any(w in body_lower for w in ["partnership", "synergy"]):
        category = EmailCategory.PARTNERSHIP
    elif "ourcompany.com" in sender:
        category = EmailCategory.INTERNAL
    elif any(w in body_lower for w in ["recruit", "opportunity", "position", "parental", "leave policy"]):
        category = EmailCategory.HR
    elif any(w in body_lower for w in ["discount", "license", "renewal", "offer"]):
        category = EmailCategory.SALES

    # Sentiment
    sentiment = Sentiment.NEUTRAL
    if any(w in body_lower for w in ["frustrated", "unacceptable", "kidding", "last chance", "done with"]):
        sentiment = Sentiment.ANGRY
    elif any(w in body_lower for w in ["issue", "problem", "error", "discrepancy", "declining", "can't"]):
        sentiment = Sentiment.NEGATIVE
    elif any(w in body_lower for w in ["love", "excited", "looking forward", "pleased", "open to"]):
        sentiment = Sentiment.POSITIVE

    # Priority
    priority = Priority.P2
    if category == EmailCategory.URGENT or sentiment == Sentiment.ANGRY:
        priority = Priority.P0
    elif category == EmailCategory.SPAM:
        priority = Priority.P3
    elif sentiment == Sentiment.NEGATIVE:
        priority = Priority.P1

    # Department
    dept_map = {
        EmailCategory.SUPPORT: Department.SUPPORT,
        EmailCategory.SALES: Department.SALES,
        EmailCategory.INTERNAL: Department.EXECUTIVE,
        EmailCategory.SPAM: Department.SPAM_FILTER,
        EmailCategory.URGENT: Department.ENGINEERING,
        EmailCategory.HR: Department.HR,
        EmailCategory.BILLING: Department.FINANCE,
        EmailCategory.PARTNERSHIP: Department.SALES,
    }
    department = dept_map.get(category, Department.SUPPORT)

    # Action items
    action_items = [ActionItem(description="Review and respond to this email")]
    if category == EmailCategory.URGENT:
        action_items = [
            ActionItem(description="Escalate to senior team immediately"),
            ActionItem(description="Investigate root cause"),
        ]
    elif category == EmailCategory.SPAM:
        action_items = [ActionItem(description="Mark as spam and block sender")]

    # Draft reply
    draft = None
    requires_follow_up = category not in (EmailCategory.SPAM,)
    if category != EmailCategory.SPAM:
        draft = (
            f"Thank you for reaching out regarding '{email.subject}'. "
            f"We have received your email and are looking into it. "
            f"We will get back to you shortly with a resolution."
        )

    return Action(
        email_id=email.id,
        category=category,
        sentiment=sentiment,
        priority=priority,
        department=department,
        action_items=action_items,
        draft_reply=draft,
        requires_follow_up=requires_follow_up,
    )


# ── Format action for logging ────────────────────────────────────────────

def format_action_str(action: Action) -> str:
    """Compact single-line action string for [STEP] log."""
    parts = [f"cat={action.category.value}"]
    if action.sentiment:
        parts.append(f"sent={action.sentiment.value}")
    if action.priority:
        parts.append(f"pri={action.priority.value}")
    if action.department:
        parts.append(f"dept={action.department.value}")
    return "|".join(parts)


# ── Run one episode ───────────────────────────────────────────────────────

def run_episode(env: EmailTriageEnv, task_id: str, client, use_mock: bool) -> dict:
    """Run a single task episode and emit structured logs."""
    task_info = env.get_task_info(task_id)
    task_name = task_id

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)
        observation = result.observation

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if result.done:
                break

            # Get action from LLM or mock
            if use_mock:
                action = get_mock_action(observation)
            else:
                action = get_llm_action(client, observation)

            # Step the environment
            result = env.step(action)
            reward = result.reward.total
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = format_action_str(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            observation = result.observation

        # Compute normalized score (average reward, already in [0,1])
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.3  # reasonable threshold for success

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "task_name": task_info["name"],
        "difficulty": task_info["difficulty"],
        "steps": steps_taken,
        "score": round(score, 4),
        "rewards": rewards,
        "success": success,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    use_mock = not API_KEY or OpenAI is None
    if use_mock:
        print("[DEBUG] No API key or openai package not found. Running with mock agent.", file=sys.stderr, flush=True)
        client = None
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = ["task_classify", "task_prioritize_route", "task_full_triage"]
    all_results = []

    for task_id in task_ids:
        env = EmailTriageEnv()
        result = run_episode(env, task_id, client, use_mock)
        all_results.append(result)

    # Print summary to stderr (not stdout, to keep stdout clean for log format)
    print("", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print("  INFERENCE SUMMARY", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    for r in all_results:
        print(f"  {r['task_id']:25s} ({r['difficulty']:6s}): score={r['score']:.4f} success={r['success']}",
              file=sys.stderr)
    overall = sum(r["score"] for r in all_results) / len(all_results)
    print(f"  {'OVERALL':25s}         : score={overall:.4f}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


if __name__ == "__main__":
    main()
