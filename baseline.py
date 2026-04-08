"""
Baseline Inference Script — runs an OpenAI model against the Email Triage environment.

Usage:
    export OPENAI_API_KEY="sk-..."
    python baseline.py                          # Run all tasks
    python baseline.py --task task_classify      # Run single task
    python baseline.py --mock                    # Run without API (mock responses)

Produces reproducible baseline scores on all 3 tasks.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from email_triage_env import EmailTriageEnv
from models import Action, ActionItem, EmailCategory, Priority, Sentiment, Department


# ── System Prompt for the Agent ────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an expert email triage assistant for a corporate organization.
You receive emails one at a time and must analyze each one.

For each email, respond with a JSON object containing:

{
  "email_id": "<the email ID from the observation>",
  "category": "<support|sales|internal|spam|urgent|hr|billing|partnership>",
  "sentiment": "<positive|neutral|negative|angry>",
  "priority": "<P0|P1|P2|P3>",
  "department": "<engineering|sales|support|hr|finance|legal|executive|marketing|spam_filter>",
  "action_items": [
    {"description": "...", "assignee": "...", "deadline": "..."}
  ],
  "draft_reply": "...",
  "requires_follow_up": true/false,
  "notes": "..."
}

Priority guide:
- P0: Critical — immediate response needed (outages, security incidents, at-risk accounts)
- P1: High — respond within hours (upset customers, time-sensitive requests)
- P2: Medium — respond within a day (standard business, renewals, meetings)
- P3: Low — respond when possible (spam, newsletters, low-priority info)

Respond ONLY with the JSON object. No extra text."""


def build_agent_message(observation) -> str:
    """Build the user message from an observation."""
    email = observation.current_email
    msg = f"""Email #{observation.step_number + 1} of {observation.max_steps}
Task: {observation.task_description}

--- EMAIL ---
ID: {email.id}
From: {email.sender}
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


def parse_agent_response(response_text: str, email_id: str) -> Action:
    """Parse the LLM response into an Action object."""
    try:
        # Clean up response
        text = response_text.strip()
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
            sentiment=Sentiment(data.get("sentiment", "neutral")) if data.get("sentiment") else None,
            priority=Priority(data.get("priority", "P2")) if data.get("priority") else None,
            department=Department(data.get("department", "support")) if data.get("department") else None,
            action_items=action_items,
            draft_reply=data.get("draft_reply"),
            requires_follow_up=data.get("requires_follow_up", False),
            notes=data.get("notes")
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fallback: return a default action
        print(f"    [WARN] Failed to parse response: {e}")
        return Action(
            email_id=email_id,
            category=EmailCategory.SUPPORT,
            sentiment=Sentiment.NEUTRAL,
            priority=Priority.P2,
            department=Department.SUPPORT,
        )


def run_openai_agent(observation, model: str = "gpt-4o") -> Action:
    """Call OpenAI API to get the agent's action."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    user_msg = build_agent_message(observation)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.1,
        max_tokens=800,
        response_format={"type": "json_object"}
    )

    return parse_agent_response(
        response.choices[0].message.content,
        observation.current_email.id
    )


def run_mock_agent(observation) -> Action:
    """Simple rule-based mock agent for testing without API."""
    email = observation.current_email
    body_lower = email.body.lower()
    subject_lower = email.subject.lower()
    sender = email.sender.lower()

    # Category detection
    category = EmailCategory.SUPPORT
    if any(w in subject_lower + body_lower for w in ["congratulations", "winner", "prize", "click now", "unsubscribe"]):
        category = EmailCategory.SPAM
    elif any(w in subject_lower for w in ["urgent", "critical", "down", "breach", "security"]):
        category = EmailCategory.URGENT
    elif any(w in body_lower for w in ["invoice", "payment", "billing", "renewal"]):
        category = EmailCategory.BILLING
    elif any(w in body_lower for w in ["partnership", "synergy", "collaboration"]):
        category = EmailCategory.PARTNERSHIP
    elif "ourcompany.com" in sender:
        category = EmailCategory.INTERNAL
    elif any(w in body_lower for w in ["recruit", "opportunity", "position", "compensation", "parental", "leave"]):
        category = EmailCategory.HR
    elif any(w in body_lower for w in ["discount", "license", "offer", "pricing"]):
        category = EmailCategory.SALES

    # Sentiment
    sentiment = Sentiment.NEUTRAL
    if any(w in body_lower for w in ["frustrated", "unacceptable", "angry", "kidding", "last chance", "done with"]):
        sentiment = Sentiment.ANGRY
    elif any(w in body_lower for w in ["sorry", "issue", "problem", "error", "discrepancy", "declining"]):
        sentiment = Sentiment.NEGATIVE
    elif any(w in body_lower for w in ["love", "excited", "great", "looking forward", "pleased"]):
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
            f"Thank you for reaching out. We have received your email regarding "
            f"'{email.subject}' and are looking into it. We will get back to you shortly."
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


def run_episode(env: EmailTriageEnv, task_id: str, use_mock: bool = False, model: str = "gpt-4o", verbose: bool = True) -> dict:
    """Run a single episode (one task) and return results."""
    if verbose:
        task_info = env.get_task_info(task_id)
        print(f"\n{'='*65}")
        print(f"  Task: {task_info['name']} ({task_id})")
        print(f"  Difficulty: {task_info['difficulty']} | Emails: {task_info['email_count']}")
        print(f"  Graded fields: {', '.join(task_info['graded_fields'])}")
        print(f"{'='*65}")

    reset_result = env.reset(task_id)
    observation = reset_result.observation
    total_rewards = []
    step_count = 0

    while True:
        step_count += 1
        if verbose:
            print(f"\n  Step {step_count}/{observation.max_steps}: {observation.current_email.subject[:50]}...")

        # Get agent action
        start = time.time()
        if use_mock:
            action = run_mock_agent(observation)
        else:
            action = run_openai_agent(observation, model)
        elapsed = time.time() - start

        if verbose:
            print(f"    Agent: {action.category.value} | {action.priority.value if action.priority else 'N/A'} "
                  f"| {action.department.value if action.department else 'N/A'} ({elapsed:.1f}s)")

        # Step environment
        result = env.step(action)
        total_rewards.append(result.reward.total)

        if verbose:
            print(f"    Reward: {result.reward.total:.4f} | {result.reward.feedback}")

        if result.done:
            break
        observation = result.observation

    # Episode summary
    state = env.state()
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0

    if verbose:
        print(f"\n  Episode Complete:")
        print(f"    Average Reward: {avg_reward:.4f}")
        print(f"    Cumulative Reward: {state.cumulative_reward:.4f}")

    return {
        "task_id": task_id,
        "task_name": env.get_task_info(task_id)["name"],
        "difficulty": env.get_task_info(task_id)["difficulty"],
        "emails_processed": len(total_rewards),
        "average_reward": round(avg_reward, 4),
        "cumulative_reward": round(state.cumulative_reward, 4),
        "per_step_rewards": [round(r, 4) for r in total_rewards],
        "model": model if not use_mock else "mock-agent",
    }


def main():
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv Baseline")
    parser.add_argument("--task", type=str, default=None, help="Run specific task (e.g. task_classify)")
    parser.add_argument("--mock", action="store_true", help="Use mock agent (no API key)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--output", type=str, default="results/baseline_results.json", help="Output file")
    args = parser.parse_args()

    env = EmailTriageEnv()

    tasks = [args.task] if args.task else env.get_task_ids()
    verbose = not args.quiet

    print(f"\n{'#'*65}")
    print(f"  EMAIL TRIAGE OPENENV — BASELINE INFERENCE")
    print(f"  Model: {args.model if not args.mock else 'mock-agent'}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*65}")

    all_results = []
    for task_id in tasks:
        result = run_episode(env, task_id, use_mock=args.mock, model=args.model, verbose=verbose)
        all_results.append(result)

    # Summary
    print(f"\n{'#'*65}")
    print(f"  BASELINE RESULTS SUMMARY")
    print(f"{'#'*65}")
    for r in all_results:
        print(f"  {r['task_id']:25s} ({r['difficulty']:6s}): avg_reward = {r['average_reward']:.4f}")
    overall_avg = sum(r["average_reward"] for r in all_results) / len(all_results)
    print(f"  {'OVERALL':25s}         : avg_reward = {overall_avg:.4f}")
    print(f"{'#'*65}\n")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model if not args.mock else "mock-agent",
        "overall_average_reward": round(overall_avg, 4),
        "task_results": all_results
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
