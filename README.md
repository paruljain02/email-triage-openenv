---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - agent
  - email-triage
---

# Email Triage OpenEnv

A real-world email triage environment where an AI agent must classify, prioritize, route, and respond to corporate emails. Built to the OpenEnv spec for training and evaluating LLM agents on genuine productivity tasks.

## Motivation

Every knowledge worker spends hours daily triaging email. This environment models that workflow: the agent receives emails one at a time from a mixed corporate inbox (support tickets, executive escalations, billing disputes, partnership proposals, spam) and must make fast, accurate triage decisions. It tests classification accuracy, priority judgment, routing knowledge, action item extraction, and communication skills — all in one environment.

## Architecture

```
Agent → receives email (Observation)
     → outputs triage decision (Action)
     → receives score + next email (Reward + Observation)
     → repeats until inbox is empty (done=True)
```

## OpenEnv API

| Method | Description |
|--------|-------------|
| `reset(task_id)` | Initialize a new episode. Returns `StepResult` with `.observation` and `.done`. |
| `step(action)` | Submit triage `Action`. Returns `StepResult(observation, reward, done, info)`. |
| `state()` | Returns full serializable `EnvState`. |
| `close()` | Clean up environment resources. |

## Observation Space

Each observation provides the agent with:

| Field | Type | Description |
|-------|------|-------------|
| `current_email` | `Email` | Full email object (sender, subject, body, attachments, thread history) |
| `inbox_size` | `int` | Total emails in this episode |
| `emails_processed` | `int` | How many already triaged |
| `emails_remaining` | `int` | How many left |
| `task_id` | `str` | Current task identifier |
| `task_description` | `str` | What the agent needs to do |
| `step_number` | `int` | Current step index |
| `max_steps` | `int` | Total steps in episode |

## Action Space

The agent outputs a structured JSON action:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `email_id` | `str` | Yes | Must match the current email's ID |
| `category` | `enum` | Yes | `support\|sales\|internal\|spam\|urgent\|hr\|billing\|partnership` |
| `sentiment` | `enum` | Task 1+ | `positive\|neutral\|negative\|angry` |
| `priority` | `enum` | Task 2+ | `P0\|P1\|P2\|P3` |
| `department` | `enum` | Task 2+ | `engineering\|sales\|support\|hr\|finance\|legal\|executive\|marketing\|spam_filter` |
| `action_items` | `list` | Task 3 | List of `{description, assignee?, deadline?}` |
| `draft_reply` | `str` | Task 3 | Draft customer/internal reply |
| `requires_follow_up` | `bool` | Task 3 | Whether this email needs follow-up |

## Tasks

### Task 1: Email Classification (Easy)
Classify each email by category and detect sender sentiment. 10 emails per episode. Graded on category accuracy (65% weight) and sentiment detection (35% weight). Partial credit for related categories (e.g., `urgent` vs `support`).

### Task 2: Priority Assignment & Routing (Medium)
All of Task 1, plus assign priority level (P0-P3) and route to the correct department. 10 emails. Graded across 4 dimensions with heavier weight on priority (30%) and routing (30%). Extra penalty for under-prioritizing critical P0 items. Bonus for perfect scores.

### Task 3: Full Email Triage with Draft Response (Hard)
Complete triage: classify, prioritize, route, extract action items, and draft an appropriate reply. 12 emails. Graded across 6 dimensions including action item coverage and reply quality. Tests the full range of agent capabilities.

**Expected difficulty:**
- Easy task: frontier models should score 0.80+
- Medium task: frontier models should score 0.65-0.80
- Hard task: frontier models should score 0.50-0.70

## Reward Function

Rewards are computed per-step (per-email) with values in [0.0, 1.0]:

- **Partial credit**: Close-but-wrong answers earn partial scores (e.g., classifying `urgent` as `support` earns 0.4 instead of 0.0; priority off by one level earns 0.5)
- **Graduated penalties**: Under-prioritizing critical items (calling a P0 "P3") is penalized more heavily than over-prioritizing. Replying to spam is penalized. Missing follow-up flags costs points.
- **Bonuses**: Perfect scores on all dimensions earn a 5% bonus.
- **Deterministic**: Same inputs always produce the same scores.

## Environment Variables (MANDATORY)

Before running inference, set these environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # LLM API endpoint
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # Model identifier
export HF_TOKEN="hf_..."                                   # Your HF / API key
```

## Setup & Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Inference Script (MANDATORY — produces structured stdout logs)
```bash
# With API key:
export HF_TOKEN="hf_..."
python inference.py

# Without API key (mock agent for testing):
python inference.py
```

The inference script emits structured stdout in the required format:
```
[START] task=task_classify env=email-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=cat=spam|sent=neutral|pri=P3|dept=spam_filter reward=1.00 done=false error=null
[STEP] step=2 action=cat=urgent|sent=negative|pri=P0|dept=engineering reward=0.83 done=false error=null
...
[END] success=true steps=10 score=0.835 rewards=1.00,0.83,...
```

### Run Validation
```bash
python validate.py
```

### Run Baseline (detailed output)
```bash
python baseline.py --mock           # Mock mode (no API key)
python baseline.py --model gpt-4o   # Live mode
```

### Start HTTP Server
```bash
python server.py
# Server runs at http://localhost:7860
```

### Docker
```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 email-triage-openenv
```

### HTTP API Examples
```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start episode (empty body defaults to task_classify)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{}'

# Start specific task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_full_triage"}'

# Submit action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "action": {"email_id": "...", "category": "support", "sentiment": "negative", "priority": "P1", "department": "support"}}'
```

## Baseline Scores

| Task | Mock Agent | Expected (frontier LLM) |
|------|-----------|------------------------|
| task_classify (easy) | ~0.84 | ~0.90 |
| task_prioritize_route (medium) | ~0.86 | ~0.80 |
| task_full_triage (hard) | ~0.62 | ~0.65 |

## Hugging Face Deployment

1. Create a new HF Space (Docker type)
2. Upload all files to the Space repo
3. Add the tag `openenv` to the Space metadata
4. The Space will auto-build from the Dockerfile and serve on port 7860
5. Verify: `curl -X POST -d '{}' -H 'Content-Type: application/json' https://your-space.hf.space/reset`

## Pre-Submission Checklist

- [ ] `python validate.py` — all checks pass
- [ ] `python inference.py` — completes without error, emits [START]/[STEP]/[END] format
- [ ] `docker build -t email-triage-openenv .` — builds successfully
- [ ] `docker run -p 7860:7860 email-triage-openenv` — starts and responds to `/reset`
- [ ] HF Space deployed and responding to POST `/reset` with `{}` body
- [ ] `openenv.yaml` present with 3+ tasks
- [ ] All rewards in 0.0-1.0 range
- [ ] Graders are deterministic and reproducible

## Project Structure
```
email-triage-openenv/
├── openenv.yaml              # OpenEnv metadata and config
├── models.py                 # Typed models (Observation, Action, Reward, StepResult, EnvState)
├── email_triage_env.py       # Core environment (step/reset/state/close)
├── server.py                 # HTTP server (FastAPI + stdlib fallback)
├── inference.py              # MANDATORY inference script with structured stdout
├── baseline.py               # Detailed baseline script (optional, for development)
├── validate.py               # Spec validation script
├── Dockerfile                # Container (vcpu=2, 8GB optimized)
├── requirements.txt          # Python dependencies
├── data/
│   ├── __init__.py
│   └── email_generator.py    # Synthetic email dataset with ground truth
├── graders/
│   ├── __init__.py
│   ├── grader_utils.py       # Shared scoring (partial credit, penalties, bonuses)
│   ├── classify_grader.py    # Task 1 grader (easy)
│   ├── prioritize_route_grader.py  # Task 2 grader (medium)
│   └── full_triage_grader.py # Task 3 grader (hard)
└── results/                  # Output directory
```

## License

MIT
