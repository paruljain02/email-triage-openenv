"""
OpenEnv Validator — checks that the environment meets all spec requirements.

Run: python validate.py
"""

import sys
import os
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def validate():
    """Run all validation checks."""
    checks_passed = 0
    checks_failed = 0
    total_checks = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal checks_passed, checks_failed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
            print(f"  ✅ {name}")
        else:
            checks_failed += 1
            print(f"  ❌ {name}: {detail}")

    print("\n" + "=" * 60)
    print("  OpenEnv Validation: email-triage")
    print("=" * 60)

    # 1. openenv.yaml exists and is valid
    print("\n📄 openenv.yaml")
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        check("openenv.yaml exists and parses", True)
        check("Has name field", "name" in config)
        check("Has version field", "version" in config)
        check("Has description", "description" in config)
        check("Has tasks list", "tasks" in config and len(config["tasks"]) >= 3,
              f"Found {len(config.get('tasks', []))} tasks, need >= 3")
        check("Has action_space", "action_space" in config)
        check("Has observation_space", "observation_space" in config)
        check("Has reward_range", "reward_range" in config)
    except Exception as e:
        check("openenv.yaml exists and parses", False, str(e))

    # 2. Typed models
    print("\n📐 Pydantic Models")
    try:
        from models import Observation, Action, Reward, StepResult, EnvState
        check("Observation model imports", True)
        check("Action model imports", True)
        check("Reward model imports", True)
        check("StepResult model imports", True)
        check("EnvState model imports", True)

        # Test instantiation
        from models import EmailCategory, Sentiment, Priority, Department, Email, RewardBreakdown
        test_action = Action(
            email_id="test",
            category=EmailCategory.SUPPORT,
            sentiment=Sentiment.NEUTRAL,
            priority=Priority.P2,
            department=Department.SUPPORT
        )
        check("Action instantiation works", test_action.email_id == "test")

        test_reward = Reward(total=0.75, breakdown=RewardBreakdown(category_score=0.8))
        check("Reward instantiation works", 0.0 <= test_reward.total <= 1.0)
    except Exception as e:
        check("Model imports", False, str(e))

    # 3. Environment API
    print("\n🔧 Environment API")
    try:
        from email_triage_env import EmailTriageEnv
        env = EmailTriageEnv()
        check("EmailTriageEnv instantiates", True)

        # reset()
        result = env.reset("task_classify")
        check("reset() returns StepResult", hasattr(result, "observation") and hasattr(result, "done"))
        obs = result.observation
        check("reset() observation has current_email", hasattr(obs, "current_email"))
        check("reset() sets task", obs.task_id == "task_classify")

        # state()
        state = env.state()
        check("state() returns EnvState", hasattr(state, "task_id"))
        check("state() has correct task", state.task_id == "task_classify")
        check("state() done is False after reset", not state.done)

        # step()
        action = Action(
            email_id=obs.current_email.id,
            category=EmailCategory.SUPPORT,
            sentiment=Sentiment.NEUTRAL,
            priority=Priority.P2,
            department=Department.SUPPORT
        )
        result = env.step(action)
        check("step() returns StepResult", hasattr(result, "reward"))
        check("step() reward in [0, 1]", 0.0 <= result.reward.total <= 1.0)
        check("step() reward has breakdown", hasattr(result.reward, "breakdown"))
        check("step() has done flag", isinstance(result.done, bool))

        # Run full episode
        reset_result = env.reset("task_classify")
        obs2 = reset_result.observation
        rewards = []
        while True:
            a = Action(
                email_id=obs2.current_email.id,
                category=EmailCategory.SUPPORT,
                sentiment=Sentiment.NEUTRAL,
            )
            r = env.step(a)
            rewards.append(r.reward.total)
            if r.done:
                break
            obs2 = r.observation

        check("Full episode completes", r.done)
        check("All rewards in [0, 1]", all(0.0 <= r <= 1.0 for r in rewards))
        check("Rewards vary (not constant)", len(set(f"{r:.4f}" for r in rewards)) > 1,
              "All rewards are identical — grader may be broken")

        # close()
        env3 = EmailTriageEnv()
        env3.reset("task_classify")
        env3.close()
        check("close() works without error", env3.done)
    except Exception as e:
        check("Environment API", False, str(e))

    # 4. Three tasks with graders
    print("\n📋 Tasks & Graders")
    try:
        env = EmailTriageEnv()
        task_ids = env.get_task_ids()
        check("Has >= 3 tasks", len(task_ids) >= 3, f"Found {len(task_ids)}")

        difficulties = set()
        for tid in task_ids:
            info = env.get_task_info(tid)
            difficulties.add(info["difficulty"])
            reset_res = env.reset(tid)
            check(f"Task '{tid}' resets cleanly", reset_res is not None and reset_res.observation is not None)

        check("Difficulty range (easy/medium/hard)", difficulties == {"easy", "medium", "hard"},
              f"Found: {difficulties}")
    except Exception as e:
        check("Tasks & Graders", False, str(e))

    # 5. Mandatory files
    print("\n🐳 Deployment & Mandatory Files")
    base = os.path.dirname(__file__)

    dockerfile = os.path.join(base, "Dockerfile")
    check("Dockerfile exists", os.path.exists(dockerfile))

    readme = os.path.join(base, "README.md")
    check("README.md exists", os.path.exists(readme))

    requirements = os.path.join(base, "requirements.txt")
    check("requirements.txt exists", os.path.exists(requirements))

    server = os.path.join(base, "server.py")
    check("server.py exists", os.path.exists(server))

    inference = os.path.join(base, "inference.py")
    check("inference.py exists (MANDATORY)", os.path.exists(inference))

    openenv_yaml = os.path.join(base, "openenv.yaml")
    check("openenv.yaml exists", os.path.exists(openenv_yaml))

    # 6. inference.py format compliance
    print("\n📝 Inference Script Compliance")
    if os.path.exists(inference):
        with open(inference) as f:
            inf_code = f.read()
        check("Uses OpenAI client", "from openai import OpenAI" in inf_code or "import openai" in inf_code)
        check("Reads API_BASE_URL env var", "API_BASE_URL" in inf_code)
        check("Reads MODEL_NAME env var", "MODEL_NAME" in inf_code)
        check("Reads HF_TOKEN env var", "HF_TOKEN" in inf_code)
        check("Emits [START] log", "[START]" in inf_code)
        check("Emits [STEP] log", "[STEP]" in inf_code)
        check("Emits [END] log", "[END]" in inf_code)
        check("Has log_start function", "def log_start" in inf_code)
        check("Has log_step function", "def log_step" in inf_code)
        check("Has log_end function", "def log_end" in inf_code)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION RESULT: {checks_passed}/{total_checks} checks passed")
    if checks_failed == 0:
        print("  ✅ ALL CHECKS PASSED — Environment is spec-compliant!")
    else:
        print(f"  ⚠️  {checks_failed} checks failed")
    print(f"{'=' * 60}\n")

    return checks_failed == 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
