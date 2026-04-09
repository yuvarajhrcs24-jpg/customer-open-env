#!/usr/bin/env python3
"""
Pre-submission validation checklist for the Customer Support OpenEnv.

Run this before submitting to verify all requirements are met.
"""

import os
import sys
from pathlib import Path


def check_file(filename: str, description: str) -> bool:
    """Check if a required file exists."""
    exists = Path(filename).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filename}")
    return exists


def check_directory(dirname: str, description: str) -> bool:
    """Check if a required directory exists."""
    exists = Path(dirname).is_dir()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirname}/")
    return exists


def check_content(filename: str, search_text: str, description: str) -> bool:
    """Check if a file contains specific text."""
    try:
        with open(filename) as f:
            content = f.read()
        found = search_text in content
        status = "✅" if found else "❌"
        print(f"{status} {description}: {filename}")
        return found
    except FileNotFoundError:
        print(f"❌ {description}: {filename} (file not found)")
        return False


def main() -> None:
    print("\n" + "=" * 70)
    print("SUBMISSION CHECKLIST - Customer Support OpenEnv")
    print("=" * 70 + "\n")

    all_passed = True

    # ========== Project Structure ==========
    print("📁 PROJECT STRUCTURE:")
    all_passed &= check_file("inference.py", "Main inference script")
    all_passed &= check_file("app.py", "Gradio web interface")
    all_passed &= check_file("openenv.yaml", "OpenEnv metadata")
    all_passed &= check_file("Dockerfile", "Docker container")
    all_passed &= check_file("requirements.txt", "Dependencies")
    all_passed &= check_file("README.md", "Documentation")
    all_passed &= check_directory("customer_support_env", "Environment package")
    all_passed &= check_directory("examples", "Example scripts")
    all_passed &= check_directory("scripts", "Utility scripts")

    # ========== Inference Script Requirements ==========
    print("\n🔧 INFERENCE SCRIPT (inference.py):")
    all_passed &= check_content(
        "inference.py",
        "API_BASE_URL = os.getenv",
        "API_BASE_URL configured via environment variable",
    )
    all_passed &= check_content(
        "inference.py",
        "MODEL_NAME = os.getenv",
        "MODEL_NAME configured via environment variable",
    )
    all_passed &= check_content(
        "inference.py",
        "HF_TOKEN = os.getenv",
        "HF_TOKEN environment variable support",
    )
    all_passed &= check_content(
        "inference.py",
        "from openai import OpenAI",
        "OpenAI client imported",
    )
    all_passed &= check_content(
        "inference.py",
        "OpenAI(",
        "OpenAI client instantiated",
    )
    all_passed &= check_content(
        "inference.py",
        'temperature=0',
        "Temperature set to 0 for reproducibility",
    )
    all_passed &= check_content(
        "inference.py",
        'seed=17',
        "Seed set to 17 for reproducibility",
    )

    # ========== Structured Logging ==========
    print("\n📊 STRUCTURED LOGGING (START/STEP/END):")
    all_passed &= check_content(
        "inference.py",
        '"type": "START"',
        "START log format",
    )
    all_passed &= check_content(
        "inference.py",
        '"type": "STEP"',
        "STEP log format",
    )
    all_passed &= check_content(
        "inference.py",
        '"type": "END"',
        "END log format",
    )

    # ========== Documentation ==========
    print("\n📖 DOCUMENTATION:")
    all_passed &= check_content(
        "README.md",
        "OPENAI_API_KEY",
        "README documents environment variables",
    )
    all_passed &= check_content(
        "README.md",
        "inference.py",
        "README mentions inference.py",
    )
    all_passed &= check_content(
        "README.md",
        "openenv",
        "README mentions OpenEnv",
    )

    # ========== Environment Tests ==========
    print("\n🧪 ENVIRONMENT TESTS:")
    try:
        from customer_support_env import CustomerSupportEnv, Action, Observation, Reward
        print("✅ Environment imports successfully")
        all_passed &= True
    except Exception as e:
        print(f"❌ Environment import failed: {e}")
        all_passed = False

    try:
        env = CustomerSupportEnv(default_task_id="easy_password_reset")
        obs = env.reset()
        print("✅ Environment instantiation works")
        all_passed &= True
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        all_passed = False

    try:
        from customer_support_env import Action
        from customer_support_env.models import ActionType
        action = Action(action_type=ActionType.CLASSIFY_TICKET, ticket_id="T-1001", category="account")
        obs, reward, done, info = env.step(action)
        print("✅ Environment step works")
        all_passed &= True
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        all_passed = False

    # ========== Task Validation ==========
    print("\n📝 TASKS:")
    try:
        tasks = ["easy_password_reset", "medium_billing_and_outage", "hard_security_and_retention"]
        for task_id in tasks:
            env = CustomerSupportEnv(default_task_id=task_id)
            obs = env.reset()
            status = "✅" if obs.task_id == task_id else "❌"
            print(f"{status} Task '{task_id}' instantiates")
            all_passed &= (obs.task_id == task_id)
    except Exception as e:
        print(f"❌ Task validation failed: {e}")
        all_passed = False

    # ========== Final Result ==========
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready for submission!")
        print("\nSubmission Requirements:")
        print("1. Push this repo to GitHub: https://github.com/username/customer-open-env")
        print("2. Deploy to HF Spaces: https://huggingface.co/spaces/username/customer-open-env")
        print("3. Fill in the submission form with both URLs")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
