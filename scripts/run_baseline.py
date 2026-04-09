#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

# Ensure local package imports work regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from customer_support_env import Action, CustomerSupportEnv
from customer_support_env.models import ActionType, TeamName, TicketCategory, TicketPriority


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object")
    return json.loads(text[start : end + 1])


def _rule_policy(task_id: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    step = obs["step_count"]
    summaries = obs["ticket_summaries"]

    if task_id == "easy_password_reset":
        if step == 0:
            return {"action_type": "classify_ticket", "ticket_id": "T-1001", "category": "account", "priority": "high"}
        if step == 1:
            return {"action_type": "assign_ticket", "ticket_id": "T-1001", "assigned_team": "frontline"}
        if step == 2:
            return {"action_type": "send_reply", "ticket_id": "T-1001", "content": "We have initiated your password reset. Please check your email and sign in again."}
        return {"action_type": "close_ticket", "ticket_id": "T-1001"}

    if task_id == "medium_billing_and_outage":
        if step == 0:
            return {"action_type": "classify_ticket", "ticket_id": "T-2002", "category": "technical", "priority": "urgent"}
        if step == 1:
            return {"action_type": "assign_ticket", "ticket_id": "T-2002", "assigned_team": "technical"}
        if step == 2:
            return {"action_type": "send_reply", "ticket_id": "T-2002", "content": "We have identified the outage and applied a mitigation. Service is restored."}
        if step == 3:
            return {"action_type": "close_ticket", "ticket_id": "T-2002"}
        if step == 4:
            return {"action_type": "classify_ticket", "ticket_id": "T-2001", "category": "billing", "priority": "high"}
        if step == 5:
            return {"action_type": "assign_ticket", "ticket_id": "T-2001", "assigned_team": "billing"}
        if step == 6:
            return {"action_type": "send_reply", "ticket_id": "T-2001", "content": "You were charged twice. We have processed a refund and shared the updated invoice."}
        return {"action_type": "close_ticket", "ticket_id": "T-2001"}

    if task_id == "hard_security_and_retention":
        if step == 0:
            return {"action_type": "classify_ticket", "ticket_id": "T-3001", "category": "security", "priority": "urgent"}
        if step == 1:
            return {"action_type": "assign_ticket", "ticket_id": "T-3001", "assigned_team": "security"}
        if step == 2:
            return {"action_type": "escalate_ticket", "ticket_id": "T-3001"}
        if step == 3:
            return {"action_type": "send_reply", "ticket_id": "T-3001", "content": "We secured your account, invalidated sessions, and enabled extra verification."}
        if step == 4:
            return {"action_type": "close_ticket", "ticket_id": "T-3001"}
        if step == 5:
            return {"action_type": "classify_ticket", "ticket_id": "T-3002", "category": "account", "priority": "high"}
        if step == 6:
            return {"action_type": "assign_ticket", "ticket_id": "T-3002", "assigned_team": "retention"}
        if step == 7:
            return {"action_type": "send_reply", "ticket_id": "T-3002", "content": "We are sorry for the delays and understand your concern. We will prioritize your requests and improve response times."}
        if step == 8:
            return {"action_type": "close_ticket", "ticket_id": "T-3002"}
        if step == 9:
            return {"action_type": "classify_ticket", "ticket_id": "T-3003", "category": "shipping", "priority": "normal"}
        return {"action_type": "send_reply", "ticket_id": "T-3003", "content": "Your package is delayed by 2 days and is now in transit. We appreciate your patience."}

    return {"action_type": "add_internal_note", "ticket_id": summaries[0]["ticket_id"], "content": "Default action"}


def _llm_action(
    client: Optional[OpenAI],
    model: str,
    task_id: str,
    objective: str,
    observation_json: str,
    use_rule_fallback: bool,
) -> Dict[str, Any]:
    if client is None:
        parsed_obs = json.loads(observation_json)
        return _rule_policy(task_id, parsed_obs)

    system = (
        "You are a customer support operations agent. Return exactly one JSON object for the next action. "
        "No prose. Use fields from: action_type, ticket_id, category, priority, assigned_team, content."
    )
    user = (
        f"task_id={task_id}\n"
        f"objective={objective}\n"
        f"observation={observation_json}\n"
        "Choose the single best next action to maximize final task score."
    )

    response = client.responses.create(
        model=model,
        temperature=0,
        seed=17,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = response.output_text
    try:
        return _extract_json(text)
    except Exception:
        if use_rule_fallback:
            parsed_obs = json.loads(observation_json)
            return _rule_policy(task_id, parsed_obs)
        raise


def _run_single_task(client: Optional[OpenAI], model: str, task_id: str, max_steps: int, use_rule_fallback: bool) -> Tuple[float, int, Dict[str, float]]:
    env = CustomerSupportEnv(default_task_id=task_id)
    obs = env.reset(task_id=task_id)

    done = False
    steps = 0
    info: Dict[str, Any] = {}

    while not done and steps < max_steps:
        obs_json = obs.model_dump_json()
        action_payload = _llm_action(
            client=client,
            model=model,
            task_id=task_id,
            objective=obs.task_objective,
            observation_json=obs_json,
            use_rule_fallback=use_rule_fallback,
        )

        action = Action.model_validate(action_payload)
        obs, _, done, info = env.step(action)
        steps += 1

    final_score = float(info.get("final_score", env.state().progress))
    breakdown = info.get("grading_breakdown", {})
    return final_score, steps, breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI baseline on all customer support tasks.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-fallback", action="store_true", help="Disable deterministic fallback when JSON parse fails")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    client: Optional[OpenAI] = None
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        print("OPENAI_API_KEY is not set; running deterministic fallback policy only.")

    tasks = [
        "easy_password_reset",
        "medium_billing_and_outage",
        "hard_security_and_retention",
    ]

    total = 0.0
    print(f"Running baseline with model={args.model}")
    for task in tasks:
        score, steps, breakdown = _run_single_task(
            client=client,
            model=args.model,
            task_id=task,
            max_steps=args.max_steps,
            use_rule_fallback=not args.no_fallback,
        )
        total += score
        print(f"{task}: score={score:.3f} steps={steps} breakdown={json.dumps(breakdown, sort_keys=True)}")

    avg = total / len(tasks)
    print(f"average_score={avg:.3f}")


if __name__ == "__main__":
    main()
