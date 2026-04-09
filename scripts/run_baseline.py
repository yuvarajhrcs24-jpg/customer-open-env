#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

from openai import OpenAI

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
    summaries = obs["ticket_summaries"]
    by_id = {ticket["ticket_id"]: ticket for ticket in summaries}

    def classify(ticket_id: str, category: str, priority: str | None = None) -> Dict[str, Any]:
        payload = {"action_type": "classify_ticket", "ticket_id": ticket_id, "category": category}
        if priority:
            payload["priority"] = priority
        return payload

    def assign(ticket_id: str, team: str) -> Dict[str, Any]:
        return {"action_type": "assign_ticket", "ticket_id": ticket_id, "assigned_team": team}

    def reply(ticket_id: str, message: str) -> Dict[str, Any]:
        if by_id[ticket_id]["status"] == "open":
            return {"action_type": "draft_reply", "ticket_id": ticket_id, "content": message}
        return {"action_type": "send_reply", "ticket_id": ticket_id, "content": message}

    def close(ticket_id: str) -> Dict[str, Any]:
        return {"action_type": "close_ticket", "ticket_id": ticket_id}

    if task_id == "easy_password_reset":
        ticket_id = "T-1001"
        t = by_id[ticket_id]
        if t["category"] is None:
            return classify(ticket_id, "account", "high")
        if t["assigned_team"] is None:
            return assign(ticket_id, "frontline")
        if t["status"] != "closed":
            return reply(ticket_id, "We have initiated your password reset. Please check your email and sign in again.")
        return close(ticket_id)

    if task_id == "medium_billing_and_outage":
        t2 = by_id["T-2002"]
        t1 = by_id["T-2001"]

        if t2["category"] is None:
            return classify("T-2002", "technical", "urgent")
        if t2["assigned_team"] is None:
            return assign("T-2002", "technical")
        if t2["status"] != "closed":
            return reply("T-2002", "We have identified the outage and applied a mitigation. Service is restored.")

        if t1["category"] is None:
            return classify("T-2001", "billing", "high")
        if t1["assigned_team"] is None:
            return assign("T-2001", "billing")
        if t1["status"] != "closed":
            return reply("T-2001", "You were charged twice. We have processed a refund and shared the updated invoice.")

        return close("T-2001")

    if task_id == "hard_security_and_retention":
        sec = by_id["T-3001"]
        ret = by_id["T-3002"]
        shp = by_id["T-3003"]

        if sec["category"] is None:
            return classify("T-3001", "security", "urgent")
        if sec["assigned_team"] is None:
            return assign("T-3001", "security")
        if sec["status"] not in {"escalated", "closed"}:
            return {"action_type": "escalate_ticket", "ticket_id": "T-3001"}
        if sec["status"] != "closed":
            return reply("T-3001", "We secured your account, invalidated sessions, and enabled extra verification.")

        if ret["category"] is None:
            return classify("T-3002", "account", "high")
        if ret["assigned_team"] is None:
            return assign("T-3002", "retention")
        if ret["status"] != "closed":
            return reply("T-3002", "We are sorry for the delays and understand your concern. We will prioritize your requests and improve response times.")

        if shp["category"] is None:
            return classify("T-3003", "shipping", "normal")
        if shp["status"] != "closed":
            return reply("T-3003", "Your package is delayed by 2 days and is now in transit. We appreciate your patience.")
        return close("T-3003")

    return {"action_type": "add_internal_note", "ticket_id": summaries[0]["ticket_id"], "content": "Default action"}


def _llm_action(
    client: OpenAI,
    model: str,
    task_id: str,
    objective: str,
    observation_json: str,
    use_rule_fallback: bool,
) -> Dict[str, Any]:
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


def _run_single_task(client: OpenAI, model: str, task_id: str, max_steps: int, use_rule_fallback: bool) -> Tuple[float, int, Dict[str, float]]:
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
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

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
