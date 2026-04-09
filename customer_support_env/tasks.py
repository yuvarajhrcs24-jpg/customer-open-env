from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

from customer_support_env.models import Ticket


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    title: str
    objective: str
    success_criteria: str
    max_steps: int
    initial_tickets: List[Ticket]


def _easy_task() -> TaskSpec:
    return TaskSpec(
        task_id="easy_password_reset",
        difficulty="easy",
        title="Password Reset Triage",
        objective=(
            "Resolve a single account access ticket end-to-end: classify it, assign to the right team, "
            "reply with a reset confirmation, and close the ticket."
        ),
        success_criteria=(
            "Ticket T-1001 is categorized as account, assigned to frontline, receives a customer reply "
            "mentioning password reset, and is closed."
        ),
        max_steps=12,
        initial_tickets=[
            Ticket(
                ticket_id="T-1001",
                customer_name="Ava Johnson",
                customer_email="ava@example.com",
                subject="Can't sign in",
                body="I forgot my password and cannot access my dashboard.",
                sla_minutes_remaining=180,
            )
        ],
    )


def _medium_task() -> TaskSpec:
    return TaskSpec(
        task_id="medium_billing_and_outage",
        difficulty="medium",
        title="Billing + Outage Queue",
        objective=(
            "Handle a mixed queue by prioritizing urgent technical outage first, then resolve a billing refund, "
            "while avoiding unnecessary escalations."
        ),
        success_criteria=(
            "T-2002 outage is marked urgent, assigned technical, replied and closed first. "
            "T-2001 billing is assigned billing with a refund response and closed."
        ),
        max_steps=22,
        initial_tickets=[
            Ticket(
                ticket_id="T-2001",
                customer_name="Maya Lee",
                customer_email="maya@example.com",
                subject="Charged twice",
                body="I was charged twice for the Pro plan this month.",
                sla_minutes_remaining=220,
            ),
            Ticket(
                ticket_id="T-2002",
                customer_name="Leo Patel",
                customer_email="leo@example.com",
                subject="Production down",
                body="Our team cannot access the app. This is blocking all agents right now.",
                sla_minutes_remaining=45,
            ),
            Ticket(
                ticket_id="T-2003",
                customer_name="Noah Cruz",
                customer_email="noah@example.com",
                subject="How to change profile photo",
                body="Where do I update my avatar?",
                sla_minutes_remaining=400,
            ),
        ],
    )


def _hard_task() -> TaskSpec:
    return TaskSpec(
        task_id="hard_security_and_retention",
        difficulty="hard",
        title="Security Incident with Retention Risk",
        objective=(
            "Process a high-risk queue: escalate a suspicious login to security, de-escalate a churn-risk account, "
            "and resolve routine shipping noise while respecting priorities and SLAs."
        ),
        success_criteria=(
            "Security ticket is escalated correctly before closure, churn-risk ticket gets retention assignment and "
            "empathetic response, and low-priority shipping ticket is handled without blocking critical work."
        ),
        max_steps=30,
        initial_tickets=[
            Ticket(
                ticket_id="T-3001",
                customer_name="Iris Chen",
                customer_email="iris@example.com",
                subject="Unknown login attempt",
                body="I got a login alert from another country. Please secure my account.",
                sla_minutes_remaining=30,
            ),
            Ticket(
                ticket_id="T-3002",
                customer_name="Ethan Brown",
                customer_email="ethan@example.com",
                subject="Thinking of canceling",
                body="Support has been slow. I might cancel unless this improves.",
                sla_minutes_remaining=120,
            ),
            Ticket(
                ticket_id="T-3003",
                customer_name="Sofia Kim",
                customer_email="sofia@example.com",
                subject="Package delayed",
                body="My shipment is delayed by 2 days. Any update?",
                sla_minutes_remaining=320,
            ),
        ],
    )


def build_task_registry() -> Dict[str, TaskSpec]:
    tasks = [_easy_task(), _medium_task(), _hard_task()]
    return {task.task_id: task for task in tasks}


def clone_initial_tickets(task: TaskSpec) -> Dict[str, Ticket]:
    return {ticket.ticket_id: deepcopy(ticket) for ticket in task.initial_tickets}
