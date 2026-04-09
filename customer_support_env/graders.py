from __future__ import annotations

from typing import Dict, List

from customer_support_env.models import Action, ActionType, EnvironmentState, TaskResult, TeamName, TicketCategory, TicketPriority


def _history_index(history: List[Action], action_type: ActionType, ticket_id: str) -> int:
    for index, action in enumerate(history):
        if action.action_type == action_type and action.ticket_id == ticket_id:
            return index
    return 10_000


def _contains_any(texts: List[str], words: List[str]) -> bool:
    joined = " ".join(texts).lower()
    return any(word in joined for word in words)


class TaskGrader:
    def partial_score(self, state: EnvironmentState) -> float:
        task_id = state.task_id
        if task_id == "easy_password_reset":
            return self._easy_progress(state)
        if task_id == "medium_billing_and_outage":
            return self._medium_progress(state)
        if task_id == "hard_security_and_retention":
            return self._hard_progress(state)
        return 0.0

    def final_result(self, state: EnvironmentState) -> TaskResult:
        task_id = state.task_id
        if task_id == "easy_password_reset":
            breakdown = self._easy_breakdown(state)
        elif task_id == "medium_billing_and_outage":
            breakdown = self._medium_breakdown(state)
        elif task_id == "hard_security_and_retention":
            breakdown = self._hard_breakdown(state)
        else:
            breakdown = {"unknown_task": 0.0}

        total = max(0.0, min(1.0, sum(breakdown.values())))
        return TaskResult(
            task_id=task_id,
            score=round(total, 4),
            done=state.done,
            total_steps=state.step_count,
            grading_breakdown=breakdown,
            metadata={"task": task_id},
        )

    def _easy_progress(self, state: EnvironmentState) -> float:
        return min(1.0, sum(self._easy_breakdown(state).values()))

    def _medium_progress(self, state: EnvironmentState) -> float:
        return min(1.0, sum(self._medium_breakdown(state).values()))

    def _hard_progress(self, state: EnvironmentState) -> float:
        return min(1.0, sum(self._hard_breakdown(state).values()))

    def _easy_breakdown(self, state: EnvironmentState) -> Dict[str, float]:
        ticket = state.tickets["T-1001"]
        score = {
            "classified_account": 0.2 if ticket.category == TicketCategory.ACCOUNT else 0.0,
            "assigned_frontline": 0.2 if ticket.assigned_team == TeamName.FRONTLINE else 0.0,
            "sent_password_reply": 0.3
            if _contains_any(ticket.public_replies, ["password", "reset", "signin", "sign in"])
            else 0.0,
            "closed": 0.3 if ticket.status.value == "closed" else 0.0,
        }
        return score

    def _medium_breakdown(self, state: EnvironmentState) -> Dict[str, float]:
        billing = state.tickets["T-2001"]
        outage = state.tickets["T-2002"]

        outage_first = _history_index(state.action_history, ActionType.CLOSE_TICKET, "T-2002")
        billing_first = _history_index(state.action_history, ActionType.CLOSE_TICKET, "T-2001")

        score = {
            "outage_priority_and_team": 0.25
            if outage.priority == TicketPriority.URGENT and outage.assigned_team == TeamName.TECHNICAL
            else 0.0,
            "outage_resolved_first": 0.15 if outage_first < billing_first else 0.0,
            "outage_closed": 0.2 if outage.status.value == "closed" else 0.0,
            "billing_assigned_and_refund_reply": 0.2
            if billing.assigned_team == TeamName.BILLING
            and _contains_any(billing.public_replies, ["refund", "charged", "invoice"])
            else 0.0,
            "billing_closed": 0.15 if billing.status.value == "closed" else 0.0,
            "avoid_unnecessary_escalation": 0.05
            if state.tickets["T-2003"].status.value != "escalated"
            else 0.0,
        }
        return score

    def _hard_breakdown(self, state: EnvironmentState) -> Dict[str, float]:
        security = state.tickets["T-3001"]
        retention = state.tickets["T-3002"]
        shipping = state.tickets["T-3003"]

        security_escalated = _history_index(state.action_history, ActionType.ESCALATE_TICKET, "T-3001") < 10_000
        security_closed = _history_index(state.action_history, ActionType.CLOSE_TICKET, "T-3001")
        secure_order_bonus = 0.15 if security_escalated and _history_index(state.action_history, ActionType.ESCALATE_TICKET, "T-3001") < security_closed else 0.0

        score = {
            "security_classified_escalated": 0.2
            if security.category == TicketCategory.SECURITY
            and security.assigned_team == TeamName.SECURITY
            and security.status.value in {"escalated", "closed"}
            else 0.0,
            "security_ordering": secure_order_bonus,
            "security_closed": 0.15 if security.status.value == "closed" else 0.0,
            "retention_assignment": 0.15 if retention.assigned_team == TeamName.RETENTION else 0.0,
            "retention_empathetic_reply": 0.15
            if _contains_any(retention.public_replies, ["sorry", "understand", "improve", "priority"])
            else 0.0,
            "retention_closed": 0.1 if retention.status.value == "closed" else 0.0,
            "shipping_handled": 0.1
            if shipping.status.value in {"pending", "closed"} and len(shipping.public_replies) > 0
            else 0.0,
        }
        return score
