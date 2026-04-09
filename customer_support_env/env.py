from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

from customer_support_env.graders import TaskGrader
from customer_support_env.models import (
    Action,
    ActionType,
    EnvironmentState,
    InfoPayload,
    Observation,
    Reward,
    TeamName,
    Ticket,
    TicketSummary,
    TicketCategory,
)
from customer_support_env.tasks import TaskSpec, build_task_registry, clone_initial_tickets


class CustomerSupportEnv:
    """OpenEnv-compliant environment for customer support ticket operations.
    
    Simulates a customer support queue where an AI agent must:
    - Classify incoming tickets
    - Route to appropriate teams
    - Communicate with customers
    - Escalate high-risk cases
    - Close tickets appropriately
    
    Provides shaped reward signals throughout episodes and deterministic
    grading based on task-specific success criteria.
    
    Attributes:
        tasks: Available task specs by ID
        default_task_id: Default task when not specified
        grader: TaskGrader for computing scores
    """

    def __init__(self, default_task_id: str = "easy_password_reset") -> None:
        self.tasks: Dict[str, TaskSpec] = build_task_registry()
        if default_task_id not in self.tasks:
            raise ValueError(f"Unknown default task: {default_task_id}")

        self.default_task_id = default_task_id
        self.grader = TaskGrader()

        self._state: EnvironmentState | None = None
        self._max_steps = 0
        self._last_progress = 0.0

    def reset(self, task_id: str | None = None) -> Observation:
        """Reset environment to initial state for a task.
        
        Args:
            task_id: Task identifier ("easy_password_reset", "medium_billing_and_outage",
                    "hard_security_and_retention"). If None, uses default_task_id.
        
        Returns:
            Initial Observation with task metadata and ticket queue.
        
        Raises:
            ValueError: If task_id is unknown.
        """
        task_key = task_id or self.default_task_id
        if task_key not in self.tasks:
            raise ValueError(f"Unknown task_id: {task_key}")

        task = self.tasks[task_key]
        self._max_steps = task.max_steps

        self._state = EnvironmentState(
            task_id=task.task_id,
            task_objective=task.objective,
            step_count=0,
            max_steps=task.max_steps,
            tickets=clone_initial_tickets(task),
            action_history=[],
            progress=0.0,
            done=False,
        )
        self._last_progress = 0.0
        return self._build_observation(hints=["Start by prioritizing urgent or security-sensitive tickets."])

    def state(self) -> EnvironmentState:
        """Get complete current environment state.
        
        Returns:
            EnvironmentState with all ticket details, action history, and progress.
        
        Raises:
            RuntimeError: If reset() not called yet.
        """
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return deepcopy(self._state)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, InfoPayload]:
        """Execute one agent action in the environment.
        
        Args:
            action: The Action to execute (must be valid per _validate_action).
        
        Returns:
            - observation: Updated Observation after action
            - reward: Reward with score, progress, penalties, and reason
            - done: Whether episode is complete (goal reached or max steps)
            - info: Dict with step metadata, final_score (if done), and grading_breakdown (if done)
        
        Raises:
            RuntimeError: If reset() not called or episode already done.
        
        Notes:
            Reward is shaped:
            - Base: -0.01 per step (time cost)
            - Progress: difference in grader.partial_score()
            - Invalid action: up to -0.12 penalty
            - Loop detection: -0.05 if repeating same action 3x on same ticket
        """
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode already completed. Call reset() for a new episode.")

        self._state.step_count += 1
        self._state.action_history.append(action)

        action_penalty = 0.0
        reason = "Action applied"

        is_valid, validation_reason = self._validate_action(action)
        if not is_valid:
            action_penalty -= 0.12
            reason = validation_reason
        else:
            self._apply_action(action)

        loop_penalty = self._loop_penalty()
        action_penalty += loop_penalty

        current_progress = self.grader.partial_score(self._state)
        delta_progress = current_progress - self._last_progress

        step_reward = delta_progress - 0.01 + action_penalty

        done = current_progress >= 0.999 or self._state.step_count >= self._max_steps
        self._state.done = done

        info: InfoPayload = {
            "task_id": self._state.task_id,
            "step": self._state.step_count,
            "validation_reason": validation_reason,
            "progress": round(current_progress, 4),
        }

        if done:
            final_result = self.grader.final_result(self._state)
            final_score = final_result.score
            step_reward += final_score - current_progress
            self._state.progress = final_score
            current_progress = final_score
            info["final_score"] = final_score
            info["grading_breakdown"] = final_result.grading_breakdown
        else:
            self._state.progress = current_progress

        step_reward = max(-1.0, min(1.0, step_reward))
        self._last_progress = current_progress

        reward = Reward(
            score=round(step_reward, 4),
            progress=round(current_progress, 4),
            penalties=round(action_penalty, 4),
            reason=reason,
        )

        observation = self._build_observation(hints=self._hints_for_state())
        return observation, reward, done, info

    def _validate_action(self, action: Action) -> Tuple[bool, str]:
        """Validate action is well-formed and legal in current state.
        
        Returns:
            (is_valid, reason_string)
        """
        if action.action_type == ActionType.OPEN_TICKET:
            required = [action.customer_name, action.customer_email, action.subject, action.body]
            if not all(required):
                return False, "open_ticket requires customer_name, customer_email, subject, body"
            return True, "ok"

        if not action.ticket_id or action.ticket_id not in self._state.tickets:
            return False, "Action references unknown ticket_id"

        ticket = self._state.tickets[action.ticket_id]
        if ticket.status.value == "closed" and action.action_type != ActionType.ADD_INTERNAL_NOTE:
            return False, "Cannot perform action on closed ticket"

        if action.action_type == ActionType.CLASSIFY_TICKET and action.category is None:
            return False, f"classify_ticket requires category field"
        if action.action_type == ActionType.ASSIGN_TICKET and action.assigned_team is None:
            return False, f"assign_ticket requires assigned_team field; available: {', '.join(t.value for t in TeamName)}"
        if action.action_type in {ActionType.ADD_INTERNAL_NOTE, ActionType.DRAFT_REPLY} and not action.content:
            return False, f"{action.action_type.value} requires content"
        if action.action_type == ActionType.SEND_REPLY and not (action.content or ticket.draft_reply):
            return False, "send_reply requires content or an existing draft"

        return True, "ok"

    def _apply_action(self, action: Action) -> None:
        """Apply valid action to internal state."""
        if action.action_type == ActionType.OPEN_TICKET:
            new_id = f"T-{9000 + len(self._state.tickets) + 1}"
            self._state.tickets[new_id] = Ticket(
                ticket_id=new_id,
                customer_name=action.customer_name or "Unknown",
                customer_email=action.customer_email or "unknown@example.com",
                subject=action.subject or "No subject",
                body=action.body or "",
            )
            return

        ticket = self._state.tickets[action.ticket_id]

        if action.action_type == ActionType.CLASSIFY_TICKET:
            ticket.category = action.category
            if action.priority is not None:
                ticket.priority = action.priority
            if ticket.status.value == "open":
                ticket.status = ticket.status.PENDING
            return

        if action.action_type == ActionType.ASSIGN_TICKET:
            ticket.assigned_team = action.assigned_team
            if ticket.status.value == "open":
                ticket.status = ticket.status.PENDING
            return

        if action.action_type == ActionType.ADD_INTERNAL_NOTE:
            ticket.internal_notes.append(action.content or "")
            return

        if action.action_type == ActionType.DRAFT_REPLY:
            ticket.draft_reply = action.content or ""
            ticket.status = ticket.status.PENDING
            return

        if action.action_type == ActionType.SEND_REPLY:
            reply_text = action.content or ticket.draft_reply or ""
            ticket.public_replies.append(reply_text)
            ticket.draft_reply = None
            ticket.status = ticket.status.PENDING
            return

        if action.action_type == ActionType.ESCALATE_TICKET:
            ticket.status = ticket.status.ESCALATED
            if ticket.assigned_team is None:
                ticket.assigned_team = TeamName.TECHNICAL
            return

        if action.action_type == ActionType.CLOSE_TICKET:
            if len(ticket.public_replies) == 0:
                # Keep status pending if there was no customer communication.
                ticket.status = ticket.status.PENDING
            else:
                ticket.status = ticket.status.CLOSED
            return

    def _loop_penalty(self) -> float:
        """Detect and penalize looping (same action 3+ times on same ticket)."""
        if len(self._state.action_history) < 3:
            return 0.0

        last_three = self._state.action_history[-3:]
        first = last_three[0]
        # Check if last 3 actions are identical (same type + ticket)
        if all(a.action_type == first.action_type and a.ticket_id == first.ticket_id for a in last_three):
            return -0.05
        return 0.0

    def _build_observation(self, hints: List[str]) -> Observation:
        """Construct Observation from current state."""
        summaries = []
        open_count = 0
        escalated_count = 0
        closed_count = 0

        for ticket in self._state.tickets.values():
            if ticket.status.value in {"open", "pending"}:
                open_count += 1
            if ticket.status.value == "escalated":
                escalated_count += 1
            if ticket.status.value == "closed":
                closed_count += 1

            summaries.append(
                TicketSummary(
                    ticket_id=ticket.ticket_id,
                    subject=ticket.subject,
                    status=ticket.status,
                    priority=ticket.priority,
                    category=ticket.category,
                    assigned_team=ticket.assigned_team,
                    sla_minutes_remaining=ticket.sla_minutes_remaining,
                )
            )

        summaries.sort(key=lambda t: t.sla_minutes_remaining)

        return Observation(
            task_id=self._state.task_id,
            task_objective=self._state.task_objective,
            step_count=self._state.step_count,
            steps_remaining=max(0, self._max_steps - self._state.step_count),
            queue_size=len(summaries),
            open_count=open_count,
            escalated_count=escalated_count,
            closed_count=closed_count,
            ticket_summaries=summaries,
            available_actions=[
                ActionType.OPEN_TICKET,
                ActionType.CLASSIFY_TICKET,
                ActionType.ASSIGN_TICKET,
                ActionType.ADD_INTERNAL_NOTE,
                ActionType.DRAFT_REPLY,
                ActionType.SEND_REPLY,
                ActionType.ESCALATE_TICKET,
                ActionType.CLOSE_TICKET,
            ],
            hints=hints,
        )

    def _hints_for_state(self) -> List[str]:
        """Generate contextual hints for current state."""
        hints: List[str] = []
        if any(ticket.sla_minutes_remaining <= 60 for ticket in self._state.tickets.values()):
            hints.append("At least one ticket has <= 60 minutes remaining SLA; prioritize it.")
        if self._state.task_id == "hard_security_and_retention":
            hints.append("Security incidents should usually be escalated before closure.")
        if self._state.step_count > self._max_steps // 2:
            hints.append("Episode is past halfway. Focus on closure and avoid extra actions.")
        return hints
