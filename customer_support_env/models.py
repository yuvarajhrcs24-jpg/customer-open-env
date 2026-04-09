from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    ESCALATED = "escalated"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SECURITY = "security"
    ACCOUNT = "account"
    SHIPPING = "shipping"
    OTHER = "other"


class TeamName(str, Enum):
    FRONTLINE = "frontline"
    BILLING = "billing"
    TECHNICAL = "technical"
    SECURITY = "security"
    RETENTION = "retention"


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    customer_email: str
    subject: str
    body: str
    status: TicketStatus = TicketStatus.OPEN
    priority: TicketPriority = TicketPriority.NORMAL
    category: Optional[TicketCategory] = None
    assigned_team: Optional[TeamName] = None
    tags: List[str] = Field(default_factory=list)
    sla_minutes_remaining: int = 240
    internal_notes: List[str] = Field(default_factory=list)
    draft_reply: Optional[str] = None
    public_replies: List[str] = Field(default_factory=list)


class ActionType(str, Enum):
    OPEN_TICKET = "open_ticket"
    CLASSIFY_TICKET = "classify_ticket"
    ASSIGN_TICKET = "assign_ticket"
    ADD_INTERNAL_NOTE = "add_internal_note"
    DRAFT_REPLY = "draft_reply"
    SEND_REPLY = "send_reply"
    ESCALATE_TICKET = "escalate_ticket"
    CLOSE_TICKET = "close_ticket"


class Action(BaseModel):
    action_type: ActionType
    ticket_id: Optional[str] = None
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    assigned_team: Optional[TeamName] = None
    content: Optional[str] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None


class TicketSummary(BaseModel):
    ticket_id: str
    subject: str
    status: TicketStatus
    priority: TicketPriority
    category: Optional[TicketCategory] = None
    assigned_team: Optional[TeamName] = None
    sla_minutes_remaining: int


class Observation(BaseModel):
    task_id: str
    task_objective: str
    step_count: int
    steps_remaining: int
    queue_size: int
    open_count: int
    escalated_count: int
    closed_count: int
    ticket_summaries: List[TicketSummary]
    available_actions: List[ActionType]
    hints: List[str] = Field(default_factory=list)


class Reward(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    progress: float = Field(ge=0.0, le=1.0)
    penalties: float
    reason: str


class EnvironmentState(BaseModel):
    task_id: str
    task_objective: str
    step_count: int
    max_steps: int
    tickets: Dict[str, Ticket]
    action_history: List[Action] = Field(default_factory=list)
    progress: float = 0.0
    done: bool = False


class TaskResult(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    done: bool
    total_steps: int
    grading_breakdown: Dict[str, float]
    metadata: Dict[str, str] = Field(default_factory=dict)


GradingBreakdown = Dict[str, float]
InfoPayload = Dict[str, str | float | int | bool | Dict[str, float] | List[str]]
ObservationType = Literal["queue", "ticket"]
