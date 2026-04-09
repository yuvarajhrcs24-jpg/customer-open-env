"""Customer Support Ticket OpenEnv Environment

A complete, real-world OpenEnv environment that simulates customer support
ticket triage and resolution workflows.

Main components:
- CustomerSupportEnv: Main environment class (reset/step/state API)
- Action, Observation, Reward: Typed interfaces (Pydantic models)
- 3 deterministic tasks with increasing difficulty
- Shaped reward function with trajectory-level signals
- Deterministic graders with partial progress and final breakdowns

Example usage:
	from customer_support_env import CustomerSupportEnv, Action
	from customer_support_env.models import ActionType, TeamName, TicketCategory
    
	env = CustomerSupportEnv(default_task_id="easy_password_reset")
	obs = env.reset()
    
	action = Action(
		action_type=ActionType.CLASSIFY_TICKET,
		ticket_id="T-1001",
		category=TicketCategory.ACCOUNT
	)
	observation, reward, done, info = env.step(action)
"""

from customer_support_env.env import CustomerSupportEnv
from customer_support_env.models import Action, Observation, Reward

__all__ = ["CustomerSupportEnv", "Action", "Observation", "Reward"]
