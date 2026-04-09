from __future__ import annotations

import os
import json
from typing import Any

import gradio as gr

from customer_support_env import Action, CustomerSupportEnv


"""Gradio web interface for Customer Support OpenEnv.

Provides interactive exploration of the environment:
- Select difficulty level (easy, medium, hard)
- Reset task and see initial observation
- Submit JSON actions and observe results
- Monitor reward signals and progress
"""

env = CustomerSupportEnv(default_task_id="easy_password_reset")


def start_task(task_id: str) -> tuple[str, str]:
    try:
        obs = env.reset(task_id=task_id)
        return obs.model_dump_json(indent=2), "✓ Task reset. Submit an action JSON below."
    except Exception as e:
        return "", f"❌ Error: {e}"


def apply_action(action_json: str) -> tuple[str, str]:
    """Execute one action and show result."""
    try:
        payload: dict[str, Any] = json.loads(action_json)
        action = Action.model_validate(payload)
        obs, reward, done, info = env.step(action)
        result = {
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
        return obs.model_dump_json(indent=2), json.dumps(result, indent=2)
    except json.JSONDecodeError as e:
        return "", f"❌ Invalid JSON: {str(e)[:100]}"
    except Exception as exc:
        return "", f"❌ Error: {str(exc)[:200]}"


with gr.Blocks(title="Customer Support OpenEnv") as demo:
    gr.Markdown("""# 📨 Customer Support OpenEnv

Interactive environment for customer support ticket operations.

**Instructions:**
1. Select a difficulty level
2. Click "Reset Task" to initialize
3. Submit JSON actions to step the environment
4. Monitor rewards and progress in real-time

**Sample action:** `{"action_type": "classify_ticket", "ticket_id": "T-1001", "category": "account"}`
""")

    with gr.Row():
        task_picker = gr.Dropdown(
            choices=["easy_password_reset", "medium_billing_and_outage", "hard_security_and_retention"],
            value="easy_password_reset",
            label="Task",
        )
        reset_btn = gr.Button("Reset Task")

    observation = gr.Code(label="Observation", language="json")
    log = gr.Code(label="Step Result", language="json")

    action_input = gr.Code(
        value='{"action_type": "classify_ticket", "ticket_id": "T-1001", "category": "account"}',
        label="Action JSON",
        language="json",
    )
    step_btn = gr.Button("Step")

    reset_btn.click(start_task, inputs=[task_picker], outputs=[observation, log])
    step_btn.click(apply_action, inputs=[action_input], outputs=[observation, log])


if __name__ == "__main__":
    # Spaces should run without a tunnel/share proxy to avoid abuse flags.
    share_enabled = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share_enabled)
