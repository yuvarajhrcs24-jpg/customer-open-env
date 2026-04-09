from __future__ import annotations

import json
from typing import Any

import gradio as gr

from customer_support_env import Action, CustomerSupportEnv


env = CustomerSupportEnv(default_task_id="easy_password_reset")


def start_task(task_id: str) -> tuple[str, str]:
    obs = env.reset(task_id=task_id)
    return obs.model_dump_json(indent=2), "Task reset complete. Submit an action JSON to step the environment."


def apply_action(action_json: str) -> tuple[str, str]:
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
    except Exception as exc:
        return "", f"Error: {exc}"


with gr.Blocks(title="Customer Support OpenEnv") as demo:
    gr.Markdown("# Customer Support OpenEnv\nUse this panel to interact with the environment step by step.")

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
    demo.launch(server_name="0.0.0.0", server_port=7860)
