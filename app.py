from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime, timezone
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


ACTION_TEMPLATES: dict[str, dict[str, Any]] = {
    "classify_account": {
        "action_type": "classify_ticket",
        "ticket_id": "T-1001",
        "category": "account",
        "priority": "high",
    },
    "assign_frontline": {
        "action_type": "assign_ticket",
        "ticket_id": "T-1001",
        "assigned_team": "frontline",
    },
    "send_reset_reply": {
        "action_type": "send_reply",
        "ticket_id": "T-1001",
        "content": "We initiated your password reset. Please check your email and try signing in again.",
    },
    "close_ticket": {
        "action_type": "close_ticket",
        "ticket_id": "T-1001",
    },
}


PLAYBOOKS: dict[str, list[dict[str, Any]]] = {
    "easy_password_reset": [
        {"action_type": "classify_ticket", "ticket_id": "T-1001", "category": "account", "priority": "high"},
        {"action_type": "assign_ticket", "ticket_id": "T-1001", "assigned_team": "frontline"},
        {
            "action_type": "send_reply",
            "ticket_id": "T-1001",
            "content": "We initiated your password reset. Please check your email and sign in again.",
        },
        {"action_type": "close_ticket", "ticket_id": "T-1001"},
    ],
    "medium_billing_and_outage": [
        {"action_type": "classify_ticket", "ticket_id": "T-2002", "category": "technical", "priority": "urgent"},
        {"action_type": "assign_ticket", "ticket_id": "T-2002", "assigned_team": "technical"},
        {
            "action_type": "send_reply",
            "ticket_id": "T-2002",
            "content": "We identified the outage and applied mitigation. Service is restored.",
        },
        {"action_type": "close_ticket", "ticket_id": "T-2002"},
        {"action_type": "assign_ticket", "ticket_id": "T-2001", "assigned_team": "billing"},
        {
            "action_type": "send_reply",
            "ticket_id": "T-2001",
            "content": "Sorry for the double charge. We issued a refund and updated your invoice.",
        },
        {"action_type": "close_ticket", "ticket_id": "T-2001"},
    ],
    "hard_security_and_retention": [
        {"action_type": "classify_ticket", "ticket_id": "T-3001", "category": "security", "priority": "urgent"},
        {"action_type": "assign_ticket", "ticket_id": "T-3001", "assigned_team": "security"},
        {"action_type": "escalate_ticket", "ticket_id": "T-3001"},
        {
            "action_type": "send_reply",
            "ticket_id": "T-3001",
            "content": "We secured your account and escalated this security incident for immediate investigation.",
        },
        {"action_type": "close_ticket", "ticket_id": "T-3001"},
        {"action_type": "classify_ticket", "ticket_id": "T-3002", "category": "account", "priority": "high"},
        {"action_type": "assign_ticket", "ticket_id": "T-3002", "assigned_team": "retention"},
        {
            "action_type": "send_reply",
            "ticket_id": "T-3002",
            "content": "We are sorry for the delays and understand your frustration. We will improve support speed and prioritize your account.",
        },
        {"action_type": "close_ticket", "ticket_id": "T-3002"},
        {
            "action_type": "send_reply",
            "ticket_id": "T-3003",
            "content": "Thanks for your patience. Your delayed package is now in transit and prioritized for delivery.",
        },
        {"action_type": "close_ticket", "ticket_id": "T-3003"},
    ],
}


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --surface: #f6faf2;
    --surface-2: #eef7e7;
    --ink: #173225;
    --muted: #436153;
    --accent: #18826e;
    --accent-2: #245e87;
    --ok-bg: #e8f9ef;
    --err-bg: #ffeceb;
}

body {
    background:
        radial-gradient(circle at 10% 15%, #dff5df 0%, transparent 32%),
        radial-gradient(circle at 90% 0%, #cfe9f6 0%, transparent 34%),
        linear-gradient(180deg, #f8fcf5 0%, #edf5ef 100%);
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink);
}

.hero {
    background: linear-gradient(115deg, #e2f6e7 0%, #d7edf8 100%);
    border: 1px solid #c8e3d4;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 10px;
}

.hero h1 {
    margin: 0;
    font-size: 30px;
    line-height: 1.1;
}

.hero p {
    margin: 8px 0 0 0;
    color: var(--muted);
}

.badge-row {
    margin-top: 12px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    border: 1px solid #bfdccc;
    background: rgba(255, 255, 255, 0.6);
}

.panel-title {
    margin-bottom: 4px;
    color: var(--muted);
}

.status-card {
    border-radius: 12px;
    padding: 10px 12px;
    border: 1px solid #cae1d4;
    background: var(--surface);
}

.status-ok {
    background: var(--ok-bg);
    border-color: #b7dfc5;
}

.status-err {
    background: var(--err-bg);
    border-color: #f2c6c2;
}

code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
}

.feature-box {
    border: 1px solid #cae1d4;
    border-radius: 12px;
    background: #fbfef8;
    padding: 10px;
}
"""


def _status(msg: str, *, error: bool = False) -> str:
    cls = "status-card status-err" if error else "status-card status-ok"
    return f"<div class='{cls}'>{msg}</div>"


def _build_episode_summary(obs: dict[str, Any]) -> str:
    if not obs:
        return "### Episode Snapshot\nNo observation yet. Reset a task to begin."

    lines = [
        "### Episode Snapshot",
        f"Task: {obs.get('task_id', '-')}",
        f"Step: {obs.get('step_count', '-')} | Remaining: {obs.get('steps_remaining', '-')}",
        f"Queue: {obs.get('queue_size', '-')} | Open: {obs.get('open_count', '-')} | Escalated: {obs.get('escalated_count', '-')} | Closed: {obs.get('closed_count', '-')}",
        "",
        "Ticket queue:",
    ]

    for t in obs.get("ticket_summaries", []):
        lines.append(
            "- "
            + f"{t.get('ticket_id', '?')} | {t.get('subject', 'No subject')} | "
            + f"status={t.get('status', '?')} | priority={t.get('priority', '?')} | "
            + f"SLA={t.get('sla_minutes_remaining', '?')}m"
        )

    if not obs.get("ticket_summaries"):
        lines.append("- No tickets in queue")

    return "\n".join(lines)


def _build_step_summary(result: dict[str, Any]) -> str:
    if not result:
        return "### Step Insight\nNo actions submitted yet."

    reward = result.get("reward") or {}
    info = result.get("info") or {}
    done = result.get("done")

    lines = [
        "### Step Insight",
        f"Reward score: {reward.get('score', '-')}",
        f"Progress: {reward.get('progress', info.get('progress', '-'))}",
        f"Penalties: {reward.get('penalties', '-')}",
        f"Done: {done}",
    ]

    reason = reward.get("reason") or info.get("validation_reason")
    if reason:
        lines.append(f"Reason: {reason}")

    return "\n".join(lines)


def _build_action_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "### Action History\nNo steps yet."

    lines = ["### Action History"]
    for i, action in enumerate(history, start=1):
        action_type = action.get("action_type", "?")
        ticket = action.get("ticket_id", "-")
        content = action.get("content") or ""
        extra = f" | note={content[:44]}" if content else ""
        lines.append(f"{i}. {action_type} | ticket={ticket}{extra}")
    return "\n".join(lines)


def _build_action_guidance(action_json: str, obs: dict[str, Any]) -> str:
    required_map = {
        "open_ticket": ["customer_name", "customer_email", "subject", "body"],
        "classify_ticket": ["ticket_id", "category", "priority(optional)"],
        "assign_ticket": ["ticket_id", "assigned_team"],
        "add_internal_note": ["ticket_id", "content"],
        "draft_reply": ["ticket_id", "content"],
        "send_reply": ["ticket_id", "content (or existing draft)"],
        "escalate_ticket": ["ticket_id"],
        "close_ticket": ["ticket_id"],
    }

    try:
        payload = json.loads(action_json)
        action_type = str(payload.get("action_type", "")).strip()
    except Exception:
        action_type = ""

    if not action_type:
        return "### Action Guidance\nProvide action_type in JSON to see required fields."

    lines = ["### Action Guidance", f"Action: {action_type}"]
    required = required_map.get(action_type, [])
    if required:
        lines.append("Required: " + ", ".join(required))

    tickets = [t.get("ticket_id", "?") for t in obs.get("ticket_summaries", [])] if obs else []
    if tickets:
        lines.append("Known ticket ids: " + ", ".join(tickets))

    available = obs.get("available_actions", []) if obs else []
    if available:
        lines.append("Available actions: " + ", ".join(str(a) for a in available))

    return "\n".join(lines)


def _build_playbook_preview(task_id: str) -> str:
    playbook = PLAYBOOKS.get(task_id, [])
    if not playbook:
        return "### Task Playbook\nNo playbook available for this task."

    lines = [
        "### Task Playbook",
        f"Task: {task_id}",
        f"Steps: {len(playbook)}",
        "",
    ]

    for i, action in enumerate(playbook, start=1):
        action_type = action.get("action_type", "?")
        ticket = action.get("ticket_id", "-")
        lines.append(f"{i}. {action_type} | ticket={ticket}")

    return "\n".join(lines)


def _build_playbook_timeline(events: list[str], final_result: dict[str, Any]) -> str:
    lines = ["### Playbook Timeline"]
    if not events:
        lines.append("No playbook run yet.")
        return "\n".join(lines)

    lines.extend(events)
    info = final_result.get("info") or {}
    if "final_score" in info:
        lines.append("")
        lines.append(f"Final score: {info.get('final_score')}")
    return "\n".join(lines)


def _build_session_log(task_id: str, history: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "step_count": len(history),
        "actions": history,
        "last_result": result,
    }


def _pick_team(category: str | None) -> str:
    mapping = {
        "billing": "billing",
        "technical": "technical",
        "security": "security",
        "account": "frontline",
        "shipping": "frontline",
        "other": "frontline",
    }
    return mapping.get(str(category), "frontline")


def _suggest_action(obs: dict[str, Any]) -> dict[str, Any]:
    tickets = obs.get("ticket_summaries", []) if obs else []
    active = [t for t in tickets if t.get("status") != "closed"]
    if not active:
        return {"action_type": "open_ticket", "customer_name": "New Customer", "customer_email": "customer@example.com", "subject": "New issue", "body": "Describe customer issue."}

    # Prioritize lowest SLA first.
    target = sorted(active, key=lambda t: int(t.get("sla_minutes_remaining", 10_000)))[0]
    tid = target.get("ticket_id")

    if not target.get("category"):
        subject = str(target.get("subject", "")).lower()
        guessed_category = "account"
        if "bill" in subject or "refund" in subject or "charge" in subject:
            guessed_category = "billing"
        elif "down" in subject or "outage" in subject or "error" in subject:
            guessed_category = "technical"
        elif "security" in subject or "login" in subject or "unknown" in subject:
            guessed_category = "security"
        return {
            "action_type": "classify_ticket",
            "ticket_id": tid,
            "category": guessed_category,
            "priority": "urgent" if int(target.get("sla_minutes_remaining", 9999)) <= 60 else "high",
        }

    if not target.get("assigned_team"):
        return {
            "action_type": "assign_ticket",
            "ticket_id": tid,
            "assigned_team": _pick_team(target.get("category")),
        }

    if target.get("status") != "closed":
        return {
            "action_type": "send_reply",
            "ticket_id": tid,
            "content": "Thanks for your patience. We have investigated your issue and are taking care of it now.",
        }

    return {"action_type": "close_ticket", "ticket_id": tid}


def load_action_template(template_key: str) -> str:
    template = ACTION_TEMPLATES.get(template_key, ACTION_TEMPLATES["classify_account"])
    return json.dumps(template, indent=2)


def load_task_playbook(task_id: str, obs: dict[str, Any]) -> tuple[str, str, str, str]:
    playbook = PLAYBOOKS.get(task_id, [])
    if not playbook:
        msg = f"No playbook found for task: {task_id}"
        return "{}", _status(msg, error=True), _build_playbook_preview(task_id), _build_action_guidance("{}", obs or {})

    action_json = json.dumps(playbook[0], indent=2)
    return (
        action_json,
        _status(f"Loaded playbook for {task_id}. First step is now in the editor."),
        _build_playbook_preview(task_id),
        _build_action_guidance(action_json, obs or {}),
    )


def start_task(task_id: str) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    str,
    str,
    list[dict[str, Any]],
    str,
    dict[str, Any],
]:
    try:
        obs = env.reset(task_id=task_id)
        obs_data = obs.model_dump()
        result_data = {"message": "Task reset. Submit an action JSON below."}
        history: list[dict[str, Any]] = []
        return (
            obs_data,
            result_data,
            _status("Task reset successfully. You can now submit an action."),
            _build_episode_summary(obs_data),
            _build_step_summary({}),
            _build_action_guidance(load_action_template("classify_account"), obs_data),
            _build_action_history(history),
            history,
            task_id,
            _build_session_log(task_id, history, result_data),
        )
    except Exception as e:
        err = f"Error while resetting task: {str(e)[:200]}"
        return (
            {},
            {"error": err},
            _status(err, error=True),
            _build_episode_summary({}),
            _build_step_summary({}),
            _build_action_guidance("{}", {}),
            _build_action_history([]),
            [],
            task_id,
            _build_session_log(task_id, [], {"error": err}),
        )


def apply_action(
    action_json: str,
    task_id: str,
    history: list[dict[str, Any]],
    current_obs: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    str,
    str,
    list[dict[str, Any]],
    str,
    dict[str, Any],
]:
    """Execute one action and show result."""
    try:
        payload: dict[str, Any] = json.loads(action_json)
        action = Action.model_validate(payload)
        obs, reward, done, info = env.step(action)
        obs_data = obs.model_dump()
        updated_history = list(history or []) + [payload]
        result_data = {
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
        status_msg = "Action applied. Episode complete." if done else "Action applied. Continue to the next step."
        return (
            obs_data,
            result_data,
            _status(status_msg),
            _build_episode_summary(obs_data),
            _build_step_summary(result_data),
            _build_action_guidance(action_json, obs_data),
            _build_action_history(updated_history),
            updated_history,
            task_id,
            _build_session_log(task_id, updated_history, result_data),
        )
    except json.JSONDecodeError as e:
        err = f"Invalid JSON: {str(e)[:100]}"
        safe_obs = current_obs or {}
        result_data = {"error": err}
        history_safe = list(history or [])
        return (
            safe_obs,
            result_data,
            _status(err, error=True),
            _build_episode_summary(safe_obs),
            _build_step_summary(result_data),
            _build_action_guidance(action_json, safe_obs),
            _build_action_history(history_safe),
            history_safe,
            task_id,
            _build_session_log(task_id, history_safe, result_data),
        )
    except Exception as exc:
        err = f"Error: {str(exc)[:200]}"
        safe_obs = current_obs or {}
        result_data = {"error": err}
        history_safe = list(history or [])
        return (
            safe_obs,
            result_data,
            _status(err, error=True),
            _build_episode_summary(safe_obs),
            _build_step_summary(result_data),
            _build_action_guidance(action_json, safe_obs),
            _build_action_history(history_safe),
            history_safe,
            task_id,
            _build_session_log(task_id, history_safe, result_data),
        )


def validate_action_input(action_json: str, obs: dict[str, Any]) -> tuple[str, str]:
    try:
        payload: dict[str, Any] = json.loads(action_json)
        Action.model_validate(payload)
        action_type = str(payload.get("action_type", ""))
        if obs and action_type and action_type not in [str(a) for a in obs.get("available_actions", [])]:
            msg = f"Action JSON is valid, but action_type '{action_type}' is not available right now."
            return _status(msg, error=True), _build_action_guidance(action_json, obs)
        return _status("Action JSON is valid."), _build_action_guidance(action_json, obs or {})
    except Exception as e:
        return _status(f"Validation error: {str(e)[:200]}", error=True), _build_action_guidance(action_json, obs or {})


def suggest_next_action(obs: dict[str, Any]) -> tuple[str, str]:
    if not obs:
        default = json.dumps(ACTION_TEMPLATES["classify_account"], indent=2)
        return default, _status("No observation yet. Reset task first, then suggestions become context-aware.")
    suggestion = json.dumps(_suggest_action(obs), indent=2)
    return suggestion, _status("Suggested next action generated from current queue state.")


def copy_suggestion_to_editor(suggested_action_json: str, obs: dict[str, Any]) -> tuple[str, str]:
    action_json = suggested_action_json or json.dumps(ACTION_TEMPLATES["classify_account"], indent=2)
    return action_json, _build_action_guidance(action_json, obs or {})


def undo_last_action(task_id: str, history: list[dict[str, Any]]) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    str,
    str,
    list[dict[str, Any]],
    str,
    dict[str, Any],
]:
    if not task_id:
        err = "Cannot undo before a task is initialized."
        return (
            {},
            {"error": err},
            _status(err, error=True),
            _build_episode_summary({}),
            _build_step_summary({"error": err}),
            _build_action_guidance("{}", {}),
            _build_action_history([]),
            [],
            "",
            _build_session_log("", [], {"error": err}),
        )

    history_safe = list(history or [])
    if not history_safe:
        obs = env.reset(task_id=task_id).model_dump()
        result_data = {"message": "Nothing to undo. Task is at initial state."}
        return (
            obs,
            result_data,
            _status("Nothing to undo. You are already at step 0."),
            _build_episode_summary(obs),
            _build_step_summary(result_data),
            _build_action_guidance(json.dumps(ACTION_TEMPLATES["classify_account"], indent=2), obs),
            _build_action_history([]),
            [],
            task_id,
            _build_session_log(task_id, [], result_data),
        )

    replay_actions = history_safe[:-1]
    obs_obj = env.reset(task_id=task_id)
    obs_data = obs_obj.model_dump()
    result_data: dict[str, Any] = {"message": "Replayed to previous step."}

    for payload in replay_actions:
        action = Action.model_validate(payload)
        obs_obj, reward, done, info = env.step(action)
        obs_data = obs_obj.model_dump()
        result_data = {
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
        if done:
            break

    status_msg = f"Undid last step. Active steps: {len(replay_actions)}."
    action_seed = json.dumps(_suggest_action(obs_data), indent=2)
    return (
        obs_data,
        result_data,
        _status(status_msg),
        _build_episode_summary(obs_data),
        _build_step_summary(result_data),
        _build_action_guidance(action_seed, obs_data),
        _build_action_history(replay_actions),
        replay_actions,
        task_id,
        _build_session_log(task_id, replay_actions, result_data),
    )


def export_session_log(session_data: dict[str, Any]) -> tuple[str | None, str]:
    if not session_data:
        return None, _status("No session data available to export.", error=True)

    fd, path = tempfile.mkstemp(prefix="openenv-session-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    return path, _status("Session log exported. Download it from the file output.")


def run_task_playbook(task_id: str) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    str,
    str,
    list[dict[str, Any]],
    str,
    dict[str, Any],
    str,
    str,
]:
    playbook = PLAYBOOKS.get(task_id, [])
    if not playbook:
        err = f"No playbook configured for task: {task_id}"
        return (
            {},
            {"error": err},
            _status(err, error=True),
            _build_episode_summary({}),
            _build_step_summary({"error": err}),
            _build_action_guidance("{}", {}),
            _build_action_history([]),
            [],
            task_id,
            _build_session_log(task_id, [], {"error": err}),
            _build_playbook_timeline([], {"error": err}),
            _build_playbook_preview(task_id),
        )

    obs_obj = env.reset(task_id=task_id)
    obs_data = obs_obj.model_dump()
    result_data: dict[str, Any] = {"message": "Playbook initialized."}
    events: list[str] = []
    history: list[dict[str, Any]] = []

    for i, payload in enumerate(playbook, start=1):
        action = Action.model_validate(payload)
        obs_obj, reward, done, info = env.step(action)
        obs_data = obs_obj.model_dump()
        result_data = {
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
        history.append(payload)
        events.append(
            f"{i}. {payload.get('action_type')} {payload.get('ticket_id', '')} | "
            + f"reward={reward.score} | progress={reward.progress} | done={done}"
        )
        if done:
            break

    status_msg = "Playbook execution finished."
    if result_data.get("done"):
        status_msg = "Playbook execution finished with completed episode."

    action_seed = json.dumps(_suggest_action(obs_data), indent=2)
    session_log = _build_session_log(task_id, history, result_data)
    return (
        obs_data,
        result_data,
        _status(status_msg),
        _build_episode_summary(obs_data),
        _build_step_summary(result_data),
        _build_action_guidance(action_seed, obs_data),
        _build_action_history(history),
        history,
        task_id,
        session_log,
        _build_playbook_timeline(events, result_data),
        _build_playbook_preview(task_id),
    )


with gr.Blocks(title="Customer Support OpenEnv", css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <section class='hero'>
          <h1>Customer Support OpenEnv</h1>
          <p>Practice ticket triage with real-time rewards, progress tracking, and action validation.</p>
          <div class='badge-row'>
            <span class='badge'>Deterministic tasks</span>
            <span class='badge'>Step-by-step grading</span>
            <span class='badge'>JSON-native interaction</span>
          </div>
        </section>
        """
    )

    status = gr.HTML(value=_status("Ready. Pick a task and reset to start."), elem_classes=["panel-title"])
    history_state = gr.State([])
    task_state = gr.State("easy_password_reset")

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            task_picker = gr.Dropdown(
                choices=["easy_password_reset", "medium_billing_and_outage", "hard_security_and_retention"],
                value="easy_password_reset",
                label="Scenario",
            )

            with gr.Row():
                reset_btn = gr.Button("Reset Task", variant="secondary")
                step_btn = gr.Button("Step", variant="primary")
                undo_btn = gr.Button("Undo Last Step")

            template_picker = gr.Dropdown(
                choices=[
                    ("Classify account ticket", "classify_account"),
                    ("Assign to frontline", "assign_frontline"),
                    ("Send reset reply", "send_reset_reply"),
                    ("Close ticket", "close_ticket"),
                ],
                value="classify_account",
                label="Quick action template",
            )
            template_btn = gr.Button("Load Template")

            with gr.Row():
                validate_btn = gr.Button("Validate JSON")
                suggest_btn = gr.Button("Suggest Next")

            with gr.Row():
                load_playbook_btn = gr.Button("Load Full Playbook")
                run_playbook_btn = gr.Button("Run Full Playbook", variant="primary")

            suggested_action = gr.Code(
                value=json.dumps(ACTION_TEMPLATES["classify_account"], indent=2),
                label="Suggested Action",
                language="json",
                lines=8,
            )
            use_suggestion_btn = gr.Button("Use Suggested Action")

            action_input = gr.Code(
                value=json.dumps(ACTION_TEMPLATES["classify_account"], indent=2),
                label="Action JSON",
                language="json",
                lines=14,
            )

            playbook_preview = gr.Markdown(value=_build_playbook_preview("easy_password_reset"), elem_classes=["feature-box"])

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.Tab("Observation"):
                    observation = gr.JSON(label="Observation")
                with gr.Tab("Step Result"):
                    log = gr.JSON(label="Result")

            with gr.Row():
                episode_summary = gr.Markdown(value=_build_episode_summary({}))
                step_summary = gr.Markdown(value=_build_step_summary({}))

            with gr.Row():
                action_guidance = gr.Markdown(value=_build_action_guidance("{}", {}), elem_classes=["feature-box"])
                action_history = gr.Markdown(value=_build_action_history([]), elem_classes=["feature-box"])

            session_log = gr.JSON(label="Session Log")
            playbook_timeline = gr.Markdown(value=_build_playbook_timeline([], {}), elem_classes=["feature-box"])
            with gr.Row():
                export_btn = gr.Button("Export Session Log")
                export_file = gr.File(label="Exported JSON", file_count="single")

    template_btn.click(load_action_template, inputs=[template_picker], outputs=[action_input])

    reset_btn.click(
        start_task,
        inputs=[task_picker],
        outputs=[
            observation,
            log,
            status,
            episode_summary,
            step_summary,
            action_guidance,
            action_history,
            history_state,
            task_state,
            session_log,
        ],
    )

    step_btn.click(
        apply_action,
        inputs=[action_input, task_state, history_state, observation],
        outputs=[
            observation,
            log,
            status,
            episode_summary,
            step_summary,
            action_guidance,
            action_history,
            history_state,
            task_state,
            session_log,
        ],
    )

    validate_btn.click(validate_action_input, inputs=[action_input, observation], outputs=[status, action_guidance])
    suggest_btn.click(suggest_next_action, inputs=[observation], outputs=[suggested_action, status])
    use_suggestion_btn.click(copy_suggestion_to_editor, inputs=[suggested_action, observation], outputs=[action_input, action_guidance])

    load_playbook_btn.click(
        load_task_playbook,
        inputs=[task_picker, observation],
        outputs=[action_input, status, playbook_preview, action_guidance],
    )

    run_playbook_btn.click(
        run_task_playbook,
        inputs=[task_picker],
        outputs=[
            observation,
            log,
            status,
            episode_summary,
            step_summary,
            action_guidance,
            action_history,
            history_state,
            task_state,
            session_log,
            playbook_timeline,
            playbook_preview,
        ],
    )

    undo_btn.click(
        undo_last_action,
        inputs=[task_state, history_state],
        outputs=[
            observation,
            log,
            status,
            episode_summary,
            step_summary,
            action_guidance,
            action_history,
            history_state,
            task_state,
            session_log,
        ],
    )

    export_btn.click(export_session_log, inputs=[session_log], outputs=[export_file, status])


if __name__ == "__main__":
    # Never enable share tunnel in Spaces; this can trigger abuse/proxy flags.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
