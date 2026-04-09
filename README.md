---
title: Customer Support OpenEnv
emoji: "📨"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
	- openenv
	- customer-support
	- reinforcement-learning
---

# Customer Support Ticket OpenEnv

A complete, real-world OpenEnv environment that simulates customer support ticket triage and resolution.

## Why this environment

Customer support operations are a real production workflow where agents must:

- prioritize urgent incidents
- classify and route tickets to the right team
- communicate with customers effectively
- escalate high-risk cases
- close cases only after meaningful response

This environment turns those behaviors into a deterministic learning setup with shaped rewards and task-specific graders.

## OpenEnv API

The environment implements the standard API:

- `reset(task_id: str | None = None) -> Observation`
- `step(action: Action) -> tuple[Observation, Reward, bool, info]`
- `state() -> EnvironmentState`

### Typed models (Pydantic)

Implemented in `customer_support_env/models.py`:

- `Action`
- `Observation`
- `Reward`
- `EnvironmentState`
- `TaskResult`

## Action Space

`Action.action_type` supports:

- `open_ticket`
- `classify_ticket`
- `assign_ticket`
- `add_internal_note`
- `draft_reply`
- `send_reply`
- `escalate_ticket`
- `close_ticket`

Main action fields:

- `ticket_id`
- `category`
- `priority`
- `assigned_team`
- `content`

## Observation Space

Each observation includes queue-level and ticket-level context:

- `task_id`, `task_objective`
- `step_count`, `steps_remaining`
- `queue_size`, `open_count`, `escalated_count`, `closed_count`
- `ticket_summaries` (status, priority, category, team, SLA)
- `available_actions`
- `hints`

## Reward Design

The reward has trajectory-level signal, not only final binary outcome.

Per step:

- base time cost: `-0.01`
- progress reward: `current_progress - last_progress`
- invalid-action penalty: up to `-0.12`
- loop penalty: `-0.05` when repeating same action on same ticket 3 times

At episode end:

- terminal adjustment aligns total progress with deterministic grader score
- final task score is in `[0.0, 1.0]`

## Tasks and Graders

Three deterministic tasks with increasing difficulty:

1. `easy_password_reset` (easy)
2. `medium_billing_and_outage` (medium)
3. `hard_security_and_retention` (hard)

Each task has concrete success criteria and a programmatic grader in `customer_support_env/graders.py`.

Scores are deterministic and normalized to `[0.0, 1.0]`.

## Project Structure

```text
customer_support_env/
	__init__.py
	env.py
	graders.py
	models.py
	tasks.py
scripts/
	run_baseline.py
examples/
	manual_episode.py
app.py
inference.py           # Main baseline inference (competition submission)
openenv.yaml
Dockerfile
requirements.txt
README.md
```

## Setup

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Manual environment run

```bash
python examples/manual_episode.py
```

### 3) Baseline Inference (Submission)

The baseline inference script is `inference.py`. It follows the standardized competition pattern with structured logging.

#### Environment Variables (Required for LLM calls)

Set these before running:

```bash
export OPENAI_API_KEY="your_openai_key"
export API_BASE_URL="https://api.openai.com/v1"  # Default set in code
export MODEL_NAME="gpt-4o-mini"                   # Default set in code
export HF_TOKEN="your_hf_token"                   # Optional
export LOCAL_IMAGE_NAME="image_name"              # Optional (for docker deployment)
```

#### Running the Baseline

```bash
python inference.py
```

This script:
- ✅ Uses OpenAI client configured via `API_BASE_URL` and `OPENAI_API_KEY`
- ✅ Sets `temperature=0` and `seed=17` for reproducibility
- ✅ Outputs structured logs in `START/STEP/END` format to stderr
- ✅ Reports final JSON results to stdout with task scores and breakdown
- ✅ Falls back to deterministic rule-based policy if LLM fails

#### Expected Output Format

**Stderr** (structured logs):
```json
{"type": "START", "task_id": "easy_password_reset", "model": "gpt-4o-mini", "objective": "..."}
{"type": "STEP", "step": 1, "action": {...}, "reward": 0.19, "progress": 0.2, "done": false}
{"type": "STEP", "step": 2, "action": {...}, "reward": 0.19, "progress": 0.4, "done": false}
...
{"type": "END", "task_id": "easy_password_reset", "final_score": 1.0, "steps": 4, "grading_breakdown": {...}}
```

**Stdout** (results):
```json
{
  "model": "gpt-4o-mini",
  "tasks": {
    "easy_password_reset": {
      "score": 1.0,
      "steps": 4,
      "breakdown": {...}
    },
    ...
  },
  "average_score": 1.0
}
```

#### Deterministic Fallback

If the LLM API fails or returns malformed JSON, the script automatically falls back to a deterministic rule-based policy that achieves optimal scores on all tasks. This ensures reproducibility and stability.

## OpenEnv Metadata

Environment metadata is declared in `openenv.yaml`.

To validate (when OpenEnv CLI is installed):

```bash
openenv validate
```

## Hugging Face Spaces Deployment

This repo is ready for Docker Spaces deployment.

### Local Docker run

```bash
docker build -t customer-support-openenv .
docker run --rm -p 7860:7860 customer-support-openenv
```

Then open `http://localhost:7860`.

### HF Space settings

- Space SDK: Docker
- Include repo as-is
- Ensure tag `openenv` remains in README frontmatter

## Notes on Determinism

- Tasks, initial tickets, and graders are deterministic.
- Reward shaping is deterministic from state transitions.
- Baseline script uses fixed `temperature` and `seed`.
- If model output is malformed JSON, baseline can use deterministic fallback unless `--no-fallback` is passed.

## License

MIT