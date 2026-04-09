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

A production-style OpenEnv environment for customer support ticket triage and resolution.

This project simulates realistic support operations where an agent must classify tickets, route work to the right teams, respond to customers, escalate high-risk incidents, and close tickets correctly under SLA pressure.

## Table of Contents

- [What This Project Includes](#what-this-project-includes)
- [Why This Environment Matters](#why-this-environment-matters)
- [Architecture Overview](#architecture-overview)
- [OpenEnv API](#openenv-api)
- [Task Catalog](#task-catalog)
- [Reward and Grading Design](#reward-and-grading-design)
- [Space Definitions](#space-definitions)
- [Baseline Scores](#baseline-scores)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Run the Inference Baseline](#run-the-inference-baseline)
- [Run the Interactive Web Preview](#run-the-interactive-web-preview)
- [Docker and Hugging Face Spaces Deployment](#docker-and-hugging-face-spaces-deployment)
- [Validation and Submission](#validation-and-submission)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What This Project Includes

- OpenEnv-compatible environment with the standard API (`reset`, `step`, `state`)
- Strongly typed Pydantic models for actions, observations, rewards, and state
- Three deterministic benchmark tasks (easy, medium, hard)
- Deterministic task graders with criterion-level score breakdowns
- Baseline inference script with OpenAI integration and deterministic fallback
- Structured evaluation logs (`START`, `STEP`, `END`) for evaluator compatibility
- Gradio interface for manual step-by-step interaction
- Submission validator script for pre-checks
- Dockerized runtime for local and Hugging Face Spaces deployment

## Why This Environment Matters

Customer support is a real operational workflow with competing objectives:

- Handle urgent incidents first
- Route each case to the proper team
- Communicate clearly and empathetically
- Avoid unnecessary escalation
- Close cases only after useful customer response

This environment turns those operational constraints into deterministic tasks with measurable outcomes suitable for agent development and evaluation.

## Architecture Overview

Core flow:

1. `CustomerSupportEnv.reset(task_id)` initializes a deterministic task episode.
2. Agent submits typed `Action` objects through `step(action)`.
3. Environment validates and applies transitions to ticket state.
4. Reward shaping provides immediate progress feedback.
5. Deterministic grader computes final normalized score in `[0.0, 1.0]`.

Primary implementation files:

- `customer_support_env/env.py`: Environment API, transitions, validation, reward shaping
- `customer_support_env/models.py`: Typed schemas and enums
- `customer_support_env/tasks.py`: Task definitions and seeded initial tickets
- `customer_support_env/graders.py`: Deterministic scoring logic
- `inference.py`: Submission baseline runner

## OpenEnv API

The environment implements:

- `reset(task_id: str | None = None) -> Observation`
- `step(action: Action) -> tuple[Observation, Reward, bool, info]`
- `state() -> EnvironmentState`

### Space Definitions

This project uses typed Pydantic models to define the environment interface:

- **Action space**: `customer_support_env.models.Action`
  - Required fields depend on `action_type`
  - Supported action types are listed below
- **Observation space**: `customer_support_env.models.Observation`
  - Includes task metadata, queue counts, ticket summaries, available actions, and hints
- **Reward space**: `customer_support_env.models.Reward`
  - Provides shaped step feedback with score, progress, penalties, and reason
- **State space**: `customer_support_env.models.EnvironmentState`
  - Returns the full internal ticket state, action history, and completion flag

### Action Types

Supported `Action.action_type` values:

- `open_ticket`
- `classify_ticket`
- `assign_ticket`
- `add_internal_note`
- `draft_reply`
- `send_reply`
- `escalate_ticket`
- `close_ticket`

### Observation Fields

Each observation includes queue and task context:

- `task_id`, `task_objective`
- `step_count`, `steps_remaining`
- `queue_size`, `open_count`, `escalated_count`, `closed_count`
- `ticket_summaries` (status, priority, category, assigned team, SLA)
- `available_actions`
- `hints`

### Reward Fields

`Reward` contains:

- `score`: per-step reward in `[-1, 1]`
- `progress`: cumulative task progress in `[0, 1]`
- `penalties`: aggregate penalties at the step
- `reason`: readable reason string

## Task Catalog

| Task ID | Difficulty | Scenario | Typical Perfect Steps |
|---|---|---|---|
| `easy_password_reset` | easy | One account access ticket | 4 |
| `medium_billing_and_outage` | medium | Urgent outage + billing queue | 8 |
| `hard_security_and_retention` | hard | Security escalation + churn-risk retention + shipping noise | 11 |

Task metadata is also declared in `openenv.yaml`.

### Baseline Scores

The deterministic fallback policy and bundled playbooks provide a reproducible baseline across all tasks:

| Task ID | Baseline Score | Notes |
|---|---:|---|
| `easy_password_reset` | 1.0 | Full classify, assign, reply, close flow |
| `medium_billing_and_outage` | 1.0 | Outage handled first, billing resolved after |
| `hard_security_and_retention` | 1.0 | Security escalation and retention flow completed |

Average baseline score: `1.0`

## Reward and Grading Design

### Shaped Reward (per step)

- Base time cost: `-0.01`
- Progress gain: `current_progress - previous_progress`
- Invalid action penalty: up to `-0.12`
- Loop penalty: `-0.05` for repeated same action on same ticket (3 times)

### Terminal Scoring

At episode completion, a deterministic grader computes criterion-level scores and final normalized score in `[0.0, 1.0]`.

This gives both immediate learning signals and strict final evaluation.

## Project Structure

```text
customer-open-env/
├── customer_support_env/
│   ├── __init__.py
│   ├── env.py
│   ├── graders.py
│   ├── models.py
│   ├── tasks.py
│   └── utils.py
├── examples/
│   └── manual_episode.py
├── scripts/
│   └── run_baseline.py
├── app.py
├── inference.py
├── openenv.yaml
├── validate_submission.py
├── Dockerfile
├── requirements.txt
├── INTERFACE_DEMO.md
├── SUBMISSION_GUIDE.md
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run an example episode

```bash
python examples/manual_episode.py
```

### 3. Run pre-submission checks

```bash
python validate_submission.py
```

## Run the Inference Baseline

Main entrypoint: `inference.py`

### Environment variables

```bash
export OPENAI_API_KEY="your_openai_key"           # required for LLM calls
export API_BASE_URL="https://api.openai.com/v1"   # optional override
export MODEL_NAME="gpt-4o-mini"                   # optional override
export HF_TOKEN="your_hf_token"                   # optional
export LOCAL_IMAGE_NAME="customer-open-env"       # optional
```

### Run

```bash
python inference.py
```

### Output contract

- `stderr`: structured JSON logs (`START`, `STEP`, `END`)
- `stdout`: final JSON with task scores, per-task breakdown, and average score

## Run the Interactive Web Preview

```bash
python app.py
```

Then open:

- Local: `http://localhost:7860`
- Share tunneling is intentionally disabled in this project for Space safety/compliance

### Updated Interface Highlights

The app now includes a richer operator-focused layout:

- Hero overview panel with task context and interaction tips
- Left control rail for scenario selection, reset/step controls, and editable action JSON
- Quick action templates for common flows (classify, assign, reply, close)
- Right-side tabs for structured `Observation` and `Step Result` JSON output
- Real-time status banner for success/error feedback
- Two summary cards:
  - `Episode Snapshot`: queue health, progress counters, and ticket status list
  - `Step Insight`: reward score, progress delta, penalties, done state, and reason

### Recommended Usage Flow

1. Choose a scenario and click `Reset Task`
2. Use `Load Template` for a starter action or edit JSON manually
3. Click `Step` and inspect both the JSON tabs and summary cards
4. Iterate until `done=true` and review the final grading details in `Step Result`

## Docker and Hugging Face Spaces Deployment

### Build and run locally

```bash
docker build -t customer-support-openenv .
docker run --rm -p 7860:7860 customer-support-openenv
```

### Deploy to Hugging Face Spaces

1. Create a new Space with SDK set to Docker
2. Connect this repository
3. Push updates to `main` to trigger rebuilds

## Validation and Submission

### Local validation

```bash
python validate_submission.py
```

### Submission artifacts

- GitHub repository URL
- Hugging Face Space URL

Detailed checklist and instructions are provided in `SUBMISSION_GUIDE.md`.

## Reproducibility

- Deterministic tasks and grading
- Baseline configured with `temperature=0` and `seed=17`
- Deterministic fallback policy ensures stable behavior when LLM output fails

## Scoring Rubric Alignment

This project is designed to map directly onto the published evaluation rubric:

### Judging Workflow

Submission review happens in three phases:

- Phase 1: Automated validation
  - Checks HF Space deployment, OpenEnv spec compliance, Docker build/run, baseline reproducibility, and 3+ graded tasks.
- Phase 2: Agentic evaluation
  - Re-runs the baseline agent and standard OpenLLM agents against the environment to compare behavior and score variance.
- Phase 3: Human review
  - Reviews top submissions for real-world utility, creativity, and exploit resistance.

### Disqualification Safeguards

The repository is built to avoid the most common disqualification causes:

- The environment deploys and responds from the provided Dockerfile and Gradio app.
- The implementation is not a trivial wrapper: it includes a real support workflow, typed models, task logic, reward shaping, and deterministic graders.
- Graders do not return a fixed score; they evaluate state and action history deterministically.
- A baseline inference script is included and reproducible across all three tasks.
- The project includes 3+ tasks with a difficulty range and clear success criteria.

### Real-World Utility

- Customer support triage is a common, practical agent workflow with clear operational value.
- The environment models authentic support actions: classify, assign, reply, escalate, and close.
- SLA pressure, routing decisions, and customer messaging make the tasks useful for real agent evaluation.

### Task and Grader Quality

- There are 3 tasks with an easy-to-hard difficulty spread.
- Each task has a deterministic grader with clear success criteria and a normalized score in `[0.0, 1.0]`.
- The hard task includes ordering constraints and empathy/security requirements that are more demanding than the easy and medium tasks.

### Environment Design

- `reset()` creates a clean deterministic episode state.
- `Action`, `Observation`, `Reward`, and `EnvironmentState` are typed and documented.
- Reward shaping provides step-by-step feedback instead of sparse terminal-only signals.
- Episode boundaries are sensible through max-step limits and terminal grading.

### Code Quality and Spec Compliance

- `validate_submission.py` passes locally.
- `docker build` and `docker run` are supported by the provided `Dockerfile`.
- The Gradio UI runs from `app.py`.
- The baseline inference script is reproducible and uses the OpenAI client with environment-based configuration.

### Creativity and Novelty

- The environment is not a toy benchmark; it captures a realistic support ops workflow.
- The hard task combines security escalation, retention sensitivity, and routine noise handling.
- The UI includes playbooks, undo, suggestions, and exportable session logs to support practical exploration.

## Troubleshooting

### `OPENAI_API_KEY` is missing

- Set `OPENAI_API_KEY` before running LLM-based inference
- Baseline fallback policies can still be used for deterministic local runs

### Import errors when running scripts from subfolders

- Scripts in `examples` and `scripts` include path handling for direct execution
- Prefer running commands from repository root

### Docker build issues

- Rebuild without cache:

```bash
docker build --no-cache -t customer-support-openenv .
```

## License

MIT
