# Customer Support OpenEnv - Interface Demo

This document shows what happens when you interact with the web interface.

## Task 1: Easy - Password Reset

### Initial State (After Reset)

```json
{
  "task_id": "easy_password_reset",
  "task_objective": "Resolve a single account access ticket end-to-end: classify it, assign to the right team, reply with a reset confirmation, and close the ticket.",
  "step_count": 0,
  "steps_remaining": 12,
  "queue_size": 1,
  "open_count": 1,
  "escalated_count": 0,
  "closed_count": 0,
  "ticket_summaries": [
    {
      "ticket_id": "T-1001",
      "subject": "Can't sign in",
      "status": "open",
      "priority": "normal",
      "category": null,
      "assigned_team": null,
      "sla_minutes_remaining": 180
    }
  ],
  "available_actions": [
    "open_ticket", "classify_ticket", "assign_ticket", 
    "add_internal_note", "draft_reply", "send_reply", 
    "escalate_ticket", "close_ticket"
  ],
  "hints": ["Start by prioritizing urgent or security-sensitive tickets."]
}
```

### Step 1 - Classify as Account Issue

**Action:**
```json
{
  "action_type": "classify_ticket",
  "ticket_id": "T-1001",
  "category": "account",
  "priority": "high"
}
```

**Result:**
```json
{
  "reward": {
    "score": 0.19,
    "progress": 0.2,
    "penalties": 0.0,
    "reason": "Action applied"
  },
  "done": false,
  "info": {
    "task_id": "easy_password_reset",
    "step": 1,
    "validation_reason": "ok",
    "progress": 0.2
  }
}
```

✅ **Progress:** 20% (Classified ✓)

---

### Step 2 - Assign to Frontline Team

**Action:**
```json
{
  "action_type": "assign_ticket",
  "ticket_id": "T-1001",
  "assigned_team": "frontline"
}
```

**Result:**
```json
{
  "reward": {
    "score": 0.19,
    "progress": 0.4,
    "penalties": 0.0,
    "reason": "Action applied"
  },
  "done": false,
  "info": {
    "task_id": "easy_password_reset",
    "step": 2,
    "validation_reason": "ok",
    "progress": 0.4
  }
}
```

✅ **Progress:** 40% (Assigned ✓)

---

### Step 3 - Send Customer Reply

**Action:**
```json
{
  "action_type": "send_reply",
  "ticket_id": "T-1001",
  "content": "We have initiated your password reset. Please check your email and sign in again."
}
```

**Result:**
```json
{
  "reward": {
    "score": 0.29,
    "progress": 0.7,
    "penalties": 0.0,
    "reason": "Action applied"
  },
  "done": false,
  "info": {
    "task_id": "easy_password_reset",
    "step": 3,
    "validation_reason": "ok",
    "progress": 0.7
  }
}
```

✅ **Progress:** 70% (Replied ✓)

---

### Step 4 - Close Ticket

**Action:**
```json
{
  "action_type": "close_ticket",
  "ticket_id": "T-1001"
}
```

**Result:**
```json
{
  "reward": {
    "score": 0.29,
    "progress": 1.0,
    "penalties": 0.0,
    "reason": "Action applied"
  },
  "done": true,
  "info": {
    "task_id": "easy_password_reset",
    "step": 4,
    "validation_reason": "ok",
    "progress": 1.0,
    "final_score": 1.0,
    "grading_breakdown": {
      "classified_account": 0.2,
      "assigned_frontline": 0.2,
      "sent_password_reply": 0.3,
      "closed": 0.3
    }
  }
}
```

✅ **EPISODE COMPLETE: Score 1.0 (Perfect)**

---

## Task 2: Medium - Billing + Outage

### Initial State

```json
{
  "task_id": "medium_billing_and_outage",
  "task_objective": "Handle a mixed queue by prioritizing urgent technical outage first, then resolve a billing refund...",
  "queue_size": 3,
  "ticket_summaries": [
    {
      "ticket_id": "T-2001",
      "subject": "Charged twice",
      "status": "open",
      "priority": "normal",
      "sla_minutes_remaining": 220
    },
    {
      "ticket_id": "T-2002",
      "subject": "Production down",
      "status": "open",
      "priority": "normal",
      "sla_minutes_remaining": 45  // ⚠️ URGENT
    },
    {
      "ticket_id": "T-2003",
      "subject": "How to change profile photo",
      "status": "open",
      "priority": "normal",
      "sla_minutes_remaining": 400
    }
  ]
}
```

### Key Actions (Optimal Path)

1. **Classify & Prioritize Outage (45 min SLA!):**
```json
{
  "action_type": "classify_ticket",
  "ticket_id": "T-2002",
  "category": "technical",
  "priority": "urgent"
}
```

2. **Assign to Technical Team:**
```json
{
  "action_type": "assign_ticket",
  "ticket_id": "T-2002",
  "assigned_team": "technical"
}
```

3. **Reply Before Closing:**
```json
{
  "action_type": "send_reply",
  "ticket_id": "T-2002",
  "content": "We have identified the outage and applied a mitigation. Service is restored."
}
```

4. **Close First Critical Ticket:**
```json
{
  "action_type": "close_ticket",
  "ticket_id": "T-2002"
}
```

5. **Then Handle Billing (Similar steps)...**

### Final Result

```json
{
  "final_score": 1.0,
  "grading_breakdown": {
    "outage_priority_and_team": 0.25,
    "outage_resolved_first": 0.15,
    "outage_closed": 0.2,
    "billing_assigned_and_refund_reply": 0.2,
    "billing_closed": 0.15,
    "avoid_unnecessary_escalation": 0.05
  }
}
```

✅ **Perfect Score for Correct Prioritization**

---

## Task 3: Hard - Security & Retention

### Complexity: 3 Tickets with Different Priorities

- **T-3001** (30 min SLA): Unknown login - REQUIRES ESCALATION
- **T-3002** (120 min SLA): Churn risk - REQUIRES CAREFUL EMPATHY
- **T-3003** (320 min SLA): Package delayed - ROUTINE

### Key Challenge Points

1. **Security ticket MUST be escalated before closure** (ordering matters)
2. **Retention ticket needs empathetic language** (score checks for keywords)
3. **SLA awareness** (30 min is critical)

### Optimal Actions

```json
// 1. Escalate security properly
{"action_type": "classify_ticket", "ticket_id": "T-3001", "category": "security", "priority": "urgent"}
{"action_type": "assign_ticket", "ticket_id": "T-3001", "assigned_team": "security"}
{"action_type": "escalate_ticket", "ticket_id": "T-3001"}

// 2. De-escalate with retention assignment
{"action_type": "classify_ticket", "ticket_id": "T-3002", "category": "account", "priority": "high"}
{"action_type": "assign_ticket", "ticket_id": "T-3002", "assigned_team": "retention"}
// Must include empathetic keywords: "sorry", "understand", "improve", "priority"
{"action_type": "send_reply", "ticket_id": "T-3002", "content": "We are sorry for the delays and understand your concern. We will prioritize your requests and improve response times."}
```

### Final Breakdown

```json
{
  "final_score": 1.0,
  "grading_breakdown": {
    "security_classified_escalated": 0.2,
    "security_ordering": 0.15,
    "security_closed": 0.15,
    "retention_assignment": 0.15,
    "retention_empathetic_reply": 0.15,
    "retention_closed": 0.1,
    "shipping_handled": 0.1
  }
}
```

✅ **Perfect Score for Complex Multi-Task Planning**

---

## How to Run Locally

```bash
# Start the interface
python app.py

# Navigate to http://localhost:7860 in your browser
```

## Key Observations

- **Reward is shaped step-by-step**, not just at episode end
- **Invalid actions get -0.12 penalty** (tight validation)
- **Repeating same action 3x in a row gets -0.05 loop penalty**
- **Grading is deterministic** - same actions always produce same score
- **All 3 tasks achievable at 1.0** with optimal play
