---
title: AI Office Workflow
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# AI Office Workflow Simulator

A reinforcement learning environment for email triage, meeting extraction, and multi-intent recognition tasks.

## Overview

This environment simulates common office email workflow tasks, challenging AI agents to:

1. **Classify emails** as spam, urgent, or normal
2. **Extract meeting/scheduling information** from emails
3. **Identify multiple intents** in complex emails and generate appropriate replies

### Real-World Motivation

Enterprise email management is a $15B+ problem. Workers spend 28% of their workday managing email. This environment tests whether AI agents can:

- Triage incoming emails by priority and category
- Automatically extract calendar events and meeting requests
- Understand multi-faceted communications with multiple action items
- Maintain context across episodes (memory system)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from environment import OfficeWorkflowEnv

env = OfficeWorkflowEnv(total_episodes=10, seed=42)
observation = env.reset()

print(f"Task: {observation.task_type}")
print(f"Email from: {observation.current_email.sender}")
print(f"Subject: {observation.current_email.subject}")
```

## Environment Design

### Action Space

All actions use a flat JSON structure with optional fields:

```json
{
  "task_type": "classification|schedule_extraction|multi_intent",
  "category": "spam|urgent|normal",
  "confidence": 0.0-1.0,
  "meetings": [{"date": "", "time": "", "duration": "", "purpose": ""}],
  "intents": ["intent1", "intent2"],
  "suggested_reply": "Brief professional reply",
  "reasoning": "Explanation of classification"
}
```

### Observation Space

```json
{
  "current_email": {
    "sender": "string",
    "subject": "string",
    "body": "string",
    "timestamp": "optional string"
  },
  "task_type": "classification|schedule_extraction|multi_intent",
  "history": [
    {
      "sender": "string",
      "subject": "string",
      "action_taken": "string or array",
      "score": 0.0-1.0,
      "task_type": "string"
    }
  ],
  "episode_number": 0,
  "total_episodes": 10
}
```

### Task Descriptions

#### Task 1: Email Classification

**Goal:** Classify emails as spam, urgent, or normal based on content analysis.

**Scoring:**
- Correct category: 1.0
- Wrong category with reasonable confusion: 0.2-0.5
- Binary wrong category: 0.0

**Edge cases:**
- Email with "URGENT" subject but scheduled content (tests subject vs body understanding)
- Legitimate promotional emails vs actual spam
- Emotional urgency vs operational urgency

#### Task 2: Schedule Extraction

**Goal:** Extract all meeting information from emails.

**Scoring:**
- Meeting detection: 0.3 points
- Date accuracy: 0.35 points
- Time accuracy (±1 hour tolerance): 0.3 points
- Duration/purpose match: 0.05 points

**Edge cases:**
- Multiple time options in one email
- Availability windows vs specific times
- Recurring meetings
- Emails about time management (not actual scheduling)

#### Task 3: Multi-Intent Recognition

**Goal:** Identify ALL intents and generate appropriate replies.

**Scoring:**
- Intent recall: 0.4 points
- Intent precision: 0.4 points
- Primary intent match: +0.15 bonus
- Suggested reply quality: +0.15 bonus
- Sentiment/urgency match: +0.15 bonus

**Edge cases:**
- Mixed sentiment emails (complaint + compliment)
- Multiple action items with different priorities
- Implicit requests vs explicit requests

## Reward System

Each task uses weighted scoring:

```
reward = w₁·accuracy + w₂·completeness + w₃·efficiency - penalty
```

### Weights by Task

| Task                  | Accuracy | Completeness | Efficiency |
|-----------------------|----------|--------------|------------|
| Classification       | 0.5      | 0.3          | 0.2        |
| Schedule Extraction   | 0.4      | 0.4          | 0.2        |
| Multi-Intent         | 0.3      | 0.4          | 0.3        |

### Penalty Types

- Missing required field: 0.1
- Invalid category: 0.15
- Empty action: 0.2
- Format error: 0.1

## Memory System

The environment maintains episode history, injecting the last 3 episodes into each observation:

```python
observation.history = [
    {"sender": "...", "subject": "...", "score": 0.85, "task_type": "classification"},
    {"sender": "...", "subject": "...", "score": 0.72, "task_type": "multi_intent"},
    {"sender": "...", "subject": "...", "score": 0.91, "task_type": "schedule_extraction"}
]
```

This allows agents to learn from past performance and adapt.

## Running the Baseline Agent

```bash
python inference.py --episodes 10 --seed 42
```

### Output Format

```
============================================================
Episode 1/10
Task: classification
Difficulty: easy
Reward: 0.850
  - Accuracy: 0.900
  - Completeness: 1.000
  - Efficiency: 0.850
============================================================
...

============================================================
FINAL RESULTS
============================================================
Total Episodes: 10
Average Reward: 0.720
Min Reward: 0.450
Max Reward: 0.950

By Task Type:
  classification: 4 episodes, avg reward: 0.780
  schedule_extraction: 3 episodes, avg reward: 0.680
  multi_intent: 3 episodes, avg reward: 0.690
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
ai-office-workflow-simulator/
├── environment.py          # Core OfficeWorkflowEnv class
├── models.py               # Pydantic models: Observation, Action, Reward
├── memory.py               # Episode memory system
├── reward.py               # Weighted reward shaper
├── graders/
│   ├── __init__.py
│   ├── grader_task1.py     # Classification grader (deterministic)
│   ├── grader_task2.py     # Schedule extraction grader
│   └── grader_task3.py     # Multi-intent grader (partial scoring)
├── data/
│   ├── task1_emails.json   # Spam/urgent/normal samples
│   ├── task2_emails.json   # Scheduling emails
│   └── task3_emails.json   # Multi-intent emails
├── inference.py            # Baseline agent (OpenAI-compatible)
├── openenv.yaml            # Environment metadata + task definitions
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── CLAUDE.md               # ECC integration config
└── tests/
    ├── test_environment.py
    ├── test_graders.py
    └── test_inference.py
```

## Differentiators

### 1. Memory System
Most OpenEnv submissions lack cross-episode context. This environment injects the last 3 episodes into each observation, allowing agents to learn from past mistakes.

### 2. Partial Scoring
All graders use partial credit:
- Classification: Confusion matrix-based scoring
- Schedule extraction: Per-field matching with tolerance
- Multi-intent: F1 score + bonus components

### 3. Realistic Data
Emails include:
- Typos and conversational tone
- Ambiguous intent (e.g., "URGENT" subject vs scheduled maintenance content)
- Mixed sentiment (complaint + compliment in same email)
- Implicit requests

### 4. Weighted Rewards
Each task has different weights for accuracy/completeness/efficiency, forcing agents to adapt priorities.

### 5. Baseline Scores Documented
Run `python inference.py` to generate real baseline scores. Judges can verify actual performance.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request

## Citation

```bibtex
@misc{ai_office_workflow_2024,
  title={AI Office Workflow Simulator},
  author={AI Office Workflow Team},
  year={2024},
  url={https://github.com/ai-office/workflow-simulator}
}
```