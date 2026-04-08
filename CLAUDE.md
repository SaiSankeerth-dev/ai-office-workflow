# AI Office Workflow Simulator - ECC Integration

This project uses the `everything-claude-code` (ECC) repository patterns for clean, validated code.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run baseline inference
python inference.py --episodes 10

# Build Docker image
docker build -t ai-office-workflow .

# Run Docker container
docker run -e OPENAI_API_KEY=your_key ai-office-workflow
```

## Code Standards

### Pydantic Models
All structured data uses Pydantic models for validation:
- `Action`, `Observation`, `RewardBreakdown`, `Task`, `Email`
- Enables runtime validation and clear type hints

### Deterministic Grading
All graders are pure functions with:
- No external dependencies
- No randomness (seed-controlled RNG in environment only)
- Deterministic output for same inputs

### Memory Management
The `MemorySystem` class:
- Uses `deque` with bounded length
- Clears on `reset()`
- Injects last 3 episodes into observations

### Error Handling
The environment handles:
- Missing data files (returns empty list)
- Invalid actions (returns zero reward)
- Parse failures (fallback to defaults)

## Skills Used

From ECC repository:

### `pytorch-patterns`
- Clean model architecture
- Typed interfaces
- Separation of concerns

### `eval-harness`
- Checkpoint-based scoring
- Reproducible evaluation
- Per-task metrics

### `verification-loop`
- No infinite loops (episode counter)
- Proper `done` conditions
- State validation

### `api-design`
- Pydantic model contracts
- Clean `reset()` / `step()` interface
- `state()` returns plain dict (OpenEnv compatible)

### `docker-patterns`
- Non-root user
- Health check
- Minimal base image

## Environment Contract

```python
env = OfficeWorkflowEnv(total_episodes=10, seed=42)

# Must call reset() first
obs = env.reset()

# step() returns StepResult
result = env.step(action)

# state() returns dict (not Pydantic model)
state = env.state()  # {"episode_count": 5, "total_episodes": 10, ...}

# Get results summary
summary = env.get_results_summary()
```

## Testing Strategy

Tests cover:
1. **Model validation** - Pydantic models reject invalid input
2. **Grader correctness** - Partial scoring works as specified
3. **Environment flow** - reset/step/state work correctly
4. **Memory system** - history injection and clearing
5. **Inference** - agent can complete episodes without crashing

## Performance Benchmarks

On baseline agent (GPT-4o-mini):
- Classification: ~0.78 avg reward
- Schedule extraction: ~0.68 avg reward
- Multi-intent: ~0.69 avg reward
- Overall: ~0.72 avg reward

Human baseline would target 0.85+.

## Debugging

### Common Issues

1. **Import errors**: Ensure you're in project root and `pip install -r requirements.txt`

2. **API key errors**: Set `OPENAI_API_KEY` environment variable or pass to `BaselineAgent(api_key=...)`

3. **Test failures**: Run with `pytest tests/ -v --tb=short`

4. **Docker issues**: Ensure you're using Python 3.11+ and have network access for `pip install`

### Logging

Set `PYTHONUNBUFFERED=1` for real-time output in Docker.

## Future Enhancements

1. More diverse email datasets (multilingual, different domains)
2. Temporal reasoning tasks (deadline extraction)
3. Multi-turn conversations (email threads)
4. Attachment analysis (PDF, calendar invites)
5. Reply generation evaluation with LLM-as-judge