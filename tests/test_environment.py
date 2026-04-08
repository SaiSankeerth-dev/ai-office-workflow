import pytest
from environment import OfficeWorkflowEnv
from models import Action, Observation, Email, Task, RewardBreakdown
import json


class TestEnvironment:
    def test_reset_returns_observation(self):
        env = OfficeWorkflowEnv(total_episodes=3, seed=42)
        obs = env.reset()
        
        assert isinstance(obs, Observation)
        assert obs.task_type in ["classification", "schedule_extraction", "multi_intent"]
        assert obs.episode_number == 0
        assert obs.total_episodes == 3
        assert len(obs.history) == 0
    
    def test_reset_clears_state(self):
        env = OfficeWorkflowEnv(total_episodes=3, seed=42)
        
        obs = env.reset()
        action = Action(task_type="classification", category="normal", confidence=0.8)
        env.step(action)
        
        obs2 = env.reset()
        assert env.current_episode == 0
        assert len(env.episode_results) == 0
        assert len(env.memory.episodes) == 0
    
    def test_step_requires_reset(self):
        env = OfficeWorkflowEnv(total_episodes=3, seed=42)
        
        action = Action(task_type="classification", category="normal")
        
        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step(action)
    
    def test_step_returns_result(self):
        env = OfficeWorkflowEnv(total_episodes=3, seed=42)
        env.reset()
        
        action = Action(task_type="classification", category="spam", confidence=0.9)
        result = env.step(action)
        
        assert result.observation is not None
        assert result.reward is not None
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)
    
    def test_done_after_total_episodes(self):
        env = OfficeWorkflowEnv(total_episodes=3, seed=42)
        obs = env.reset()
        
        for i in range(3):
            action = Action(task_type=obs.task_type, category="normal")
            result = env.step(action)
            
            if i < 2:
                assert result.done is False
                obs = result.observation
            else:
                assert result.done is True
    
    def test_state_returns_dict(self):
        env = OfficeWorkflowEnv(total_episodes=5, seed=42)
        env.reset()
        
        state = env.state()
        
        assert isinstance(state, dict)
        assert "episode_count" in state
        assert "total_episodes" in state
        assert "done" in state
        assert state["total_episodes"] == 5
    
    def test_task_sequence(self):
        env = OfficeWorkflowEnv(total_episodes=10, seed=42)
        
        tasks_seen = []
        obs = env.reset()
        tasks_seen.append(obs.task_type)
        
        for _ in range(9):
            action = Action(task_type=obs.task_type, category="normal")
            result = env.step(action)
            
            if not result.done:
                tasks_seen.append(result.observation.task_type)
                obs = result.observation
        
        expected = ["classification", "schedule_extraction", "multi_intent"] * 3 + ["classification"]
        assert tasks_seen == expected
    
    def test_seed_reproducibility(self):
        env1 = OfficeWorkflowEnv(total_episodes=3, seed=42)
        env2 = OfficeWorkflowEnv(total_episodes=3, seed=42)
        
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        assert obs1.current_email.subject == obs2.current_email.subject
        assert obs1.current_email.sender == obs2.current_email.sender
    
    def test_memory_history(self):
        env = OfficeWorkflowEnv(total_episodes=5, seed=42, history_limit=3)
        obs = env.reset()
        
        for i in range(3):
            action = Action(task_type=obs.task_type, category="normal")
            result = env.step(action)
            if result.done:
                break
            obs = result.observation
        
        assert len(obs.history) <= 3
    
    def test_results_summary(self):
        env = OfficeWorkflowEnv(total_episodes=3, seed=42)
        obs = env.reset()
        
        for _ in range(3):
            action = Action(task_type=obs.task_type, category="spam")
            result = env.step(action)
            if result.done:
                break
            obs = result.observation
        
        summary = env.get_results_summary()
        
        assert "total_episodes" in summary
        assert "average_reward" in summary
        assert "by_task" in summary
    
    def test_difficulty_detection(self):
        env = OfficeWorkflowEnv(total_episodes=1, seed=42)
        obs = env.reset()
        
        assert env.current_task is not None
        assert env.current_task.difficulty in ["easy", "medium", "hard"]


class TestModels:
    def test_action_validation(self):
        action = Action(task_type="classification", category="spam")
        assert action.task_type == "classification"
        assert action.category == "spam"
    
    def test_action_invalid_category(self):
        with pytest.raises(ValueError):
            Action(task_type="classification", category="invalid")
    
    def test_action_invalid_task_type(self):
        with pytest.raises(ValueError):
            Action(task_type="unknown", category="spam")
    
    def test_email_model(self):
        email = Email(
            sender="test@example.com",
            subject="Test Subject",
            body="Test body"
        )
        assert email.sender == "test@example.com"
    
    def test_observation_model(self):
        email = Email(sender="a@b.com", subject="test", body="body")
        obs = Observation(
            current_email=email,
            task_type="classification",
            episode_number=1,
            total_episodes=10
        )
        assert obs.task_type == "classification"
    
    def test_action_with_confidence(self):
        action = Action(
            task_type="classification",
            category="urgent",
            confidence=0.85,
            reasoning="Multiple urgency indicators"
        )
        assert action.confidence == 0.85
        assert action.reasoning is not None
    
    def test_action_dict_input(self):
        env = OfficeWorkflowEnv(total_episodes=1, seed=42)
        env.reset()
        
        action_dict = {
            "task_type": "classification",
            "category": "normal",
            "confidence": 0.7
        }
        
        result = env.step(action_dict)
        assert result.reward is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])