import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from inference import BaselineAgent, run_inference
from models import Action, Observation, Email


class TestBaselineAgent:
    def test_format_prompt_classification(self):
        agent = BaselineAgent()
        
        email = Email(
            sender="test@example.com",
            subject="Test Email",
            body="This is a test email body."
        )
        obs = Observation(
            current_email=email,
            task_type="classification",
            episode_number=0,
            total_episodes=10
        )
        
        prompt = agent.format_prompt(obs)
        
        assert "classification" in prompt.lower()
        assert "spam" in prompt.lower() or "urgent" in prompt.lower() or "normal" in prompt.lower()
        assert "JSON" in prompt
    
    def test_format_prompt_schedule_extraction(self):
        agent = BaselineAgent()
        
        email = Email(
            sender="scheduler@company.com",
            subject="Meeting Request",
            body="Can we meet Thursday at 2pm?"
        )
        obs = Observation(
            current_email=email,
            task_type="schedule_extraction",
            episode_number=0,
            total_episodes=10
        )
        
        prompt = agent.format_prompt(obs)
        
        assert "schedule" in prompt.lower()
        assert "meeting" in prompt.lower() or "meetings" in prompt.lower()
        assert "meetings" in prompt
    
    def test_format_prompt_multi_intent(self):
        agent = BaselineAgent()
        
        email = Email(
            sender="user@company.com",
            subject="Update and Question",
            body="Here's the update. Also, can we meet?"
        )
        obs = Observation(
            current_email=email,
            task_type="multi_intent",
            episode_number=0,
            total_episodes=10
        )
        
        prompt = agent.format_prompt(obs)
        
        assert "intent" in prompt.lower()
        assert "intents" in prompt.lower()
    
    def test_parse_response_valid_json(self):
        agent = BaselineAgent()
        
        response = '{"task_type": "classification", "category": "spam", "confidence": 0.9}'
        
        parsed = agent.parse_response(response, "classification")
        
        assert parsed["task_type"] == "classification"
        assert parsed["category"] == "spam"
        assert parsed["confidence"] == 0.9
    
    def test_parse_response_with_code_blocks(self):
        agent = BaselineAgent()
        
        response = '''```json
{"task_type": "schedule_extraction", "meetings": []}
```'''
        
        parsed = agent.parse_response(response, "schedule_extraction")
        
        assert parsed["task_type"] == "schedule_extraction"
    
    @patch('inference.OpenAI')
    def test_act_with_mock_client(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"task_type": "classification", "category": "normal", "confidence": 0.7}'
        
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = BaselineAgent()
        agent.client = mock_client
        
        email = Email(sender="test@example.com", subject="Test", body="Body")
        obs = Observation(current_email=email, task_type="classification", episode_number=0, total_episodes=10)
        
        action = agent.act(obs)
        
        assert isinstance(action, Action)
        assert action.task_type == "classification"
        assert action.category in ["spam", "urgent", "normal"]
    
    def test_mock_response_classification(self):
        agent = BaselineAgent()
        agent.client = None
        
        email = Email(sender="test@example.com", subject="URGENT: Test", body="This is urgent!")
        response = agent._mock_response("classification", email)
        
        parsed = agent.parse_response(response, "classification")
        assert "category" in parsed
        assert parsed["category"] in ["spam", "urgent", "normal"]
    
    def test_mock_response_schedule_extraction(self):
        agent = BaselineAgent()
        agent.client = None
    
        email = Email(
            sender="scheduler@company.com",
            subject="Meeting Request",
            body="Can we meet Thursday at 2pm for 1 hour?"
        )
        response = agent._mock_response("schedule_extraction", email)
        
        parsed = agent.parse_response(response, "schedule_extraction")
        assert "meetings" in parsed
        assert isinstance(parsed["meetings"], list)
    
    def test_mock_response_multi_intent(self):
        agent = BaselineAgent()
        agent.client = None
        
        email = Email(
            sender="user@company.com",
            subject="Complex request",
            body="I need to schedule a meeting and request documentation."
        )
        response = agent._mock_response("multi_intent", email)
        
        parsed = agent.parse_response(response, "multi_intent")
        assert "intents" in parsed
        assert isinstance(parsed["intents"], list)
    
    def test_history_in_prompt(self):
        agent = BaselineAgent()
        
        email = Email(
            sender="test@example.com",
            subject="Test Subject",
            body="Body content"
        )
        
        history = [
            {"sender": "prev@sender.com", "subject": "Previous email", "score": 0.85, "task_type": "classification"}
        ]
        
        obs = Observation(
            current_email=email,
            task_type="classification",
            history=history,
            episode_number=1,
            total_episodes=10
        )
        
        prompt = agent.format_prompt(obs)
        
        assert "0.85" in prompt or "history" in prompt.lower()


class TestInferenceFunction:
    def test_run_inference_returns_summary(self):
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            summary = run_inference(total_episodes=3, seed=42)
        finally:
            sys.stdout = old_stdout
        
        assert "total_episodes" in summary
        assert "average_reward" in summary
        assert "by_task" in summary
        assert summary["total_episodes"] == 3
    
    def test_run_inference_seed_reproducibility(self):
        import io
        import sys
        old_stdout = sys.stdout
        
        sys.stdout = io.StringIO()
        summary1 = run_inference(total_episodes=3, seed=42)
        sys.stdout = old_stdout
        
        sys.stdout = io.StringIO()
        summary2 = run_inference(total_episodes=3, seed=42)
        sys.stdout = old_stdout
        
        assert summary1["total_episodes"] == summary2["total_episodes"]
        assert summary1["average_reward"] == summary2["average_reward"]
    
    def test_run_inference_different_seeds(self):
        import io
        import sys
        old_stdout = sys.stdout
        
        sys.stdout = io.StringIO()
        summary1 = run_inference(total_episodes=3, seed=42)
        sys.stdout = old_stdout
        
        sys.stdout = io.StringIO()
        summary2 = run_inference(total_episodes=3, seed=999)
        sys.stdout = old_stdout
        
        assert summary1["total_episodes"] == summary2["total_episodes"]
    
    def test_run_inference_empty_episodes(self):
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            summary = run_inference(total_episodes=1, seed=42)
            assert summary["total_episodes"] == 1
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])