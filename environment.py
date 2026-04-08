import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from models import Action, Observation, RewardBreakdown, StepResult, Task, Email
from memory import MemorySystem
from reward import RewardShaper
from graders import grade_classification, grade_schedule_extraction, grade_multi_intent


DATA_DIR = Path(__file__).parent / "data"


class OfficeWorkflowEnv:
    TASK_SEQUENCE = ["classification", "schedule_extraction", "multi_intent"]
    
    def __init__(
        self,
        total_episodes: int = 10,
        seed: Optional[int] = None,
        history_limit: int = 3,
        adaptive_difficulty: bool = True
    ):
        self.total_episodes = total_episodes
        self.seed_value = seed
        self.history_limit = history_limit
        self.adaptive_difficulty = adaptive_difficulty
        
        self.rng = random.Random(seed)
        
        self.memory = MemorySystem(max_history=total_episodes)
        self.reward_shaper = RewardShaper()
        
        self._load_data()
        
        self.current_episode = 0
        self.episode_results: List[Dict[str, Any]] = []
        self.current_task: Optional[Task] = None
        self.current_email_data: Optional[Dict[str, Any]] = None
        
        self.performance_history: Dict[str, List[float]] = {
            "classification": [],
            "schedule_extraction": [],
            "multi_intent": []
        }
        self.difficulty_multiplier: Dict[str, float] = {
            "classification": 1.0,
            "schedule_extraction": 1.0,
            "multi_intent": 1.0
        }
    
    def _load_data(self):
        self.task_data = {
            "classification": self._load_json(DATA_DIR / "task1_emails.json"),
            "schedule_extraction": self._load_json(DATA_DIR / "task2_emails.json"),
            "multi_intent": self._load_json(DATA_DIR / "task3_emails.json"),
        }
    
    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def reset(self, seed: int = None, episode_id: str = None, **kwargs) -> Observation:
        self.current_episode = 0
        self.episode_results = []
        self.memory.clear()
        
        if seed is not None:
            self.rng = random.Random(seed)
        
        obs = self._get_next_observation()
        obs.reward = None
        obs.done = False
        return obs
    
    async def reset_async(self, seed: int = None, episode_id: str = None, **kwargs) -> Observation:
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)
    
    def step(self, action: Action, timeout_s: float = None, **kwargs) -> StepResult:
        if not self.current_task:
            self.reset()
        
        if isinstance(action, dict):
            action = Action(**action)
        
        current_task_type = self.current_task.task_type
        current_difficulty = self.current_task.difficulty
        
        grader_score = self._grade_action(action)
        
        reward_dict = self.reward_shaper.compute(
            task_type=current_task_type,
            grader_score=grader_score,
            action=action.model_dump(),
            ground_truth=self.current_task.ground_truth
        )
        
        reward = RewardBreakdown(**reward_dict)
        
        self.memory.add(
            email=self.current_task.email.model_dump(),
            action=action.model_dump(),
            score=reward.final_reward,
            task_type=current_task_type
        )
        
        self.episode_results.append({
            "task_id": self.current_task.task_id,
            "task_type": current_task_type,
            "email": self.current_task.email.model_dump(),
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "difficulty": current_difficulty,
            "grader_score": grader_score
        })
        
        if self.adaptive_difficulty:
            self._update_performance_tracking(current_task_type, reward.final_reward)
        
        self.current_episode += 1
        done = self.current_episode >= self.total_episodes
        
        if not done:
            next_obs = self._get_next_observation()
        else:
            next_obs = Observation(
                current_email=Email(sender="", subject="", body=""),
                task_type="",
                history=self.memory.get_history(self.history_limit),
                episode_number=self.current_episode,
                total_episodes=self.total_episodes
            )
            self.current_task = None
            self.current_email_data = None
        
        info = {
            "task_type": current_task_type,
            "difficulty": current_difficulty,
            "grader_score": grader_score
        }
        
        next_obs.reward = reward.final_reward
        next_obs.done = done
        
        return next_obs
    
    async def step_async(self, action: Action, timeout_s: float = None, **kwargs) -> Observation:
        return self.step(action, timeout_s=timeout_s, **kwargs)
    
    def close(self) -> None:
        pass
    
    async def step_async(self, action: Action, timeout_s: float = None, **kwargs) -> Observation:
        return self.step(action, timeout_s=timeout_s, **kwargs)
    
    def _get_next_observation(self) -> Observation:
        task_idx = self.current_episode % len(self.TASK_SEQUENCE)
        task_type = self.TASK_SEQUENCE[task_idx]
        
        email_data, task = self._sample_task(task_type)
        
        self.current_task = task
        self.current_email_data = email_data
        
        return Observation(
            current_email=task.email,
            task_type=task_type,
            history=self.memory.get_history(self.history_limit),
            episode_number=self.current_episode,
            total_episodes=self.total_episodes
        )
    
    def _sample_task(self, task_type: str) -> Tuple[Dict[str, Any], Task]:
        data = self.task_data.get(task_type, [])
        
        if not data:
            placeholder = {
                "id": f"default_{task_type}",
                "sender": "system@test.com",
                "subject": "Test email",
                "body": "This is a placeholder email."
            }
            return placeholder, self._create_task(placeholder, task_type)
        
        email_data = self.rng.choice(data)
        return email_data, self._create_task(email_data, task_type)
    
    def _create_task(self, email_data: Dict[str, Any], task_type: str) -> Task:
        email = Email(
            sender=email_data.get("sender", "unknown@example.com"),
            subject=email_data.get("subject", ""),
            body=email_data.get("body", ""),
            timestamp=email_data.get("timestamp")
        )
        
        ground_truth = self._extract_ground_truth(email_data, task_type)
        
        difficulty = self._get_difficulty(email_data, task_type)
        
        return Task(
            task_id=email_data.get("id", f"task_{task_type}_{self.current_episode}"),
            task_type=task_type,
            email=email,
            ground_truth=ground_truth,
            difficulty=difficulty
        )
    
    def _extract_ground_truth(self, email_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        if task_type == "classification":
            return {
                "expected_category": email_data.get("expected_category", "normal"),
                "tone": email_data.get("tone", "")
            }
        elif task_type == "schedule_extraction":
            return {
                "scheduling_intent": email_data.get("scheduling_intent", False),
                "meetings": email_data.get("meetings", [])
            }
        elif task_type == "multi_intent":
            return {
                "expected_intents": email_data.get("expected_intents", []),
                "primary_intent": email_data.get("primary_intent"),
                "sentiment": email_data.get("sentiment"),
                "urgency": email_data.get("urgency")
            }
        return {}
    
    def _get_difficulty(self, email_data: Dict[str, Any], task_type: str) -> str:
        if task_type == "classification":
            category = email_data.get("expected_category", "")
            if category == "urgent":
                return "easy"
            elif category == "spam":
                return "medium"
            else:
                return "easy"
        
        elif task_type == "schedule_extraction":
            meetings = email_data.get("meetings", [])
            if len(meetings) == 0:
                return "easy"
            elif len(meetings) <= 2:
                return "medium"
            else:
                return "hard"
        
        elif task_type == "multi_intent":
            intents = email_data.get("expected_intents", [])
            if len(intents) <= 2:
                return "easy"
            elif len(intents) <= 3:
                return "medium"
            else:
                return "hard"
        
        return "medium"
    
    def _grade_action(self, action: Action) -> float:
        if self.current_task.task_type == "classification":
            return grade_classification(
                action.model_dump(),
                self.current_task.ground_truth
            )
        elif self.current_task.task_type == "schedule_extraction":
            return grade_schedule_extraction(
                action.model_dump(),
                self.current_task.ground_truth
            )
        elif self.current_task.task_type == "multi_intent":
            return grade_multi_intent(
                action.model_dump(),
                self.current_task.ground_truth
            )
        return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "episode_count": self.current_episode,
            "total_episodes": self.total_episodes,
            "current_task": self.current_task.task_type if self.current_task else "",
            "cumulative_reward": sum(
                r["reward"]["final_reward"] for r in self.episode_results
            ) if self.episode_results else 0.0,
            "done": self.current_episode >= self.total_episodes
        }
    
    @property
    def state(self) -> Dict[str, Any]:
        return self.get_state()
    
    def get_results_summary(self) -> Dict[str, Any]:
        if not self.episode_results:
            return {"total_episodes": 0, "average_reward": 0.0}
        
        rewards = [r["reward"]["final_reward"] for r in self.episode_results]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        by_task = {}
        for result in self.episode_results:
            task_type = result["task_type"]
            if task_type not in by_task:
                by_task[task_type] = {"count": 0, "total_reward": 0.0}
            by_task[task_type]["count"] += 1
            by_task[task_type]["total_reward"] += result["reward"]["final_reward"]
        
        for task_type in by_task:
            count = by_task[task_type]["count"]
            by_task[task_type]["average_reward"] = (
                by_task[task_type]["total_reward"] / count if count > 0 else 0.0
            )
        
        return {
            "total_episodes": len(self.episode_results),
            "average_reward": round(avg_reward, 3),
            "min_reward": round(min(rewards), 3) if rewards else 0.0,
            "max_reward": round(max(rewards), 3) if rewards else 0.0,
            "by_task": {
                k: {
                    "count": v["count"],
                    "average_reward": round(v["average_reward"], 3)
                } for k, v in by_task.items()
            },
            "difficulty_multiplier": self.difficulty_multiplier.copy()
        }
    
    def _update_performance_tracking(self, task_type: str, reward: float):
        if task_type not in self.performance_history:
            self.performance_history[task_type] = []
        
        self.performance_history[task_type].append(reward)
        
        if len(self.performance_history[task_type]) > 6:
            self.performance_history[task_type] = self.performance_history[task_type][-6:]
        
        if len(self.performance_history[task_type]) >= 3:
            recent = self.performance_history[task_type][-3:]
            older = self.performance_history[task_type][:max(1, len(self.performance_history[task_type])-3)]
            
            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                
                if recent_avg > older_avg + 0.1:
                    self.difficulty_multiplier[task_type] = min(1.3, self.difficulty_multiplier[task_type] * 1.05)
                elif recent_avg < older_avg - 0.1:
                    self.difficulty_multiplier[task_type] = max(0.7, self.difficulty_multiplier[task_type] * 0.95)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        stats = {}
        for task_type, rewards in self.performance_history.items():
            if rewards:
                stats[task_type] = {
                    "episodes": len(rewards),
                    "avg_reward": sum(rewards) / len(rewards),
                    "difficulty_multiplier": self.difficulty_multiplier.get(task_type, 1.0)
                }
        return stats