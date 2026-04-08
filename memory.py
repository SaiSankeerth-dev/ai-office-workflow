from typing import List, Dict, Any
from collections import deque


class MemorySystem:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.episodes: deque = deque(maxlen=max_history)
    
    def add(self, email: dict, action: dict, score: float, task_type: str):
        self.episodes.append({
            "email": email,
            "action": action,
            "score": score,
            "task_type": task_type
        })
    
    def get_history(self, limit: int = 3) -> List[Dict[str, Any]]:
        recent = list(self.episodes)[-limit:]
        return [
            {
                "sender": ep["email"].get("sender", "unknown"),
                "subject": ep["email"].get("subject", ""),
                "action_taken": ep["action"].get("category") or ep["action"].get("intents", []),
                "score": ep["score"],
                "task_type": ep["task_type"]
            }
            for ep in recent
        ]
    
    def clear(self):
        self.episodes.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.episodes:
            return {"total_episodes": 0, "avg_score": 0.0, "by_task": {}}
        
        total = len(self.episodes)
        avg = sum(ep["score"] for ep in self.episodes) / total
        
        by_task = {}
        for ep in self.episodes:
            task = ep["task_type"]
            if task not in by_task:
                by_task[task] = {"count": 0, "avg_score": 0.0, "scores": []}
            by_task[task]["count"] += 1
            by_task[task]["scores"].append(ep["score"])
        
        for task in by_task:
            scores = by_task[task]["scores"]
            by_task[task]["avg_score"] = sum(scores) / len(scores)
            del by_task[task]["scores"]
        
        return {"total_episodes": total, "avg_score": avg, "by_task": by_task}