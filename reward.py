from typing import Dict, Any


class RewardShaper:
    WEIGHTS = {
        "classification": {"accuracy": 0.5, "completeness": 0.3, "efficiency": 0.2},
        "schedule_extraction": {"accuracy": 0.4, "completeness": 0.4, "efficiency": 0.2},
        "multi_intent": {"accuracy": 0.3, "completeness": 0.4, "efficiency": 0.3},
    }
    
    PENALTIES = {
        "missing_field": 0.1,
        "invalid_category": 0.15,
        "empty_action": 0.2,
        "format_error": 0.1,
    }
    
    def __init__(self, weights: Dict[str, Dict[str, float]] = None):
        if weights:
            self.WEIGHTS.update(weights)
    
    def compute(
        self,
        task_type: str,
        grader_score: float,
        action: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        weights = self.WEIGHTS.get(task_type, {"accuracy": 0.33, "completeness": 0.33, "efficiency": 0.34})
        
        accuracy = grader_score
        completeness = self._compute_completeness(action, ground_truth, task_type)
        efficiency = self._compute_efficiency(action)
        penalty = self._compute_penalty(action, task_type)
        
        final = max(0.0, min(1.0,
            weights["accuracy"] * accuracy +
            weights["completeness"] * completeness +
            weights["efficiency"] * efficiency -
            penalty
        ))
        
        return {
            "accuracy": round(accuracy, 3),
            "completeness": round(completeness, 3),
            "efficiency": round(efficiency, 3),
            "penalty": round(penalty, 3),
            "final_reward": round(final, 3)
        }
    
    def _compute_completeness(self, action: Dict[str, Any], ground_truth: Dict[str, Any], task_type: str) -> float:
        if task_type == "classification":
            return 1.0 if action.get("category") else 0.0
        
        elif task_type == "schedule_extraction":
            expected_count = len(ground_truth.get("meetings", []))
            actual_count = len(action.get("meetings", [])) if action.get("meetings") else 0
            if expected_count == 0:
                return 1.0 if actual_count == 0 else 0.5
            return min(1.0, actual_count / expected_count)
        
        elif task_type == "multi_intent":
            expected = set(ground_truth.get("expected_intents", []))
            actual = set(action.get("intents", [])) if action.get("intents") else set()
            if not expected:
                return 1.0
            intersection = len(expected & actual)
            return intersection / len(expected)
        
        return 0.5
    
    def _compute_efficiency(self, action: Dict[str, Any]) -> float:
        if not action:
            return 0.0
        
        has_reasoning = action.get("reasoning") is not None
        has_confidence = action.get("confidence") is not None
        
        score = 0.55
        
        if has_reasoning:
            reasoning = str(action.get("reasoning", ""))
            reasoning_length = len(reasoning)
            
            if reasoning_length > 10:
                score += 0.08
            if reasoning_length > 50:
                score += 0.08
            if reasoning_length > 100:
                score += 0.04
            
            reasoning_quality = self._assess_reasoning_quality(reasoning)
            score += reasoning_quality * 0.15
            
            explainability = self._assess_explainability(reasoning, action)
            score += explainability * 0.1
        
        if has_confidence and 0 <= action.get("confidence", 0) <= 1:
            score += 0.05
        
        return min(1.0, score)
    
    def _assess_reasoning_quality(self, reasoning: str) -> float:
        reasoning_lower = reasoning.lower()
        
        quality_indicators = [
            "because", "since", "therefore", "indicates", "suggests",
            "based on", "due to", "shows", "reveals", "evidence"
        ]
        
        length_bonus = min(1.0, len(reasoning) / 200)
        indicator_bonus = min(1.0, sum(1 for ind in quality_indicators if ind in reasoning_lower) * 0.2)
        
        return (length_bonus + indicator_bonus) / 2
    
    def _assess_explainability(self, reasoning: str, action: Dict[str, Any]) -> float:
        reasoning_lower = reasoning.lower()
        
        causal_words = sum(1 for w in ["because", "since", "therefore", "thus", "hence"] if w in reasoning_lower)
        evidence_words = sum(1 for w in ["evidence", "shows", "indicates", "suggests", "demonstrates"] if w in reasoning_lower)
        
        specific_words = sum(1 for w in ["specifically", "particular", "exact", "precisely", "this"] if w in reasoning_lower)
        
        action_words = sum(1 for w in ["will", "would", "should", "may", "could"] if w in reasoning_lower)
        
        total = causal_words * 0.3 + evidence_words * 0.3 + specific_words * 0.2 + action_words * 0.2
        
        if total >= 3:
            return 0.9
        elif total >= 2:
            return 0.7
        elif total >= 1:
            return 0.5
        return 0.2
    
    def _compute_penalty(self, action: Dict[str, Any], task_type: str) -> float:
        penalty = 0.0
        
        if not action:
            return self.PENALTIES["empty_action"]
        
        if task_type == "classification":
            if not action.get("category"):
                penalty += self.PENALTIES["missing_field"]
            elif action["category"] not in ["spam", "urgent", "normal"]:
                penalty += self.PENALTIES["invalid_category"]
        
        elif task_type == "schedule_extraction":
            if action.get("meetings") is None:
                penalty += self.PENALTIES["missing_field"] * 0.5
        
        elif task_type == "multi_intent":
            if not action.get("intents"):
                penalty += self.PENALTIES["missing_field"]
        
        return penalty