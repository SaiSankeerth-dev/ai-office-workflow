def grade_classification(action: dict, ground_truth: dict) -> float:
    predicted = action.get("category")
    if predicted is None:
        return 0.0
    predicted = str(predicted).lower().strip()
    expected = ground_truth.get("expected_category", "").lower().strip()
    
    if not predicted:
        return 0.0
    
    if predicted == expected:
        base_score = 1.0
    else:
        wrong_category_penalty = 0.3
        
        if expected == "urgent" and predicted == "normal":
            base_score = 0.3
        elif expected == "normal" and predicted == "urgent":
            base_score = 0.5
        elif expected == "spam" and predicted == "normal":
            base_score = 0.2
        elif expected == "normal" and predicted == "spam":
            base_score = 0.4
        elif expected in ["urgent", "spam"] and predicted == "spam":
            base_score = 0.1
        else:
            base_score = 0.3 - wrong_category_penalty
        
        base_score = max(0.0, base_score)
    
    confidence = action.get("confidence")
    if confidence is not None:
        if 0.0 <= confidence <= 1.0:
            if predicted == expected:
                if confidence >= 0.7:
                    base_score = min(1.0, base_score + 0.1)
            else:
                if confidence >= 0.8:
                    base_score = max(0.0, base_score - 0.1)
                elif confidence <= 0.5:
                    base_score = base_score + 0.05
    
    reasoning = action.get("reasoning")
    if reasoning and len(str(reasoning)) > 20:
        base_score = min(1.0, base_score + 0.05)
    
    return max(0.0, min(1.0, base_score))