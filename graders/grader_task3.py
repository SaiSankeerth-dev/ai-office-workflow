from typing import List, Dict, Any, Set
from difflib import SequenceMatcher


VALID_INTENTS: Set[str] = {
    "schedule_meeting", "request_meeting", "request_information", "request_document", "request_documents",
    "request_documentation", "request_approval", "complaint", "compliment", "cancel_subscription", "request_refund",
    "request_replacement", "out_of_office_notification", "status_update", "deadline_notification",
    "escalate_issue", "request_action", "request_review", "request_feedback", "social_invitation",
    "provide_information", "confirm_action", "information_request", "interview_invitation",
    "request_schedule", "request_status", "request_update", "request_resolution"
}


def grade_multi_intent(action: dict, ground_truth: dict) -> float:
    predicted_intents = action.get("intents", [])
    expected_intents = ground_truth.get("expected_intents", [])
    
    if not expected_intents:
        if not predicted_intents:
            return 1.0
        else:
            return max(0.0, 1.0 - len(predicted_intents) * 0.1)
    
    if not predicted_intents and expected_intents:
        return 0.0
    
    valid_predicted = []
    for intent in predicted_intents:
        intent_lower = intent.lower().strip()
        
        if intent_lower in VALID_INTENTS:
            valid_predicted.append(intent_lower)
            continue
        
        alt_form = intent_lower.replace(" ", "_")
        if alt_form in VALID_INTENTS:
            valid_predicted.append(alt_form)
            continue
        
        alt_form2 = intent_lower.replace("_", " ")
        if alt_form2 in VALID_INTENTS:
            valid_predicted.append(alt_form2)
            continue
        
        for valid_intent in VALID_INTENTS:
            similarity = SequenceMatcher(None, intent_lower, valid_intent).ratio()
            if similarity > 0.7:
                valid_predicted.append(valid_intent)
                break
    
    invalid_count = len(predicted_intents) - len(valid_predicted)
    gaming_penalty = 0.0
    if invalid_count > 0:
        gaming_penalty = min(0.3, invalid_count * 0.1)
    
    exact_matches = 0
    fuzzy_matches = 0
    matched_expected = set()
    
    predicted_normalized = [_normalize_intent(i) for i in valid_predicted]
    expected_normalized = [_normalize_intent(i) for i in expected_intents]
    
    for pred_norm, pred_orig in zip(predicted_normalized, valid_predicted):
        if pred_norm in expected_normalized:
            idx = expected_normalized.index(pred_norm)
            if idx not in matched_expected:
                exact_matches += 1
                matched_expected.add(idx)
            continue
        
        best_score = 0
        best_idx = -1
        for idx, exp_norm in enumerate(expected_normalized):
            if idx in matched_expected:
                continue
            similarity = SequenceMatcher(None, pred_norm, exp_norm).ratio()
            if similarity > best_score:
                best_score = similarity
                best_idx = idx
        
        if best_score > 0.75 and best_idx >= 0:
            fuzzy_matches += 1
            matched_expected.add(best_idx)
    
    total_matches = exact_matches + fuzzy_matches
    precision = total_matches / len(valid_predicted) if valid_predicted else 0.0
    recall = total_matches / len(expected_intents) if expected_intents else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    score = 0.4 * precision + 0.4 * recall + 0.2 * f1
    
    score = max(0.0, score - gaming_penalty)
    
    sentiment = action.get("sentiment")
    expected_sentiment = ground_truth.get("sentiment")
    if sentiment and expected_sentiment:
        if sentiment.lower().strip() == expected_sentiment.lower().strip():
            score = min(1.0, score + 0.1)
    
    urgency = action.get("urgency")
    expected_urgency = ground_truth.get("urgency")
    if urgency and expected_urgency:
        if urgency.lower().strip() == expected_urgency.lower().strip():
            score = min(1.0, score + 0.05)
    
    suggested_reply = action.get("suggested_reply")
    if suggested_reply and isinstance(suggested_reply, str):
        reply_length = len(suggested_reply.strip())
        
        if reply_length > 20:
            score = min(1.0, score + 0.05)
        
        reply_quality = _assess_reply_quality(suggested_reply)
        if reply_quality > 0.5:
            score = min(1.0, score + 0.1)
    
    reasoning = action.get("reasoning")
    if reasoning and isinstance(reasoning, str) and len(reasoning.strip()) > 30:
        score = min(1.0, score + 0.05)
    
    return max(0.0, min(1.0, round(score, 3)))


def _normalize_intent(intent: str) -> str:
    if not intent:
        return ""
    
    normalized = intent.lower().strip()
    normalized = normalized.replace("_", " ").replace("-", " ")
    
    aliases = {
        "request docs": "request_document",
        "request doc": "request_document",
        "request documentation": "request_document",
        "docs request": "request_document",
    }
    if normalized in aliases:
        normalized = aliases[normalized]
    
    stopwords = ['a', 'an', 'the', 'to', 'for', 'and', 'or']
    words = [w for w in normalized.split() if w not in stopwords]
    
    return ' '.join(words)


def intent_similarity(intent1: str, intent2: str) -> float:
    norm1 = _normalize_intent(intent1)
    norm2 = _normalize_intent(intent2)
    
    if norm1 == norm2:
        return 1.0
    
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return similarity


def _assess_reply_quality(reply: str) -> float:
    reply_lower = reply.lower()
    
    quality_indicators = [
        "thank", "appreciate", "happy", "glad", "would", "could",
        "respond", "contact", "reach", "follow up", "let me"
    ]
    
    action_words = sum(1 for word in ["will", "would", "could", "should", "may"] if word in reply_lower)
    courtesy_words = sum(1 for word in ["thank", "please", "appreciate", "regards"] if word in reply_lower)
    specificity = sum(1 for word in ["tomorrow", "today", "monday", "confirm", "schedule"] if word in reply_lower)
    
    total = action_words + courtesy_words + specificity
    
    if total >= 4:
        return 0.9
    elif total >= 3:
        return 0.7
    elif total >= 2:
        return 0.5
    elif total >= 1:
        return 0.3
    return 0.1