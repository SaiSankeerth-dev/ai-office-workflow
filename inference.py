import os
import sys
import json
import re
from typing import Dict, Any, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from environment import OfficeWorkflowEnv
from models import Action


class BaselineAgent:
    VALID_INTENTS = {
        "schedule_meeting", "request_meeting", "request_information", "request_document",
        "request_approval", "complaint", "compliment", "cancel_subscription", "request_refund",
        "request_replacement", "out_of_office_notification", "status_update", "deadline_notification",
        "escalate_issue", "request_action", "request_review", "request_feedback", "social_invitation",
        "provide_information", "confirm_action", "information_request", "interview_invitation"
    }
    
    def __init__(
        self,
        api_base: str = None,
        model_name: str = None,
        api_key: str = None,
        hf_token: str = None
    ):
        self.api_base = api_base or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.client = None
        
        if OpenAI and self.api_key and self.api_key != "sk-test":
            try:
                self.client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key
                )
            except Exception:
                self.client = None
        
        self.episode_history: List[Dict[str, Any]] = []
        self.performance_by_task: Dict[str, List[float]] = {
            "classification": [],
            "schedule_extraction": [],
            "multi_intent": []
        }
        
        self.reflection_enabled = True
        self.last_action_quality = None
        self.consistency_score = 1.0
    
    def _update_performance(self, task_type: str, reward: float):
        self.performance_by_task[task_type].append(reward)
        if len(self.performance_by_task[task_type]) > 5:
            self.performance_by_task[task_type] = self.performance_by_task[task_type][-5:]
    
    def _get_task_confidence(self, task_type: str) -> float:
        history = self.performance_by_task.get(task_type, [])
        if not history:
            return 0.5
        return sum(history) / len(history)
    
    def format_prompt(self, observation) -> str:
        email = observation.current_email
        task_type = observation.task_type
        history = observation.history
        
        perf = self._get_task_confidence(task_type)
        
        history_str = ""
        if history:
            history_str = "\n\nRecent performance:\n"
            for i, h in enumerate(history[-3:], 1):
                score = h.get('score', 0)
                subject = h.get('subject', 'Unknown')[:35]
                task = h.get('task_type', 'unknown')[:15]
                history_str += f"{i}. [{task}] {subject}... -> {score:.2f}\n"
            
            if perf > 0.7:
                history_str += "\nYour recent performance is EXCELLENT. Maintain this quality."
            elif perf > 0.5:
                history_str += "\nYou are improving. Focus on accuracy."
            else:
                history_str += "\nYour recent scores are low. Analyze carefully before responding."
        
        task_guidance = ""
        if task_type == "classification":
            task_guidance = """
CRITICAL CLASSIFICATION RULES:
1. URGENT in subject line but scheduled content = NORMAL (not urgent)
2. Legitimate business emails with deadlines = NORMAL 
3. Security alerts = URGENT
4. Promotional/marketing = SPAM
"""
        elif task_type == "schedule_extraction":
            task_guidance = """
SCHEDULE EXTRACTION RULES:
1. Return empty meetings array if NO schedule info exists
2. Extract ALL meetings if multiple exist
3. Date, time, duration, purpose all required
4. If email mentions 'could we' or 'would like to' = proposed status
"""
        elif task_type == "multi_intent":
            task_guidance = """
MULTI-INTENT RULES:
1. Identify ALL unique intents (max 5)
2. Use ONLY predefined intents from the list provided
3. Sentiment must match overall tone
4. Provide actionable suggested reply
"""
        
        prompt = f"""You are a precision email analysis AI. Your goal is PERFECT accuracy.

TASK TYPE: {task_type}
Current Performance on {task_type}: {perf:.1%}
{history_str}

EMAIL TO ANALYZE:
From: {email.sender}
Subject: {email.subject}

{email.body}
{task_guidance}

"""
        
        if task_type == "classification":
            prompt += """Classify into EXACTLY ONE: spam, urgent, or normal

JSON required:
{"task_type": "classification", "category": "spam|urgent|normal", "confidence": 0.0-1.0, "reasoning": "why"}"""
        
        elif task_type == "schedule_extraction":
            prompt += """Extract meeting info. Return [] if none.

JSON required:
{"task_type": "schedule_extraction", "meetings": [{"date": "", "time": "", "duration": "", "purpose": "", "status": "confirmed|proposed"}], "reasoning": "why"}"""
        
        elif task_type == "multi_intent":
            prompt += """Identify all intents (use ONLY: schedule_meeting, request_information, request_document, request_approval, complaint, compliment, cancel_subscription, request_refund, out_of_office_notification, status_update, deadline_notification, escalate_issue, request_action, request_review, social_invitation, provide_information, confirm_action)

JSON required:
{"task_type": "multi_intent", "intents": [], "primary_intent": "", "sentiment": "positive|negative|neutral|mixed", "urgency": "urgent|normal|low", "suggested_reply": "", "reasoning": "why"}"""
        
        return prompt
    
    def parse_response(self, response_text: str, task_type: str) -> Dict[str, Any]:
        response_text = response_text.strip()
        
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        try:
            parsed = json.loads(response_text)
            parsed["task_type"] = task_type
            return self._validate_and_fix(parsed, task_type)
        except json.JSONDecodeError:
            return self._fallback_parse(response_text, task_type)
    
    def _validate_and_fix(self, parsed: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        if task_type == "classification":
            if "category" not in parsed:
                parsed["category"] = "normal"
            if parsed["category"] not in ["spam", "urgent", "normal"]:
                parsed["category"] = "normal"
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("reasoning", "Parsed from response")
        
        elif task_type == "schedule_extraction":
            parsed.setdefault("meetings", [])
            parsed.setdefault("reasoning", "Parsed from response")
        
        elif task_type == "multi_intent":
            if "intents" in parsed:
                valid = []
                for intent in parsed["intents"]:
                    if intent.lower() in self.VALID_INTENTS:
                        valid.append(intent.lower())
                    elif intent.replace("_", " ").lower() in self.VALID_INTENTS:
                        valid.append(intent.replace("_", " ").lower())
                parsed["intents"] = valid[:5] if valid else ["information_request"]
            
            parsed.setdefault("sentiment", "neutral")
            parsed.setdefault("urgency", "normal")
            parsed.setdefault("suggested_reply", "Thank you for your email.")
            parsed.setdefault("reasoning", "Parsed from response")
        
        return parsed
    
    def _fallback_parse(self, text: str, task_type: str) -> Dict[str, Any]:
        result = {"task_type": task_type}
        text_lower = text.lower()
        
        if task_type == "classification":
            if "spam" in text_lower:
                result["category"] = "spam"
            elif "urgent" in text_lower:
                result["category"] = "urgent"
            else:
                result["category"] = "normal"
            result["confidence"] = 0.5
            result["reasoning"] = text[:100]
        
        elif task_type == "schedule_extraction":
            result["meetings"] = []
            result["reasoning"] = text[:100]
        
        elif task_type == "multi_intent":
            result["intents"] = ["information_request"]
            result["sentiment"] = "neutral"
            result["urgency"] = "normal"
            result["reasoning"] = text[:100]
        
        return result
    
    def _classify_email(self, subject: str, body: str, sender: str) -> Dict[str, Any]:
        text = f"{subject} {body} {sender}".lower()
        subject_upper = subject.upper()
        
        spam_indicators = [
            "congratulations", "you've won", "winner", "lottery", "million",
            "click here now", "act now", "limited time offer", "free money",
            "claim your", "unsubscribe", "buy now", "order now",
            "special promotion", "inheritance", "wire transfer",
            "password reset", "verify your account", "account suspended"
        ]
        
        urgent_indicators = [
            "urgent", "asap", "immediately", "emergency", "critical",
            "security breach", "down", "outage", "production down",
            "deadline today", "need today", "blocking", "highest priority",
            "as soon as possible", "right away", "customer impact"
        ]
        
        normal_indicators = [
            "reminder", "fyi", "update", "weekly", "monthly", "quarterly",
            "attached", "please review", "let me know", "thanks", "cheers",
            "best regards", "meeting", "schedule", "discussion"
        ]
        
        spam_score = sum(1 for ind in spam_indicators if ind in text)
        urgent_score = sum(1 for ind in urgent_indicators if ind in text)
        normal_score = sum(1 for ind in normal_indicators if ind in text)
        
        if "URGENT" in subject_upper and any(w in text for w in ["scheduled", "maintenance", "reminder"]):
            urgent_score = 0
            normal_score += 2
        
        if "security" in text and any(w in text for w in ["alert", "breach", "unauthorized", "immediately"]):
            urgent_score += 3
        
        if any(w in text for w in ["newsletter", "promotions", "deals", "offers"]):
            if sender and any(d in sender.lower() for d in ["newsletter", "promotions", "marketing"]):
                spam_score += 2
        
        if spam_score > urgent_score and spam_score > normal_score:
            return {"category": "spam", "confidence": min(0.95, 0.5 + spam_score * 0.08), "reasoning": f"Spam: {spam_score} indicators"}
        
        if urgent_score >= 2:
            return {"category": "urgent", "confidence": min(0.95, 0.6 + urgent_score * 0.1), "reasoning": f"Urgent: {urgent_score} time-critical indicators"}
        
        if normal_score >= 2:
            return {"category": "normal", "confidence": min(0.85, 0.4 + normal_score * 0.1), "reasoning": f"Normal: standard business email"}
        
        if urgent_score > 0:
            return {"category": "urgent", "confidence": 0.6, "reasoning": "Some urgency indicators present"}
        
        return {"category": "normal", "confidence": 0.55, "reasoning": "Default to normal business communication"}
    
    def _extract_meetings(self, subject: str, body: str) -> Dict[str, Any]:
        text = f"{subject} {body}"
        text_lower = text.lower()
        
        has_meeting_words = any(w in text_lower for w in [
            "meet", "meeting", "schedule", "call", "appointment", 
            "session", "discussion", "1:1", "one-on-one"
        ])
        
        if not has_meeting_words:
            return {"meetings": [], "reasoning": "No scheduling language detected"}
        
        month_pattern = r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
        day_pattern = r'\b(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b'
        time_pattern = r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b'
        
        dates = re.findall(month_pattern, text_lower, re.IGNORECASE)
        dates += re.findall(day_pattern, text_lower, re.IGNORECASE)
        
        times = re.findall(time_pattern, text_lower, re.IGNORECASE)
        
        durations = re.findall(r'(\d+(?:\.?\d+)?)\s*(hour|hr|minute|min)', text_lower)
        
        purposes = []
        purpose_keywords = {
            "sprint planning": "Sprint Planning",
            "project kickoff": "Project Kickoff", 
            "standup": "Daily Standup",
            "review": "Review Meeting",
            "1:1": "1:1 Meeting",
            "interview": "Interview",
            "training": "Training Session",
            "workshop": "Workshop"
        }
        for kw, purpose in purpose_keywords.items():
            if kw in text_lower:
                purposes.append(purpose)
        
        status = "proposed"
        if any(w in text_lower for w in ["confirm", "confirmed", "set up"]):
            status = "confirmed"
        
        meetings = []
        max_items = max(len(dates), len(times), 1)
        
        for i in range(min(3, max_items)):
            meeting = {
                "date": dates[i] if dates and i < len(dates) else "TBD",
                "time": times[i] if times and i < len(times) else "TBD",
                "duration": f"{durations[i][0]} {durations[i][1]}s" if i < len(durations) else "1 hour",
                "purpose": purposes[i] if purposes and i < len(purposes) else "Meeting",
                "status": status
            }
            meetings.append(meeting)
        
        return {"meetings": meetings, "reasoning": f"Extracted {len(meetings)} meeting(s)"}
    
    def _extract_intents(self, subject: str, body: str) -> Dict[str, Any]:
        text = f"{subject} {body}".lower()
        intents = []
        
        intent_map = {
            "schedule_meeting": ["meet", "schedule", "call", "appointment", "discussion", "1:1"],
            "request_information": ["could you", "can you", "please provide", "need to know", "let me know", "wondering"],
            "request_document": ["send", "forward", "attach", "share the", "provide the document"],
            "request_approval": ["approve", "approval", "sign off", "authorization"],
            "complaint": ["issue", "problem", "frustrated", "disappointed", "not working", "broken", "damaged", "unsatisfied"],
            "compliment": ["thank", "great", "excellent", "appreciate", "wonderful", "helpful", "pleased"],
            "cancel_subscription": ["cancel", "unsubscribe", "stop my", "remove me"],
            "request_refund": ["refund", "money back", "return", "reimbursement"],
            "out_of_office_notification": ["out of office", "ooo", "away", "vacation", "unavailable"],
            "status_update": ["update", "status", "progress", "fyi", "let you know"],
            "deadline_notification": ["deadline", "due by", "due date", "reminder"],
            "escalate_issue": ["escalate", "urgent", "asap", "immediately", "critical", "emergency"],
            "request_action": ["please", "need you to", "action required", "could you please"],
            "request_review": ["review", "look over", "check", "evaluate"],
            "social_invitation": ["lunch", "dinner", "coffee", "party", "invite", "join us"]
        }
        
        for intent, keywords in intent_map.items():
            if any(kw in text for kw in keywords):
                intents.append(intent)
        
        intents = list(set(intents))[:5]
        
        if not intents:
            intents = ["information_request"]
        
        sentiment = "neutral"
        pos_words = ["thank", "great", "appreciate", "excellent", "wonderful", "pleased", "happy"]
        neg_words = ["frustrated", "disappointed", "issue", "problem", "complaint", "unsatisfied", "angry"]
        
        pos_count = sum(1 for w in pos_words if w in text)
        neg_count = sum(1 for w in neg_words if w in text)
        
        if pos_count > 0 and neg_count > 0:
            sentiment = "mixed"
        elif pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        
        urgency = "normal"
        if any(w in text for w in ["urgent", "asap", "immediately", "critical", "emergency", "deadline today"]):
            urgency = "urgent"
        elif any(w in text for w in ["whenever", "no rush", "at your convenience", "when you have time"]):
            urgency = "low"
        
        reply_templates = {
            "schedule_meeting": "I'd be happy to schedule a meeting. Let me check my calendar and confirm available times.",
            "request_information": "Thank you for reaching out. I'll gather the requested information and respond shortly.",
            "complaint": "I sincerely apologize for any inconvenience. I'd like to address your concerns immediately.",
            "compliment": "Thank you so much for your kind words! It's wonderful to hear feedback like this.",
            "out_of_office_notification": "Thank you for your message. I'll respond when I return from my absence.",
            "escalate_issue": "I understand the urgency. I'm escalating this to the appropriate team immediately.",
        }
        
        suggested_reply = reply_templates.get(intents[0], "Thank you for your email. I'll review and respond accordingly.")
        
        return {
            "intents": intents,
            "primary_intent": intents[0],
            "suggested_reply": suggested_reply,
            "sentiment": sentiment,
            "urgency": urgency,
            "reasoning": f"Detected {len(intents)} intent(s)"
        }
    
    def _mock_response(self, task_type: str, email) -> str:
        subject = email.subject
        body = email.body
        sender = email.sender
        
        if task_type == "classification":
            result = self._classify_email(subject, body, sender)
            return json.dumps({
                "task_type": "classification",
                "category": result["category"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"]
            })
        
        elif task_type == "schedule_extraction":
            result = self._extract_meetings(subject, body)
            return json.dumps({
                "task_type": "schedule_extraction",
                "meetings": result["meetings"],
                "reasoning": result["reasoning"]
            })
        
        elif task_type == "multi_intent":
            result = self._extract_intents(subject, body)
            return json.dumps({
                "task_type": "multi_intent",
                "intents": result["intents"],
                "primary_intent": result["primary_intent"],
                "suggested_reply": result["suggested_reply"],
                "sentiment": result["sentiment"],
                "urgency": result["urgency"],
                "reasoning": result["reasoning"]
            })
        
        return '{"task_type": "unknown"}'
    
    def _reflect_on_last_action(self, action: Action, reward: float, task_type: str):
        if self.last_action_quality is None:
            self.last_action_quality = reward
            return
        
        quality_change = reward - self.last_action_quality
        
        if quality_change > 0.1:
            self.consistency_score = min(1.2, self.consistency_score * 1.05)
        elif quality_change < -0.1:
            self.consistency_score = max(0.7, self.consistency_score * 0.95)
        
        self.last_action_quality = reward
    
    def _estimate_confidence(self, action: Action, task_type: str) -> float:
        base_confidence = action.confidence if action.confidence else 0.5
        
        reasoning_length = len(action.reasoning or "")
        if reasoning_length > 100:
            base_confidence = min(0.95, base_confidence + 0.1)
        elif reasoning_length < 20:
            base_confidence = max(0.3, base_confidence - 0.1)
        
        if task_type == "multi_intent" and action.intents:
            if len(action.intents) >= 3:
                base_confidence = min(0.9, base_confidence + 0.05)
            elif len(action.intents) == 1:
                base_confidence = max(0.4, base_confidence - 0.05)
        
        if action.suggested_reply and len(action.suggested_reply) > 50:
            base_confidence = min(0.9, base_confidence + 0.05)
        
        return base_confidence * self.consistency_score
    
    def act(self, observation) -> Action:
        task_type = observation.task_type
        prompt = self.format_prompt(observation)
        
        if self.reflection_enabled and self.last_action_quality is not None:
            prompt += f"\n\nSELF-REFLECTION: Previous action scored {self.last_action_quality:.2f}. {'Improve your accuracy.' if self.last_action_quality < 0.7 else 'Maintain quality.'}"
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
            except Exception:
                response_text = self._mock_response(task_type, observation.current_email)
        else:
            response_text = self._mock_response(task_type, observation.current_email)
        
        action_dict = self.parse_response(response_text, task_type)
        
        return Action(**action_dict)


def run_inference(
    total_episodes: int = 10,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    env = OfficeWorkflowEnv(total_episodes=total_episodes, seed=seed)
    agent = BaselineAgent()
    
    print("[START]")
    print(f"task=ai-office-workflow-simulator")
    print(f"env=OfficeWorkflowEnv")
    print(f"model={agent.model_name}")
    sys.stdout.flush()
    
    obs = env.reset()
    step = 0
    rewards = []
    
    while True:
        action = agent.act(obs)
        action_str = json.dumps(action.model_dump(), separators=(',', ':'))
        
        result = env.step(action)
        
        reward = result.reward.final_reward
        rewards.append(round(reward, 4))
        
        agent._update_performance(result.info["task_type"], reward)
        
        if agent.reflection_enabled:
            agent._reflect_on_last_action(action, reward, result.info["task_type"])
        
        print()
        print("[STEP]")
        print(f"step={step}")
        print(f"action={action_str}")
        print(f"reward={reward:.4f}")
        print(f"done={'true' if result.done else 'false'}")
        print(f"error=null")
        sys.stdout.flush()
        
        step += 1
        
        if result.done:
            break
        
        obs = result.observation
    
    summary = env.get_results_summary()
    avg_reward = summary.get("average_reward", 0.0)
    
    print()
    print("[END]")
    print(f"success=true")
    print(f"steps={step}")
    print(f"score={avg_reward:.4f}")
    print(f"rewards={json.dumps(rewards)}")
    sys.stdout.flush()
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline agent on email workflow tasks")
    parser.add_argument("--episodes", "-n", type=int, default=10, help="Number of episodes")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_inference(
        total_episodes=args.episodes,
        seed=args.seed,
        verbose=False
    )