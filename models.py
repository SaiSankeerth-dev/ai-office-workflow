from typing import List, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    model_config = {"extra": "ignore"}
    
    task_type: str = Field(..., pattern="^(classification|schedule_extraction|multi_intent)$")
    category: Optional[str] = Field(None, pattern="^(spam|urgent|normal)$")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    meetings: Optional[List[dict]] = Field(default_factory=list)
    intents: Optional[List[str]] = Field(default_factory=list)
    suggested_reply: Optional[str] = None
    reasoning: Optional[str] = None
    sentiment: Optional[str] = None
    urgency: Optional[str] = None
    primary_intent: Optional[str] = None


class Email(BaseModel):
    sender: str
    subject: str
    body: str
    timestamp: Optional[str] = None


class Task(BaseModel):
    task_id: str
    task_type: str
    email: Email
    ground_truth: dict
    difficulty: str = "medium"


class Observation(BaseModel):
    current_email: Email
    task_type: str
    history: List[dict] = Field(default_factory=list)
    episode_number: int = 0
    total_episodes: int = 10
    reward: Optional[float] = None
    done: bool = False


class RewardBreakdown(BaseModel):
    accuracy: float = 0.0
    completeness: float = 0.0
    efficiency: float = 1.0
    penalty: float = 0.0
    final_reward: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: RewardBreakdown
    done: bool
    info: dict