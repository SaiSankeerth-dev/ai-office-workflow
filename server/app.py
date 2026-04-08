import os
from typing import Dict, Any

from fastapi import FastAPI
from openenv.core.env_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action as OpenEnvAction, Observation as OpenEnvObservation, State as OpenEnvState

from environment import OfficeWorkflowEnv
from models import Action, Observation, RewardBreakdown, StepResult


def create_office_env() -> OfficeWorkflowEnv:
    """Factory function to create new environment instances."""
    return OfficeWorkflowEnv(total_episodes=10, seed=42)


app = create_app(
    env=create_office_env,
    action_cls=Action,
    observation_cls=Observation,
    env_name="ai-office-workflow-simulator"
)