from openenv.core.env_server import create_app
from environment import OfficeWorkflowEnv

app = create_app(OfficeWorkflowEnv())