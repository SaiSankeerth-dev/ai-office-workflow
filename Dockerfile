FROM python:3.11-slim

LABEL maintainer="AI Office Workflow Team"
LABEL description="OpenEnv Email Workflow Simulator"
LABEL version="1.0.0"

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app

USER appuser

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV OPENAI_API_KEY=""
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "from environment import OfficeWorkflowEnv; env = OfficeWorkflowEnv(); env.reset(); print('OK')" || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]