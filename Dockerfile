FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml || true

COPY . .

ENV AGENT_WORKSPACE_ROOT=/workspace
RUN mkdir -p /workspace

CMD ["python", "-m", "agent_platform.cli", "serve"]
