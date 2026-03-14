# GEMINI.md - Agent Platform

## Project Overview

**agent-platform** is a LangGraph-based multi-agent runtime environment designed for service-oriented, multi-tenant deployments. It uses a structured filesystem-based mailbox system (`.pagent`) for communication and isolation.

### Main Technologies
- **Runtime:** Python >= 3.11, LangGraph, LangChain
- **Orchestration:** Typer (CLI), FastAPI (planned for Service Mode)
- **Logging:** Structlog (JSON structured logging)
- **Deployment:** Docker (slim-based Python 3.11)
- **Workspace:** Hierarchical filesystem-based mailbox with snapshotting.

---

## Architectural Principles

1.  **Hierarchical Workspace (`.pagent`):**
    - `global/` -> `user_{id}/` -> `user_{id}/{session_id}/`
    - Resources (skills, prompts) are **COPIED** (not symlinked) from higher levels to lower levels upon initialization to allow for disruption-free upgrades.

2.  **Mailbox Orchestration:**
    - Agents communicate exclusively via JSON messages.
    - LangGraph units receive absolute "pointers" (paths) to their `inbox/` and `outbox/` folders.

3.  **Snapshotting:**
    - Each session context is a self-contained unit capable of being snapshotted via directory copying.

---

## Building and Running

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Installation
```bash
pip install -e .
```

### Configuration
Set the workspace root via environment variable:
```bash
export AGENT_WORKSPACE_ROOT=~/.pagent
```

### Running (CLI Mode)
```bash
python -m agent_platform.cli serve
```

---

## Development Conventions

### Documentation Rule
Whenever architecture or runtime behavior changes, update:
- `docs/architecture.md`
- `docs/tasks/tasks.md`
- `docs/guidelines.md`
- `GEMINI.md`

### Logging Guidelines
Every log entry **must** include:
- `user_id`
- `session_id`
- `agent_id`

### Resource Rule
- **No Symlinks:** Always copy resources from higher layers into the session folder to ensure version immutability for running agents.

### Mailbox Rule
- **Isolation:** Agents communicate **only** through mailbox messages. Direct inter-agent logic calls are prohibited.
