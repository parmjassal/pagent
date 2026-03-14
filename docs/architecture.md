# Architecture

This platform implements a LangGraph-based multi-agent runtime designed for service-oriented, multi-tenant deployments.

## The Hierarchical Workspace (`.pagent`)

The platform uses a structured filesystem root (defaulting to `~/.pagent` or `AGENT_WORKSPACE_ROOT`) to manage isolation and resource inheritance.

```text
.pagent/
├── global/                 # System-wide shared resources (skills, prompts)
├── user_{user_id}/         # User-specific persistent data
└── user_{user_id}/{session_id}/  # Session-specific execution context
    ├── snapshot/           # Point-in-time state of the session
    ├── skills/             # COPIED from global/user for this session
    ├── prompts/            # COPIED from global/user for this session
    └── agents/             # The active "Basic Units"
        └── {agent_id}/
            ├── inbox/      # Mailbox pointer for LangGraph
            ├── outbox/     # Mailbox pointer for LangGraph
            └── state.db    # LangGraph checkpoint (Includes In-Memory Quota)
```

## Core Architectural Components

1.  **MailboxProvider Interface:**
    - Communication is abstracted via a `MailboxProvider` interface.
    - **Current Implementation:** `FilesystemMailboxProvider`.

2.  **ResourceManager Interface:**
    - Handles the "Copy Rule" for skills and prompts.
    - **Current Implementation:** `SimpleCopyResourceManager` (shutil-based).
    - **Future-Proofing:** Abstraction allows for later implementation of Content-Addressable Storage (CAS) or versioned symlinks.

3.  **In-Memory Session Quota:**
    - Quota tracking (agent count, messages, tokens) is maintained in the **LangGraph State Object**.
    - This allows all agents in a session to share and update the quota with zero disk I/O latency.
    - Persistence is handled by the LangGraph checkpointer (`state.db`).

4.  **Context-Aware Policy Caching:**
    - Guardrail and policy results are cached based on `(user_id, context, action, history)`.
    - Prevents redundant LLM/SMT calls while maintaining security boundaries.

## Dynamic Agent Orchestration

1.  **Recursive Dynamic Spawning:**
    - Any agent can spawn sub-agents up to a `max_spawn_depth` (default 5).
    - Total agents per session are governed by the in-memory **Global Session Quota**.

2.  **Default System Agents:**
    - **Supervisor:** The primary orchestrator.
    - **Dynamic Prompt Writer/Validator:** For context-aware instructions.
    - **Guardrail Policy Generator/Validator:** For safety enforcement.
