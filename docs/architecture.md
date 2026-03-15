# Architecture

This document describes the design and implementation of the LangGraph-based multi-agent runtime.

## Core Tenets

The platform is built on the following architectural pillars:

1.  **Isolation-first [Y]:** Agents operate in strictly bounded filesystem and process contexts.
2.  **Immutable Execution Contexts [Y]:** Sessions are hydrated via copying (not symlinking) to ensure global upgrades do not disrupt active work.
3.  **Service-Oriented Multi-tenancy [Y]:** Native support for `user_id` and `session_id` throughout the stack.
4.  **Resource-bounded Autonomy [Y]:** Recursive spawning is governed by depth limits, session quotas, and loop detection.
5.  **State Persistence & Resumability [Y]:** The ability to recover graph state from disk after process failure via SQLiteSaver.
6.  **Persistent Human-in-the-Loop [Y]:** Asynchronous approval requests that survive process restarts.

---

## 1. Project Package Structure

The `runtime` package is organized into four domain-isolated sub-packages.

```text
src/agent_platform/runtime/
├── core/           # Low-level system primitives
│   ├── workspace.py        # Workspace hierarchy management
│   ├── sandbox.py          # Process-level isolation
│   ├── dispatcher.py       # Tool execution routing
│   ├── hitl.py             # [NEW] Interaction management (HITL)
│   ├── mailbox.py          # Filesystem transport logic
│   └── agent_factory.py    # Recursive agent initialization
│
├── orch/           # LangGraph plumbing & State
│   ├── state.py            # Reducer-based AgentState
│   ├── quota.py            # Session usage models
│   └── logic.py            # Loop and repetition monitor
│
├── agents/         # System agent implementations
│   ├── supervisor.py       # Task decomposition & orchestration
│   └── validator.py        # Output safety verification
│
├── storage/        # Memory & Persistence
│   ├── knowledge.py        # Markdown FactSheet management
│   └── semantic_search.py  # Hybrid Sparse/LSH indexing engine
```

---

## 2. Hierarchical Workspace (`.pagent`)

The workspace root organizes data to support multi-tenancy and knowledge persistence.

### Directory Structure
-   `user_{user_id}/{session_id}/`: The atomic unit of execution.
    -   `guidelines.md`: Session-specific safety rules.
    -   `knowledge/`: Extracted "Fact Sheets" (Markdown).
    -   `interactions/`: **[NEW]** Persistent HITL (Human-in-the-Loop) JSON requests.
    -   `platform.log`: Persistent session-level execution trace.
    -   `agents/{agent_id}/`: Individual agent sandbox.
        -   `state.db`: LangGraph SQLite checkpointer.

---

## 3. Human-in-the-Loop (HITL) Model

The platform uses a **State-Interrupt** model for human approvals to ensure non-blocking, parallel-friendly execution.

### The Flow
1.  **Trigger:** An agent hits a sensitive operation (e.g., destructive tool).
2.  **Request:** The agent submits a `HITLRequest` via the `InteractionManager`.
3.  **Suspend:** The agent calls LangGraph `interrupt()`, saving its state to `state.db` and halting.
4.  **Resolve:** A human (via CLI/API) provides a `HITLResponse`. The `InteractionManager` updates the persistent JSON.
5.  **Resume:** The Scheduler detects the resolution and re-triggers the agent graph from the last checkpoint.

---

## 4. Current Gaps & Roadmap

| Feature | Status | Priority |
| :--- | :--- | :--- |
| **Scheduler/Listener** | **N** | **High** - Automated triggering of Mailbox and HITL messages. |
| **Formal Methods Validator** | **N** | Medium - SMT-based code verification. |
| **Branching Snapshots** | **N** | Low - Session rewinding and auditing. |
