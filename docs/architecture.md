# Architecture

This document describes the design and implementation of the LangGraph-based multi-agent runtime.

## Core Tenets

The platform is built on the following architectural pillars:

1.  **Isolation-first [Y]:** Agents operate in strictly bounded filesystem and process contexts.
2.  **Immutable Execution Contexts [Y]:** Sessions are hydrated via copying (not symlinking) to ensure global upgrades do not disrupt active work.
3.  **Service-Oriented Multi-tenancy [Y]:** Native support for `user_id` and `session_id` throughout the stack.
4.  **Resource-bounded Autonomy [Y]:** Recursive spawning is governed by depth limits, session quotas, and loop detection.
5.  **State Persistence & Resumability [Y]:** The ability to recover graph state from disk after process failure via SQLiteSaver.
6.  **Recursive Hierarchical Units [Y]:** Unified graph execution with single `thread_id` and "Result-only" state merging.
7.  **Lexical Scoping [Y]:** Downward visibility of `global_context` folders (children see parent facts).

---

## 1. Project Package Structure

The `runtime` package is organized into four domain-isolated sub-packages.

```text
src/agent_platform/runtime/
├── core/           # Low-level system primitives
│   ├── workspace.py        # Workspace hierarchy management
│   ├── sandbox.py          # Process-level isolation
│   ├── dispatcher.py       # Tool execution routing (State-aware)
│   ├── context_store.py    # [NEW] Hierarchical knowledge visibility
│   ├── mailbox.py          # Filesystem transport logic
│   ├── agent_factory.py    # Hierarchical agent initialization
│   └── tools/              # Native Core Tools
│       └── filesystem.py   # Secure ls, read, write
│
├── orch/           # LangGraph plumbing & State
│   ├── state.py            # Reducer-based AgentState with final_result
│   ├── quota.py            # Session usage models
│   ├── logic.py            # Loop and repetition monitor
│   ├── unit_compiler.py    # [NEW] Dynamic Supervisor/Worker graph compiler
│   └── result_hook.py      # [NEW] Big-data offloading logic
│
├── agents/         # System agent implementations
│   ├── supervisor.py       # Strategy-based Planning & Orchestration
│   ├── worker.py           # [NEW] LLM-driven tool execution
│   └── validator.py        # Output safety verification
│
├── storage/        # Persistence & Specialized Tools
│   ├── knowledge.py        # Markdown FactSheet management
│   ├── semantic_search.py  # Hybrid Sparse/LSH indexing engine
│   ├── search_tool.py      # [NEW] Semantic search tools
│   └── context_tool.py     # [NEW] Hierarchical context tools
```

---

## 2. Hierarchical Workspace (`.pagent`)

The workspace root organizes data to support multi-tenancy and lexical scoping.

### Directory Structure
-   `user_{user_id}/{session_id}/`: The atomic unit of execution.
    -   `guidelines.md`: Session-specific safety rules.
    -   `knowledge/`: Offloaded large results (Markdown).
    -   `agents/{parent_id}/{child_id}/`: Recursive agent sandboxes.
        -   `inbox/`, `outbox/`: Communication channels.
        -   `todo/`: Agent-specific task list.
        -   `global_context/`: Shared facts (Lexical scoping).
        -   `state.db`: LangGraph SQLite checkpointer.

---

## 3. Recursive Orchestration Model

The platform uses a **Recursive Unit** model where a single thread manages a tree of subgraphs.

### The Flow
1.  **Planning:** Supervisor decides between `DECOMPOSE`, `TOOL_USE`, or `FINISH`.
2.  **Spawning:** Supervisor awaits a subgraph compiled by the `UnitCompiler`.
3.  **Lexical Visibility:** Sub-agents recursively lookup `global_context` folders up to the session root.
4.  **Result Offloading:** Large outputs are intercepted by the `ResultHook`, stored as files, and returned as references.
5.  **Merge:** Only the `final_result` of a child is merged back into the parent's message history.

---

## 4. Current Gaps & Roadmap

| Feature | Status | Priority |
| :--- | :--- | :--- |
| **Branching Snapshots** | **N** | High - Session rewinding and auditing. |
| **Formal Methods Validator** | **N** | Medium - SMT-based code verification. |
| **Autonomous Tool Generator**| **P** | Medium - Refinement of LLM tool-writing logic. |
