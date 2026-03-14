# Tasks

## Workspace & Core Runtime
- [x] Workspace Root Resolution (Env Var / Home Dir)
- [x] Session Context Initialization (Path Hierarchy Setup)
- [x] **MailboxProvider Interface** (Filesystem implementation)
- [x] **ResourceManager Interface** (Simple copy-based implementation)
- [x] **In-Memory Session Quota:** Integration into the LangGraph state object.
- [x] Mailbox Pointer Injection (for LangGraph units)
- [x] **Hierarchical Knowledge Base:** Persistent fact sheets in `knowledge/`.
- [ ] Session Snapshot Engine (DEFERRED)
- [ ] **SQLite Checkpointer:** Real graph state persistence in `state.db`.
- [ ] Persistent Runtime Loop (Service Mode / Scheduler)

## Dynamic Agent & Tool Orchestration
- [x] Recursive Spawning Logic (with `max_spawn_depth` and In-Memory Quota checks)
- [x] Agent Lifecycle Manager (Creation/Cleanup of Agent Pointers)
- [x] **Tool Source Registry:** Categorize tools as COMMUNITY or DYNAMIC (Persistent metadata).
- [x] **Tool Execution Dispatcher:** Direct tools to Native or Sandbox based on source.
- [x] **Agent Role Segregation:** SUPERVISOR vs WORKER logic.

## Default System Agents & Guardrails
- [x] **Supervisor Agent:** Structured task decomposition and recursive spawning.
- [x] **Generic Code Generator Agent:** Used for Prompts and Tools (Injected LLM).
- [x] **Generic Code Validator Agent:** Structured safety verification (Injected LLM).
- [x] **Semantic Search Agent:** Hybrid Sparse/LSH indexing with Chunking support.
- [x] **FactSheet Agent:** Knowledge extraction from big files.
- [x] **Guardrail Policy Generator & Validator:**
    - [x] Injected Policy Generator interface.
    - [x] **Context-Aware Policy Cache:** Pluggable lookup providers (Hash/Semantic).

## Safety, Connectivity & Observability
- [x] **Role-Aware Loop Detection:** Thresholds for node and content repetition.
- [x] **Sandbox Runner:** Process-level isolation for DYNAMIC tools.
- [x] **Proxy Support:** Custom base URL and Redirect detection hooks.
- [x] **Rich CLI UI:** Live Tree and Thinking states using `rich`.
- [x] **Persistent Logging:** Dual stderr/file trace in the session context.
- [ ] Quota Manager (Global rate limiting across users)
- [ ] Multi-tenant Authentication & Authorization
