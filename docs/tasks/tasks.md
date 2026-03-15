# Tasks

## Workspace & Core Runtime
- [x] Workspace Root Resolution (Env Var / Home Dir)
- [x] Session Context Initialization (Path Hierarchy Setup)
- [x] **MailboxProvider Interface** (Filesystem implementation)
- [x] **ResourceManager Interface** (Simple copy-based implementation)
- [x] **In-Memory Session Quota:** Integration into the LangGraph state object.
- [x] Mailbox Pointer Injection (for LangGraph units)
- [x] **Hierarchical Knowledge Base:** Persistent fact sheets in `knowledge/`.
- [x] **SQLite Checkpointer:** Real graph state persistence in `state.db`.
- [ ] **Autonomous Scheduler:** Automated triggering of Mailbox and HITL events.

## Dynamic Agent & Tool Orchestration
- [x] Recursive Spawning Logic (with `max_spawn_depth` and In-Memory Quota checks)
- [x] Agent Lifecycle Manager (Creation/Cleanup of Agent Pointers)
- [x] **Tool Source Registry:** Categorize tools as COMMUNITY or DYNAMIC (Persistent metadata).
- [x] **Tool Execution Dispatcher:** Direct tools to Native or Sandbox based on source.
- [x] **Human-in-the-Loop (HITL):** Persistent approval queue and state-interrupt model.

## Default System Agents & Guardrails
- [x] **Supervisor Agent:** Structured task decomposition and recursive spawning.
- [x] **Generic Code Generator Agent:** Used for Prompts and Tools (Injected LLM).
- [x] **Generic Code Validator Agent:** Structured safety verification (Injected LLM).
- [x] **Semantic Search Agent:** Hybrid Sparse/LSH indexing with Chunking support.
- [x] **FactSheet Agent:** Knowledge extraction from big files.
- [x] **Guardrail Policy Generator & Validator:** Pluggable Policy Generator interface.

## Safety, Connectivity & Observability
- [x] **Role-Aware Loop Detection:** Thresholds for node and content repetition.
- [x] **Sandbox Runner:** Process-level isolation for DYNAMIC tools.
- [x] **Proxy Support:** Custom base URL and Redirect detection hooks.
- [x] **Rich CLI UI:** Live Tree and Thinking states using `rich`.
- [x] **Persistent Logging:** Dual stderr/file trace in the session context.
