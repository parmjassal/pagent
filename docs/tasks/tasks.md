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
- [x] **Autonomous Scheduler:** Automated triggering of Mailbox and HITL events.

## Prompt Engineering & Tool-Use
- [x] **Formalize Supervisor Prompts:** Detailed task decomposition logic.
- [x] **Formalize Generator Prompts:** Best practices for Prompt and Tool generation.
- [x] **Formalize Validator Prompts:** Quality and safety audit logic.
- [x] **Formalize Guardrail Policy Prompts:** Rules for dynamic policy generation.
- [x] **Agent Tool-Use Implementation:** Standardized `AgentToolNode` utilizing the `ToolDispatcher`.

## Dynamic Agent & Tool Orchestration
- [x] Recursive Spawning Logic (with `max_spawn_depth` and In-Memory Quota checks)
- [x] Agent Lifecycle Manager (Creation/Cleanup of Agent Pointers)
- [x] **Tool Source Registry:** Categorize tools as COMMUNITY or DYNAMIC (Persistent metadata).
- [x] **Tool Execution Dispatcher:** Direct tools to Native or Sandbox based on source.
- [x] **Agent Role Segregation:** SUPERVISOR vs WORKER logic.
- [x] **Human-in-the-Loop (HITL):** Persistent approval queue and state-interrupt model.

## Milestone: Recursive Hierarchical Units (Unified Graph)
- [x] **Agent-Level Workspace Refactor:** Move `todo/`, `inbox/`, and `global_context/` under `agents/{agent_id}/`.
- [x] **ContextStore Abstraction:** Interface for hierarchical, read-only visibility of `global_context`.
- [ ] **ResultHook Abstraction:** Interface for payload size checking and automatic file offloading (Summary + File Ref).
- [ ] **Dynamic Unit Compiler:** Logic to compile a `SupervisorUnit` or `WorkerUnit` graph based on role metadata.
- [ ] **Wait & Merge Spawn Node:** Implementation of recursive subgraph call sharing `thread_id` with Result-only merging.
- [ ] **Supervisor Planning Node:** Decision logic for "Decompose vs. Self-Execute."
- [x] **ScopedFileSystem Tool:** Hierarchical read-only access tool.
- [x] **ContextUpdate Tool:** Restricted promotion of facts to `global_context`.

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
