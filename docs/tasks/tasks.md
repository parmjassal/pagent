## Refined Context & Knowledge Orchestration (v2.1)
- [ ] **Unified Context Interface:** Refactor `ContextTools` to handle hierarchical `update_context` and global `update_knowledge`.
- [ ] **Reasoning-driven Global Knowledge:** Implement `update_knowledge` with `[8-char-hex-prefix]_[contextual_name].json` naming logic.
- [ ] **Scratchpad Offloading Refactor:** Update `OffloadingResultHook` to use a distinct `offload_` prefix and write to the global `knowledge/` folder.
- [ ] **Filesystem Restriction:** Prevent `FilesystemTools.write_file` from writing directly to the `knowledge/` directory to enforce reasoning-driven knowledge promotion.
- [ ] **Initialization Alignment:** Correctly initialize `ContextTools` with both `ContextStore` and `knowledge_path` in `bootstrap.py`.
- [ ] **Prompt Updates:** Update Supervisor and Worker prompts to distinguish between Downward (Branch) and Global (Reasoned) visibility.
- [ ] **Verification:** Add unit tests for prefix logic and integration tests for filesystem write restrictions.

# Original Tasks (Archive/Complete)

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
- [x] **ResultHook Abstraction:** Interface for payload size checking and automatic file offloading (Summary + File Ref).
- [x] **Dynamic Unit Compiler:** Logic to compile a `SupervisorUnit` or `WorkerUnit` graph based on role metadata.
- [x] **Wait & Merge Spawn Node:** Implementation of recursive subgraph call sharing `thread_id` with Result-only merging.
- [x] **Supervisor Planning Node:** Decision logic for "Decompose vs. Self-Execute."
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

## Global Context & Guardrail Refinement
- [x] **Guardrail Context-Awareness:** Inject visible facts into the Policy Generator.
- [x] **Supervisor Intent Persistence:** Record high-level tasks as persistent facts in Supervisor's context.
- [x] **Error Hardening:** Standardize error messages and types for structured reasoning in results.
- [x] **Generator Context-Awareness:** Inject visible facts into sub-agent prompts.
- [x] **Robustness Fix:** Prevent `KeyError: tool_call_id` on LLM hallucinations and handle non-JSON responses.
