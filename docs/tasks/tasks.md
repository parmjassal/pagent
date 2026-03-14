# Tasks

## Workspace & Core Runtime
- [x] Workspace Root Resolution (Env Var / Home Dir)
- [x] Session Context Initialization (Path Hierarchy Setup)
- [x] **MailboxProvider Interface** (Filesystem implementation)
- [x] **ResourceManager Interface** (Simple copy-based implementation)
- [x] **In-Memory Session Quota:** Integration into the LangGraph state object.
- [x] Mailbox Pointer Injection (for LangGraph units)
- [ ] Session Snapshot Engine
- [ ] Persistent Runtime Loop (Service Mode)

## Dynamic Agent Orchestration
- [x] Recursive Spawning Logic (with `max_spawn_depth` and In-Memory Quota checks)
- [x] Agent Lifecycle Manager (Creation/Cleanup of Agent Pointers)

## Default System Agents & Guardrails
- [x] **Supervisor Agent:** Task decomposition and sub-agent spawning.
- [x] **Dynamic Prompt Writer & Validator:** Context-aware instructions.
- [x] **Guardrail Policy Generator & Validator:**
    - [x] Implementation with LLM or SMT (simulated).
    - [x] **Context-Aware Policy Cache:** Logic for `(user_id, context, action, history)`.

## Safety & Resource Management
- [ ] Quota Manager (Global rate limiting across users)
- [ ] Sandbox Runner (Isolated tool/agent execution)
- [ ] Multi-tenant Authentication & Authorization
