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
- [ ] Recursive Spawning Logic (with `max_spawn_depth` and In-Memory Quota checks)
- [ ] Agent Lifecycle Manager (Creation/Cleanup of Agent Pointers)

## Default System Agents & Guardrails
- [ ] **Supervisor Agent:** Task decomposition and sub-agent spawning.
- [ ] **Dynamic Prompt Writer & Validator:** Context-aware instructions.
- [ ] **Guardrail Policy Generator & Validator:**
    - [ ] Implementation with LLM or SMT.
    - [ ] **Context-Aware Policy Cache:** Logic for `(user_id, context, action, history)`.

## Safety & Resource Management
- [ ] Quota Manager (Global rate limiting across users)
- [ ] Sandbox Runner (Isolated tool/agent execution)
- [ ] Multi-tenant Authentication & Authorization
