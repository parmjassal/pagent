# Project Guidelines

## Documentation Rule
Whenever architecture or runtime behavior changes:
- Update architecture.md
- Update tasks.md (Mark tasks as complete [x] as they are implemented and verified)
- Update GEMINI.md
- Update related task docs
- Update this file if standards evolve

## Pre-Implementation Discussion Rule
For any architectural change, new component, or complex logic update, the agent MUST first provide a technical analysis, reasoning, and risk assessment. Code implementation shall only begin after receiving user confirmation.

## Abstraction & Dependency Injection Rule
- **No Hardcoding:** Production code (`src/`) must never contain hardcoded mock data or test-specific logic.
- **Interfaces First:** All external or replaceable components (Storage, LLMs, Tool Loaders) must be defined as ABC (Abstract Base Class) interfaces.
- **Injected Dependencies:** Implementations must be injected via constructors to facilitate testing and future replacement (e.g., swapping Filesystem for Database).

## Lego-Style Integration Rule
- **Incremental Building:** Build components as independent "blocks" that are immediately integrated and verified.
- **Mandatory Integration Tests:** Every new feature must be accompanied by an integration test (V-series) that verifies its behavior within the full system stack.
- **Assertion Persistence:** Established integration test assertions (V2+) must never be changed without explicit reasoning and user permission.

## Hierarchical Scoping & Visibility
- **Agent Isolation:** Agents operate in private workspaces. 
- **Read-Only Context:** Hierarchical visibility of `global_context` folders is read-only for sub-agents. 
- **Supervisor Authority:** Only Supervisor agents have the authority to update the `global_context` via specific tools.

## Logging & UI Rule
- **Rich UI:** Use `rich.tree` and `rich.live` for stdout-based user visualization.
- **Developer Trace:** Use `stderr` for functional logs.
- **Persistence:** Every session must maintain a `platform.log` file in its root directory.

## Code Principles
### Stability & Integrity
- **No Disastrous Changes:** Do not perform large-scale structural refactors if existing components (like `ls`) are working fine. If a component appears to fail, investigate the *caller* (e.g., LLM hallucinations) before modifying the *platform*.
- **Empirical Validation:** Before "fixing" a supposed bug, reproduce it with an integration test. Never assume a platform tool is broken based on a single failed LLM call.
- **Protocol Compliance:** Always maintain strict OpenAI Chat Completion schema compatibility. Never inject `system` or `tool` roles in a way that violates sequential ordering.
- **Incremental Refinement:** Prefer surgical updates over architectural shifts.

### Context & Logic
- **Isolation-first:** Strictly bounded filesystem and process contexts.
- **Strategic Skepticism:** No structural change without identifying potential failure modes.
- **Service-Ready:** All components must handle `user_id` and `session_id`.
