# Project Guidelines

## Documentation Rule
Whenever architecture or runtime behavior changes:
- Update architecture.md
- Update tasks.md (Mark tasks as complete [x] as they are implemented and verified)
- Update GEMINI.md
- Update related task docs
- Update this file if standards evolve

## Logging Guidelines
Use structlog with JSON logs.
Log entries **must** include:
- `user_id`
- `session_id`
- `agent_id`

## Resource Persistence (The "Copy" Rule)
Skills, prompts, and configurations **must not** be symlinked between hierarchy levels (`global` -> `user` -> `session`). They **must be copied** to ensure that active sessions are immutable and unaffected by global system upgrades.

## Mailbox Rule
Agents communicate **only** through mailbox messages. Direct function calls between agent logic are prohibited.

## Testing Rule
Every code revision **must** be verified with `pytest`. A change is not considered complete until:
- All existing tests pass.
- New test cases are added to verify the specific revision or bug fix.
- The project-specific `pytest` command is executed and confirmed successful.

## Critical Reasoning Rules
To maintain the platform's architectural integrity, the following rules apply to all design and implementation discussions:
1.  **Strategic Skepticism:** No structural change is implemented without first identifying at least two potential failure modes or bottlenecks.
2.  **Feasibility Reasoning:** Every request must be reasoned through (e.g., "Why this way?", "What's the alternative?") before being executed.
3.  **Performance Cost Evaluation:** Any feature involving disk I/O, recursion, or LLM-based validation must have an associated performance "cost" evaluation.
4.  **Consistency Check:** Verify if a new requirement contradicts a previously established mandate in `docs/architecture.md` or `GEMINI.md`.

## Code Principles
- **Isolation-first:** Prioritize filesystem and resource isolation.
- **KISS, DRY.**
- **Service-Ready:** All components must handle `user_id` and `session_id` context.
