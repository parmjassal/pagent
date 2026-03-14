# Project Guidelines

## Documentation Rule
Whenever architecture or runtime behavior changes:
- Update architecture.md
- Update tasks.md (Mark tasks as complete [x] as they are implemented and verified)
- Update GEMINI.md
- Update related task docs
- Update this file if standards evolve

## Dependency Injection Rule
Production code **must never** contain hardcoded simulations or placeholders for LLMs, tools, or external services.
- Always use ABC interfaces for external components (e.g., `KnowledgeManager`, `PolicyGenerator`).
- System agents must accept an optional `llm` parameter in their constructor to facilitate mock injection in tests.

## Testing Rule
Every code revision **must** be verified with `pytest`. A change is not considered complete until:
- All existing tests pass.
- New test cases are added to verify the specific revision or bug fix.
- Integration tests (V0-V4) are executed and confirmed successful.

### Test Assertion Persistence
Established integration test assertions **must never be changed** without explicit justification. Request user permission before modifying any V-series test code.

## Logging & UI Rule
- **Rich UI:** Use `rich.tree` and `rich.live` for stdout-based user visualization.
- **Developer Trace:** Use `stderr` for functional logs.
- **Persistence:** Every session must maintain a `platform.log` file in its root directory for full traceability.

## Resource Persistence (The "Copy" Rule)
Resources (skills, prompts, guidelines) **must be copied** (not symlinked) from global to session levels upon initialization to ensure active session immutability.

## Code Principles
- **Isolation-first:** Strictly bounded filesystem and process contexts.
- **Strategic Skepticism:** No structural change without identifying potential failure modes.
- **Service-Ready:** All components must handle `user_id` and `session_id`.
