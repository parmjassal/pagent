# pagent - Multi-Agent Runtime & Orchestration

**pagent** is a LangGraph-based multi-agent runtime environment designed for service-oriented, multi-tenant deployments. It provides a secure and scalable foundation for agents to communicate, spawn recursively, and execute tools in isolated environments.

## 🚀 Key Features

- **Hierarchical Workspace (`.pagent`):** Automated multi-tenant directory structure for Users and Sessions.
- **Mailbox Orchestration:** Asynchronous, filesystem-based communication bus for agent isolation.
- **Selective Sandboxing:** Dual-path execution model. Native for trusted community tools, and process-level sandboxing for dynamically generated code.
- **Semantic Repo Analysis:** Hybrid Sparse/LSH engine for indexing local folders with chunking support for big files.
- **Knowledge Management:** Persistent Markdown-based "Fact Sheets" extracted from large context analysis.
- **Recursive Dynamic Spawning:** Agents can spawn sub-agents up to a configurable depth, with automated handover.
- **Persistent Human-in-the-Loop:** Asynchronous approval queue for sensitive operations, allowing agents to suspend and resume statefully.
- **Rich CLI UI:** Real-time visualization of the orchestration tree and agent "thinking" states.
- **In-Memory Quota Management:** Strict session-level tracking of agent counts and resource usage.

---

## 🛠 Prerequisites

- **Python 3.11+**
- **OpenAI API Key** (or any LLM provider supported by LangChain)
- **Git** (for version control)

### Quick Setup

We provide setup scripts to automate the creation of a virtual environment and installation of dependencies.

**Unix / Mac:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
source .vcli/bin/activate
```

**Windows:**
```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

---

## 📖 How to Use

### Env Variables
```bash
export AGENT_WORKSPACE_ROOT=.pagenttest
export TAVILY_API_KEY="xyz"
export OPENAI_API_KEY="xyz"
```

### Starting the Platform
Launch the runtime and provide an optional initial task:
```bash
python -m agent_platform.cli --openai-base-url "xxxx" --model-name "glm-5"  "Find most important 5 system design interview question for Staff Software Engineer role. Check recency of these questions as in 2026.  Provide summary and topi
c pointers for each question .  Dump into /tmp/StaffSystemInterview.md"
```

### Configuration Options
You can configure user, session, and model via flags:
```bash
python -m agent_platform.cli --user-id "dev_user" --model-name "gpt-4-turbo"
```

### Resume an Existing Session
```bash
python -m agent_platform.cli --session-id <your-session-id>
```

---

## 🛠 Proxy & Endpoint Configuration

**pagent** is designed to work in restricted corporate environments and with local LLM proxies.

### Custom API Endpoints & Models
You can point the platform to a custom OpenAI-compatible API and specify the model:
```bash
python -m agent_platform.cli --openai-base-url "https://my-proxy.internal/v1" --model-name "gpt-4-turbo"
```
Or via environment variables:
```bash
export AGENT_MODEL_NAME="claude-3-5-sonnet"
python -m agent_platform.cli
```

### Corporate Proxy & Redirects
If your corporate proxy intercepts requests (e.g., Captive Portals), the platform will:
1.  **Detect** the 3xx redirect.
2.  **Output** the redirect link to the console for you to authenticate or accept terms.
3.  **Prevent** silent failures by not following redirects automatically.

---

## 🏗 Project Structure

- `src/agent_platform/runtime/`:
    - `core/`: Infrastructure primitives (Workspace, Sandbox, Dispatcher, Mailbox).
    - `orch/`: LangGraph orchestration (State, Quota, Logic).
    - `agents/`: System agent implementations (Supervisor, Generator, Validator, Search).
    - `storage/`: Memory & Knowledge management (Semantic Index, Fact Sheets).
- `docs/`: Architecture, guidelines, and roadmap.
- `scripts/`: OS-specific installation helpers.
- `tests/`: Unit and full-lifecycle integration tests (V0-V4).

---

## 📜 License
MIT
