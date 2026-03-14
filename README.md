# pagent - Multi-Agent Runtime & Orchestration

**pagent** is a LangGraph-based multi-agent runtime environment designed for service-oriented, multi-tenant deployments. It provides a secure and scalable foundation for agents to communicate, spawn recursively, and execute tools in isolated environments.

## 🚀 Key Features

- **Hierarchical Workspace (`.pagent`):** Automated multi-tenant directory structure for Users and Sessions.
- **Mailbox Orchestration:** Asynchronous, filesystem-based communication bus for agent isolation.
- **Selective Sandboxing:** Dual-path execution model. Native for trusted community tools, and process-level sandboxing for dynamically generated code.
- **Recursive Dynamic Spawning:** Agents can spawn sub-agents up to a configurable depth, with automated handover.
- **In-Memory Quota Management:** Strict session-level tracking of agent counts and resource usage.
- **System Validation:** Context-aware validation of generated prompts and code against session guidelines.

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
source .venv/bin/activate
```

**Windows:**
```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

---

## 📖 How to Use

### Starting the Platform
Launch the runtime for the current system user:
```bash
export AGENT_WORKSPACE_ROOT=~/.pagent
export OPENAI_API_KEY=your_key_here
python -m agent_platform.cli
```

### Resume an Existing Session
```bash
python -m agent_platform.cli --session-id <your-session-id>
```

---

## 🛠 Proxy & Endpoint Configuration

**pagent** is designed to work in restricted corporate environments and with local LLM proxies.

### Custom API Endpoints
You can point the platform to a custom OpenAI-compatible API (e.g., Azure, LiteLLM, Ollama):
```bash
python -m agent_platform.cli --openai-base-url "https://my-proxy.internal/v1"
```

### Corporate Proxy & Redirects
If your corporate proxy intercepts requests (e.g., Captive Portals), the platform will:
1.  **Detect** the 3xx redirect.
2.  **Output** the redirect link to the console for you to authenticate or accept terms.
3.  **Prevent** silent failures by not following redirects automatically.

---

## 🌟 Example Scenario: Secure Task Delegation

Imagine you are building a **Security Audit Platform**.

1.  **Supervisor:** You provide a high-level task: "Audit this repository for SQL injection vulnerabilities."
2.  **Decomposition:** The Supervisor spawns a `vulnerability_scanner` sub-agent.
3.  **Prompt Writing:** The `SystemGeneratorAgent` creates a specialized system prompt for the scanner based on the repository's tech stack.
4.  **Tool Generation:** The scanner realizes it needs a custom parser. It requests a `parse_sql` tool.
5.  **Validation:** The `SystemValidatorAgent` checks the generated `parse_sql` code. It detects a `shutil.rmtree` call (violation of destructive policy) and rejects it.
6.  **Secure Execution:** Once a safe tool is generated, it runs in the **Sandbox Runner** (isolated process), ensuring it cannot access sensitive host files.
7.  **Handover:** The scanner writes its findings to its `outbox`, which the Supervisor picks up via the **Mailbox** to compile the final report.

---

## 🏗 Project Structure

- `src/agent_platform/`: Core platform logic.
- `docs/`: Architecture, guidelines, and roadmap.
- `scripts/`: OS-specific installation helpers.
- `tests/`: Unit and full-lifecycle integration tests (V0, V1, V2).

---

## 📜 License
MIT
