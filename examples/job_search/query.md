```
export AGENT_WORKSPACE_ROOT=.pagenttest
export TAVILY_API_KEY="xyz"
export OPENAI_API_KEY="xyz"
python -m agent_platform.cli --openai-base-url "xxxx" --model-name "glm-5"  "Find a list of Principal Engineer or Staff level roles in Dublin, Ireland.  Dump into /tmp/jobs.md"
```