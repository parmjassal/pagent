```
export AGENT_WORKSPACE_ROOT=.pagenttest
export TAVILY_API_KEY="xyz"
export OPENAI_API_KEY="xyz"
python -m agent_platform.cli --openai-base-url "xxxx" --model-name "glm-5"  "Find most important 5 system design interview question for Staff Software Engineer role. Check recency of these questions as in 2026.  Provide summary and topi
c pointers for each question .  Dump into /tmp/StaffSystemInterview.md"
```