import pytest
from agent_platform.runtime.quota import SessionQuota, update_quota

def test_quota_spawning_limit():
    quota = SessionQuota(agent_count=4, max_agents=5)
    assert quota.can_spawn() is True
    
    # Simulate spawning an agent
    quota.agent_count += 1
    assert quota.can_spawn() is False

def test_quota_reducer_logic():
    initial = SessionQuota(agent_count=1, message_count=10, token_usage=100)
    update = SessionQuota(agent_count=2, message_count=5, token_usage=50)
    
    # Reducer sums usage but maintains constraints
    result = update_quota(initial, update)
    
    assert result.agent_count == 3
    assert result.message_count == 15
    assert result.token_usage == 150
    assert result.max_agents == 50 # Default from left (initial)
