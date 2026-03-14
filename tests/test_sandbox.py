import pytest
import time
from agent_platform.runtime.sandbox import ProcessSandboxRunner

def dummy_tool(x, y):
    return x + y

def error_tool():
    raise ValueError("Tool failure")

def infinite_tool():
    while True:
        time.sleep(0.1)

def test_sandbox_success():
    runner = ProcessSandboxRunner()
    result = runner.run(dummy_tool, 10, 20)
    
    assert result.success is True
    assert result.output == 30
    assert result.duration > 0

def test_sandbox_error_capture():
    runner = ProcessSandboxRunner()
    result = runner.run(error_tool)
    
    assert result.success is False
    assert "ValueError: Tool failure" in result.error

def test_sandbox_timeout_enforcement():
    runner = ProcessSandboxRunner()
    # Setting a very short timeout
    result = runner.run(infinite_tool, timeout=1)
    
    assert result.success is False
    assert "Timeout" in result.error
    assert result.duration >= 1
