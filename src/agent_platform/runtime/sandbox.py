import multiprocessing
import time
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SandboxResult:
    """Encapsulates the result of a sandboxed execution."""
    def __init__(
        self, 
        success: bool, 
        output: Any = None, 
        error: Optional[str] = None, 
        duration: float = 0.0
    ):
        self.success = success
        self.output = output
        self.error = error
        self.duration = duration

class SandboxRunner(ABC):
    """Base interface for sandboxed execution environments."""
    
    @abstractmethod
    def run(self, func: Callable, *args, timeout: int = 30, **kwargs) -> SandboxResult:
        """Executes a function within the sandbox."""
        pass

def _proc_wrapper(func, queue, args, kwargs):
    """Internal helper to run the function and capture output/errors in a separate process."""
    try:
        result = func(*args, **kwargs)
        queue.put((True, result, None))
    except Exception:
        error_msg = traceback.format_exc()
        queue.put((False, None, error_msg))

class ProcessSandboxRunner(SandboxRunner):
    """
    Provides process-level isolation for Python function execution.
    Prevents a tool from crashing the main runtime or hanging indefinitely.
    """

    def run(self, func: Callable, *args, timeout: int = 30, **kwargs) -> SandboxResult:
        start_time = time.time()
        queue = multiprocessing.Queue()
        
        # Create a separate process for the function
        proc = multiprocessing.Process(
            target=_proc_wrapper, 
            args=(func, queue, args, kwargs)
        )
        
        try:
            proc.start()
            
            # Wait for result or timeout
            try:
                success, output, error = queue.get(timeout=timeout)
                duration = time.time() - start_time
                return SandboxResult(success, output, error, duration)
            except multiprocessing.queues.Empty:
                # Timeout reached
                proc.terminate()
                proc.join()
                return SandboxResult(
                    False, 
                    error=f"Timeout: Execution exceeded {timeout} seconds",
                    duration=time.time() - start_time
                )
        except Exception as e:
            return SandboxResult(False, error=str(e), duration=time.time() - start_time)
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join()
