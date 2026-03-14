import structlog
from ..logging_config import configure_logging

log = structlog.get_logger()

def start_runtime():
    configure_logging()
    log.info("runtime_start")
