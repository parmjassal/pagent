import json
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class InteractionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"

class HITLRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str
    context: str
    data: Dict[str, Any] = Field(default_factory=dict)
    status: InteractionStatus = InteractionStatus.PENDING

class HITLResponse(BaseModel):
    request_id: str
    approved: bool
    feedback: Optional[str] = None

class InteractionManager:
    """
    Manages persistent Human-in-the-Loop requests across a session.
    Acts as a bridge between the isolated agents and the human UI (CLI/API).
    """

    def __init__(self, session_path: Path):
        self.root = session_path / "interactions"
        self.root.mkdir(parents=True, exist_ok=True)

    def submit_request(self, request: HITLRequest):
        """Persists a new request for human approval."""
        path = self.root / f"req_{request.request_id}.json"
        path.write_text(request.model_dump_json(indent=2))
        logger.info(f"HITL Request Submitted: {request.request_id} from {request.agent_id}")

    def list_pending(self) -> List[HITLRequest]:
        """Lists all requests currently awaiting approval."""
        pending = []
        for path in self.root.glob("req_*.json"):
            try:
                req = HITLRequest.model_validate_json(path.read_text())
                if req.status == InteractionStatus.PENDING:
                    pending.append(req)
            except Exception as e:
                logger.error(f"Failed to parse HITL request {path}: {e}")
        return pending

    def resolve_request(self, response: HITLResponse):
        """Updates a request with the human's decision and archives it."""
        path = self.root / f"req_{response.request_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Request {response.request_id} not found.")

        req = HITLRequest.model_validate_json(path.read_text())
        req.status = InteractionStatus.APPROVED if response.approved else InteractionStatus.DENIED
        
        # Save updated status
        path.write_text(req.model_dump_json(indent=2))
        
        # In a full implementation, we'd also write a message to the agent's inbox
        logger.info(f"HITL Request Resolved: {response.request_id} (Approved: {response.approved})")
        return req
