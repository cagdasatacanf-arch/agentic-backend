from typing import Dict

import anyio
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from app.main import api_key_auth
from app.services.agent_service import agent_service

router = APIRouter(prefix="/api/v1", tags=["docs"])


class DocIn(BaseModel):
    text: str
    metadata: Dict | None = None


class DocOut(BaseModel):
    id: str
    status: str


def _run_index_document(text: str, metadata: Dict | None):
    # Run async index_document inside background task
    anyio.run(agent_service.index_document, text, metadata)


@router.post(
    "/docs",
    response_model=DocOut,
    dependencies=[Depends(api_key_auth)],
)
async def add_doc(body: DocIn, bg: BackgroundTasks):
    """
    Accepts a document and queues it for background indexing.
    Returns quickly with a queued status.
    """
    bg.add_task(_run_index_document, body.text, body.metadata)
    # In a more advanced setup you would return the real doc_id or job_id.
    return DocOut(id="pending", status="queued")
