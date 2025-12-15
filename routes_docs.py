from typing import List, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.main import api_key_auth
from app.services.agent_service import agent_service

router = APIRouter(prefix="/api/v1", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    stream: bool = False
    session_id: str | None = None


class Source(BaseModel):
    id: str | None = None
    text: str
    score: float
    metadata: Dict | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    usage: Dict | None = None


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(api_key_auth)],
)
async def query_agent(body: QueryRequest):
    answer, docs, usage = await agent_service.answer(
        question=body.question,
        top_k=body.top_k,
        session_id=body.session_id,
    )
    return QueryResponse(
        answer=answer,
        sources=[Source(**d) for d in docs],
        usage=usage,
    )
