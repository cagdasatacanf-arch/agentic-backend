from typing import Dict, List, Tuple
from uuid import uuid4

from qdrant_client.http.models import PointStruct

from app.rag import get_agent_answer, client, COLLECTION, embed


class AgentService:
    """
    Facade for agent-related operations.
    Extend here with tools, planning, memory, multi-agent, etc.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, callable] = {}

    def register_tool(self, name: str, fn: callable) -> None:
        self.tools[name] = fn

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> Tuple[str, List[Dict], Dict]:
        # session_id reserved for future memory / conversation logic
        answer, docs, usage = await get_agent_answer(question=question, top_k=top_k)
        return answer, docs, usage

    async def index_document(self, text: str, metadata: Dict | None = None) -> str:
        """
        Embed and store a single document in Qdrant.
        Intended to run in a background task or worker.
        """
        vec = (await embed([text]))[0]
        doc_id = str(uuid4())
        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=vec,
                    payload={"text": text, "metadata": metadata or {}},
                )
            ],
        )
        return doc_id


agent_service = AgentService()
