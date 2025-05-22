from pydantic import BaseModel, Field
from typing import List, Dict, Any

class SimilarDoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: float

class ChunkResult(BaseModel):
    chunk: str
    summary: str
    similar_documents: List[SimilarDoc] = Field(default_factory=list)
    elapsed_time: float
    error: str | None = None
4