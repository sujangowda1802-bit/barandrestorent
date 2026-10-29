from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(default=None, max_length=80)


class Citation(BaseModel):
    chunk_id: str
    source: str
    page_number: Optional[int] = None
    excerpt: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    is_evidence_sufficient: bool
    citations: list[Citation]
    retrieval_trace: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    gemini_configured: bool
    embedding_mode: str
    documents: int
    pages: int
    chunks: int


class DocumentInfo(BaseModel):
    filename: str
    title: str
    file_type: str
    pages: int
    chunks: int
    status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int
