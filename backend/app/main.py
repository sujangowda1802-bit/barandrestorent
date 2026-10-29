import logging
import re
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, settings
from app.rag import (
    RAGError,
    answer_generator,
    document_manager,
    embedding_service,
    hybrid_retriever,
    lexical_index,
    vector_store,
)
from app.schemas import ChatRequest, ChatResponse, DocumentInfo, DocumentListResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        report = document_manager.sync_managed_directory(settings.documents_dir)
        logger.info(
            "Knowledge base synchronized: %s documents scanned; %s removed.",
            report["documents_processed"],
            report["documents_removed"],
        )
    except Exception:
        logger.exception("Could not synchronize the document directory at startup.")
        raise
    yield


app = FastAPI(title="Bar & Restaurant Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/api/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    documents = list(document_manager.documents.values())
    return HealthResponse(
        status="healthy",
        app_name="Bar & Restaurant Assistant",
        gemini_configured=bool(settings.google_api_key and settings.google_api_key.strip()),
        embedding_mode=embedding_service.mode,
        documents=len(documents),
        pages=sum(int(item.get("page_count", 0)) for item in documents),
        chunks=vector_store.count(),
    )


@app.get("/api/documents", response_model=DocumentListResponse)
@app.get("/documents", response_model=DocumentListResponse, include_in_schema=False)
def list_documents() -> DocumentListResponse:
    items = [
        DocumentInfo(
            filename=doc["filename"],
            title=doc["title"],
            file_type=doc["file_type"],
            pages=doc["page_count"],
            chunks=doc["chunk_count"],
            status=doc["status"],
        )
        for doc in document_manager.documents.values()
    ]
    return DocumentListResponse(
        documents=items,
        total_documents=len(items),
        total_chunks=vector_store.count(),
    )


@app.post("/api/chat", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse, include_in_schema=False)
def chat(payload: ChatRequest) -> ChatResponse:
    query = payload.message.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a question.")
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    history = conversations[conversation_id]
    context_history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    try:
        retrieval_query = query
        prior_question = next(
            (turn["content"] for turn in reversed(history) if turn["role"] == "user"),
            None,
        )
        if (
            prior_question
            and len(query.split()) <= 6
            and re.search(r"\b(it|that|those|they|them|there|same|its|their)\b", query, re.I)
        ):
            retrieval_query = f"{prior_question}\nFollow-up question: {query}"
        chunks, trace = hybrid_retriever.retrieve(retrieval_query)
        trace["user_question"] = query
        answer, citations, mode = answer_generator.generate(
            query,
            chunks,
            trace["is_evidence_sufficient"],
            context_history,
        )
    except RAGError as exc:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat request failure")
        raise HTTPException(status_code=500, detail="The assistant could not process that question.") from exc

    history.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer, "response": ChatResponse(
                conversation_id=conversation_id,
                answer=answer,
                is_evidence_sufficient=trace["is_evidence_sufficient"],
                citations=citations,
                retrieval_trace=trace,
            ).model_dump()},
        ]
    )
    del history[:-12]
    trace["answer_mode"] = mode
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        is_evidence_sufficient=trace["is_evidence_sufficient"],
        citations=citations,
        retrieval_trace=trace,
    )


@app.get("/api/chat/conversations")
@app.get("/chat/conversations", include_in_schema=False)
def list_conversations() -> list[dict[str, str | int]]:
    return [
        {
            "conversation_id": conversation_id,
            "title": next(
                (turn["content"][:60] for turn in turns if turn["role"] == "user"),
                "New conversation",
            ),
            "message_count": len(turns),
        }
        for conversation_id, turns in conversations.items()
    ]


@app.get("/api/chat/conversations/{conversation_id}")
@app.get("/chat/conversations/{conversation_id}", include_in_schema=False)
def get_conversation(conversation_id: str) -> list[dict[str, Any]]:
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversations[conversation_id]


@app.post("/api/documents/upload")
@app.post("/documents/upload", include_in_schema=False)
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    _require_loopback(request)
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.casefold()
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF, Markdown, and TXT documents are supported.")
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    total = 0
    try:
        with destination.open("wb") as target:
            while block := await file.read(1024 * 1024):
                total += len(block)
                if total > settings.max_upload_size_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File exceeds the upload size limit.")
                target.write(block)
        metadata = document_manager.ingest_file(destination, source_filename=original_name)
        return {
            "filename": metadata["filename"],
            "pages": metadata["page_count"],
            "chunks": metadata["chunk_count"],
            "status": metadata["status"],
        }
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (ValueError, RAGError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc
    finally:
        destination.unlink(missing_ok=True)
        await file.close()


@app.post("/api/admin/reindex")
@app.post("/admin/reindex", include_in_schema=False)
def reindex(request: Request) -> dict[str, int]:
    _require_loopback(request)
    try:
        return document_manager.sync_managed_directory(settings.documents_dir, force=True)
    except (ValueError, RAGError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_loopback(request: Request) -> None:
    client = request.client.host if request.client else ""
    if client not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(status_code=403, detail="This local admin action is restricted to loopback clients.")


def _find_frontend_dir() -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "public",
        PROJECT_ROOT / "frontend",
        Path(__file__).resolve().parents[2] / "public",
        Path(__file__).resolve().parents[2] / "frontend",
        Path("public"),
        Path("frontend"),
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "index.html").exists():
            return candidate
    return None


frontend_dir = _find_frontend_dir()
if frontend_dir:
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
@app.get("/api/index.py", include_in_schema=False)
@app.get("/api/index", include_in_schema=False)
def root():
    fdir = _find_frontend_dir()
    if fdir and (fdir / "index.html").exists():
        return FileResponse(fdir / "index.html")
    for fallback in [Path("public/index.html"), Path("frontend/index.html"), PROJECT_ROOT / "frontend" / "index.html"]:
        if fallback.exists():
            return FileResponse(fallback)
    return HTMLResponse("<!doctype html><html><body><h1>Bar & Restaurant Assistant</h1><p>Frontend loading...</p></body></html>")

