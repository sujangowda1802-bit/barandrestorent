import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app
from app.config import settings
import app.rag as rag


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_app_imports_and_has_title():
    assert app is not None
    assert app.title == "Bar & Restaurant Assistant"


def test_root_serves_frontend_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Bar &amp; Restaurant Assistant" in response.text or "Bar & Restaurant Assistant" in response.text


def test_root_aliases_serve_frontend(client):
    for path in ["/", "/index.html"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


def test_rewritten_post_chat_returns_200_not_405(client):
    # Verifies that when Vercel rewrites POST /api/chat to /api/index.py,
    # the middleware correctly restores the path so it does not return 405 Method Not Allowed
    response = client.post(
        "/api/index.py",
        headers={"x-matched-path": "/api/chat"},
        json={"message": "What vegetarian dishes cost less than 250?"},
    )
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/json")
    data = response.json()
    assert "answer" in data
    assert "citations" in data


def test_rewritten_get_health_returns_json_not_html(client):
    # Verifies that when Vercel rewrites GET /api/health to /api/index.py,
    # it returns the HealthResponse JSON, NOT index.html!
    response = client.get("/api/index.py", headers={"x-matched-path": "/api/health"})
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/json")
    data = response.json()
    assert data["status"] == "healthy"
    assert data["documents"] == 3
    assert data["chunks"] > 0


def test_rewritten_get_conversations(client):
    response = client.get("/api/index.py", headers={"x-matched-path": "/api/chat/conversations"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_root_main_entrypoint_matches_app():
    import main as root_main
    assert root_main.app is app


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "Bar & Restaurant Assistant"
    assert "documents" in data
    assert "pages" in data
    assert "chunks" in data
    assert "embedding_mode" in data


def test_health_alias_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_chat_reaches_handler(client):
    response = client.post("/api/chat", json={"message": "What vegetarian dishes cost less than 250?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "conversation_id" in data
    assert data["is_evidence_sufficient"] is True


def test_chat_alias_endpoint(client):
    response = client.post("/chat", json={"message": "What are the weekend opening hours?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_unknown_api_routes_return_json_404(client):
    response = client.get("/api/nonexistent-endpoint-test")
    assert response.status_code == 404
    assert response.headers.get("content-type", "").startswith("application/json")
    assert response.json() == {"detail": "Not Found"}


def test_static_assets_served(client):
    css_res = client.get("/static/style.css")
    assert css_res.status_code == 200
    assert len(css_res.text) > 100

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert len(js_res.text) > 100


def test_vector_store_operates_without_chromadb(tmp_path):
    embedder = rag.EmbeddingService(api_key="")
    # Create store using numpy engine
    store = rag.VectorStore(path=tmp_path / "vecs", embedder=embedder)
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "document_title": "Menu",
            "filename": "menu.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Paneer Tikka price is INR 230",
            "record_type": "table_row",
            "entity": "Paneer Tikka",
        },
        {
            "chunk_id": "c2",
            "document_id": "doc1",
            "document_title": "Menu",
            "filename": "menu.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "content": "Cold Coffee beverage price is INR 120",
            "record_type": "table_row",
            "entity": "Cold Coffee",
        },
    ]
    store.replace_document("doc1", chunks)
    assert store.count() == 2

    # Query matching Paneer Tikka
    hits = store.search("Paneer Tikka", limit=1)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["entity"] == "Paneer Tikka"
    assert hits[0]["dense_score"] > 0

    # Delete doc
    store.delete_document("doc1")
    assert store.count() == 0


def test_missing_google_api_key_handled_gracefully():
    embedder = rag.EmbeddingService(api_key="")
    assert embedder.mode == "local-hash"
    vec = embedder.embed_query("Sample menu question")
    assert len(vec) == 768
    assert isinstance(vec[0], float)


def test_production_requirements_exclude_heavy_unused_packages():
    req_path = PROJECT_ROOT / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8")
    # Verify heavy packages are not present in production requirements
    assert "chromadb" not in content.lower()
    assert "onnxruntime" not in content.lower()
    assert "pytest" not in content.lower()
    # Verify core packages are present
    assert "fastapi" in content
    assert "pymupdf" in content
    assert "rank-bm25" in content
