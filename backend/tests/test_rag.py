import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app
from app.rag import (
    AnswerGenerator,
    DocumentManager,
    EmbeddingService,
    HybridRetriever,
    LexicalIndex,
    RAGError,
    VectorStore,
    GEMINI_REQUEST_TIMEOUT_MS,
    _create_genai_client,
    chunk_pages,
    extract_pdf_pages,
    answer_generator,
    hybrid_retriever,
    vector_store,
)


@pytest.fixture(scope="module")
def client():
    original_search = vector_store.search
    original_generator = answer_generator.client
    # API tests must stay deterministic and must not make live embedding or generation calls.
    vector_store.search = lambda _query, _limit: []
    answer_generator.client = None
    with TestClient(app) as test_client:
        try:
            yield test_client
        finally:
            vector_store.search = original_search
            answer_generator.client = original_generator


def test_demo_pdfs_extract_pages_and_page_numbers():
    for name in ("restaurant_menu.pdf", "ingredients_allergens.pdf", "restaurant_policies.pdf"):
        pages = extract_pdf_pages(PROJECT_ROOT / "data" / "documents" / name)
        assert len(pages) >= 1
        assert all(page["page_number"] == index for index, page in enumerate(pages, start=1))
        assert any(page["text"] for page in pages)


def test_empty_scanned_and_malformed_pdfs_are_reported(tmp_path):
    import pymupdf

    blank = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(blank)
    document.close()
    with pytest.raises(ValueError, match="no extractable text"):
        extract_pdf_pages(blank)

    malformed = tmp_path / "bad.pdf"
    malformed.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="Could not read PDF"):
        extract_pdf_pages(malformed)


def test_chunk_metadata_and_overlap():
    chunks = chunk_pages(
        [{"page_number": 2, "text": "A " * 1200}],
        "stable-id",
        "Sample",
        "sample.pdf",
        chunk_size=300,
        overlap=40,
    )
    assert len(chunks) > 1
    assert all(chunk["document_id"] == "stable-id" for chunk in chunks)
    assert all(chunk["filename"] == "sample.pdf" for chunk in chunks)
    assert all(chunk["page_number"] == 2 for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_pdf_table_chunks_keep_complete_individual_records():
    menu_pages = extract_pdf_pages(PROJECT_ROOT / "data" / "documents" / "restaurant_menu.pdf")
    menu_chunks = chunk_pages(menu_pages, "menu", "Menu", "restaurant_menu.pdf")
    paneer = next(chunk for chunk in menu_chunks if chunk.get("entity") == "Paneer Tikka")
    assert "Price: INR 230" in paneer["content"]
    assert "Paneer, yogurt marinade, peppers, spices" in paneer["content"]
    assert "Chicken Pepper Fry" not in paneer["content"]
    assert paneer["page_number"] == 1
    repeated_chunks = chunk_pages(menu_pages, "menu", "Menu", "restaurant_menu.pdf")
    assert next(chunk for chunk in repeated_chunks if chunk.get("entity") == "Paneer Tikka")[
        "chunk_id"
    ] == paneer["chunk_id"]

    allergen_pages = extract_pdf_pages(
        PROJECT_ROOT / "data" / "documents" / "ingredients_allergens.pdf"
    )
    allergen_chunks = chunk_pages(allergen_pages, "allergens", "Allergens", "ingredients_allergens.pdf")
    paneer_allergen = next(chunk for chunk in allergen_chunks if chunk.get("entity") == "Paneer Tikka")
    assert "Known listed ingredients: Paneer, yogurt marinade, peppers, spices" in paneer_allergen["content"]
    assert "Potential allergen flags in this sample: Milk" in paneer_allergen["content"]
    assert "Crispy Corn" not in paneer_allergen["content"]
    assert not any("Useful test questions" in chunk["content"] for chunk in allergen_chunks)


def test_hybrid_retrieval_covers_every_entity_and_requested_allergen_facet(tmp_path):
    embedder = EmbeddingService(api_key="")
    store = VectorStore(tmp_path / "chroma-hybrid", embedder)
    lexical = LexicalIndex(tmp_path / "lexical-hybrid.json")
    manager = DocumentManager(store, lexical, tmp_path / "documents-hybrid.json")
    manager.sync_managed_directory(PROJECT_ROOT / "data" / "documents")
    retriever = HybridRetriever(store, lexical)

    comparison, _ = retriever.retrieve(
        "Compare Paneer Tikka and Paneer Butter Masala by price and ingredients."
    )
    comparison_evidence = {
        (item["filename"], item.get("entity"))
        for item in comparison
    }
    assert {
        ("restaurant_menu.pdf", "Paneer Tikka"),
        ("restaurant_menu.pdf", "Paneer Butter Masala"),
        ("ingredients_allergens.pdf", "Paneer Tikka"),
        ("ingredients_allergens.pdf", "Paneer Butter Masala"),
    }.issubset(comparison_evidence)

    facets, _ = retriever.retrieve("Which dishes contain wheat, and which contain dairy?")
    allergen_entities = {
        item.get("entity")
        for item in facets
        if item["filename"] == "ingredients_allergens.pdf"
    }
    assert {
        "Plain Roti",
        "Gulab Jamun",
        "Chocolate Brownie",
        "Paneer Tikka",
        "Paneer Butter Masala",
        "Tomato Basil Soup",
        "Dal Tadka with Jeera Rice",
        "Masala Chai",
        "Filter Coffee",
    }.issubset(allergen_entities)


def test_ingestion_is_repeatable_and_changed_document_replaces_chunks(tmp_path):
    docs = tmp_path / "source"
    docs.mkdir()
    source = docs / "notes.txt"
    source.write_text("A small menu with paneer and tomato.", encoding="utf-8")
    embedder = EmbeddingService(api_key="")
    store = VectorStore(tmp_path / "chroma", embedder)
    lexical = LexicalIndex(tmp_path / "lexical.json")
    manager = DocumentManager(store, lexical, tmp_path / "documents.json")

    first = manager.ingest_file(source, managed=True)
    second = manager.ingest_file(source, managed=True)
    assert first["id"] == second["id"]
    assert store.count() == first["chunk_count"]

    source.write_text("Changed content with wheat and flour.", encoding="utf-8")
    changed = manager.ingest_file(source, managed=True)
    assert changed["id"] == first["id"]
    assert store.count() == changed["chunk_count"]
    assert lexical.search("wheat flour", 3)


def test_managed_sync_removes_missing_source_documents(tmp_path):
    docs = tmp_path / "managed"
    docs.mkdir()
    source = docs / "sample.txt"
    source.write_text("A documented sample payment policy mentions UPI.", encoding="utf-8")
    embedder = EmbeddingService(api_key="")
    store = VectorStore(tmp_path / "chroma-sync", embedder)
    lexical = LexicalIndex(tmp_path / "lexical-sync.json")
    manager = DocumentManager(store, lexical, tmp_path / "registry-sync.json")

    assert manager.sync_managed_directory(docs)["documents_processed"] == 1
    assert store.count() == 1
    source.unlink()
    assert manager.sync_managed_directory(docs)["documents_removed"] == 1
    assert store.count() == 0
    assert lexical.chunks == {}


def test_lexical_and_vector_retrieval(client, tmp_path):
    lexical_results = hybrid_retriever.lexical.search("Paneer Butter Masala ingredients", 3)
    assert lexical_results
    assert any("paneer" in item["content"].casefold() for item in lexical_results)

    embedder = EmbeddingService(api_key="")
    store = VectorStore(tmp_path / "chroma-vector", embedder)
    chunk = {
        "chunk_id": "test-vector-chunk",
        "document_id": "test-vector-doc",
        "document_title": "Test",
        "filename": "test.txt",
        "page_number": 1,
        "chunk_index": 0,
        "content": "Paneer ingredients include milk, cream, and spices.",
    }
    store.replace_document(chunk["document_id"], [chunk])
    assert store.search("paneer cream ingredients", 1)[0]["chunk_id"] == chunk["chunk_id"]


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("What vegetarian main courses cost less than ₹250?", "restaurant_menu.pdf"),
        ("What ingredients are listed for Paneer Butter Masala?", "ingredients_allergens.pdf"),
        ("Which listed dishes contain wheat?", "ingredients_allergens.pdf"),
        ("What are the weekend opening hours?", "restaurant_policies.pdf"),
        ("Does the sample restaurant accept UPI?", "restaurant_policies.pdf"),
    ],
)
def test_supported_questions_have_cited_evidence(client, question, expected_source):
    result = client.post("/api/chat", json={"message": question}).json()
    assert result["is_evidence_sufficient"]
    assert result["citations"]
    assert expected_source in {citation["source"] for citation in result["citations"]}
    assert all(citation["chunk_id"] in {
        chunk["chunk_id"] for chunk in result["retrieval_trace"]["selected_chunks"]
    } for citation in result["citations"])


@pytest.mark.parametrize(
    "question",
    [
        "What is the price of an unlisted dish?",
    ],
)
def test_missing_or_live_actions_do_not_get_fabricated_answers(client, question):
    result = client.post("/api/chat", json={"message": question}).json()
    assert "not contain enough information" in result["answer"].casefold()
    assert not result["citations"]
    assert result["is_evidence_sufficient"] is False


def test_booking_request_is_never_confirmed(client):
    result = client.post(
        "/api/chat", json={"message": "Can you confirm a table for 7 PM tonight?"}
    ).json()
    assert "cannot confirm or make a reservation" in result["answer"].casefold()
    assert result["citations"]
    assert result["citations"][0]["source"] == "restaurant_policies.pdf"
    assert "not confirmed until staff confirms" in " ".join(
        result["citations"][0]["excerpt"].casefold().split()
    )


def test_delivery_fee_is_not_invented(client):
    result = client.post("/api/chat", json={"message": "What is the delivery fee?"}).json()
    assert "does not specify a delivery fee" in result["answer"].casefold()
    assert result["citations"]
    assert result["citations"][0]["source"] == "restaurant_policies.pdf"
    assert "delivery fees" in result["citations"][0]["excerpt"].casefold()
    assert "not specified in this document" in result["citations"][0]["excerpt"].casefold()
    assert "upi" not in result["citations"][0]["excerpt"].casefold()


def test_paneer_tikka_price_cites_its_menu_record(client):
    result = client.post(
        "/api/chat", json={"message": "What is the price of Paneer Tikka?"}
    ).json()
    assert "₹230" in result["answer"]
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["source"] == "restaurant_menu.pdf"
    assert citation["page_number"] == 1
    assert "Paneer Tikka" in citation["excerpt"]
    assert "INR 230" in citation["excerpt"]
    assert "Chicken Pepper Fry" not in citation["excerpt"]


def test_weekend_hours_citation_is_the_policy_passage(client):
    result = client.post(
        "/api/chat", json={"message": "What are the weekend opening hours?"}
    ).json()
    assert result["answer"] == (
        "The sample restaurant's weekend hours are Saturday and Sunday, 10:00 AM–10:30 PM."
    )
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["source"] == "restaurant_policies.pdf"
    assert citation["page_number"] == 1
    excerpt = citation["excerpt"].casefold()
    assert "saturday" in excerpt and "sunday" in excerpt
    assert "10:00 am" in excerpt and "10:30 pm" in excerpt


def test_citation_excerpt_contains_the_claimed_payment_method(client):
    result = client.post(
        "/api/chat", json={"message": "Does the sample restaurant accept UPI?"}
    ).json()
    assert any("upi" in citation["excerpt"].casefold() for citation in result["citations"])


def test_allergen_question_never_guarantees_safety(client):
    result = client.post(
        "/api/chat",
        json={"message": "Is Paneer Butter Masala safe for someone with a milk allergy?"},
    ).json()
    answer = result["answer"].casefold()
    assert "cannot establish allergy safety" in answer
    assert "confirm with restaurant staff" in answer
    assert result["citations"]


def test_comparison_answers_both_dishes_with_prices_ingredients_and_relevant_sources(client):
    result = client.post(
        "/api/chat",
        json={
            "message": "Compare Paneer Tikka and Paneer Butter Masala by price and ingredients."
        },
    ).json()
    answer = result["answer"].casefold()
    assert "paneer tikka" in answer and "inr 230" in answer
    assert "paneer, yogurt marinade, peppers, spices" in answer
    assert "paneer butter masala" in answer and "inr 280" in answer
    assert "paneer, tomato gravy, butter, cream, spices" in answer
    assert answer.index("paneer tikka") < answer.index("paneer butter masala")
    assert result["answer"].count("\n| ") == 2
    menu_citations = [
        citation for citation in result["citations"]
        if citation["source"] == "restaurant_menu.pdf"
    ]
    assert len(menu_citations) == 2
    assert "restaurant_menu.pdf" in {citation["source"] for citation in result["citations"]}
    assert all(
        citation["chunk_id"] in {
            chunk["chunk_id"] for chunk in result["retrieval_trace"]["selected_chunks"]
        }
        for citation in result["citations"]
    )
    assert all("Paneer" in citation["excerpt"] for citation in result["citations"])


def test_wheat_and_dairy_answer_separates_explicit_and_conditional_entries(client):
    result = client.post(
        "/api/chat",
        json={"message": "Which dishes contain wheat, and which contain dairy?"},
    ).json()
    answer = result["answer"].casefold()
    wheat, dairy = answer.split("## dairy", maxsplit=1)
    for dish in ("plain roti", "gulab jamun", "chocolate brownie"):
        assert dish in wheat
    for dish in (
        "paneer tikka",
        "paneer butter masala",
        "gulab jamun",
        "chocolate brownie",
        "tomato basil soup",
        "dal tadka with jeera rice",
        "masala chai",
        "filter coffee",
    ):
        assert dish in dairy
    assert "conditional dairy ingredients and flags" in dairy
    assert "**potential allergen flags:**" in dairy
    assert "chocolate brownie — wheat / gluten, milk, egg" in dairy
    assert "tomato, basil, vegetable stock; cream may be added" in dairy
    assert "milk if prepared with milk" in dairy
    assert "allergen flag is not an ingredient list" in dairy
    assert "selected ingredient" in answer
    assert result["citations"]
    assert all(citation["source"] == "ingredients_allergens.pdf" for citation in result["citations"])
    assert all(citation["page_number"] in {1, 2} for citation in result["citations"])


def test_paneer_tikka_milk_allergy_uses_only_relevant_evidence(client):
    result = client.post(
        "/api/chat",
        json={
            "message": "Can you guarantee that Paneer Tikka is safe for someone with a milk allergy?"
        },
    ).json()
    answer = result["answer"].casefold()
    assert "cannot guarantee" in answer
    assert "paneer, yogurt marinade, peppers, spices" in answer
    assert "milk" in answer and "cross-contact" in answer
    assert "confirm" in answer and "restaurant staff" in answer
    assert result["citations"]
    assert all(citation["source"] == "ingredients_allergens.pdf" for citation in result["citations"])
    excerpts = " ".join(citation["excerpt"].casefold() for citation in result["citations"])
    assert "paneer tikka" in excerpts
    assert "cross-contact conditions can vary" in excerpts
    assert "crispy corn" not in excerpts
    assert "chicken pepper fry" not in excerpts
    selected_ids = {
        chunk["chunk_id"] for chunk in result["retrieval_trace"]["selected_chunks"]
    }
    assert all(citation["chunk_id"] in selected_ids for citation in result["citations"])


def test_health_and_document_status(client):
    health = client.get("/api/health").json()
    docs = client.get("/api/documents").json()
    assert health["status"] == "healthy"
    assert health["documents"] == 3
    assert health["pages"] >= 6
    assert health["chunks"] == docs["total_chunks"]


def test_conversation_history_preserves_answer_sources(client):
    first = client.post("/api/chat", json={"message": "Does the sample restaurant accept UPI?"}).json()
    second = client.post(
        "/api/chat",
        json={"message": "What payment methods were listed?", "conversation_id": first["conversation_id"]},
    ).json()
    history = client.get(f"/api/chat/conversations/{first['conversation_id']}").json()
    assert second["conversation_id"] == first["conversation_id"]
    assert len(history) == 4
    assert history[1]["response"]["citations"]


def test_repeated_chat_sequence_keeps_grounded_menu_answer(client):
    questions = [
        "Does the sample restaurant accept UPI?",
        "What is the price of Paneer Tikka?",
        "What vegetarian main courses cost less than INR 250?",
        "What is the delivery fee?",
        "What is the price of an unlisted dish?",
        "What is the price of Paneer Tikka?",
    ]
    conversation_id = None
    results = []
    for question in questions:
        response = client.post(
            "/api/chat",
            json={"message": question, "conversation_id": conversation_id},
        )
        assert response.status_code == 200
        result = response.json()
        conversation_id = result["conversation_id"]
        results.append(result)

    assert "UPI" in results[0]["answer"]
    assert results[1]["answer"] == "The sample menu lists Paneer Tikka for INR 230 (₹230)."
    assert [citation["source"] for citation in results[1]["citations"]] == ["restaurant_menu.pdf"]
    assert "Vegetable Pulao" in results[2]["answer"]
    assert "Dal Tadka with Jeera Rice" in results[2]["answer"]
    assert "does not specify a delivery fee" in results[3]["answer"]
    assert results[4]["is_evidence_sufficient"] is False
    assert not results[4]["citations"]
    assert results[5]["answer"] == results[1]["answer"]
    assert [citation["source"] for citation in results[5]["citations"]] == ["restaurant_menu.pdf"]


def test_eight_requested_questions_remain_independent_in_one_conversation(client):
    questions = [
        "What is the price of Paneer Tikka?",
        "Compare Paneer Tikka and Paneer Butter Masala by price and ingredients.",
        "Which dishes contain wheat, and which contain dairy?",
        "Can you guarantee that Paneer Tikka is safe for someone with a milk allergy?",
        "What are the weekend opening hours?",
        "What is the delivery fee?",
        "What is the price of an unlisted dish?",
        "Does the sample restaurant accept UPI?",
    ]
    conversation_id = None
    results = []
    for question in questions:
        response = client.post(
            "/api/chat",
            json={"message": question, "conversation_id": conversation_id},
        )
        assert response.status_code == 200
        result = response.json()
        conversation_id = result["conversation_id"]
        results.append(result)

    assert "₹230" in results[0]["answer"]
    assert "Paneer Tikka" in results[1]["answer"]
    assert "Paneer Butter Masala" in results[1]["answer"]
    assert "₹280" in results[1]["answer"]
    assert "Plain Roti" in results[2]["answer"]
    assert "Filter Coffee" in results[2]["answer"]
    assert "conditional dairy" in results[2]["answer"].casefold()
    assert "cannot guarantee" in results[3]["answer"].casefold()
    assert "cross-contact" in results[3]["answer"].casefold()
    assert "10:00 AM" in results[4]["answer"]
    assert "does not specify a delivery fee" in results[5]["answer"].casefold()
    assert results[6]["is_evidence_sufficient"] is False
    assert not results[6]["citations"]
    assert "UPI" in results[7]["answer"]
    assert [citation["source"] for citation in results[4]["citations"]] == [
        "restaurant_policies.pdf"
    ]
    assert [citation["source"] for citation in results[7]["citations"]] == [
        "restaurant_policies.pdf"
    ]


def test_api_rejects_empty_question(client):
    assert client.post("/api/chat", json={"message": " "}).status_code == 422


def test_upload_rejects_unsupported_and_malformed_files(client):
    unsupported = client.post(
        "/api/documents/upload",
        files={"file": ("notes.exe", b"no", "application/octet-stream")},
    )
    assert unsupported.status_code == 400

    malformed = client.post(
        "/api/documents/upload",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )
    assert malformed.status_code == 400


def test_gemini_missing_configuration_uses_explicit_local_mode():
    generator = AnswerGenerator(api_key="")
    assert generator.client is None


def test_gemini_client_uses_bounded_request_timeout(monkeypatch):
    from google import genai

    captured = {}

    def create_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(genai, "Client", create_client)
    _create_genai_client("test-key")
    assert captured["http_options"].timeout == GEMINI_REQUEST_TIMEOUT_MS


def test_embedding_failure_does_not_silently_switch_models():
    embedder = EmbeddingService(api_key="test-key")

    def fail_embed(**_kwargs):
        raise RuntimeError("simulated network failure")

    embedder.client = SimpleNamespace(models=SimpleNamespace(embed_content=fail_embed))
    with pytest.raises(RAGError, match="Gemini embedding request failed"):
        embedder.embed_documents(["test"])


def test_generation_failure_returns_a_cited_local_excerpt():
    generator = AnswerGenerator(api_key="test-key")

    class FailedModels:
        @staticmethod
        def generate_content(**_kwargs):
            raise RuntimeError("simulated model failure")

    generator.client = SimpleNamespace(models=FailedModels())
    chunk = {
        "chunk_id": "actual_chunk",
        "filename": "restaurant_menu.pdf",
        "page_number": 1,
        "content": "Crispy Corn costs INR 180 in this fictional sample menu.",
    }
    answer, citations, mode = generator.generate(
        "What is Crispy Corn's price?", [chunk], True, []
    )
    assert "could not be reached" in answer
    assert citations[0].chunk_id == "actual_chunk"
    assert mode == "local_extractive_after_gemini_error"


def test_gemini_is_used_when_configured_with_verified_chunk_ids():
    generator = AnswerGenerator(api_key="test-key")
    chunk = {
        "chunk_id": "verified_chunk",
        "filename": "restaurant_menu.pdf",
        "page_number": 1,
        "content": "Crispy Corn is listed for INR 180.",
    }

    class Models:
        @staticmethod
        def generate_content(**_kwargs):
            return SimpleNamespace(
                text=json.dumps(
                    {"answer": "Crispy Corn is listed for INR 180.", "citations": ["verified_chunk"]}
                )
            )

    generator.client = SimpleNamespace(models=Models())
    answer, citations, mode = generator.generate("What does Crispy Corn cost?", [chunk], True, [])
    assert "INR 180" in answer
    assert citations[0].chunk_id == "verified_chunk"
    assert mode == "gemini"
