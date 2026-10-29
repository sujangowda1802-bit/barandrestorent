# Bar & Restaurant Assistant

A university capstone demonstration of a document-grounded restaurant chatbot for the fictional **Green Fork Bistro**. It answers using the three supplied sample PDFs, shows verified source excerpts and page numbers, and abstains when the documents do not establish an answer. The data is fictional and must not be used to make real restaurant, dietary, or food-safety decisions.

## Architecture

```text
PDF/TXT/Markdown
  -> PyMuPDF page extraction
  -> overlapping page-aware chunks with stable document/chunk IDs
  -> Gemini embeddings (or deterministic local embeddings when no key is configured)
  -> persistent ChromaDB + persisted BM25 index
  -> Reciprocal Rank Fusion
  -> bounded evidence context -> Gemini answer generation
  -> citation IDs checked against retrieved chunks -> browser UI
```

FastAPI serves the API and static frontend. The browser does not receive the Gemini key. The re-index action and uploads are restricted to loopback clients; the server defaults to `127.0.0.1`.

PDF tables are split into labeled, page-aware row chunks so menu items, allergen records, and policy entries stay together; non-table page text is retained as narrative chunks. Hybrid retrieval keeps ChromaDB and BM25, adds exact-entity and requested allergen-facet coverage for multi-part questions, and includes only focused source excerpts with answers. A chunking-version change triggers managed-document re-indexing at startup; run `python scripts\ingest_documents.py` to force a rebuild.

## Project structure

```text
backend/app/config.py       Environment and project paths
backend/app/main.py         FastAPI endpoints and local admin actions
backend/app/rag.py          PDF ingestion, ChromaDB, BM25, hybrid retrieval, Gemini
backend/app/schemas.py      Request and response models
backend/tests/test_rag.py   Ingestion, retrieval, citations, and API checks
data/documents/             Managed source documents
data/chroma/                Persistent vector collection (generated)
data/metadata/              Document registry and BM25 data (generated)
data/uploads/               Safe temporary copies of uploaded files (generated)
frontend/                   Responsive HTML/CSS/JavaScript chat
scripts/                    Ingestion and manual evaluation commands
```

## Windows setup

Python 3.11 or newer is recommended.

```powershell
Set-Location 'D:\Bar&restorent'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `GOOGLE_API_KEY` in `.env` to enable Gemini generation and embeddings. Keep that value private; `.env` is ignored by Git. Without a key, the app uses local deterministic embeddings and returns relevant source text extractively rather than claiming Gemini generated an answer.
Gemini embedding and generation requests have a 30-second HTTP timeout. Generation falls back to cited document text on timeout; embedding failures are reported rather than silently switching embedding models.

The configured embedding identity is persisted with the ChromaDB collection. If you change between local and Gemini embeddings, do not mix vectors: use a new `data/chroma` directory and re-index all PDFs.

## Ingest and run

The three demonstration PDFs are already in `data/documents`. On startup, the server synchronizes that directory, indexes new or changed files, and removes previously managed documents no longer present.

```powershell
python scripts\ingest_documents.py
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001). If that port is already in use, select a free port and use it in both the command and URL. Ask about sample prices, ingredients, allergen notes, hours, UPI, reservations, and delivery policies. Answers include source filenames and page numbers plus expandable hybrid retrieval diagnostics. Upload accepts PDF, Markdown, and TXT within the configured size limit. Re-index scans the managed PDF directory.

## Tests and evaluation

```powershell
python -m pytest backend\tests -q
python scripts\run_evaluation.py
```

The evaluation script asks the eight requested sample questions and prints answer evidence sufficiency, retrieved chunks, and citations. Verify the retrieval trace/citations alongside the answer; plausible wording alone is not a pass.

## Safety and limitations

- All Green Fork Bistro data is fictional. Contact/address details in the sample policy PDF are explicitly demo-only.
- This application cannot make reservations, take orders or payments, or track deliveries.
- Delivery coverage, fees, live availability, holiday hours, and final taxes are not specified.
- Ingredient lists and sample allergen notes do not prove allergy safety or absence of cross-contact. Confirm with restaurant staff.
- Scanned PDFs without an extractable text layer are reported; OCR is not installed.
- Chat history is held in server memory and is lost when the process restarts.
- Local embedding fallback supports an offline demonstration but is less semantically capable than a configured embedding API.
- There is no user-login/authentication system. Upload and re-index endpoints only accept loopback connections; expose the app publicly only after adding proper authentication and deployment hardening.

## Presentation screenshots

Capture (1) the welcome screen and sample-question buttons, (2) a menu answer with its PDF/page citation, (3) the index status and document list, (4) an explicit insufficient-evidence answer for a delivery fee or booking request, and (5) the evaluation terminal output.
