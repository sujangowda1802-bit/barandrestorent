import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.rag import document_manager


if __name__ == "__main__":
    result = document_manager.sync_managed_directory(settings.documents_dir, force=True)
    print(
        f"Scanned {result['documents_processed']} documents; "
        f"removed {result['documents_removed']} missing managed documents."
    )
    for document in document_manager.documents.values():
        print(
            f"{document['filename']}: {document['page_count']} pages, "
            f"{document['chunk_count']} chunks ({document['status']})"
        )
