import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app

QUESTIONS = [
    "What is the price of Paneer Tikka?",
    "Compare Paneer Tikka and Paneer Butter Masala by price and ingredients.",
    "Which dishes contain wheat, and which contain dairy?",
    "Can you guarantee that Paneer Tikka is safe for someone with a milk allergy?",
    "What are the weekend opening hours?",
    "What is the delivery fee?",
    "What is the price of an unlisted dish?",
    "Does the sample restaurant accept UPI?",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with TestClient(app) as client:
        for question in QUESTIONS:
            response = client.post("/api/chat", json={"message": question})
            if response.status_code != 200:
                print(f"ERROR {response.status_code}: {question}: {response.text}")
                continue
            result = response.json()
            cites = ", ".join(
                f"{item['source']} p.{item['page_number'] or 'N/A'}"
                for item in result["citations"]
            )
            print(f"\nQ: {question}\nA: {result['answer']}")
            print(
                f"Evidence sufficient: {result['is_evidence_sufficient']} | "
                f"retrieved chunks: {result['retrieval_trace']['fused_count']} | Sources: {cites or 'none'}"
            )


if __name__ == "__main__":
    main()
