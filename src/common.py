from pathlib import Path
import json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "synthetic_tickets.json"
load_dotenv(ROOT / ".env")

def load_tickets():
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def ticket_documents():
    from langchain_core.documents import Document
    return [Document(
        page_content=f"Ticket ID: {t['ticket_id']}\nTitle: {t['title']}\n"
                     f"Category: {t['category']}\nPriority: {t['priority']}\n"
                     f"Description: {t['description']}\nResolution: {t['resolution']}",
        metadata={"ticket_id":t["ticket_id"],"category":t["category"],"priority":t["priority"]}
    ) for t in load_tickets()]
