"""
Script debug: cek isi ChromaDB memory untuk user tertentu.
Jalankan: uv run python scripts/check_memory.py [user_id]
"""
import sys
import os

os.environ.setdefault("CHROMA_DB_PATH", "./data/chroma_db")
# pyrefly: ignore [missing-import]
import chromadb

USER_ID = sys.argv[1] if len(sys.argv) > 1 else "usr_e2e_1779293242"
DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_or_create_collection("user_memories")

results = col.get(where={"user_id": USER_ID}, include=["documents", "metadatas"])

print(f"\n{'='*60}")
print(f"ChromaDB Memories — User: {USER_ID}")
print(f"Total entries: {len(results['ids'])}")
print(f"{'='*60}\n")

# Group by type
from collections import defaultdict
by_type = defaultdict(list)
for doc, meta in zip(results["documents"], results["metadatas"]):
    mem_type = meta.get("type", "unknown")
    by_type[mem_type].append((meta.get("timestamp", "")[:19], doc))

for mem_type, entries in sorted(by_type.items()):
    print(f"▶ [{mem_type}] — {len(entries)} entry")
    for ts, doc in sorted(entries):
        print(f"  [{ts}] {doc[:120]}")
    print()
