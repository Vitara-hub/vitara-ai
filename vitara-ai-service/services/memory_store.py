import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from chromadb.config import Settings

class MemoryStore:
    def __init__(self):
        # Load database path from environment variable, default to local directory
        self.db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        
        # Ensure path exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Create or fetch collection 'user_memories'
        # ChromaDB will use its default lightweight ONNX-based embedding function
        self.collection = self.client.get_or_create_collection(
            name="user_memories"
        )
        
    def add_memory(self, user_id: str, text: str, mem_type: str = "chat", timestamp: str = None, doc_id: str = None) -> str:
        """
        Saves a text document as a memory vector with associated metadata.
        Supports deterministic doc_id and upsert operation for idempotency.
        """
        if not text or not text.strip():
            return ""
            
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Default to current ISO format timestamp if not provided
        if not timestamp:
            timestamp = datetime.now().isoformat()
            
        metadata = {
            "user_id": user_id,
            "timestamp": timestamp,
            "type": mem_type
        }
        
        try:
            self.collection.upsert(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return doc_id
        except Exception as e:
            print(f"⚠️ [MemoryStore] Error adding/updating memory in ChromaDB: {e}")
            return ""

    def retrieve_memories(self, user_id: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant memories for a specific user using vector similarity.
        """
        if not query_text or not query_text.strip():
            return []
            
        try:
            # Query collection with filter by user_id
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={"user_id": user_id}
            )
            
            # Format results into a clean list of memory dicts
            memories = []
            
            if not results or "documents" not in results or not results["documents"]:
                return memories
                
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else None
            
            for idx in range(len(documents)):
                memory_item = {
                    "id": ids[idx],
                    "document": documents[idx],
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                }
                if distances is not None and idx < len(distances):
                    memory_item["distance"] = distances[idx]
                    
                memories.append(memory_item)
                
            return memories
            
        except Exception as e:
            print(f"⚠️ [MemoryStore] Error querying memories from ChromaDB: {e}")
            return []

    def retrieve_diverse_memories(
        self,
        user_id: str,
        query_text: str,
        per_type_limit: int = 2,
        mem_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves memories per memory type using a Hybrid Retrieval strategy:
        1. Gets the top 1 most semantically similar memory via collection.query.
        2. Gets the 1 absolute newest memory chronologically via collection.get and sorting.
        3. Unions both results (eliminating duplicate IDs) to ensure the LLM receives
           both the latest health state and relevant historic context.
        """
        if not query_text or not query_text.strip():
            return []

        if mem_types is None:
            mem_types = [
                "user_journal",       # Teks asli jurnal pengguna
                "nlp_prediction",     # Hasil analisis emosi & stres (journal)
                "food_prediction",    # Hasil analisis makanan (vision)
                "sleep_prediction",   # Hasil analisis kualitas tidur
                "typing_prediction",  # Hasil analisis stres dari pengetikan
                "health_score",       # Skor kesehatan keseluruhan
                "user_chat",          # Riwayat pesan percakapan pengguna
                "companion_response", # Riwayat respons companion
            ]

        all_memories: Dict[str, Dict[str, Any]] = {}  # keyed by id to avoid duplicates

        for mem_type in mem_types:
            try:
                # Path A: Semantic Query (Top 1)
                semantic_limit = max(1, per_type_limit - 1)
                semantic_results = self.collection.query(
                    query_texts=[query_text],
                    n_results=semantic_limit,
                    where={"$and": [{"user_id": user_id}, {"type": mem_type}]}
                )

                if semantic_results and "documents" in semantic_results and semantic_results["documents"]:
                    documents = semantic_results["documents"][0]
                    metadatas = semantic_results.get("metadatas", [[]])[0]
                    ids = semantic_results.get("ids", [[]])[0]
                    distances = semantic_results.get("distances", [[]])[0] if "distances" in semantic_results else None

                    for idx in range(len(documents)):
                        mem_id = ids[idx]
                        memory_item = {
                            "id": mem_id,
                            "document": documents[idx],
                            "metadata": metadatas[idx] if idx < len(metadatas) else {},
                        }
                        if distances is not None and idx < len(distances):
                            memory_item["distance"] = distances[idx]
                        all_memories[mem_id] = memory_item

                # Path B: Absolute Newest Query (Top 1 chronologically)
                type_results = self.collection.get(
                    where={"$and": [{"user_id": user_id}, {"type": mem_type}]}
                )

                if type_results and "documents" in type_results and type_results["documents"]:
                    get_docs = type_results["documents"]
                    get_metas = type_results.get("metadatas", [])
                    get_ids = type_results.get("ids", [])

                    # Convert to list of dicts for sorting
                    items_to_sort = []
                    for idx in range(len(get_docs)):
                        items_to_sort.append({
                            "id": get_ids[idx],
                            "document": get_docs[idx],
                            "metadata": get_metas[idx] if idx < len(get_metas) else {},
                        })

                    if items_to_sort:
                        # Sort by timestamp descending to find newest
                        items_to_sort.sort(key=lambda x: x.get("metadata", {}).get("timestamp", ""), reverse=True)
                        newest_item = items_to_sort[0]
                        newest_id = newest_item["id"]
                        all_memories[newest_id] = newest_item

            except Exception as e:
                print(f"⚠️ [MemoryStore] Skipping type '{mem_type}' in hybrid retrieval: {e}")
                continue

        # Sort final list chronologically by timestamp ascending for standard RAG context stitching
        unique_memories = list(all_memories.values())
        try:
            unique_memories.sort(key=lambda x: x.get("metadata", {}).get("timestamp", ""))
        except Exception:
            pass

        return unique_memories
            
    def delete_user_memories(self, user_id: str):
        """
        Deletes all memories stored for a specific user. Useful for privacy/reset features.
        """
        try:
            self.collection.delete(where={"user_id": user_id})
        except Exception as e:
            print(f"⚠️ [MemoryStore] Error deleting user memories: {e}")
