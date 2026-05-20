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
            
    def delete_user_memories(self, user_id: str):
        """
        Deletes all memories stored for a specific user. Useful for privacy/reset features.
        """
        try:
            self.collection.delete(where={"user_id": user_id})
        except Exception as e:
            print(f"⚠️ [MemoryStore] Error deleting user memories: {e}")
