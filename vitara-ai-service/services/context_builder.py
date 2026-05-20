from typing import List, Dict, Any

class ContextBuilder:
    @staticmethod
    def construct_query_text(user_message: str) -> str:
        """
        Prepares and cleans the user message to be used as a query for ChromaDB.
        In this implementation, it returns the raw message, but this can be extended 
        to extract keywords, remove stop words, or formulate a hybrid search query.
        """
        return user_message.strip()

    @staticmethod
    def stitch_memories(memories: List[Dict[str, Any]]) -> str:
        """
        Stitches retrieved memories from ChromaDB into a clean, cohesive string context.
        
        Each memory should be a dict with:
        - 'document': The actual text content
        - 'metadata': A dict containing 'timestamp', 'type', and optional health details
        """
        if not memories:
            return ""
        
        # Sort memories chronologically by timestamp if possible
        try:
            sorted_memories = sorted(
                memories, 
                key=lambda x: x.get("metadata", {}).get("timestamp", "")
            )
        except Exception:
            sorted_memories = memories
            
        context_parts = []
        context_parts.append("Konteks riwayat kesehatan dan percakapan pengguna:")
        
        for idx, mem in enumerate(sorted_memories, 1):
            doc = mem.get("document", "").strip()
            meta = mem.get("metadata", {})
            timestamp = meta.get("timestamp", "Waktu tidak diketahui")
            mem_type = meta.get("type", "Umum").upper()
            
            # Format nicely: - [Timestamp] [TYPE] Content
            part = f"- [{timestamp}] [Tipe: {mem_type}] {doc}"
            context_parts.append(part)
            
        return "\n".join(context_parts)
