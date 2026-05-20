import os
import sys
import json
import asyncio
from datetime import datetime
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set Python path to find services and routers packages properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.memory_store import MemoryStore
from services.llm_companion import LLMCompanionService

async def run_tests():
    print("🚀 Running LLM Companion Integration Tests...")
    
    # 1. Test ChromaDB MemoryStore
    print("\n--- 1. Testing Memory Store (ChromaDB) ---")
    try:
        store = MemoryStore()
        test_user = "user_test_123"
        test_text = "Saya merasa cemas karena ada ujian besok pagi dan saya kurang tidur."
        test_type = "user_chat"
        timestamp = datetime.now().isoformat()
        
        print("Adding memory to ChromaDB...")
        doc_id = store.add_memory(
            user_id=test_user,
            text=test_text,
            mem_type=test_type,
            timestamp=timestamp
        )
        print(f"✅ Memory added successfully with ID: {doc_id}")
        
        print("Querying memories from ChromaDB...")
        memories = store.retrieve_memories(
            user_id=test_user,
            query_text="ujian dan cemas",
            limit=2
        )
        
        print(f"Found {len(memories)} memories:")
        for idx, m in enumerate(memories, 1):
            print(f"  {idx}. [{m['metadata'].get('type')}] -> {m['document']}")
            
        assert len(memories) > 0, "No memories were returned!"
        print("✅ MemoryStore retrieval test passed!")
        
        # Clean up test user memories
        print("Cleaning up test memories...")
        store.delete_user_memories(test_user)
        print("✅ MemoryStore cleanup test passed!")
        
    except Exception as e:
        print(f"❌ MemoryStore Test failed: {e}")
        
    # 2. Test LLM Companion Service & Gemini Integration (Streaming SSE)
    print("\n--- 2. Testing LLM Companion Service (Gemini RAG SSE) ---")
    try:
        service = LLMCompanionService()
        test_user_2 = "user_chat_test"
        test_message = "Aku sedang merasa sangat stres karena deadline tugas besok, apa saranmu?"
        
        print(f"Is GEMINI_API_KEY Configured? {service.is_configured}")
        print("Sending SSE streaming chat request to service...")
        
        print("\n--- Streaming Response (Real-time Token Visualizer) ---")
        accumulated_text = ""
        recommendations = []
        
        async for event_chunk in service.chat_stream(user_id=test_user_2, user_message=test_message):
            # Parse SSE format
            lines = event_chunk.strip().split("\n")
            event_type = ""
            data_content = ""
            for line in lines:
                if line.startswith("event:"):
                    event_type = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
            
            if event_type == "delta" and data_content:
                data_json = json.loads(data_content)
                token = data_json.get("token", "")
                accumulated_text += token
                print(token, end="", flush=True)
            elif event_type == "final" and data_content:
                data_json = json.loads(data_content)
                recommendations = data_json.get("recommendations", [])
                
        print("\n\n--- Recommendations Received ---")
        for rec in recommendations:
            print(f" - {rec}")
        print("---------------------------------")
        
        assert len(accumulated_text) > 0, "No response tokens were accumulated!"
        assert len(recommendations) > 0, "No recommendations were generated!"
        print("✅ LLM Companion SSE Service test completed successfully!")
        
        # Clean up test user memories
        store = MemoryStore()
        store.delete_user_memories(test_user_2)
        
    except Exception as e:
        print(f"❌ LLM Companion Service Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_tests())
