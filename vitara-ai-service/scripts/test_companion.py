import os
import sys
import asyncio
from datetime import datetime

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
        
    # 2. Test LLM Companion Service & Gemini Integration
    print("\n--- 2. Testing LLM Companion Service (Gemini RAG) ---")
    try:
        service = LLMCompanionService()
        test_user_2 = "user_chat_test"
        test_message = "Aku sedang merasa sangat stres karena deadline tugas besok, apa saranmu?"
        
        print(f"Is GEMINI_API_KEY Configured? {service.is_configured}")
        print("Sending chat request to service...")
        
        # Run chat service
        response = await service.chat(user_id=test_user_2, user_message=test_message)
        
        print("\n--- Model Response Output ---")
        print(f"Response:\n{response.get('response')}")
        print("\nRecommendations:")
        for rec in response.get("recommendations", []):
            print(f" - {rec}")
        print("----------------------------")
        
        assert "response" in response, "Response key missing!"
        assert "recommendations" in response, "Recommendations key missing!"
        print("✅ LLM Companion Service test completed successfully!")
        
        # Clean up test user memories
        store = MemoryStore()
        store.delete_user_memories(test_user_2)
        
    except Exception as e:
        print(f"❌ LLM Companion Service Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_tests())
