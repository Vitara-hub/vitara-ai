import os
import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Set Python path to find services and routers packages properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env
load_dotenv()

from services.memory_store import MemoryStore

def clear_all_memories():
    print("🧹 Initializing MemoryStore...")
    try:
        store = MemoryStore()
        
        # Count items before clearing
        count_before = store.collection.count()
        print(f"Current document count in 'user_memories': {count_before}")
        
        if count_before == 0:
            print("✨ Collection is already empty. Nothing to clear!")
            return
            
        print("Clearing all documents in 'user_memories' collection...")
        # In ChromaDB, calling delete() without where deletes all documents in the collection
        store.collection.delete()
        
        count_after = store.collection.count()
        print(f"✅ Successfully cleared all memories! New count: {count_after}")
        
    except Exception as e:
        print(f"❌ Failed to clear ChromaDB: {e}")

if __name__ == "__main__":
    clear_all_memories()
