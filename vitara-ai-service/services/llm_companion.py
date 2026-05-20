import os
import json
from datetime import datetime
from typing import Dict, Any, List
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
import google.generativeai as genai

from services.memory_store import MemoryStore
from services.context_builder import ContextBuilder
from services.prompts import SYSTEM_INSTRUCTION, get_user_chat_prompt

# Initialize MemoryStore as a shared singleton or instance
memory_store = MemoryStore()

# Define the expected schema for Gemini Structured Output
class CompanionResponseSchema(BaseModel):
    response: str = Field(description="The main warm, empathetic, and health-focused conversational response in Indonesian.")
    recommendations: List[str] = Field(description="2 to 4 concrete, highly actionable, and personalized wellness tips for the user today.")

class LLMCompanionService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_configured = self.api_key is not None and self.api_key != "your_gemini_api_key_here"
        
        if self.is_configured:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_INSTRUCTION
            )
        else:
            self.model = None

    async def chat(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """
        Processes a chat request using RAG pipeline:
        1. Query ChromaDB for past memories
        2. Format context with ContextBuilder
        3. Assemble system and user prompts
        4. Call Gemini with Structured JSON Output
        5. Store new message and response into ChromaDB
        """
        # Step 1: Retrieve memories from ChromaDB using user_message as query
        query_text = ContextBuilder.construct_query_text(user_message)
        raw_memories = memory_store.retrieve_memories(user_id=user_id, query_text=query_text, limit=4)
        
        # Step 2: Stitch memories into a cohesive context string
        context_str = ContextBuilder.stitch_memories(raw_memories)
        
        # Step 3: Build user prompt
        prompt = get_user_chat_prompt(user_message=user_message, context_str=context_str)
        
        # Step 4: Query Gemini API
        if not self.is_configured or not self.model:
            # Emulated local fallback mode for testing if API Key is not configured
            fallback_response = {
                "response": f"[Demo Mode] Halo! Saya menerima pesanmu: '{user_message}'. Saat ini GEMINI_API_KEY belum dikonfigurasi di file .env. Silakan tambahkan API key Anda untuk mendapatkan respons cerdas dari Gemini.",
                "recommendations": [
                    "Konfigurasikan GEMINI_API_KEY di file .env",
                    "Pastikan koneksi internet aktif",
                    "Coba jalankan ulang server setelah menambahkan key"
                ]
            }
            # Save message even in demo mode
            self._save_interaction_to_memory(user_id, user_message, fallback_response["response"])
            return fallback_response

        try:
            # Query the model using Structured Output
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CompanionResponseSchema,
                    temperature=0.7
                )
            )
            
            # Parse the response text
            parsed_data = json.loads(response.text)
            
            # Ensure the output has correct keys
            final_response = {
                "response": parsed_data.get("response", ""),
                "recommendations": parsed_data.get("recommendations", [])
            }
            
            # Step 5: Save current interaction to vector memory asynchronously/in background
            self._save_interaction_to_memory(user_id, user_message, final_response["response"])
            
            return final_response
            
        except Exception as e:
            print(f" [LLMCompanion] Error calling Gemini API: {e}")
            # Robust fallback on error
            err_response = {
                "response": "Maaf, terjadi kesalahan saat menghubungi asisten AI saya. Namun, cobalah untuk tetap rileks, minum segelas air putih, dan beristirahat sejenak.",
                "recommendations": [
                    "Beri jeda sejenak sebelum mencoba lagi",
                    "Pastikan koneksi internet stabil",
                    "Minum air putih untuk menenangkan pikiran"
                ]
            }
            self._save_interaction_to_memory(user_id, user_message, err_response["response"])
            return err_response

    def _save_interaction_to_memory(self, user_id: str, user_message: str, companion_response: str):
        """
        Helper to save the current conversation turn to ChromaDB.
        Saves both user's input and companion's response to keep rich chronological context.
        """
        now = datetime.now().isoformat()
        
        # Save User Input
        memory_store.add_memory(
            user_id=user_id,
            text=f"Pengguna berkata: \"{user_message}\"",
            mem_type="user_chat",
            timestamp=now
        )
        
        # Save Companion Response
        memory_store.add_memory(
            user_id=user_id,
            text=f"Vitara Companion merespons: \"{companion_response}\"",
            mem_type="companion_response",
            timestamp=now
        )
