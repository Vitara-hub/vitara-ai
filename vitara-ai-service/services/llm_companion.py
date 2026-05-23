import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

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
        self.model_name = os.getenv("COMPANION_MODEL_NAME", "gemini-3.1-flash-lite")
        
        if self.is_configured:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    async def chat_stream(self, user_id: str, user_message: str):
        """
        Processes a chat request using RAG pipeline with SSE streaming.
        1. Query ChromaDB for past memories.
        2. Format context with ContextBuilder.
        3. Assemble system and user prompts.
        4. Call Gemini with Streaming (chunk-by-chunk) yielding 'delta' events.
        5. Generate 2 to 4 health recommendations in a fast second call yielding 'final' event.
        6. Store new message and response into ChromaDB.
        """
        # Step 1: Retrieve diverse memories from ChromaDB (per memory type)
        query_text = ContextBuilder.construct_query_text(user_message)
        raw_memories = memory_store.retrieve_diverse_memories(
            user_id=user_id,
            query_text=query_text,
            per_type_limit=2,
            mem_types=[
                "user_journal",       # Teks asli jurnal pengguna
                "nlp_prediction",     # Hasil analisis emosi & stres (journal)
                "food_prediction",    # Hasil analisis makanan (vision)
                "sleep_prediction",   # Hasil analisis kualitas tidur
                "typing_prediction",  # Hasil analisis stres dari typing
                "health_score",       # Skor kesehatan keseluruhan
                "user_chat",          # Riwayat pesan percakapan pengguna
                "companion_response", # Riwayat respons companion
            ]
        )
        
        # Step 2: Stitch memories into a cohesive context string
        context_str = ContextBuilder.stitch_memories(raw_memories)
        
        # Step 3: Build user prompt
        prompt = get_user_chat_prompt(user_message=user_message, context_str=context_str)
        
        # Step 4: Query Gemini API
        if not self.is_configured or not self.client:
            # Emulated local fallback mode for testing if API Key is not configured
            fallback_text = f"[Demo Mode] Halo! Saya menerima pesanmu: '{user_message}'. Saat ini GEMINI_API_KEY belum dikonfigurasi di file .env. Silakan tambahkan API key Anda untuk mendapatkan respons cerdas dari Gemini."
            fallback_recs = [
                "Konfigurasikan GEMINI_API_KEY di file .env",
                "Pastikan koneksi internet aktif",
                "Coba jalankan ulang server setelah menambahkan key"
            ]
            
            # Stream the fallback text in chunks to simulate typing speed
            words = fallback_text.split(" ")
            accumulated = ""
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                accumulated += token
                yield f"event: delta\ndata: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.03) # 30ms delay
                
            yield f"event: final\ndata: {json.dumps({'full_response': accumulated, 'recommendations': fallback_recs})}\n\n"
            
            # Save message even in demo mode
            self._save_interaction_to_memory(user_id, user_message, accumulated)
            return

        accumulated_response = ""
        try:
            # Step 4.1: Stream response from Gemini using Client.aio
            response = await self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7
                )
            )
            
            async for chunk in response:
                token = chunk.text or ""
                if token:
                    accumulated_response += token
                    yield f"event: delta\ndata: {json.dumps({'token': token})}\n\n"
            
            # If for some reason we got an empty response, use a gentle fallback message
            if not accumulated_response:
                accumulated_response = "Maaf, asisten AI tidak memberikan respons. Silakan coba kirim pesan lain."
                yield f"event: delta\ndata: {json.dumps({'token': accumulated_response})}\n\n"
                
        except Exception as e:
            print(f" [LLMCompanion] Error during Gemini streaming: {e}")
            accumulated_response = "Maaf, terjadi kesalahan saat menghubungi asisten AI saya. Namun, cobalah untuk tetap rileks, minum segelas air putih, dan beristirahat sejenak."
            yield f"event: delta\ndata: {json.dumps({'token': accumulated_response})}\n\n"

        # Step 4.2: Fast second call to Gemini to generate Pydantic-validated recommendations
        recommendations = []
        try:
            # We construct a swift recommendations prompt
            rec_prompt = (
                f"Berdasarkan percakapan berikut, berikan 2 sampai 4 rekomendasi tindakan kesehatan konkret, "
                f"praktis, dan personal dalam bahasa Indonesia.\n\n"
                f"Pesan Pengguna: {user_message}\n"
                f"Tanggapan Asisten: {accumulated_response}\n\n"
                f"Format output harus berupa list JSON berisi string rekomendasi."
            )
            
            class RecommendationsSchema(BaseModel):
                recommendations: List[str] = Field(description="2 to 4 concrete, actionable health recommendations in Indonesian.")
                
            rec_response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=rec_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RecommendationsSchema,
                    temperature=0.3
                )
            )
            parsed_recs = json.loads(rec_response.text)
            recommendations = parsed_recs.get("recommendations", [])
        except Exception as e:
            print(f" [LLMCompanion] Error generating recommendations: {e}")
            # Fallback recommendations
            recommendations = [
                "Beri jeda sejenak sebelum mencoba lagi",
                "Pastikan koneksi internet stabil",
                "Minum air putih untuk menenangkan pikiran"
            ]
            
        # Yield the final SSE block
        yield f"event: final\ndata: {json.dumps({'full_response': accumulated_response, 'recommendations': recommendations})}\n\n"
        
        # Step 5: Save current interaction to vector memory asynchronously in the background
        try:
            self._save_interaction_to_memory(user_id, user_message, accumulated_response)
        except Exception as e:
            print(f" [LLMCompanion] Error saving to vector memory: {e}")

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
