import uuid
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse

from config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
)
from memory.research_memory import ResearchMemory
from agents.search_agent import SearchAgent
from agents.summarize_agent import SummarizeAgent
from agents.writing_agent import WritingAgent


class Orchestrator:
    def __init__(self):
        print("🚀 Menginisialisasi Research Assistant...")

        self.session_id = str(uuid.uuid4())
        self.user_id = "user-hasan"
        self.request_count = 0

        self.memory = ResearchMemory()
        self.search_agent = SearchAgent(self.memory)
        self.summarize_agent = SummarizeAgent(self.memory)
        self.writing_agent = WritingAgent(self.memory)

        # Langfuse Initialization
        self.lf = self._init_langfuse()

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )

        # Intent Detection Chain
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Kamu adalah orchestrator yang menganalisis permintaan user.
Tugasmu: tentukan apakah user ingin:
1. "full_research" - penelitian lengkap (search + summarize + write)
2. "search_only" - hanya cari informasi
3. "summarize" - ringkas informasi yang sudah ada
4. "write" - tulis laporan dari info yang sudah ada
5. "chat" - percakapan biasa / tanya jawab

Balas HANYA dengan salah satu kata kunci di atas, tanpa penjelasan tambahan."""),
            ("human", "Permintaan user: {user_input}\n\nRiwayat: {chat_history}"),
        ])
        self.intent_chain = self.intent_prompt | self.llm | StrOutputParser()

        # Chat Chain
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "Kamu adalah Research Assistant AI yang ramah dan membantu.\nRiwayat penelitian: {research_history}"),
            ("human", "{user_input}"),
        ])
        self.chat_chain = self.chat_prompt | self.llm | StrOutputParser()

        try:
            self.lf.auth_check()
            print(f" [Langfuse] ✅ Koneksi berhasil!")
            print(f" [Langfuse] 📌 Session ID: {self.session_id}")
        except Exception as e:
            print(f" [Langfuse] ⚠️ Koneksi gagal: {e}")

        print("✅ Research Assistant siap digunakan!\n")

    def _init_langfuse(self):
        """Inisialisasi Langfuse dengan fallback"""
        try:
            from langfuse_config import get_langfuse
            print(" [Langfuse] Menggunakan langfuse_config.py")
            return get_langfuse()
        except ImportError:
            print(" [Langfuse] Menggunakan Langfuse langsung")
            return Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
                debug=False,
            )

    # ==================== HELPER TRACING ====================
    def _create_generation(self, trace, name: str, input_data, **kwargs):
        """Helper untuk membuat Generation di Langfuse"""
        return trace.generation(
            name=name,
            model=GEMINI_MODEL,
            input=input_data,
            metadata=kwargs
        )

    # ==================== INTENT & TOPIC ====================
    def _detect_intent(self, user_input: str, trace):
        gen = self._create_generation(
            trace=trace,
            name="intent-detection",
            input_data={"user_input": user_input},   # ← Diperbaiki
            temperature=0.1,
        )

        t0 = time.time()
        intent_raw = self.intent_chain.invoke({
            "user_input": user_input,
            "chat_history": self.memory.get_history_as_text()[:1500],
        }).strip().lower()

        valid_intents = ["full_research", "search_only", "summarize", "write", "chat"]
        intent = intent_raw if intent_raw in valid_intents else "chat"

        gen.end(
            output=intent,
            metadata={
                "duration_s": round(time.time() - t0, 3),
                "raw_output": intent_raw
            }
        )
        return intent

    def _extract_topic(self, user_input: str, trace):
        span = trace.span(name="topic-extraction", input=user_input)
        
        gen = self._create_generation(
            trace=span,
            name="topic-extraction-llm",
            input_data=f"Ekstrak topik dari: {user_input}"   # ← Diperbaiki
        )

        t0 = time.time()
        result = self.llm.invoke(
            f"Ekstrak topik utama dari kalimat berikut, jawab 3-8 kata saja.\nKalimat: {user_input}\nTopik:"
        )
        topic = result.content.strip()

        gen.end(output=topic)
        span.end(
            output=topic,
            metadata={"duration_s": round(time.time() - t0, 3)}
        )
        return topic

    # ==================== AGENT RUNNERS ====================
    def _run_search(self, topic: str, trace):
        span = trace.span(
            name="search-agent",
            input={"topic": topic},
            metadata={
                "agent": "SearchAgent",
                "tools": ["web_search", "multi_search"],
                "tool_source": "DuckDuckGo"
            }
        )
        t0 = time.time()
        result = self.search_agent.run(topic, parent_trace=span)
        span.end(
            output=result[:600],
            metadata={"duration_s": round(time.time() - t0, 2)}
        )
        return result

    def _run_summarize(self, topic: str, trace):
        span = trace.span(
            name="summarize-agent",
            input={"topic": topic},
            metadata={
                "agent": "SummarizeAgent",
                "pattern": "LCEL chain",
                "input_source": "memory.search_results"
            }
        )
        t0 = time.time()
        result = self.summarize_agent.run(topic, parent_trace=span)
        span.end(
            output=result[:600],
            metadata={"duration_s": round(time.time() - t0, 2)}
        )
        return result

    def _run_writing(self, topic: str, trace):
        span = trace.span(
            name="writing-agent",
            input={"topic": topic},
            metadata={
                "agent": "WritingAgent",
                "tools": ["save_report"],
                "output_format": "markdown"
            }
        )
        t0 = time.time()
        result = self.writing_agent.run(topic, parent_trace=span)
        span.end(
            output=result[:600],
            metadata={"duration_s": round(time.time() - t0, 2), "file_saved": True}
        )
        return result

    # ==================== MAIN RUN ====================
    def run(self, user_input: str) -> str:
        self.request_count += 1
        t_total = time.time()

        trace = self.lf.trace(
            name="research-assistant",
            input=user_input,
            session_id=self.session_id,
            user_id=self.user_id,
            tags=["multi-agent", "research"],
            metadata={
                "model": GEMINI_MODEL,
                "request_number": self.request_count,
                "memory_status": self.memory.status(),
                "version": "1.2.0"
            }
        )

        print(f"\n{'═'*70}")
        print(f"🎯 Orchestrator menerima input: '{user_input}'")
        print(f" [Langfuse] Trace ID : {trace.id}")
        print(f"{'═'*70}")

        try:
            intent = self._detect_intent(user_input, trace)
            print(f" Intent terdeteksi: [{intent.upper()}]")

            result = ""

            if intent == "full_research":
                topic = self._extract_topic(user_input, trace)
                self.memory.set_topic(topic)

                print(f"\n📌 Topik: {topic}")
                print("📋 Alur: Search → Summarize → Writing Agent\n")

                self._run_search(topic, trace)
                summary_result = self._run_summarize(topic, trace)
                self._run_writing(topic, trace)

                result = (
                    f"✅ Penelitian selesai untuk topik: **{topic}**\n\n"
                    f"📝 **Ringkasan:**\n{summary_result}\n\n"
                    f"📄 **Laporan lengkap telah disimpan ke folder `output/`**"
                )

            elif intent == "search_only":
                topic = self._extract_topic(user_input, trace)
                result = self._run_search(topic, trace)

            elif intent == "summarize":
                topic = self.memory.session_topic or self._extract_topic(user_input, trace)
                result = self._run_summarize(topic, trace)

            elif intent == "write":
                topic = self.memory.session_topic or self._extract_topic(user_input, trace)
                result = self._run_writing(topic, trace)

            else:  # chat mode
                print("\n📋 Mode: Chat")
                span = trace.span(name="chat-response")
                t0 = time.time()
                result = self.chat_chain.invoke({
                    "user_input": user_input,
                    "research_history": self.memory.get_all_summaries()
                })
                span.end(
                    output=result[:500],
                    metadata={"duration_s": round(time.time() - t0, 2)}
                )

            total_duration = round(time.time() - t_total, 2)

            trace.update(
                output=result[:800],
                metadata={
                    "total_duration_s": total_duration,
                    "intent": intent,
                    "request_number": self.request_count,
                }
            )

            print(f"\n [Langfuse] ✅ Trace berhasil dikirim! (Total: {total_duration}s)")

        except Exception as e:
            trace.update(error=str(e))
            print(f"❌ Error di Orchestrator: {e}")
            result = f"Terjadi kesalahan: {e}"
        finally:
            self.lf.flush()
            self.memory.save_interaction(user_input, result)

        return result

    def get_status(self) -> str:
        return (
            f"📊 Status Memory : {self.memory.status()}\n"
            f"📌 Session ID : {self.session_id}\n"
            f"🔢 Total Request : {self.request_count}"
        )

    def reset(self):
        self.memory.clear()
        self.session_id = str(uuid.uuid4())
        self.request_count = 0
        print(f"🔄 Session direset. Session baru: {self.session_id}")