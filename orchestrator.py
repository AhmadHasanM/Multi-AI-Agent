from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from memory.research_memory import ResearchMemory
from agents.search_agent import SearchAgent
from agents.summarize_agent import SummarizeAgent
from agents.writing_agent import WritingAgent


class Orchestrator:
    def __init__(self):
        print("🚀 Menginisialisasi Research Assistant...")

        self.memory = ResearchMemory()
        self.search_agent    = SearchAgent(self.memory)
        self.summarize_agent = SummarizeAgent(self.memory)
        self.writing_agent   = WritingAgent(self.memory)

        self.lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )

        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Kamu adalah orchestrator yang menganalisis permintaan user.
Tugasmu: tentukan apakah user ingin:
1. "full_research" - penelitian lengkap (search + summarize + write)
2. "search_only"   - hanya cari informasi
3. "summarize"     - ringkas informasi yang sudah ada
4. "write"         - tulis laporan dari info yang sudah ada
5. "chat"          - percakapan biasa / tanya jawab
Balas HANYA dengan salah satu kata kunci di atas, tanpa penjelasan.
"""),
            ("human", "Permintaan user: {user_input}\n\nRiwayat: {chat_history}"),
        ])
        self.intent_chain = self.intent_prompt | self.llm | StrOutputParser()

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "Kamu adalah Research Assistant AI yang ramah.\nRiwayat: {research_history}"),
            ("human", "{user_input}"),
        ])
        self.chat_chain = self.chat_prompt | self.llm | StrOutputParser()

        # Test koneksi Langfuse
        try:
            self.lf.auth_check()
            print("  [Langfuse] ✅ Koneksi berhasil!")
        except Exception as e:
            print(f"  [Langfuse] ⚠️  Koneksi gagal: {e}")

        print("✅ Research Assistant siap digunakan!\n")

    def _detect_intent(self, user_input: str) -> str:
        intent = self.intent_chain.invoke({
            "user_input": user_input,
            "chat_history": self.memory.get_history_as_text(),
        }).strip().lower()
        valid = ["full_research", "search_only", "summarize", "write", "chat"]
        return intent if intent in valid else "chat"

    def _extract_topic(self, user_input: str) -> str:
        result = self.llm.invoke(
            f"Ekstrak topik utama dari kalimat berikut, jawab 3-7 kata saja.\nKalimat: {user_input}\nTopik:"
        )
        return result.content.strip()

    def run(self, user_input: str) -> str:
        # Buat trace
        trace = self.lf.trace(
            name="research-assistant",
            input=user_input,
            tags=["multi-agent"],
        )
        print(f"  [Langfuse] Trace dibuat: {trace.id}")

        print(f"\n{'═'*60}")
        print(f"🎯 Orchestrator menerima input: '{user_input}'")

        intent = self._detect_intent(user_input)
        print(f"   Intent terdeteksi: [{intent}]")
        print(f"{'═'*60}")

        result = ""

        if intent == "full_research":
            topic = self._extract_topic(user_input)
            self.memory.set_topic(topic)
            print(f"\n📌 Topik: {topic}")
            print("📋 Alur: Search Agent → Summarize Agent → Writing Agent\n")

            span = trace.span(name="search-agent", input=topic)
            search_result = self.search_agent.run(topic)
            span.end(output=search_result[:300])

            span = trace.span(name="summarize-agent", input=topic)
            summary_result = self.summarize_agent.run(topic)
            span.end(output=summary_result[:300])

            span = trace.span(name="writing-agent", input=topic)
            writing_result = self.writing_agent.run(topic)
            span.end(output=writing_result[:300])

            result = (
                f"✅ Penelitian selesai untuk topik: **{topic}**\n\n"
                f"📝 **Ringkasan:**\n{summary_result}\n\n"
                f"📄 **Laporan disimpan ke folder `output/`**"
            )

        elif intent == "search_only":
            topic = self._extract_topic(user_input)
            print(f"\n📌 Topik: {topic}\n")
            span = trace.span(name="search-agent", input=topic)
            result = self.search_agent.run(topic)
            span.end(output=result[:300])

        elif intent == "summarize":
            topic = self.memory.session_topic or self._extract_topic(user_input)
            print(f"\n📌 Topik: {topic}\n")
            span = trace.span(name="summarize-agent", input=topic)
            result = self.summarize_agent.run(topic)
            span.end(output=result[:300])

        elif intent == "write":
            topic = self.memory.session_topic or self._extract_topic(user_input)
            print(f"\n📌 Topik: {topic}\n")
            span = trace.span(name="writing-agent", input=topic)
            result = self.writing_agent.run(topic)
            span.end(output=result[:300])

        else:
            print("\n📋 Mode: Chat\n")
            result = self.chat_chain.invoke({
                "user_input": user_input,
                "research_history": self.memory.get_all_summaries(),
            })

        trace.update(output=result[:300])
        self.lf.flush()
        print("  [Langfuse] ✅ Trace terkirim!")

        self.memory.save_interaction(user_input, result)
        return result

    def get_status(self) -> str:
        return f"📊 Status Memory: {self.memory.status()}"

    def reset(self):
        self.memory.clear()
        print("🔄 Session direset.")