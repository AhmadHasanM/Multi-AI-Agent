from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from memory.research_memory import ResearchMemory
from agents.search_agent import SearchAgent
from agents.summarize_agent import SummarizeAgent
from agents.writing_agent import WritingAgent


class Orchestrator:
    """
    Orchestrator — otak utama sistem multi-agent.

    Tugasnya:
    1. Menerima permintaan dari user
    2. Menganalisis apa yang perlu dilakukan
    3. Mendelegasikan ke sub-agent yang tepat (Search → Summarize → Write)
    4. Menggabungkan hasil dan mengembalikan ke user
    5. Mengelola memory lintas sesi

    Alur kerja:
    User → Orchestrator → Search Agent → Summarize Agent → Writing Agent → Orchestrator → User
    """

    def __init__(self):
        print("🚀 Menginisialisasi Research Assistant...")

        # Shared memory — digunakan oleh semua agent
        self.memory = ResearchMemory()

        # Inisialisasi semua sub-agent dengan memory yang sama
        self.search_agent    = SearchAgent(self.memory)
        self.summarize_agent = SummarizeAgent(self.memory)
        self.writing_agent   = WritingAgent(self.memory)

        # LLM untuk orchestrator sendiri (routing & koordinasi)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )

        # Chain untuk analisis intent user
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

        # Chain untuk respons percakapan biasa
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """Kamu adalah Research Assistant AI yang ramah dan membantu.
Kamu bisa melakukan penelitian mendalam tentang topik apapun.
Riwayat penelitian sebelumnya: {research_history}
"""),
            ("human", "{user_input}"),
        ])
        self.chat_chain = self.chat_prompt | self.llm | StrOutputParser()

        print("✅ Research Assistant siap digunakan!\n")

    def _detect_intent(self, user_input: str) -> str:
        """Deteksi intent user."""
        intent = self.intent_chain.invoke({
            "user_input": user_input,
            "chat_history": self.memory.get_history_as_text(),
        }).strip().lower()
        # Fallback jika tidak cocok
        valid = ["full_research", "search_only", "summarize", "write", "chat"]
        return intent if intent in valid else "chat"

    def _extract_topic(self, user_input: str) -> str:
        """Ekstrak topik utama dari input user."""
        extract_prompt = f"""Ekstrak topik utama penelitian dari kalimat berikut.
Berikan hanya topik singkat (3-7 kata), tanpa penjelasan tambahan.
Kalimat: {user_input}
Topik:"""
        result = self.llm.invoke(extract_prompt)
        return result.content.strip()

    def run(self, user_input: str) -> str:
        """Proses satu permintaan dari user."""

        print(f"\n{'═'*60}")
        print(f"🎯 Orchestrator menerima input:")
        print(f"   '{user_input}'")

        # 1. Deteksi intent
        intent = self._detect_intent(user_input)
        print(f"   Intent terdeteksi: [{intent}]")
        print(f"{'═'*60}")

        result = ""

        if intent == "full_research":
            # Alur lengkap: Search → Summarize → Write
            topic = self._extract_topic(user_input)
            self.memory.set_topic(topic)
            print(f"\n📌 Topik: {topic}")
            print("📋 Alur: Search Agent → Summarize Agent → Writing Agent\n")

            search_result    = self.search_agent.run(topic)
            summary_result   = self.summarize_agent.run(topic)
            writing_result   = self.writing_agent.run(topic)

            result = (
                f"✅ Penelitian selesai untuk topik: **{topic}**\n\n"
                f"📝 **Ringkasan:**\n{summary_result}\n\n"
                f"📄 **Laporan telah disimpan ke folder `output/`**"
            )

        elif intent == "search_only":
            topic = self._extract_topic(user_input)
            print(f"\n📌 Topik: {topic}")
            print("📋 Alur: Search Agent saja\n")
            result = self.search_agent.run(topic)

        elif intent == "summarize":
            topic = self.memory.session_topic or self._extract_topic(user_input)
            print(f"\n📌 Topik: {topic}")
            print("📋 Alur: Summarize Agent saja\n")
            result = self.summarize_agent.run(topic)

        elif intent == "write":
            topic = self.memory.session_topic or self._extract_topic(user_input)
            print(f"\n📌 Topik: {topic}")
            print("📋 Alur: Writing Agent saja\n")
            result = self.writing_agent.run(topic)

        else:  # chat
            print("\n📋 Mode: Percakapan biasa\n")
            result = self.chat_chain.invoke({
                "user_input": user_input,
                "research_history": self.memory.get_all_summaries(),
            })

        # Simpan interaksi ke memory
        self.memory.save_interaction(user_input, result)

        return result

    def get_status(self) -> str:
        return f"📊 Status Memory: {self.memory.status()}"

    def reset(self):
        self.memory.clear()
        print("🔄 Session direset.")