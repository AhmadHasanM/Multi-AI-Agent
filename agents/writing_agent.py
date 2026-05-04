from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from tools.writing_tools import writing_tools
from memory.research_memory import ResearchMemory


class WritingAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Writing Agent"

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=writing_tools,
            prompt="""Kamu adalah Writing Agent yang ahli menulis laporan penelitian profesional.
Tulis laporan dengan struktur:
# [Judul Menarik]
## Pendahuluan
## Pembahasan Utama
## Temuan & Analisis
## Kesimpulan
---
*Laporan dibuat oleh Research Assistant AI*

Selalu simpan laporan ke file menggunakan tool save_report. Tulis dalam bahasa Indonesia.""",
        )

    def run(self, topic: str) -> str:
        print(f"\n{'─'*50}")
        print(f"✍️  {self.name} mulai bekerja...")
        print(f"   Topik: {topic}")
        print(f"{'─'*50}")

        messages = [HumanMessage(content=(
            f"Topik: {topic}\n\n"
            f"Ringkasan penelitian:\n{self.memory.get_latest_summary()}\n\n"
            f"Tulis laporan penelitian lengkap dan profesional, lalu simpan ke file."
        ))]

        result = self.agent.invoke({"messages": messages})
        output = result["messages"][-1].content

        print(f"\n✅ {self.name} selesai.")
        return output