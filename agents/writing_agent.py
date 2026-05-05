import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from memory.research_memory import ResearchMemory
from tools.writing_tools import writing_tools   # pastikan tool ini ada


class WritingAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Writing Agent"

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=writing_tools,
            prompt="""Kamu adalah Writing Agent yang profesional.
Tugasmu menulis laporan penelitian yang lengkap, terstruktur, dan berkualitas tinggi dalam bahasa Indonesia.
Struktur yang harus digunakan:
# Judul Laporan
## Pendahuluan
## Pembahasan Utama
## Temuan dan Analisis
## Kesimpulan dan Rekomendasi

Selalu gunakan tool save_report untuk menyimpan laporan ke file.""",
        )

    def run(self, topic: str, handler=None):
        print(f"\n{'─'*50}")
        print(f"✍️ {self.name} mulai bekerja...")
        print(f" Topik: {topic}")
        print(f"{'─'*50}")

        latest_summary = self.memory.get_latest_summary() or "Tidak ada ringkasan tersedia."

        messages = [HumanMessage(content=(
            f"Topik: {topic}\n\n"
            f"Ringkasan penelitian:\n{latest_summary}\n\n"
            f"Tulis laporan penelitian lengkap dan profesional dalam bahasa Indonesia. "
            f"Setelah selesai, simpan laporan menggunakan tool save_report."
        ))]

        config = {"callbacks": [handler]} if handler else {}

        t0 = time.time()
        result = self.agent.invoke({"messages": messages}, config=config)
        duration = round(time.time() - t0, 2)

        output = result["messages"][-1].content

        print(f"✅ {self.name} selesai dalam {duration} detik.")
        print(f"📄 Laporan telah disimpan ke folder output.")
        
        return output