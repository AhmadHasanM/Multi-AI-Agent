import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from memory.research_memory import ResearchMemory


class SummarizeAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Summarize Agent"

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",   # atau GEMINI_MODEL dari settings
            temperature=0.2,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Kamu adalah Summarize Agent yang ahli menganalisis dan meringkas informasi.
Format ringkasanmu selalu:
- **Gambaran Umum**: (1-2 kalimat)
- **Poin Kunci**: (bullet points)
- **Data & Fakta Penting**: (jika ada)
- **Insight Menarik**: (temuan unik)
- **Kesimpulan**: (1-2 kalimat)

Gunakan bahasa Indonesia yang jelas dan profesional."""),
            ("human", """Topik: {topic}
Hasil pencarian:
{search_results}

Buat ringkasan komprehensif dan terstruktur."""),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, topic: str, handler=None):
        print(f"\n{'─'*50}")
        print(f"📝 {self.name} mulai bekerja...")
        print(f" Topik: {topic}")
        print(f"{'─'*50}")

        search_results = self.memory.get_all_search_results() or "Tidak ada hasil pencarian."
        chat_history = self.memory.get_history_as_text()[:1000]

        config = {"callbacks": [handler]} if handler else {}

        t0 = time.time()
        result = self.chain.invoke(
            {
                "topic": topic,
                "search_results": search_results
            },
            config=config
        )
        duration = round(time.time() - t0, 2)

        self.memory.save_summary(topic, result)

        print(f"✅ {self.name} selesai dalam {duration} detik.")
        return result