from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from memory.research_memory import ResearchMemory


class SummarizeAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Summarize Agent"

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Kamu adalah Summarize Agent yang ahli menganalisis dan meringkas informasi.

Format ringkasanmu selalu:
- **Gambaran Umum**: (1-2 kalimat)
- **Poin Kunci**: (bullet points)
- **Data & Fakta Penting**: (angka, statistik)
- **Insight Menarik**: (temuan unik)
- **Kesimpulan**: (1-2 kalimat penutup)

Gunakan bahasa Indonesia yang jelas dan profesional."""),
            ("human", """Topik: {topic}

Hasil pencarian:
{search_results}

Konteks sebelumnya:
{chat_history}

Buat ringkasan komprehensif dan terstruktur."""),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, topic: str) -> str:
        print(f"\n{'─'*50}")
        print(f"📝 {self.name} mulai bekerja...")
        print(f"   Topik: {topic}")
        print(f"{'─'*50}")

        result = self.chain.invoke({
            "topic": topic,
            "search_results": self.memory.get_all_search_results(),
            "chat_history": self.memory.get_history_as_text(),
        })

        self.memory.save_summary(topic, result)
        print(f"\n✅ {self.name} selesai.")
        return result