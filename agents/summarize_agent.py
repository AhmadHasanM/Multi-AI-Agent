import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from memory.research_memory import ResearchMemory


class SummarizeAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name   = "Summarize Agent"

        self.lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            debug=False,
        )

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

    def run(self, topic: str, parent_trace=None) -> str:
        print(f"\n{'─'*50}")
        print(f"📝 {self.name} mulai bekerja...")
        print(f"   Topik: {topic}")
        print(f"{'─'*50}")

        search_results = self.memory.get_all_search_results()
        chat_history   = self.memory.get_history_as_text()

        prompt_input = {
            "topic"         : topic,
            "search_results": search_results,
            "chat_history"  : chat_history,
        }

        t0     = time.time()
        result = self.chain.invoke(prompt_input)
        duration = round(time.time() - t0, 2)

        # ── Log ke Langfuse ────────────────────────────────────────
        if parent_trace:
            parent_trace.generation(
                name="summarize-agent:llm-call",
                model=GEMINI_MODEL,
                input=[
                    {"role": "system", "content": "Kamu adalah Summarize Agent..."},
                    {"role": "user",   "content": f"Topik: {topic}\nSearch results: {search_results[:300]}..."},
                ],
                output=result[:500],
                metadata={
                    "agent"               : self.name,
                    "duration_s"          : duration,
                    "pattern"             : "LCEL chain (prompt | llm | parser)",
                    "input_search_length" : len(search_results),
                    "output_length"       : len(result),
                    "tools_used"          : [],
                },
            )

        self.memory.save_summary(topic, result)
        print(f"\n✅ {self.name} selesai.")
        return result