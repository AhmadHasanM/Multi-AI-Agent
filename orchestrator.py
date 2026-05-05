import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from config.langfuse_config import get_langfuse, get_callback_handler
from memory.research_memory import ResearchMemory
from agents.search_agent import SearchAgent
from agents.summarize_agent import SummarizeAgent
from agents.writing_agent import WritingAgent


class Orchestrator:
    def __init__(self):
        print("🚀 Menginisialisasi Research Assistant...")

        try:
            self.session_id = str(uuid.uuid4())
            self.user_id = "user-hasan"
            self.request_count = 0

            self.memory = ResearchMemory()
            self.search_agent = SearchAgent(self.memory)
            self.summarize_agent = SummarizeAgent(self.memory)
            self.writing_agent = WritingAgent(self.memory)

            self.lf = get_langfuse()

            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
            )

            self.intent_prompt = ChatPromptTemplate.from_messages([
                ("system", "Kamu adalah orchestrator. Balas HANYA dengan salah satu: full_research, search_only, summarize, write, atau chat."),
                ("human", "Permintaan user: {user_input}"),
            ])
            self.intent_chain = self.intent_prompt | self.llm | StrOutputParser()

            print("✅ Research Assistant siap digunakan!\n")

        except Exception as e:
            print(f"❌ Gagal menginisialisasi: {e}")
            raise

    def run(self, user_input: str) -> str:
        self.request_count += 1

        handler = get_callback_handler()

        print(f"\n{'═'*70}")
        print(f"🎯 Orchestrator menerima input: '{user_input}'")
        print(f"{'═'*70}")

        try:
            intent_raw = self.intent_chain.invoke(
                {"user_input": user_input},
                config={"callbacks": [handler]}
            )
            intent = intent_raw.strip().lower()
            print(f" Intent terdeteksi: [{intent.upper()}]")

            result = ""

            if intent == "full_research":
                topic = "Manfaat Kopi bagi Kesehatan"  # sementara
                self.memory.set_topic(topic)
                print(f"📌 Topik: {topic}")

                self.search_agent.run(topic, handler)
                self.summarize_agent.run(topic, handler)
                self.writing_agent.run(topic, handler)

                result = f"✅ Penelitian selesai untuk topik: **{topic}**"

            print(f"\n [Langfuse] ✅ Tracing aktif!")
            return result

        except Exception as e:
            print(f"❌ Error di Orchestrator: {e}")
            return f"Terjadi kesalahan: {e}"