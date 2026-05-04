from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, AGENT_VERBOSE
from tools.search_tools import search_tools
from memory.research_memory import ResearchMemory


class SearchAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Search Agent"

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=search_tools,
            prompt="Kamu adalah Search Agent. Tugasmu mencari informasi komprehensif dari internet. Selalu lakukan minimal 2 pencarian dari sudut pandang berbeda untuk setiap topik.",
        )

    def run(self, topic: str) -> str:
        print(f"\n{'─'*50}")
        print(f"🔍 {self.name} mulai bekerja...")
        print(f"   Topik: {topic}")
        print(f"{'─'*50}")

        messages = self.memory.get_history() + [
            HumanMessage(content=(
                f"Cari informasi komprehensif tentang: {topic}. "
                f"Lakukan minimal 2 pencarian dari sudut pandang berbeda. "
                f"Berikan hasil yang lengkap dan terstruktur."
            ))
        ]

        result = self.agent.invoke({"messages": messages})
        output = result["messages"][-1].content

        self.memory.save_search_result(topic, output)
        print(f"\n✅ {self.name} selesai.")
        return output