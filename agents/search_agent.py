from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from memory.research_memory import ResearchMemory
from tools.search_tools import search_tools  # pastikan import ini benar

class SearchAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Search Agent"
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

        self.agent = create_react_agent(
            model=self.llm,
            tools=search_tools,
            prompt="Kamu adalah Search Agent ahli. Lakukan pencarian mendalam dan berikan hasil yang lengkap."
        )

    def run(self, topic: str, handler=None):
        print(f"🔍 {self.name} mulai: {topic}")
        config = {"callbacks": [handler]} if handler else {}

        messages = [HumanMessage(content=f"Cari informasi lengkap tentang: {topic}")]
        result = self.agent.invoke({"messages": messages}, config=config)

        output = result["messages"][-1].content
        self.memory.save_search_result(topic, output)
        print("✅ Search Agent selesai")
        return output