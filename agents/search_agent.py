import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langfuse import Langfuse
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from tools.search_tools import search_tools, web_search, multi_search
from memory.research_memory import ResearchMemory
from duckduckgo_search import DDGS


class SearchAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name   = "Search Agent"

        self.lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            debug=False,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=search_tools,
            prompt=(
                "Kamu adalah Search Agent. Tugasmu mencari informasi komprehensif dari internet. "
                "Selalu lakukan minimal 2 pencarian dari sudut pandang berbeda untuk setiap topik."
            ),
        )

    def run(self, topic: str, parent_trace=None) -> str:
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

        t0     = time.time()
        result = self.agent.invoke({"messages": messages})
        duration = round(time.time() - t0, 2)

        all_messages = result["messages"]
        output = all_messages[-1].content

        # ── Log ke Langfuse jika ada parent trace ──────────────────
        if parent_trace:
            # 1. Log setiap tool call yang terjadi
            for msg in all_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_span = parent_trace.span(
                            name=f"tool-call:{tc['name']}",
                            input=tc['args'],
                            metadata={
                                "tool_name" : tc['name'],
                                "agent"     : self.name,
                                "tool_source": "DuckDuckGo",
                            },
                        )
                        # Cari hasil tool dari ToolMessage
                        for tm in all_messages:
                            if isinstance(tm, ToolMessage) and tm.tool_call_id == tc['id']:
                                tool_span.end(
                                    output=tm.content[:300],
                                    metadata={"result_length": len(tm.content)},
                                )
                                break
                        else:
                            tool_span.end(output="result not found")

            # 2. Log model generation
            parent_trace.generation(
                name="search-agent:llm-call",
                model=GEMINI_MODEL,
                input=[{"role": "user", "content": f"Research topic: {topic}"}],
                output=output[:500],
                metadata={
                    "agent"          : self.name,
                    "duration_s"     : duration,
                    "total_messages" : len(all_messages),
                    "tools_available": ["web_search", "multi_search"],
                },
            )

        self.memory.save_search_result(topic, output)
        print(f"\n✅ {self.name} selesai.")
        return output