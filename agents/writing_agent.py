import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from tools.writing_tools import writing_tools
from memory.research_memory import ResearchMemory


class WritingAgent:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.name = "Writing Agent"
        self.lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            debug=False,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
        )
        self.agent = create_react_agent(
            model=self.llm,
            tools=writing_tools,
            prompt="""Kamu adalah Writing Agent yang ahli menulis laporan penelitian profesional.
Tulis laporan dengan struktur yang jelas dan akademis dalam bahasa Indonesia.
Selalu gunakan tool `save_report` untuk menyimpan laporan ke file.""",
        )

    def run(self, topic: str, parent_trace=None) -> str:
        print(f"\n{'─'*50}")
        print(f"✍️ {self.name} mulai bekerja...")
        print(f" Topik: {topic}")
        print(f"{'─'*50}")

        messages = [HumanMessage(content=(
            f"Topik: {topic}\n\n"
            f"Ringkasan penelitian:\n{self.memory.get_latest_summary() or 'Tidak ada ringkasan'}\n\n"
            f"Tulis laporan penelitian lengkap, profesional, dan terstruktur. "
            f"Setelah selesai, simpan laporan menggunakan tool save_report."
        ))]

        t0 = time.time()

        # === TRACING ===
        if parent_trace:
            agent_span = parent_trace.span(
                name="writing-agent",
                input={"topic": topic, "has_summary": bool(self.memory.get_latest_summary())},
                metadata={
                    "agent": self.name,
                    "tools": ["save_report"],
                    "temperature": 0.7
                }
            )
        else:
            agent_span = None

        result = self.agent.invoke({"messages": messages})
        duration = round(time.time() - t0, 2)

        output = result["messages"][-1].content

        # Log LLM Generation + Tool Calls
        if agent_span:
            # Log tool calls (mirip SearchAgent)
            all_messages = result["messages"]
            for msg in all_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_span = agent_span.span(
                            name=f"tool-call:{tc['name']}",
                            input=tc['args'],
                            metadata={"tool_name": tc['name']}
                        )
                        # Cari ToolMessage
                        for tm in all_messages:
                            if hasattr(tm, 'tool_call_id') and tm.tool_call_id == tc['id']:
                                tool_span.end(
                                    output=str(tm.content)[:500],
                                    metadata={"status": "success"}
                                )
                                break
                        else:
                            tool_span.end(output="No result")

            # Log final generation
            agent_span.generation(
                name="writing-agent:llm-call",
                model=GEMINI_MODEL,
                input={"task": "write_full_report", "topic": topic},
                output=output[:600],
                metadata={
                    "agent": self.name,
                    "duration_s": duration,
                    "output_length": len(output)
                }
            )
            agent_span.end()

        print(f"\n✅ {self.name} selesai. Laporan telah disimpan.")
        return output