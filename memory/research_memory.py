from langchain_core.messages import HumanMessage, AIMessage


class ResearchMemory:
    """
    Mengelola memory untuk seluruh sesi research.
    Menyimpan:
    - Riwayat percakapan (conversation history)
    - Hasil search yang sudah ditemukan
    - Ringkasan yang sudah dibuat
    """

    def __init__(self):
        # Manual conversation history (tanpa ConversationBufferWindowMemory)
        self.chat_history: list = []
        self.max_messages = 10

        # Memory khusus untuk menyimpan hasil antar agent
        self.search_results: list[dict] = []
        self.summaries: list[dict] = []
        self.session_topic: str = ""

    # ── Conversation memory ──────────────────────────────────────────────────
    def save_interaction(self, human_input: str, ai_output: str):
        """Simpan satu pasang interaksi user-AI."""
        self.chat_history.append(HumanMessage(content=human_input))
        self.chat_history.append(AIMessage(content=ai_output))
        # Batasi ukuran history
        if len(self.chat_history) > self.max_messages * 2:
            self.chat_history = self.chat_history[-(self.max_messages * 2):]

    def get_history(self) -> list:
        """Ambil riwayat percakapan."""
        return self.chat_history

    def get_history_as_text(self) -> str:
        """Riwayat percakapan dalam format teks."""
        if not self.chat_history:
            return "Belum ada riwayat percakapan."
        lines = []
        for msg in self.chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content[:200]}")
        return "\n".join(lines)

    # ── Search result memory ─────────────────────────────────────────────────
    def save_search_result(self, query: str, results: str):
        """Simpan hasil pencarian."""
        self.search_results.append({
            "query": query,
            "results": results,
        })
        print(f"  [Memory] Hasil search disimpan: '{query}'")

    def get_all_search_results(self) -> str:
        """Gabungkan semua hasil search yang sudah disimpan."""
        if not self.search_results:
            return "Belum ada hasil pencarian."
        parts = []
        for i, item in enumerate(self.search_results, 1):
            parts.append(f"=== Search {i}: {item['query']} ===\n{item['results']}")
        return "\n\n".join(parts)

    # ── Summary memory ───────────────────────────────────────────────────────
    def save_summary(self, topic: str, summary: str):
        """Simpan ringkasan yang sudah dibuat."""
        self.summaries.append({
            "topic": topic,
            "summary": summary,
        })
        print(f"  [Memory] Ringkasan disimpan: '{topic}'")

    def get_latest_summary(self) -> str:
        """Ambil ringkasan terbaru."""
        if not self.summaries:
            return "Belum ada ringkasan."
        return self.summaries[-1]["summary"]

    def get_all_summaries(self) -> str:
        """Gabungkan semua ringkasan."""
        if not self.summaries:
            return "Belum ada ringkasan."
        parts = []
        for i, item in enumerate(self.summaries, 1):
            parts.append(f"=== Ringkasan {i}: {item['topic']} ===\n{item['summary']}")
        return "\n\n".join(parts)

    # ── Session info ─────────────────────────────────────────────────────────
    def set_topic(self, topic: str):
        self.session_topic = topic

    def clear(self):
        """Reset seluruh memory."""
        self.chat_history.clear()
        self.search_results.clear()
        self.summaries.clear()
        self.session_topic = ""
        print("  [Memory] Memory direset.")

    def status(self) -> str:
        return (
            f"Topik: {self.session_topic or '-'} | "
            f"Search: {len(self.search_results)} | "
            f"Ringkasan: {len(self.summaries)} | "
            f"Chat history: {len(self.chat_history)} pesan"
        )