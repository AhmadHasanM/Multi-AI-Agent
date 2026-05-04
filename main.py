from orchestrator import Orchestrator


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║         🔬 Research Assistant Multi-Agent AI             ║
║         LangChain + Gemini + Multi-Agent System          ║
╠══════════════════════════════════════════════════════════╣
║  Kemampuan:                                              ║
║  🔍 Search Agent    — mencari informasi dari internet    ║
║  📝 Summarize Agent — meringkas & menganalisis           ║
║  ✍️  Writing Agent   — menulis laporan profesional        ║
║  🎯 Orchestrator    — koordinasi semua agent             ║
╠══════════════════════════════════════════════════════════╣
║  Contoh perintah:                                        ║
║  > Buatkan penelitian tentang AI terbaru 2024            ║
║  > Cari informasi tentang quantum computing              ║
║  > Ringkaskan hasil pencarian tadi                       ║
║  > Tulis laporan dari informasi yang sudah ada           ║
║  > status  → lihat status memory                         ║
║  > reset   → reset session                               ║
║  > keluar  → keluar dari program                         ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()

    try:
        orchestrator = Orchestrator()
    except Exception as e:
        print(f"❌ Gagal menginisialisasi: {e}")
        print("   Pastikan GEMINI_API_KEY di file .env sudah benar!")
        return

    print("💬 Silakan masukkan permintaan penelitianmu:\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Sampai jumpa!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["keluar", "exit", "quit", "q"]:
            print("👋 Sampai jumpa!")
            break

        if user_input.lower() == "status":
            print(orchestrator.get_status())
            continue

        if user_input.lower() == "reset":
            orchestrator.reset()
            continue

        try:
            response = orchestrator.run(user_input)
            print(f"\n{'─'*60}")
            print(f"🤖 Assistant:\n{response}")
            print(f"{'─'*60}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Coba lagi atau ketik 'reset' untuk mulai ulang.\n")


if __name__ == "__main__":
    main()