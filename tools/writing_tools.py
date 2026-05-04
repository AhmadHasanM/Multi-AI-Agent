from langchain_core.tools import tool
import os
from datetime import datetime


@tool
def save_report(content: str) -> str:
    """
    Menyimpan laporan/artikel ke file .md di folder output.
    Input: konten lengkap laporan dalam format markdown.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/report_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [Tool: save_report] Laporan disimpan ke: {filename}")
    return f"Laporan berhasil disimpan ke '{filename}'"


@tool
def format_markdown(text: str) -> str:
    """
    Memformat teks menjadi markdown yang rapi dengan heading, bullet points, dan struktur yang baik.
    Input: teks mentah yang ingin diformat.
    """
    print(f"  [Tool: format_markdown] Memformat teks ({len(text)} karakter)...")
    # Tool ini akan dipanggil LLM untuk memformat — hasilnya dikembalikan langsung
    return f"Teks siap diformat (panjang: {len(text)} karakter). Gunakan kemampuanmu untuk menghasilkan markdown yang rapi."


# Kumpulan tools untuk writing agent
writing_tools = [save_report, format_markdown]