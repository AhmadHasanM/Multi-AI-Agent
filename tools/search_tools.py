from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Instance DuckDuckGo search
_search_engine = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Mencari informasi terbaru dari internet menggunakan DuckDuckGo.
    Gunakan tool ini untuk mencari fakta, berita, atau informasi umum.
    Input: query pencarian dalam bahasa Inggris atau Indonesia.
    """
    print(f"  [Tool: web_search] Query: {query}")
    try:
        result = _search_engine.run(query)
        return result if result else "Tidak ditemukan hasil untuk query ini."
    except Exception as e:
        return f"Error saat mencari: {str(e)}"


@tool
def multi_search(queries: str) -> str:
    """
    Melakukan beberapa pencarian sekaligus untuk topik yang sama dari sudut pandang berbeda.
    Input: beberapa query yang dipisahkan dengan tanda '|'
    Contoh: "AI terbaru 2024|perkembangan machine learning|deep learning trends"
    """
    print(f"  [Tool: multi_search] Queries: {queries}")
    query_list = [q.strip() for q in queries.split("|") if q.strip()]
    all_results = []
    for i, query in enumerate(query_list, 1):
        print(f"    Mencari ({i}/{len(query_list)}): {query}")
        try:
            result = _search_engine.run(query)
            all_results.append(f"--- Hasil untuk '{query}' ---\n{result}")
        except Exception as e:
            all_results.append(f"--- Hasil untuk '{query}' ---\nError: {str(e)}")
    return "\n\n".join(all_results)


# Kumpulan tools untuk search agent
search_tools = [web_search, multi_search]