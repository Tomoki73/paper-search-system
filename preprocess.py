import os
import fitz  # PyMuPDF
import json
import faiss
import numpy as np
import re  # 正規表現ライブラリをインポート
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 設定 (変更なし) ---
PAPERS_DIR = "papers"
METADATA_FILE = "metadata.json"
FAISS_INDEX_FILE = "paper_vectors.index"
MODEL_NAME = 'intfloat/multilingual-e5-large'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TITLE_BOOST_FACTOR = 3

def get_pdf_files(directory: str) -> list[str]:
    # (この関数は変更なし)
    if not os.path.isdir(directory): return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]

def extract_title_and_text_from_pdf(pdf_path: str) -> (str, str):
    # (この関数は変更なし)
    try:
        with fitz.open(pdf_path) as doc:
            first_page_text = doc[0].get_text("text")
            lines = first_page_text.strip().split('\n')
            title = next((line.strip() for line in lines if line.strip()), "No title found")
            full_text = "".join(page.get_text() for page in doc)
        return title, full_text
    except Exception as e:
        print(f"警告: '{pdf_path}' の読み込み中にエラー: {e}")
        return "Error reading file", ""

# ▼▼▼▼▼ この関数を大幅に改善 ▼▼▼▼▼
def split_text_into_chunks_with_sections(text: str, size: int, overlap: int) -> list[dict]:
    """テキストを章ごとに分割し、各チャンクに章の情報を付与する"""
    if not text:
        return []

    # 章のタイトルにマッチする正規表現パターン
    # (例: "1 はじめに", "1.2 関連研究", "謝辞")
    section_pattern = re.compile(r'^(?P<id>\d+(?:\.\d+)*\s+|\d+\s+|[緒結考謝参].*|概要|はじめに|関連研究|提案手法|実験|評価|考察|結論|謝辞|参考文献)\s*(?P<title>.*)', re.MULTILINE)
    
    chunks_with_sections = []
    last_match_end = 0
    current_section = "概要" # デフォルトのセクション名

    for match in section_pattern.finditer(text):
        # 前のセクションの本文を処理
        section_body = text[last_match_end:match.start()].strip()
        if section_body:
            start = 0
            while start < len(section_body):
                end = start + size
                chunks_with_sections.append({
                    "text": section_body[start:end],
                    "section": current_section
                })
                start += size - overlap
        
        # 現在のセクション名を更新
        current_section = match.group(0).strip().replace('\n', ' ')
        last_match_end = match.end()

    # 最後のセクションの本文を処理
    last_section_body = text[last_match_end:].strip()
    if last_section_body:
        start = 0
        while start < len(last_section_body):
            end = start + size
            chunks_with_sections.append({
                "text": last_section_body[start:end],
                "section": current_section
            })
            start += size - overlap
            
    return chunks_with_sections

def main():
    print("--- 論文データの前処理を開始します (高精度版) ---")
    pdf_files = get_pdf_files(PAPERS_DIR)
    if not pdf_files: return

    all_metadata = []
    all_chunks_text = []

    for i, pdf_path in enumerate(tqdm(pdf_files, desc="PDF処理中")):
        filename = os.path.basename(pdf_path)
        paper_title, full_text = extract_title_and_text_from_pdf(pdf_path)
        if not full_text: continue

        # 新しいチャンク分割関数を呼び出す
        body_chunks_with_sections = split_text_into_chunks_with_sections(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # タイトルチャンクにもセクション情報を付与
        boosted_title_chunk = {
            "text": ". ".join([paper_title] * TITLE_BOOST_FACTOR),
            "section": "タイトル"
        }
        
        all_paper_chunks_data = [boosted_title_chunk] + body_chunks_with_sections
        
        paper_data = {
            "id": f"paper_{i+1}",
            "filename": filename,
            "title": paper_title, 
            "chunks": all_paper_chunks_data # 辞書のリストとして保存
        }
        all_metadata.append(paper_data)
        # FAISSインデックス用にはテキストだけを抽出
        all_chunks_text.extend([chunk['text'] for chunk in all_paper_chunks_data])

    print(f"合計 {len(all_chunks_text)} 個のチャンクが生成されました。")
    
    # --- ステップ2以降 (ベクトル化、FAISS構築、保存) は変更なし ---
    print("\n--- ステップ2: Sentence Transformerによるベクトル化 ---")
    model = SentenceTransformer(MODEL_NAME)
    chunk_vectors = model.encode(all_chunks_text, show_progress_bar=True, batch_size=32)
    chunk_vectors = np.array(chunk_vectors).astype('float32')

    print("\n--- ステップ3: FAISSインデックスの構築 ---")
    index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    index.add(chunk_vectors)

    print("\n--- ステップ4: ファイルへの保存 ---")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("\n--- すべての前処理が完了しました ---")

if __name__ == "__main__":
    main()