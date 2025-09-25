import os
import fitz  # PyMuPDF
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 設定 ---
PAPERS_DIR = "papers"  # 論文PDFが格納されているフォルダ
METADATA_FILE = "metadata.json"
FAISS_INDEX_FILE = "paper_vectors.index"
MODEL_NAME = 'intfloat/multilingual-e5-large'

# チャンク分割の設定
CHUNK_SIZE = 500  # 1チャンクあたりの最大文字数
CHUNK_OVERLAP = 50  # チャンク間のオーバーラップ文字数

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 【追加】タイトルの重み付け設定
TITLE_BOOST_FACTOR = 3  # タイトルを何回繰り返して重みを付けるか
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


def get_pdf_files(directory: str) -> list[str]:
    """指定されたディレクトリからPDFファイルのリストを取得する"""
    if not os.path.isdir(directory):
        print(f"エラー: ディレクトリ '{directory}' が見つかりません。")
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]


def extract_title_and_text_from_pdf(pdf_path: str) -> (str, str):
    """PDFファイルからタイトルと全文テキストを抽出する"""
    try:
        with fitz.open(pdf_path) as doc:
            # タイトル抽出ロジック: 1ページ目のテキストを行で分割し、最初の空でない行をタイトルとする
            first_page_text = doc[0].get_text("text")
            lines = first_page_text.strip().split('\n')
            title = next((line.strip() for line in lines if line.strip()), "No title found")
            
            # 全文テキストの抽出
            full_text = "".join(page.get_text() for page in doc)
            
        return title, full_text
    except Exception as e:
        print(f"警告: '{pdf_path}' の読み込み中にエラーが発生しました: {e}")
        return "Error reading file", ""


def split_text_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    """テキストを指定されたサイズとオーバーラップでチャンクに分割する"""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        # 次の開始位置を計算（オーバーラップを考慮）
        start += size - overlap
        if start >= len(text):
            break
            
    return chunks


def main():
    """メインの前処理実行関数"""
    print("--- 論文データの前処理を開始します ---")

    pdf_files = get_pdf_files(PAPERS_DIR)
    if not pdf_files:
        print(f"'{PAPERS_DIR}' フォルダに処理対象のPDFファイルがありません。処理を終了します。")
        return

    print(f"{len(pdf_files)} 件のPDFファイルを検出しました。")

    all_metadata = []
    all_chunks_text = []

    # main()関数内
    print("\n--- ステップ1: PDFからのテキスト抽出とチャンク分割 ---")
    for i, pdf_path in enumerate(tqdm(pdf_files, desc="PDF処理中")):
        filename = os.path.basename(pdf_path)
        
        # ▼▼▼▼▼▼▼▼▼ ここから変更 ▼▼▼▼▼▼▼▼▼
        # paper_title = os.path.splitext(filename)[0] # 古いロジックを削除
        # full_text = extract_text_from_pdf(pdf_path) # 古い関数呼び出しを削除

        # 新しい関数でタイトルと本文を抽出
        paper_title, full_text = extract_title_and_text_from_pdf(pdf_path)
        
        if not full_text:
            continue
        # ▲▲▲▲▲▲▲▲▲ ここまで変更 ▲▲▲▲▲▲▲▲▲

        # タイトルを繰り返して、重要度を上げたテキストを作成
        boosted_title_chunk = ". ".join([paper_title] * TITLE_BOOST_FACTOR)

        # 本文から通常のチャンクを作成
        body_chunks = split_text_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 最終的なチャンクリストは、重み付けしたタイトル + 本文チャンク
        all_paper_chunks = [boosted_title_chunk] + body_chunks
        
        paper_data = {
            "id": f"paper_{i+1}",
            "filename": filename,
            "title": paper_title, # ✨抽出した本当のタイトルを設定
            "fulltext": full_text,
            "chunks": all_paper_chunks
        }
        all_metadata.append(paper_data)
        all_chunks_text.extend(all_paper_chunks)
        
        paper_data = {
            "id": f"paper_{i+1}",
            "filename": filename,
            "title": paper_title, 
            "fulltext": full_text,
            "chunks": all_paper_chunks # 変更後のチャンクリストを格納
        }
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        all_metadata.append(paper_data)
        all_chunks_text.extend(all_paper_chunks) # 変更後のチャンクリストを全体に追加

    if not all_chunks_text:
        print("エラー: テキストチャンクが一つも生成されませんでした。PDFの内容を確認してください。")
        return

    print(f"合計 {len(all_chunks_text)} 個のチャンクが生成されました。")

    print("\n--- ステップ2: Sentence Transformerによるベクトル化 ---")
    model = SentenceTransformer(MODEL_NAME)
    
    chunk_vectors = model.encode(
        all_chunks_text, 
        show_progress_bar=True,
        batch_size=32
    )

    chunk_vectors = np.array(chunk_vectors).astype('float32')
    vector_dim = chunk_vectors.shape[1]
    print(f"ベクトルの次元数: {vector_dim}")

    print("\n--- ステップ3: FAISSインデックスの構築 ---")
    index = faiss.IndexFlatL2(vector_dim)
    index.add(chunk_vectors)
    print(f"FAISSインデックスに {index.ntotal} 個のベクトルを追加しました。")

    # 【追加】保存前の最終チェック
    print("\n--- ステップ4: 内部整合性の検証 ---")
    num_chunks = len(all_chunks_text)
    num_vectors = index.ntotal
    if num_chunks != num_vectors:
        print("!!! 致命的エラー: 生成されたチャンク数とベクトル数が一致しません。!!!")
        print(f"チャンク数: {num_chunks}, ベクトル数: {num_vectors}")
        print("不整合なファイルは保存されません。処理を中断します。")
        print("ヒント: PDFの処理中やベクトル化中にメモリ不足などのエラーが発生した可能性があります。")
        return  # ★不整合なファイルが作られるのを防ぐ
    
    print("内部整合性の検証OK。")

    print("\n--- ステップ5: ファイルへの保存 ---")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)
    print(f"メタデータを '{METADATA_FILE}' に保存しました。")
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISSインデックスを '{FAISS_INDEX_FILE}' に保存しました。")

    print("\n--- すべての前処理が完了しました ---")


if __name__ == "__main__":
    main()

