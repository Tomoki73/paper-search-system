import faiss
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, Field
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer

# --- グローバル変数 ---
model: SentenceTransformer = None
index: faiss.Index = None
metadata: list = []
# チャンクから論文情報を逆引きするためのマップ
chunk_to_paper_map: list = [] 

app = FastAPI(
    title="論文検索API",
    description="Sentence TransformerとFAISSを利用して、学術論文のセマンティック検索を行うAPIです。",
    version="1.1.0"
)

# --- サーバー起動時の処理 ---
@app.on_event("startup")
def load_models():
    """サーバー起動時に、モデル、インデックス、メタデータを読み込み、逆引きマップを構築する"""
    global model, index, metadata, chunk_to_paper_map
    
    print("モデルとインデックスの読み込みを開始します...")
    
    try:
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        index = faiss.read_index("paper_vectors.index")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        print(f"!!! 致命的エラー: 必要なファイルが見つかりません: {e.filename} !!!")
        raise RuntimeError(f"{e.filename} が見つからないため、サーバーを起動できません。")
    except Exception as e:
        print(f"!!! 致命的エラー: 読み込み中にエラーが発生しました: {e} !!!")
        raise RuntimeError("モデルまたはデータの読み込みに失敗しました。")

    # メタデータから逆引きマップをメモリ上に構築
    print("--- チャンクから論文への逆引きマップを構築します ---")
    for paper_index, paper_data in enumerate(metadata):
        if "chunks" in paper_data and isinstance(paper_data["chunks"], list):
            for chunk_index_in_paper, _ in enumerate(paper_data["chunks"]):
                chunk_to_paper_map.append({
                    "paper_index": paper_index,
                    "chunk_index_in_paper": chunk_index_in_paper
                })
    print(f"--- 逆引きマップの構築が完了しました (総チャンク数: {len(chunk_to_paper_map)}) ---")
    
    # 起動時のデータ検証
    if index.ntotal != len(chunk_to_paper_map):
        print("!!! 致命的エラー: FAISSインデックスのベクトル数とメタデータの総チャンク数が一致しません。!!!")
        print(f"FAISSベクトル数: {index.ntotal}, メタデータ総チャンク数: {len(chunk_to_paper_map)}")
        raise RuntimeError("データ不整合のためサーバーを起動できません。preprocess.pyを再実行してください。")

    print("モデルとインデックスの読み込みが完了しました。")


# --- APIモデル定義 ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="検索したいテキストクエリ", example="Attention is All You Need")
    top_k: int = Field(5, gt=0, le=50, description="返却する検索結果の最大数")

class PaperResult(BaseModel):
    id: Any
    filename: str
    title: str
    fulltext: Optional[str] = None
    chunks: Optional[List[str]] = None

class PaperResultWithContext(PaperResult):
    score: float = Field(..., description="クエリとの関連度スコア（距離）。小さいほど類似度が高い。")
    best_chunk: str = Field(..., description="クエリに最も類似した論文内のチャンク")

class SearchResponse(BaseModel):
    results: List[PaperResultWithContext]


# --- /search エンドポイントの実装 ---
@app.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    """
    クエリに意味的に類似した論文を検索し、関連チャンクとスコアと共に返す
    """
    if not model or not index:
        raise HTTPException(status_code=503, detail="サーバーが初期化中です。")

    query_vector = model.encode([request.query])
    
    # 多くの候補を取得し、後から論文単位でユニークにする
    candidate_k = 50 
    distances, chunk_indices = index.search(query_vector.astype('float32'), candidate_k)

    found_paper_ids = set()
    unique_paper_results = []

    for score, chunk_idx in zip(distances[0], chunk_indices[0]):
        # 検索結果が request.top_k に達したらループを抜ける
        if len(unique_paper_results) >= request.top_k:
            break
        
        if chunk_idx != -1:
            try:
                # 逆引きマップを使い、チャンクのインデックスから正しい論文情報を取得
                map_info = chunk_to_paper_map[int(chunk_idx)]
                paper_index = map_info["paper_index"]
                paper_info = metadata[paper_index]
                paper_id = paper_info["id"]
                
                # まだ追加していない論文の場合のみ処理
                if paper_id not in found_paper_ids:
                    chunk_index_in_paper = map_info["chunk_index_in_paper"]
                    best_chunk_text = paper_info["chunks"][chunk_index_in_paper]
                    
                    result = PaperResultWithContext(
                        **paper_info,
                        score=float(score),
                        best_chunk=best_chunk_text
                    )
                    unique_paper_results.append(result)
                    found_paper_ids.add(paper_id)

            except IndexError:
                print(f"警告: 範囲外のインデックス {chunk_idx} が検出されました。スキップします。")
            except Exception as e:
                print(f"警告: データ処理中に予期せぬエラー: {e}。スキップします。")
    
    return SearchResponse(results=unique_paper_results)

@app.get("/")
def read_root():
    return {"message": "論文検索APIへようこそ！ /docs にアクセスしてください。"}

