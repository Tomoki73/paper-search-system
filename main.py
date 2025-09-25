import faiss
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Any
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# --- グローバル変数としてモデル、インデックス、メタデータを保持 ---
# これらはサーバー起動時に一度だけ読み込まれる
model: SentenceTransformer = None
index: faiss.Index = None
metadata: list = []


# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# --- CORSミドルウェアの設定 ---
origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    # 開発用として、任意のオリジンを許可する場合は以下を使用
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- サーバー起動時に実行される処理 ---
@app.on_event("startup")
def load_models():
    """サーバー起動時に、モデルとインデックスをメモリに読み込む"""
    global model, index, metadata
    
    print("モデルとインデックスの読み込みを開始します...")
    
    # 1. Sentence Transformerモデルの読み込み
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. FAISSインデックスの読み込み
    index = faiss.read_index("paper_vectors.index")
    
    # 3. 論文メタデータの読み込み
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 更新部分: メタデータの構造を新しいモデルで検証する
    print("--- メタデータ構造の検証を開始します ---")
    for i, paper_data in enumerate(metadata):
        try:
            # PaperResultモデルに合致するか試す
            _ = PaperResult(**paper_data)
        except ValidationError as e:
            # エラーが発生した場合、どのデータが問題か分かりやすく表示してサーバーを停止する
            print(f"!!! 致命的なエラー: dummy_metadata.json の {i} 番目のデータ形式が不正です。!!!")
            print(f"問題のデータ: {paper_data}")
            print(f"Pydantic検証エラー: {e}")
            raise RuntimeError("メタデータの検証に失敗しました。サーバーを起動できません。")
    print("--- メタデータ構造の検証が完了しました ---")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print(f"モデルとインデックスの読み込みが完了しました。")


# --- APIモデルの定義 (ステップ2で作成済み) ---
class SearchRequest(BaseModel):
    query: str

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 修正点: PaperResultモデルのフィールドをオプショナルに変更
class PaperResult(BaseModel):
    id: Any  # idは様々な形式がありうるためAnyで受け付ける
    filename: str
    title: str
    fulltext: str | None = None  # fulltextをオプショナル（任意）に変更
    chunks: List[str] | None = None # chunksも念のためオプショナルに変更
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


class SearchResponse(BaseModel):
    results: List[PaperResult]

# --- /search エンドポイントの実装 ---
@app.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    """
    クエリを受け取り、意味的に類似した論文を検索して返す
    """
    # 1. 検索クエリをベクトル化
    query_vector = model.encode([request.query])

    # 2. FAISSで類似ベクトルを検索
    k = 3
    distances, indices = index.search(query_vector.astype('float32'), k)

    # 3. 検索結果を整形
    results = []
    for i in indices[0]:
        if i != -1:
            # 修正点: numpyの整数型(i)をpythonのint型に変換
            paper_info = metadata[int(i)]
            results.append(PaperResult(**paper_info))

    return SearchResponse(results=results)

