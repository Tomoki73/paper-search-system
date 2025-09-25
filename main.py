import os
import faiss
import json
import requests  # requestsライブラリを使用
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- グローバル変数 ---
model: SentenceTransformer = None
index: faiss.Index = None
metadata: list = []
chunk_to_paper_map: list = []
# --- Hugging Face APIの設定 ---
HF_API_TOKEN = None
HF_MODEL_ID = "stabilityai/japanese-stablelm-instruct-gamma-7b"  # 先ほど選択したモデル
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

app = FastAPI(
    title="論文検索＆Q&A API (Hugging Face版)",
    description="論文のセマンティック検索と、Hugging Face Inference APIを利用したQ&Aを行います。",
    version="2.1.0"
)

# --- サーバー起動時の処理 ---
@app.on_event("startup")
def load_models():
    global model, index, metadata, chunk_to_paper_map, HF_API_TOKEN
    
    load_dotenv()
    HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
    if not HF_API_TOKEN:
        raise RuntimeError(".envファイルにHUGGING_FACE_API_TOKENが見つかりません。")
    print("Hugging Face APIトークンを読み込みました。")

    # (ここから下のデータ読み込み処理は以前と同じ)
    print("埋め込みモデルとデータの読み込みを開始します...")
    try:
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        index = faiss.read_index("paper_vectors.index")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"モデルまたはデータの読み込みに失敗しました: {e}")

    # (チャンクマップの構築も以前と同じ)
    chunk_offset = 0
    for paper_index, paper_data in enumerate(metadata):
        num_chunks = len(paper_data.get("chunks", []))
        for chunk_index_in_paper in range(num_chunks):
            chunk_to_paper_map.append({
                "paper_index": paper_index,
                "chunk_index_in_paper": chunk_index_in_paper,
                "global_chunk_index": chunk_offset + chunk_index_in_paper
            })
        chunk_offset += num_chunks
    print("すべてのモデルとデータの読み込みが完了しました。")

# --- APIのモデル定義 (Pydantic) ---
# (検索関連のモデルは変更なし)
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class PaperResult(BaseModel):
    id: Any
    filename: str
    title: str
    chunks: Optional[List[str]] = None

class PaperResultWithContext(PaperResult):
    score: float
    best_chunk: str

class SearchResponse(BaseModel):
    results: List[PaperResultWithContext]

# (チャット関連のモデルは変更なし)
class ChatRequest(BaseModel):
    paper_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    context_chunks: List[str]

# --- APIエンドポイント ---
# (`/search` エンドポイントは変更なし)
@app.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    # この関数のコードは以前と全く同じ
    if not model or not index:
        raise HTTPException(status_code=503, detail="サーバーが初期化中です。")
    query_vector = model.encode([request.query])
    distances, chunk_indices = index.search(query_vector.astype('float32'), 50)
    found_paper_ids = set()
    unique_paper_results = []
    for score, chunk_idx in zip(distances[0], chunk_indices[0]):
        if len(unique_paper_results) >= request.top_k: break
        if chunk_idx != -1:
            map_info = chunk_to_paper_map[int(chunk_idx)]
            paper_index = map_info["paper_index"]
            paper_info = metadata[paper_index]
            paper_id = paper_info["id"]
            if paper_id not in found_paper_ids:
                chunk_index_in_paper = map_info["chunk_index_in_paper"]
                best_chunk_text = paper_info["chunks"][chunk_index_in_paper]
                result = PaperResultWithContext(
                    **paper_info, score=float(score), best_chunk=best_chunk_text
                )
                unique_paper_results.append(result)
                found_paper_ids.add(paper_id)
    return SearchResponse(results=unique_paper_results)

# --- ここが重要な更新箇所です ---
@app.post("/chat", response_model=ChatResponse)
def chat_with_paper(request: ChatRequest):
    """指定された論文について、Hugging Face Inference APIを使って質問に回答する"""
    # 1. 論文とテキストチャンクを見つける (以前と同じ)
    target_paper = next((p for p in metadata if p["id"] == request.paper_id), None)
    if not target_paper:
        raise HTTPException(status_code=404, detail="指定されたIDの論文が見つかりません。")
    
    paper_chunks = target_paper.get("chunks", [])
    if not paper_chunks:
        return ChatResponse(answer="この論文には検索対象の本文がありません。", context_chunks=[])

    # 2. 質問に最も関連するチャンクを検索する (以前と同じ)
    question_vector = model.encode([request.question])
    chunk_vectors = model.encode(paper_chunks)
    temp_index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    temp_index.add(chunk_vectors.astype('float32'))
    
    k = 3
    _, indices = temp_index.search(question_vector.astype('float32'), k)
    context_chunks = [paper_chunks[i] for i in indices[0] if i != -1]
    
    if not context_chunks:
        return ChatResponse(answer="論文内に質問と関連する情報が見つかりませんでした。", context_chunks=[])

    # 3. Hugging Faceモデル用のプロンプトを作成する
    # 注意: ELYZAモデルは `[INST]` や `<<SYS>>` といった特別な形式を使います。
    context_str = "\n\n".join(context_chunks)
    prompt = f"""<|user|>
以下の「論文の抜粋」の情報だけを厳密な根拠として、ユーザーの質問に日本語で回答してください。抜粋に情報がない場合は、「その情報については、提供された資料の中には見つかりませんでした。」とだけ回答してください。

# 論文の抜粋
{context_str}

# 質問
{request.question} [/INST]"""

    # 4. Hugging Face Inference APIを呼び出す
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {  # オプション: 生成の挙動を制御
            "max_new_tokens": 512,      # 最大生成トークン数
            "temperature": 0.7,         # 生成の多様性
            "return_full_text": False   # プロンプトを含めずに、生成された部分だけを返す
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        result = response.json()
        
        # APIはリストを返すので、最初の要素の生成テキストを取得
        answer = result[0].get("generated_text", "エラー：モデルの応答を解析できませんでした。")
        
    except requests.exceptions.RequestException as e:
        print("--- APIリクエストエラー詳細 ---")
        print(f"ステータスコード: {e.response.status_code if e.response else 'N/A'}")
        print(f"エラー内容: {e}")
        # Hugging Faceからの具体的なエラーメッセージを表示
        if e.response:
            print(f"Hugging Faceからの応答: {e.response.text}")
        print("-------------------------------")
        
        detail = "Hugging Face APIの呼び出しに失敗しました。"
        if e.response and "is currently loading" in e.response.text:
             detail = "モデルが起動中です。しばらくしてから再度お試しください。"
        
        raise HTTPException(status_code=500, detail=detail)

    return ChatResponse(answer=answer, context_chunks=context_chunks)

@app.get("/")
def read_root():
    return {"message": "論文Q&A APIへようこそ！ /docs でAPIドキュメントを確認できます。"}