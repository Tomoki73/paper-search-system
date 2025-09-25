import os
import faiss
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai #  Geminiライブラリをインポート

# --- グローバル変数 ---
model: SentenceTransformer = None
index: faiss.Index = None
metadata: list = []
chunk_to_paper_map: list = []
genai_model = None # Geminiモデルを保持する変数

app = FastAPI(
    title="論文検索＆Q&A API (Gemini Edition)",
    description="論文のセマンティック検索と、Gemini APIを利用したQ&Aを行います。",
    version="3.0.0"
)

# --- サーバー起動時の処理 ---
@app.on_event("startup")
def load_models():
    """サーバー起動時に、モデル、インデックス、メタデータ、Geminiを読み込む"""
    global model, index, metadata, chunk_to_paper_map, genai_model
    
    # 1. Gemini APIの設定
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEYが.envファイルに設定されていません。")
        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel('gemini-1.5-flash') # 無料で高速なモデル
        print("✅ Google Geminiモデルの初期化が完了しました。")
    except Exception as e:
        raise RuntimeError(f"❌ Geminiモデルの初期化に失敗しました: {e}")

    # 2. 埋め込みモデルとデータの読み込み (ここは以前と同じ)
    print("埋め込みモデルとデータの読み込みを開始します...")
    try:
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        index = faiss.read_index("paper_vectors.index")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"❌ モデルまたはデータの読み込みに失敗しました: {e}")

    # 3. チャンクマップの構築 (ここも以前と同じ)
    chunk_offset = 0
    for paper_index, paper_data in enumerate(metadata):
        num_chunks = len(paper_data.get("chunks", []))
        for chunk_index_in_paper in range(num_chunks):
            chunk_to_paper_map.append({
                "paper_index": paper_index, "chunk_index_in_paper": chunk_index_in_paper
            })
        chunk_offset += num_chunks
    
    print("✅ すべてのモデルとデータの読み込みが完了しました。")

# --- APIのモデル定義 (Pydantic) ---
# (SearchとChatのモデル定義は以前と全く同じです)
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
class PaperResult(BaseModel):
    id: Any; filename: str; title: str; chunks: Optional[List[str]] = None
class PaperResultWithContext(PaperResult):
    score: float; best_chunk: str
class SearchResponse(BaseModel):
    results: List[PaperResultWithContext]
class ChatRequest(BaseModel):
    paper_id: str; question: str
class ChatResponse(BaseModel):
    answer: str; context_chunks: List[str]

# --- APIエンドポイント ---
# (`/search` エンドポイントは変更ありません)
@app.post("/search", response_model=SearchResponse, summary="論文の検索")
def search_papers(request: SearchRequest):
    # (この関数のコードは以前と全く同じです)
    if not model or not index: raise HTTPException(503, "サーバー初期化中")
    query_vector = model.encode([request.query])
    distances, chunk_indices = index.search(query_vector.astype('float32'), 50)
    found_paper_ids, results = set(), []
    for score, chunk_idx in zip(distances[0], chunk_indices[0]):
        if len(results) >= request.top_k: break
        if chunk_idx != -1:
            map_info = chunk_to_paper_map[int(chunk_idx)]
            paper_info = metadata[map_info["paper_index"]]
            if paper_info["id"] not in found_paper_ids:
                best_chunk = paper_info["chunks"][map_info["chunk_index_in_paper"]]
                results.append(PaperResultWithContext(**paper_info, score=float(score), best_chunk=best_chunk))
                found_paper_ids.add(paper_info["id"])
    return SearchResponse(results=results)

# --- ここがGemini APIを呼び出す部分です ---
@app.post("/chat", response_model=ChatResponse, summary="論文に関するQ&A")
def chat_with_paper(request: ChatRequest):
    """指定された論文の内容に基づいて、Gemini APIを使って質問に回答する"""
    if not genai_model or not model:
        raise HTTPException(status_code=503, detail="サーバーが初期化中です。")

    # 1. 論文とチャンクを取得 (ここは同じ)
    target_paper = next((p for p in metadata if p["id"] == request.paper_id), None)
    if not target_paper: raise HTTPException(404, "論文が見つかりません。")
    paper_chunks = target_paper.get("chunks", [])
    if not paper_chunks: return ChatResponse(answer="この論文に本文はありません。", context_chunks=[])

    # 2. 関連チャンクを検索 (ここも同じ)
    question_vector = model.encode([request.question])
    chunk_vectors = model.encode(paper_chunks)
    temp_index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    temp_index.add(chunk_vectors.astype('float32'))
    _, indices = temp_index.search(question_vector.astype('float32'), 3)
    context_chunks = [paper_chunks[i] for i in indices[0] if i != -1]
    if not context_chunks: return ChatResponse(answer="関連情報が見つかりませんでした。", context_chunks=[])

    # 3. Gemini API用のプロンプトを作成
    context_str = "\n\n".join(context_chunks)
    prompt = f"""あなたは優秀な研究アシスタントです。以下の論文の抜粋だけを厳密な情報源として、ユーザーの質問に日本語で回答してください。
情報が抜粋にない場合は、「その情報はこの論文内には見つかりませんでした。」とだけ回答してください。自身の知識や意見を加えてはいけません。

--- 論文の抜粋 ---
{context_str}
---

ユーザーの質問: {request.question}
"""

    # 4. Gemini APIを呼び出す
    try:
        response = genai_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini APIの呼び出し中にエラーが発生しました: {e}")

    return ChatResponse(answer=answer, context_chunks=context_chunks)