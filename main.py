import os
import faiss
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "null"  # ローカルファイル(file://)からのアクセスを許可するために重要
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # すべてのHTTPメソッドを許可
    allow_headers=["*"], # すべてのHTTPヘッダーを許可
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
        genai_model = genai.GenerativeModel('gemini-2.5-flash') # 無料で高速なモデル
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
class ChunkData(BaseModel):
    """チャンクとそのセクション情報を保持するモデル"""
    text: str
    section: str

class PaperResult(BaseModel):
    """論文の基本情報を保持するモデル"""
    id: Any
    filename: str
    title: str
    author: Optional[str] = None
    year: Optional[int] = None
    chunks: Optional[List[ChunkData]] = None # List[str] から List[ChunkData] に変更

class PaperResultWithContext(PaperResult):
    """検索結果として返す、コンテキスト付きの論文モデル"""
    score: float
    best_chunk: ChunkData # str から ChunkData に変更

class PaperDetailResponse(PaperResult):
    """論文の詳細情報を保持するモデル"""
    # PaperResultを継承しているため、id, filename, titleは継承される
    chunks: List[ChunkData] # 全てのチャンクが必要なため、Optionalを上書きして必須にする
    
    # 検索結果で使われる score と best_chunk は、詳細画面では不要だが、
    # 継承元のPaperResultが持つchunksを上書き定義するため、ここではそのまま。

class SearchResponse(BaseModel):
    results: List[PaperResultWithContext]
class ChatRequest(BaseModel):
    paper_id: str; question: str
class ChatResponse(BaseModel):
    answer: str; context_chunks: List[str]
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# --- APIエンドポイント ---
@app.post("/search", response_model=SearchResponse, summary="論文の検索")
def search_papers(request: SearchRequest):
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
                # best_chunkは辞書オブジェクトそのものになる
                best_chunk_obj = paper_info["chunks"][map_info["chunk_index_in_paper"]]
                
                # 新しいデータモデルに合わせてインスタンスを作成
                result_item = PaperResultWithContext(
                    **paper_info, 
                    score=float(score), 
                    best_chunk=best_chunk_obj # 辞書オブジェクトを渡す
                )
                results.append(result_item)
                found_paper_ids.add(paper_info["id"])
    return SearchResponse(results=results)

# --- ここがGemini APIを呼び出す部分です ---
@app.post("/chat", response_model=ChatResponse, summary="論文に関するQ&A")
def chat_with_paper(request: ChatRequest):
    """(高精度版) 論文の内容に基づいて、Gemini APIを使って質問に回答する"""
    if not genai_model or not model:
        raise HTTPException(status_code=503, detail="サーバーが初期化中です。")

    target_paper = next((p for p in metadata if p["id"] == request.paper_id), None)
    if not target_paper: raise HTTPException(404, "論文が見つかりません。")
    
    paper_chunks_data = target_paper.get("chunks", [])
    if not paper_chunks_data: return ChatResponse(answer="この論文に本文はありません。", context_chunks=[])
    
    # ▼▼▼▼▼ 検索ロジックの高度化 ▼▼▼▼▼
    target_chunks_data = paper_chunks_data
    
    # 質問のキーワードに基づいて、検索対象のチャンクを絞り込む
    question_lower = request.question.lower()
    if any(keyword in question_lower for keyword in ["目的", "概要", "はじめに", "背景"]):
        target_chunks_data = [chunk for chunk in paper_chunks_data if "概要" in chunk["section"] or "はじめに" in chunk["section"] or "背景" in chunk["section"]]
    elif any(keyword in question_lower for keyword in ["結論", "まとめ", "考察"]):
        target_chunks_data = [chunk for chunk in paper_chunks_data if "結論" in chunk["section"] or "まとめ" in chunk["section"] or "考察" in chunk["section"]]
    elif any(keyword in question_lower for keyword in ["手法", "方法"]):
        target_chunks_data = [chunk for chunk in paper_chunks_data if "手法" in chunk["section"] or "方法" in chunk["section"]]

    # 絞り込み対象がなかった場合は、全チャンクを対象に戻す
    if not target_chunks_data:
        target_chunks_data = paper_chunks_data

    # 絞り込んだチャンクのテキストだけをリストにする
    target_texts = [chunk['text'] for chunk in target_chunks_data]
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 絞り込んだチャンクに対してセマンティック検索を実行
    question_vector = model.encode([request.question])
    chunk_vectors = model.encode(target_texts)
    
    # ベクトルがない（対象チャンクが0個）場合はエラー回避
    if chunk_vectors.shape[0] == 0:
        return ChatResponse(answer="関連情報が見つかりませんでした。", context_chunks=[])

    temp_index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    temp_index.add(chunk_vectors.astype('float32'))
    
    k = min(10, len(target_texts)) # チャンク数より多くは検索しない
    _, indices = temp_index.search(question_vector.astype('float32'), k)
    
    context_chunks_text = [target_texts[i] for i in indices[0] if i != -1]
    if not context_chunks_text: return ChatResponse(answer="関連情報が見つかりませんでした。", context_chunks=[])

    # (Gemini APIへのプロンプト作成と呼び出し部分は変更なし)
    context_str = "\n\n".join(context_chunks_text)
    prompt = f"""あなたは優秀な研究アシスタントです。以下の論文の抜粋だけを厳密な情報源として、ユーザーの質問に日本語で回答してください。
情報が抜粋にない場合は、「その情報はこの論文内には見つかりませんでした。」とだけ回答してください。自身の知識や意見を加えてはいけません。

--- 論文の抜粋 ---
{context_str}
---

ユーザーの質問: {request.question}
"""
    try:
        response = genai_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini APIの呼び出し中にエラー: {e}")

    return ChatResponse(answer=answer, context_chunks=context_chunks_text)