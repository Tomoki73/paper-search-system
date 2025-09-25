import os
import requests
from dotenv import load_dotenv

# --- 設定 ---
# 正しいモデルID（作者名/モデル名）
# MODEL_ID = "stabilityai/japanese-stablelm-instruct-gamma-7b"
# API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# ...
# --- 設定 ---
# 究極のテスト用に、最も基本的なモデルを指定
MODEL_ID = "gpt2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
# ...
# payloadもgpt2用に単純化
payload = {"inputs": "Hello, my name is"}
# ...

# --- メインの処理 ---
def run_test():
    """Hugging Face APIへの接続をテストする"""
    
    # 1. .envファイルからAPIトークンを読み込む
    load_dotenv()
    api_token = os.getenv("HUGGING_FACE_API_TOKEN")

    # 2. トークンが正しく読み込めているか確認（重要）
    if not api_token:
        print("!!! エラー: .envファイルからHUGGING_FACE_API_TOKENを読み込めませんでした。")
        print(".envファイルが同じ階層にあるか、キーの名前が正しいか確認してください。")
        return
    
    print(f"読み込んだトークン (最初の5文字): {api_token[:5]}...") # トークンの一部を表示して確認
    print(f"アクセスするURL: {API_URL}")

    # 3. APIリクエストを送信
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": "<|user|>日本の首都は？<|endoftext|><|assistant|>",
        "parameters": {"max_new_tokens": 50}
    }
    
    print("\nAPIにリクエストを送信します...")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # エラーがあれば例外を発生
        
        # 成功した場合
        print("\n★★★ テスト成功！ ★★★")
        print("Hugging Faceからの応答:")
        print(response.json())

    except requests.exceptions.RequestException as e:
        # 失敗した場合
        print("\n!!! テスト失敗... !!!")
        print(f"エラー種別: {type(e).__name__}")
        print(f"エラー内容: {e}")
        if e.response:
            print(f"サーバーからの応答本文: {e.response.text}")

# --- スクリプトの実行 ---
if __name__ == "__main__":
    run_test()