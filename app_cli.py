import json
import time
import datetime
import pytz
import argparse
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
import os

PROGRESS_FILE = "progress.json"  # 進捗保存用
JST = pytz.timezone("Asia/Tokyo")  # 日本時間のタイムゾーン設定

def parse_args():
    parser = argparse.ArgumentParser(description="英語テキストを日本語に翻訳するスクリプト")
    parser.add_argument("--context", type=str, required=True, help="翻訳時の文脈情報")
    return parser.parse_args()

# LangChain の ChatOpenAI をローカルモデル用に設定
llm = ChatOpenAI(
    model_name="gemma3",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key='ollama',
    temperature=0.2,
)

# テキストを適切なサイズに分割する関数
def split_text(text, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", ",", " "]
    )
    return splitter.split_text(text)

# 翻訳関数（リトライ機能付き）
def translate_text(text, index, total, context, retries=3, timeout=30):
    now = datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 翻訳中 {index + 1}/{total}...")
    for attempt in range(retries):
        try:
            messages = [
                SystemMessage(content=f"あなたは文脈を組んで適切な口調で訳すことができる、優秀な翻訳家です。{context}"),
                HumanMessage(content=f"次の英語を自然な日本語に訳してください。訳した結果だけを返してください。\n\n英語:\n{text}")
            ]
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"警告: チャンク {index + 1} の翻訳が {elapsed_time:.2f} 秒かかりました。")
            return response.content
        except Exception as e:
            print(f"エラー (試行 {attempt+1}/{retries}): {e}")
            time.sleep(5)
    print(f"チャンク {index + 1} の翻訳に失敗しました。")
    return "[翻訳エラー]"

# バッチ翻訳（進捗保存機能付き）
def batch_translate(text, context, resume=False):
    chunks = split_text(text)
    translated_chunks = [None] * len(chunks)
    
    # 進捗ファイルがあればロード
    if resume and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
            translated_chunks = progress_data.get("translated_chunks", translated_chunks)
    
    for i, chunk in enumerate(chunks):
        if translated_chunks[i] is None:  # 未翻訳のみ実行
            translated_chunks[i] = translate_text(chunk, i, len(chunks), context)
            # 進捗保存
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({"translated_chunks": translated_chunks}, f, ensure_ascii=False, indent=2)
    
    return "\n".join(translated_chunks)

# 指定ファイルを翻訳（途中再開機能付き）
def translate_file(input_file, output_file, context, resume=False):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    translated_text = batch_translate(text, context, resume=resume)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated_text)
    print(f"翻訳が完了しました: {output_file}")
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)  # 翻訳完了後に進捗データ削除

# 実行例
if __name__ == "__main__":
    args = parse_args()
    input_file = "input.txt"  # 翻訳するファイルを指定
    output_file = "output.txt"  # 翻訳結果を保存するファイル
    translate_file(input_file, output_file, args.context, resume=True)  # 途中から再開可能
