import json
import time
import datetime
import pytz
import argparse
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
import os

# 定数定義
PROGRESS_FILE = "progress.json"  # 翻訳進捗の保存ファイル
JST = pytz.timezone("Asia/Tokyo")  # 日本時間のタイムゾーン設定
TIMEOUT = 30  # 最大許容翻訳時間（秒）

# コマンドライン引数を解析する関数
def parse_args():
    parser = argparse.ArgumentParser(description="原文のテキストを日本語に翻訳するスクリプト")
    parser.add_argument("--context", type=str, required=True, help="翻訳時の文脈情報")
    parser.add_argument("--input", type=str, required=True, help="入力ファイルのパス")
    parser.add_argument("--output", type=str, required=True, help="出力ファイルのパス")
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
    """
    長いテキストを適切なサイズに分割する。
    :param text: 分割するテキスト
    :param chunk_size: 各チャンクの最大サイズ
    :param overlap: チャンク間の重複部分のサイズ
    :return: 分割されたテキストのリスト
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", ",", " "]
    )
    return splitter.split_text(text)

# 翻訳関数（リトライ & タイムアウト対応）
def translate_text(text, index, total, context, retries=3):
    """
    テキストを翻訳する関数。
    :param text: 翻訳対象のテキスト
    :param index: チャンクのインデックス
    :param total: 全チャンク数
    :param context: 翻訳時の文脈情報
    :param retries: 最大リトライ回数
    :return: 翻訳結果またはエラー文字列
    """
    now = datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 翻訳中 {index + 1}/{total}...")
    for attempt in range(retries):
        try:
            messages = [
                SystemMessage(content=f"あなたは文脈を汲んで適切な口調で訳すことができる、優秀な翻訳家です。{context}"),
                HumanMessage(content=f"次の原文を自然な日本語に訳してください。訳した結果だけを返してください。\n\n原文:\n{text}")
            ]
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            if elapsed_time > TIMEOUT:
                print(f"警告: チャンク {index + 1} の翻訳に {elapsed_time:.2f} 秒かかりました。再試行します。")
                return None  # 翻訳が遅すぎる場合は再試行
            
            return response.content
        except Exception as e:
            print(f"エラー (試行 {attempt+1}/{retries}): {e}")
            time.sleep(5)
    print(f"チャンク {index + 1} の翻訳に失敗しました。")
    return "[翻訳エラー]"

# 翻訳進捗を保存する関数
def save_progress(translated_chunks):
    """
    翻訳進捗をファイルに保存する。
    :param translated_chunks: 翻訳済みのチャンクリスト
    """
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"translated_chunks": translated_chunks}, f, ensure_ascii=False, indent=2)

# バッチ翻訳関数
def batch_translate(text, context, resume=False):
    """
    テキスト全体を翻訳する（途中再開機能あり）。
    :param text: 翻訳対象のテキスト
    :param context: 翻訳時の文脈情報
    :param resume: 進捗を再開するかどうか
    :return: 翻訳済みテキスト
    """
    chunks = split_text(text)
    translated_chunks = [None] * len(chunks)
    
    if resume and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
            translated_chunks = progress_data.get("translated_chunks", translated_chunks)
    
    for i, chunk in enumerate(chunks):
        while translated_chunks[i] is None:  # 未翻訳または再試行必要なもののみ処理
            translated_chunks[i] = translate_text(chunk, i, len(chunks), context)
            if translated_chunks[i] is None:
                print("翻訳に時間がかかりすぎたため、再試行します。")
                time.sleep(5)
            save_progress(translated_chunks)
    
    return "\n".join(translated_chunks)

# 指定ファイルを翻訳する関数
def translate_file(input_file, output_file, context, resume=False):
    """
    指定されたテキストファイルを翻訳する。
    :param input_file: 入力ファイルのパス
    :param output_file: 出力ファイルのパス
    :param context: 翻訳時の文脈情報
    :param resume: 進捗を再開するかどうか
    """
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    translated_text = batch_translate(text, context, resume=resume)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated_text.replace("\u3000", " "))
    
    print(f"翻訳が完了しました: {output_file}")
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

# メイン実行部分
if __name__ == "__main__":
    args = parse_args()
    translate_file(args.input, args.output, args.context, resume=True)
