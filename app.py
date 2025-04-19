import json
import time
import datetime
import pytz
import argparse
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
import os
import streamlit as st

# 定数定義
PROGRESS_FILE = "progress.json"  # 翻訳進捗の保存ファイル
JST = pytz.timezone("Asia/Tokyo")  # 日本時間のタイムゾーン設定
TIMEOUT = 30  # 最大許容翻訳時間（秒）

# LangChain の ChatOpenAI をローカルモデル用に設定
llm = ChatOpenAI(
    model_name="gemma3:4b-it-qat",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key='ollama',
    temperature=0.2,
)

# テキストを指定したチャンクサイズに分割する関数
def split_text(text, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 一度に処理する文字数
        chunk_overlap=overlap,  # チャンク間で重複する文字数
        separators=["\n", ".", ",", " "]  # テキストを分割する区切り文字
    )
    return splitter.split_text(text)  # 分割されたテキストをリストで返す


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
        chunk_size=chunk_size,  # 一度に処理する文字数
        chunk_overlap=overlap,  # チャンク間で重複する文字数
        separators=["\n", ".", ",", " "]  # テキストを分割する区切り文字
    )
    return splitter.split_text(text)  # 分割されたテキストをリストで返す

# 翻訳を実行する関数（リトライ機能とタイムアウトをサポート）
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
    st.write(f"[{now}] 翻訳中 {index + 1}/{total}...")  # 現在の進捗を表示
    for attempt in range(retries):  # リトライ回数分ループ
        try:
            # 翻訳リクエストのメッセージ
            messages = [
                SystemMessage(content=f"あなたは文脈を汲んで適切な口調で訳すことができる、優秀な翻訳家です。{context}"),
                HumanMessage(content=f"次の原文を自然な日本語に訳してください。訳した結果だけを返してください。\n\n原文:\n{text}")
            ]
            start_time = time.time()  # 翻訳処理開始時刻
            response = llm.invoke(messages)  # モデルに翻訳を依頼
            elapsed_time = time.time() - start_time  # 処理時間

            # 最大許容時間を超えた場合は警告を出す
            if elapsed_time > TIMEOUT:
                st.warning(f"チャンク {index + 1} の翻訳に {elapsed_time:.2f} 秒かかりました。再試行します。")
                return None  # 超過した場合は再試行

            return response.content  # 翻訳結果を返す
        except Exception as e:
            st.error(f"エラー (試行 {attempt+1}/{retries}): {e}")  # エラーメッセージを表示
            time.sleep(5)  # 再試行前に待機
    st.error(f"チャンク {index + 1} の翻訳に失敗しました。")  # すべてのリトライが失敗した場合
    return "[翻訳エラー]"  # エラーメッセージを返す

# 進捗情報をファイルに保存する関数
def save_progress(translated_chunks):
    """
    翻訳進捗をファイルに保存する。
    :param translated_chunks: 翻訳済みのチャンクリスト
    """
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"translated_chunks": translated_chunks}, f, ensure_ascii=False, indent=2)

# バッチ翻訳処理を実行する関数（途中再開機能付き）
def batch_translate(text, context, resume=False):
    """
    テキスト全体を翻訳する（途中再開機能あり）。
    :param text: 翻訳対象のテキスト
    :param context: 翻訳時の文脈情報
    :param resume: 進捗を再開するかどうか
    :return: 翻訳済みテキスト
    """
    chunks = split_text(text)  # 入力テキストをチャンクに分割
    translated_chunks = [None] * len(chunks)  # 翻訳結果を格納するリスト（初期値はNone）

    # 進捗ファイルが存在する場合は再開処理
    if resume and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress_data = json.load(f)  # 進捗データを読み込み
            translated_chunks = progress_data.get("translated_chunks", translated_chunks)  # 翻訳済みのチャンクを再利用
    
    # 各チャンクを翻訳
    for i, chunk in enumerate(chunks):
        while translated_chunks[i] is None:  # まだ翻訳されていないチャンクを処理
            translated_chunks[i] = translate_text(chunk, i, len(chunks), context)  # 翻訳を実行
            if translated_chunks[i] is None:
                st.warning("翻訳に時間がかかりすぎたため、再試行します。")  # 再試行の警告
                time.sleep(5)  # 再試行前に待機
            save_progress(translated_chunks)  # 進捗を保存

    return "\n".join(translated_chunks)  # 翻訳されたテキストを結合して返す


def batch_translate(text, context, resume=False):
    """
    テキスト全体を翻訳する（途中再開機能あり）。
    :param text: 翻訳対象のテキスト
    :param context: 翻訳時の文脈情報
    :param resume: 進捗を再開するかどうか
    :return: 翻訳済みテキスト
    """
    chunks = split_text(text)  # 入力テキストをチャンクに分割
    translated_chunks = [None] * len(chunks)  # 翻訳結果を格納するリスト（初期値はNone）

    # 進捗ファイルが存在する場合は再開処理
    if resume and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
            translated_chunks = progress_data.get("translated_chunks", translated_chunks)

    # 各チャンクを翻訳
    for i, chunk in enumerate(chunks):
        while translated_chunks[i] is None:
            translated_chunks[i] = translate_text(chunk, i, len(chunks), context)
            if translated_chunks[i] is None:
                st.warning("翻訳に時間がかかりすぎたため、再試行します。")
                time.sleep(5)
            save_progress(translated_chunks)

    # 翻訳が正常に完了した場合、進捗ファイルを削除
    if None not in translated_chunks:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

    return "\n".join(translated_chunks)

# Streamlitのページ設定
st.set_page_config(
    page_title="生成AI翻訳",  # ページタイトル
    page_icon="📃",  # アイコン
    layout="centered"  # 中央揃えのレイアウト
)
st.title("生成AI翻訳")  # アプリケーションのタイトル

# 文脈情報の入力欄
context = st.text_area(
    "文脈情報(いつ、どこで、誰が書いた、どういう文書か)を入力してください。",
    placeholder="例: この文章は、イギリスで1895年に出版された『亀がアキレスに言ったこと』という、数学を題材にしてナンセンス小説です。作者は、『不思議の国のアリス』で有名なルイス・キャロルです。"
)

# ファイルアップロード機能
uploaded_file = st.file_uploader("翻訳するテキストファイルをアップロードしてください。", type=["txt"])

# 翻訳結果の格納変数
translated_text = ""

# ファイルがアップロードされた場合
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")  # ファイルのテキストを読み込み
    
    # 「翻訳実行」ボタンが押された場合
    if st.button("翻訳実行"):
        # 文脈情報が入力されていれば翻訳開始
        if context.strip():
            # バッチ翻訳を実行
            translated_text = batch_translate(text, context, resume=True)
        else:
            st.warning("文脈情報を入力してください。")  # 文脈情報がない場合は警告を表示

# 翻訳結果を表示
if translated_text:
    st.text_area("翻訳結果", translated_text, height=300)  # 翻訳結果を表示
    
    # 翻訳結果のダウンロードボタンを表示
    st.download_button(
        label="翻訳結果をダウンロード",
        data=translated_text,
        file_name="translated_output.txt",
        mime="text/plain"
    )
