import streamlit as st
import requests
import textwrap
import time
from io import StringIO
from tqdm import tqdm

# Ollama API のエンドポイント
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "gemma3"
DEFAULT_PROMPT = (
    "いつどこで書かれ、どういう内容で、タイトルは何か、"
    "書いたのは誰かなどを入力してください。"
)

# テキストを適切な長さに分割
def split_text(text, max_chunk_size=1000):
    return textwrap.wrap(text, max_chunk_size)

# LLM で翻訳を実行
def translate_chunk(chunk, model, prompt):
    full_prompt = f"あなたは優秀な翻訳家です。\n\n以下の英語のテキストは、{prompt}\n\n英語を自然な日本語に翻訳してください。翻訳結果だけを返してください。\n\n英語:\n{chunk}\n\n日本語:"
    data = {"model": model, "prompt": full_prompt, "stream": False}
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"エラー: {e}"

# 全文翻訳
def translate_text(text, model, prompt):
    chunks = split_text(text)
    translated_chunks = []
    
    for chunk in tqdm(chunks, desc="Translating"):
        translated = translate_chunk(chunk, model, prompt)
        if translated:
            translated_chunks.append(translated)
        time.sleep(0.5)  # 負荷軽減のための短い待機時間
    
    return "\n".join(translated_chunks)

# Streamlit UI
st.set_page_config(
    page_title="生成AI翻訳アプリ",
    page_icon="📕",
    layout="wide"
)
st.title("生成AI翻訳アプリ")

uploaded_file = st.file_uploader("翻訳するテキストファイルをアップロードしてください", type=["txt"])
model_name = st.text_input("使用するモデル名", DEFAULT_MODEL_NAME)
prompt_text = st.text_area("翻訳プロンプト", DEFAULT_PROMPT, height=150)

if uploaded_file is not None:
    text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    if st.button("翻訳を実行"):
        with st.spinner("翻訳中..."):
            translated_text = translate_text(text, model_name, prompt_text).replace("\u3000", " ")
            st.text_area("翻訳結果", translated_text, height=300)
            st.download_button("翻訳結果をダウンロード", translated_text, file_name="translated.txt", mime="text/plain")
