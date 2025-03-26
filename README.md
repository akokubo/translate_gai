# 生成AI翻訳

## 使用したもの
 - LangChain
 - [Ollama](https://ollama.com/)

## インストール
```
git clone https://github.com/akokubo/translate_gai.git
cd translate_gai
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Ollamaの準備

## Ollamaの準備
1. Ollamaをインストール
   - Windowsの場合は、WSL2で仮想環境から `curl -fsSL https://ollama.com/install.sh | sh` でインストール
   - Macの場合は、[ダウンロード](https://ollama.com/download/windows)してインストール
2. Ollamaで大規模言語モデルの `gemma3` をpullする。
```
ollama pull gemma3
```
※大規模言語モデルは、自由に選べ、他のものでもいい。

## 実行
最初に、プログラムを展開したフォルダに入る。
次に仮想環境に入っていない場合(コマンドプロンプトに(venv)と表示されていないとき)、仮想環境に入る。
```
source venv/bin/activate
```

Ollamaが起動していないかもしれないので、仮想環境に入っている状態で、大規模言語モデルのリストを表示する(すると起動していなければ、起動する)。
```
ollama list
```



仮想環境に入っている状態で、英語のテキストファイルをinput.txtという名前で同じフォルダに置く。
そして、以下のコマンドで実行する。

```
python3 app_cli.py --context "ここに文書の背景情報を書く"
```
背景情報とは、いつどこで書かれたとか、何について書かれているとか、どういう人物が書いたとか、そういう情報で、翻訳のときのシステム・プロンプトに追加される。

実行が完全に終了すると、output.txtという名前で、翻訳結果ができる。

途中で進捗が停まったら、Ctrl+cで一旦中断し、再度`python3 app_cli.py --context "ここに文書の背景情報を書く"`を実行すると、停めたところから再開される。
中断した場合の途中までの情報は`progress.json`に入っている。

![スクリーンショット](images/screenshot.png)

## 作者
[小久保 温(こくぼ・あつし)](https://akokubo.github.io/)

## ライセンス
[MIT License](LICENSE)
