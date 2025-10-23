# Semantic Network Project

帯域幅が限られたエッジデバイスでは、生の勾配ではなくタスク指向の圧縮表現を交換することでフェデレーテッドラーニングの通信量を抑えられます。このリポジトリは、そのようなアイデアを検証するための軽量な遊び場を提供します。PyTorch で記述したタスク固有モデルとセマンティック圧縮ユーティリティ、そして最小限の [Flower](https://flower.dev/) クライアント／サーバーを組み合わせ、NVIDIA Jetson をはじめとする CUDA 対応マシンで実験できます。

英語版のガイドは [README.md](README.md) を参照してください。

## リポジトリ構成

- `configs/` – 単体実験やフェデレーテッド実行用の YAML プリセット。新しい設定は `base.yaml` から派生させます。
- `src/federated/` – このプロジェクト向けに調整した Flower の NumPy クライアントと FedAvg サーバーの薄いラッパー。
- `src/semantic/` – スパース化や量子化フローを実装したエンコーダ／デコーダ、レート制御ヘルパー。
- `src/task/` – タスク固有のモデル、前処理、メトリクス。`disaster/` はセグメンテーション、`netqos/` は時系列予測をカバーします。
- `src/transport/` – 将来のカスタムトランスポートを想定した実験的メッセージング枠組み。
- `src/run_single.py` / `src/run_fed_client.py` / `src/run_fed_server.py` – コンフィグ、タスク、セマンティック圧縮を結線するエントリースクリプト。

## `uv` を使ったセットアップ

[`uv`](https://github.com/astral-sh/uv) は本プロジェクトの依存関係と仮想環境を管理します。以下のステップで `pyproject.toml` に記述されたパッケージが `.venv/` に展開され、リポジトリとロックファイルが同期します。

1. **依存関係の同期**

   ```bash
   uv sync
   ```

   Jetson など NVIDIA L4T 系デバイスでは、同梱されている Jetson 向けホイールを組み込むために次のように実行します。

   ```bash
   uv sync --extra jetson
   ```

2. **環境の利用**

   - 仮想環境を明示的にアクティブ化せずスクリプトを実行:

     ```bash
     uv run python src/run_single.py --task disaster --epochs 1
     ```

   - もしくは一度だけ環境をアクティブ化:

     ```bash
     source .venv/bin/activate
     ```

3. **依存関係の更新**

   ```bash
   uv add <package>
   uv remove <package>
   ```

   変更後は `uv sync` を再度実行してロックファイルを更新します。

## Docker イメージ (JetPack)

同梱の `Dockerfile` は `nvcr.io/nvidia/l4t-base:r36.4.0` をベースに NVIDIA 配布の PyTorch 2.5 ホイールと `wheels/` 内の torchvision ホイールを組み合わせ、Jetson デバイス向けのコンテナを構築します。

1. **ビルド**

   ```bash
   docker build -t semantic-net:jp36 .
   ```

   JetPack のリリースを変更する場合は、`pyproject.toml` の `jetson` エクストラを更新し、対応する torchvision ホイールを `wheels/` に追加してから `uv.lock` を再生成してください。

   NVIDIA が新しいホイールを公開した場合:

   ```bash
   uv lock  # pyproject や wheels を更新した後にロックファイルを再生成
   docker build -t semantic-net:jp36 .
   ```

2. **実行**

   ```bash
   docker run --rm -it --runtime nvidia --network host semantic-net:jp36 bash
   ```

   コンテナは `/workspace/semantic-net` にカレントディレクトリを設定し、仮想環境を PATH に追加した状態で起動します。`semantic-run-single` などのコンソールスクリプトを直接使用できます。

## Docker Compose

付属の `docker-compose.yml` は同じイメージを用いて対話シェル、Flower サーバー、クライアントを立ち上げるユーティリティを提供します。

- ホストのコードをマウントしたままコンテナ内にシェルを開く:

  ```bash
  docker compose --profile cli run --rm shell
  ```

- Flower サーバーとクライアントを立ち上げる (クライアント数はスケール可能):

  ```bash
  docker compose --profile gpu up --build server
  docker compose --profile gpu up --build --scale client=2 client
  ```

  実行時のパラメータは環境変数で調整できます。例: `SERVER_ROUNDS=5 CLIENT_CONFIG=configs/task_netqos.yaml docker compose --profile gpu up client`

## 使い方の例

### 単体デバイスでの実験

セグメンテーションタスクのスモークテスト:

```bash
uv run python src/run_single.py --task disaster --epochs 1
```

QoS 予測タスクに切り替え、バッチサイズと学習率を上書き:

```bash
uv run python src/run_single.py --task netqos --epochs 3 --batch_size 32 --lr 5e-4
```

トレーニング完了後に学習済み重みを保存したい場合は、保存フラグを指定します（拡張子は `.pt` もしくは `.pth` を使用してください）。

```bash
uv run python src/run_single.py --task disaster --epochs 1 --save-model --output-path checkpoints/disaster.pth
```

### フェデレーテッドシミュレーション

1. サーバーを起動 (ターミナル 1):

   ```bash
   uv run python src/run_fed_server.py --port 8080 --rounds 3
   ```

2. クライアントを起動 (ターミナル 2):

   ```bash
   uv run python src/run_fed_client.py \
       --config configs/task_disaster.yaml \
       --server localhost:8080
   ```

   `configs/task_disaster.yaml` や `configs/task_netqos.yaml` を編集して、セマンティック圧縮モードや最適化設定、データセットサイズを変更できます。

### セマンティック圧縮の探求

- `semantic/encoder` を調整して量子化器やスパース化手法を試作。
- `semantic/rate_controller.py` を拡張し、ラウンド間のビットレートスケジュールを制御。
- `transport/` を活用して Flower 以外の通信バックエンドやカスタムメッセージ形式を検証。

## 開発ワークフロー

- テストを追加したら `uv run pytest` で実行。
- タスクモジュール内では再現性確保のため乱数シードを固定します。このリポジトリの合成データセットはシード指定時に決定的になります。
- 新しいタスクを追加する際は `task/` 以下にサブフォルダを作成し、`model.py` / `preprocess.py` / `metrics.py` を実装して `task/__init__.py` に登録します。
- コンフィグの変更はコードと同時にコミットし、実行可能なプリセットは README や YAML コメントで説明します。

## ライセンス

MIT License – 詳細は `../LICENSE` を参照してください。
