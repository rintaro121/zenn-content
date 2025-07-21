---
title: "FastAPI+RedsisでつくるレコメンドAPI"
emoji: "🚀"
type: "tech"
topics: ["推薦システム", "FastAPI", "Redis"]
published: false
---

# はじめに
業務でRedisにふれる機会があり、勉強がてらFastAPI＆Redisでレコメンド用のAPIを作ってみました。  
使用するデータセットはいつもの如くMovieLensです。


プロジェクトの構成はこんな感じです。  
レコメンドモデルは`services.py`で実装してますが、今回はカウントベースの簡単なロジックになってます。
```
movielens-recommend-api/
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI router setup and entry point
│   ├── crud.py         # Functions for CRUD operations with Redis
│   ├── schemas.py      # Pydantic models (request/response schema definitions)
│   └── services.py     # Recommendation logic
├── data/               # Place the MovieLens dataset here
│   └── ml-latest-small/
│       ├── movies.csv
│       └── ratings.csv
├── scripts/
│   └── load_data.py    # Data loading script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
実装の詳細は[こちら](https://github.com/rintaro121/fastapi-movie-recommender)のレポジトリをご覧ください！

# Redisについて
今回扱うRedisについて簡単に紹介します。  

RedisとはIn-memory型のデータストアです。In-memory型である（ディスクではなくメモリ上でデータを保持する）ことから、アクセスが非常に高速という特徴を持ってます。

文字列やハッシュ、セットなど色々なデータ構造をサポートしています。

![supported-data-structure](/images/fastapi-movie-recommender/redis-image.png)
*[Redis Explained](https://architecturenotes.co/p/redis)より引用*


[`redis-py`](https://github.com/redis/redis-py)という公式のPythonクライアントも提供されており、Pythonアプリケーションから簡単にredisを操作可能です。
```python
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

r.json().set('user:1', '$', {
    'name': 'Alice',
    'emails': ['alice@example.com', 'alice@work.com'],
    'address': {'city': 'NYC', 'zip': '10001'}
})

print(r.json().get('user:1', '$'))            # [{'name': 'Alice', 'emails': [...], 'address': {...}}]
print(r.json().get('user:1', '$.name'))       # ['Alice']
print(r.json().get('user:1', '$.emails[0]'))  # ['alice@example.com']
print(r.json().get('user:1', '$.address.zip'))# ['10001']
```

今回はレコメンドアイテムを格納するデータストアとしてredisを使用しています。また、レコメンドアイテムを返却時にもredisにアクセスしています。

# レコメンドAPIの実装
## 事前準備
MovieLens Smallデータセットのダウンロード
1. [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/) から`ml-latest-small.zip`をダウンロード
2. 解凍後`ml-latest-small`フォルダを、プロジェクトルートに作成した`data`フォルダの中に配置

## STEP 1: プロジェクトの初期設定
上記のディレクトリ構造に従って、フォルダと空ファイルを作成してください。  
`requirements.txt`には、今回使うライブラリを記述します。
```text:requirements.txt
fastapi
uvicorn
redis
polars
tqdm
```

## STEP 2: データローディングスクリプトの作成
ここでは、MovieLensのCSVデータをRedisに格納するためのスクリプトを作成します。
このスクリプトはAPIサーバーの起動前に一度だけ実行されるようになっています。

```python:scripts/load_data.py
import os

import polars as pl
import redis
from tqdm import tqdm

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379

DATA_DIR = "/app/data/ml-latest-small"
MOVIELENS_CSV = os.path.join(DATA_DIR, "movies.csv")
RATING_CSV = os.path.join(DATA_DIR, "ratings.csv")


def load_data_to_redis():
    """
    MovieLensのデータを読み込み、Redisに保存する
    - 映画情報: movie:{movie_id} (Hash)
    - 映画を評価したユーザー: movie_raters:{movie_id} (Set)
    - ユーザーが評価した映画: user_ratings:{user_id} (Set)
    """
    print("Connecting to Redis...")
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        redis_client.ping()
        print("Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"Cloud not connect to Redis: {e}")
        return

    # 既存のデータをクリア
    print("Flushing all data from Redis...")
    redis_client.flushall()

    # movies.csvの読み込みと保存
    print("Loading movies.csv...")
    movies_df = pl.read_csv(MOVIELENS_CSV)
    for row in tqdm(movies_df.iter_rows(named=True), total=len(movies_df)):
        movie_id = str(row["movieId"])
        redis_client.hset(
            f"movie:{movie_id}",
            mapping={
                "title": row["title"],
                "genres": row["genres"],
            },
        )

    # ratings.csvの読み込みと保存
    print("Loading rating.csv...")
    rating_csv = pl.read_csv(RATING_CSV)
    pipe = redis_client.pipeline()

    for row in tqdm(rating_csv.iter_rows(named=True), total=len(rating_csv)):
        user_id = str(row["userId"])
        movie_id = str(row["movieId"])
        # ユーザが評価した映画のリスト
        pipe.sadd(f"user_ratings:{user_id}", movie_id)
        # 映画を評価したユーザーリスト
        pipe.sadd(f"movie_raters:{movie_id}", user_id)

    print("Excuting Redis pipeline for ratings...")
    pipe.execute()
    print("Data loading complete!")


if __name__ == "__main__":
    load_data_to_redis()
```

ここでのredisの処理について少し解説します。  

大まかに以下の流れになっています。
1. データソースとして`movies.csv`（映画情報）と`ratings.csv`ユーザーの評価情報）の2つのファイルをロード
2. スクリプト実行時に`redis_client.flushall()`を呼び出し、Redis内の既存データをすべて削除
3. Redisにデータを格納

#### 映画情報の保存（Hash）
映画情報の保存は`Hash`というデータ型を用いて、個々の映画のタイトルやジャンルを格納しています。  
Hashは、1つのキーに対して複数のフィールドと値のペアを保存できるデータ型で、`movie:1`というキーの中に`title`フィールドと`genres`フィールドを持つ、といった構造になります（つまり映画IDを指定するだけで、その映画の全ての情報を取得可能です）。
```python
redis_client.hset(
    f"movie:{movie_id}",
    mapping={
        "title": row["title"],
        "genres": row["genres"],
    },
)
```


#### 評価データの保存 (Set)
ここでは、誰が何をみたか、何を誰がみたか、という情報の登録をします。

例えば、`user_ratings`では、「ユーザーID:1が見た全ての映画ID」のような情報を格納し、`movie_raters`では、「映画ID:2を見た全てのユーザーID」を格納します。
```python
# ユーザーが評価した映画リスト
pipe.sadd(f"user_ratings:{user_id}", movie_id)
# 映画を評価したユーザーリスト
pipe.sadd(f"movie_raters:{movie_id}", user_id)
```
#### Redisに対する一括書き込み (Pipeline)
ここでは、`Pipeline`という機能を用いて、大量のコマンドを効率的に実行しています。  

![pipeline](/images/fastapi-movie-recommender/pipeline.png)
*pipelineの例 ([Redisパイプラインの概要、原則、例](https://www.alibabacloud.com/help/ja/redis/use-cases/use-pipelining-to-batch-issue-commands)より引用)*

`ratings.csv`には約10万件のデータがあり、1件ずつRedisにコマンドを送ると10万往復のネットワーク通信が発生するので非常に時間がかかります。  
Pipelineを用いることで、実行したい複数のコマンドを一旦クライアント側で溜めておき、`execute()`が呼ばれたタイミングで一括してサーバーに送信するようにします。  
これによりネットワーク通信を1往復に抑え、データ投入が爆速になります🚀
```python
# パイプラインによる書き込みの効率化
pipe = redis_client.pipeline()
for _, row in tqdm(ratings_df.iterrows(), ...):
    # ...
    pipe.sadd(...)
    pipe.sadd(...)

pipe.execute()
```

## STEP 3: アプリケーションのコンポーネント作成
appディレクトリ内の各ファイルに、APIの機能を実装していきます。

**1. スキーマ定義 (`app/schemas.py`)**  
APIのレスポンスとして使用するデータ構造をPydanticモデルで定義します。
```python:app/schemas.py
from typing import List

from pydantic import BaseModel, Field


class Movie(BaseModel):
    movie_id: str = Field(..., description="The ID of the movie.")
    title: str = Field(..., description="The title of the movie.")
    genres: str = Field(..., description="The genres of the movie.")


class Recommendation(BaseModel):
    movie: Movie
    score: int = Field(..., description="A score representing the recommendation strength, based on co-occurrence count." )


class RecommendationResponse(BaseModel):
    base_movie_id: str = Field(..., description="The ID of the base movie for recommendations.")
    recommendations: List[Recommendation]
```

**2. Redis操作のためのスクリプト (`app/crud.py`)**
Redisとのやり取りを行う関数をここにまとめます。

```python:app/curd.py
from typing import Set, Dict, Optional
import redis

def get_movie_details(redis_client: redis.Redis, movie_id: str) -> Optional[Dict[str, str]]:
    """指定されたmovie_idの映画詳細をRedisから取得する"""
    return redis_client.hgetall(f"movie:{movie_id}")

def get_users_who_rated_movie(redis_client: redis.Redis, movie_id: str) -> Set[str]:
    """指定された映画を評価した全ユーザーのIDをRedisから取得する"""
    return redis_client.smembers(f"movie_raters:{movie_id}")

def get_movies_rated_by_users(redis_client: redis.Redis, user_ids: Set[str]) -> list:
    """複数のユーザーが評価した映画のリストをまとめて取得する"""
    if not user_ids:
        return []
    
    pipeline = redis_client.pipeline()
    for user_id in user_ids:
        pipeline.smembers(f"user_ratings:{user_id}")
    return pipeline.execute()
```



**3. レコメンドのモデル (`app/services.py`)**  
推薦アイテムを作成する処理です。
`crud.py`の関数を利用しています。

レコメンドのロジックはシンプルなものになっており、ある映画IDに対して、同一映画を見たユーザの映画を全列挙していき、その回数をカウントするというものになっています。

Pythonの[Counter](https://docs.python.org/ja/3.13/library/collections.html#collections.Counter)クラスを使うと、この辺の処理が簡単に記述できて便利です。

```python:app/services.py
from collections import Counter
import redis
from . import crud, schemas

def generate_recommendations(redis_client: redis.Redis, movie_id: str, limit: int) -> list:
    """
    レコメンデーションを生成するメインロジック
    """
    # 1. 指定された映画を評価した全ユーザーを取得
    raters = crud.get_users_who_rated_movie(redis_client, movie_id)
    if not raters:
        return []

    # 2. それらのユーザーが評価した他の映画をすべて取得
    other_movies_lists = crud.get_movies_rated_by_users(redis_client, raters)

    # 3. 全映画を集計し、共起回数をカウント
    item_counter = Counter()
    for movies_list in other_movies_lists:
        item_counter.update(movies_list)

    # 4. 元の映画を推薦リストから除外
    item_counter.pop(movie_id, None)

    # 5. 共起回数が多い順にソートし、上位N件の映画詳細を取得
    recommended_items = []
    
    # パイプラインで映画詳細をまとめて取得
    pipeline = redis_client.pipeline()
    top_movies_ids = [item[0] for item in item_counter.most_common(limit * 2)] # 多めに取得
    for m_id in top_movies_ids:
       pipeline.hgetall(f"movie:{m_id}")
    movie_details_list = pipeline.execute()

    for i, common_movie_id in enumerate(top_movies_ids):
        if len(recommended_items) >= limit:
            break
        
        movie_details = movie_details_list[i]
        if movie_details:
            movie = schemas.Movie(
                movie_id=common_movie_id,
                title=movie_details.get("title", "N/A"),
                genres=movie_details.get("genres", "N/A"),
            )
            recommendation = schemas.Recommendation(
                movie=movie,
                score=item_counter[common_movie_id]
            )
            recommended_items.append(recommendation)
            
    return recommended_items
```



**4. APIのエントリーポイント (`app/main.py`)**

FastAPIアプリケーションをセットアップし、エンドポイントを定義していきます。

```python:app/main.py
import os
import redis
from fastapi import FastAPI, HTTPException, Depends

from . import services, schemas

# --- FastAPI App & Redis Connection ---
app = FastAPI(
    title="Movie Recommendation API",
    description="An API for movie recommendations based on the MovieLens dataset.",
    version="1.0.0"
)

# グローバルなRedisクライアントインスタンス
redis_client = None

@app.on_event("startup")
def startup_event():
    """アプリケーション起動時にRedisに接続する"""
    global redis_client
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    print(f"Connecting to Redis at {redis_host}:{redis_port}")
    try:
        redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=True
        )
        redis_client.ping()
        print("Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        redis_client = None

def get_redis_client():
    """DI（依存性注入）用のRedisクライアント取得関数"""
    if redis_client is None or not redis_client.ping():
         raise HTTPException(status_code=503, detail="Redis service is unavailable.")
    return redis_client

@app.on_event("shutdown")
def shutdown_event():
    """アプリケーション終了時にRedis接続を閉じる"""
    if redis_client:
        redis_client.close()
        print("Redis connection closed.")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommendation API"}

@app.get(
    "/recommendations/{movie_id}",
    response_model=schemas.RecommendationResponse,
    summary="Get movie recommendations",
    tags=["recommendations"],
)
def get_movie_recommendations(
    movie_id: str,
    limit: int = 10,
    redis_db: redis.Redis = Depends(get_redis_client)
):
    """
    指定された映画IDに基づいて、おすすめの映画リストを返します。
    - **movie_id**: 推薦の基準となる映画のID (例: 1 は 'Toy Story')
    - **limit**: 返される推薦の最大数
    """
    recommendations = services.generate_recommendations(
        redis_client=redis_db, movie_id=movie_id, limit=limit
    )
    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for movie_id: {movie_id}. It might not exist or have enough ratings."
        )
    return schemas.RecommendationResponse(recommendations=recommendations)
```

## STEP 4: Docker関連のスクリプト
アプリケーションを実行するコンテナのため、`Dockerfile`と`docker-compose.yml`を作成します。

```bash:Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./app /app/app
```

```yaml:docker-compose.yml
version: "3.8"

services:
  redis:
    image: "redis:alpine"
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  loader:
    build: .
    command: python scripts/load_data.py
    volumes:
      - ./data:/app/data
      - ./scripts:/app/scripts
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
  api:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0 --port 80 --reload
    ports:
      - "8000:80"
    volumes:
      - ./app:/app/app
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
volumes:
  redis_data:
```

## STEP 5: 実行と動作確認
一つづつサービスを立ち上げていきます。
（`docker-compose up`で一気に処理することも可能です。）
1. **Redisの起動**: まずはRedisコンテナを起動します。
```bash
docker-compose up redis
```
2. **データのロード**: 次にloaderサービスを実行して、MovieLensデータをRedisに格納します。このサービスは処理が完了すると終了します
```bash
docker-compose run loader
```
ターミナルにData loading complete!と表示されれば成功です。  
3. **APIサーバーの起動**: 最後にAPIサーバーを起動します。
```bash
docker-compose up --build api
```
4. 動作確認
   1. ブラウザで http://localhost:8000/docs にアクセスします。
   2. `/recommendations/{movie_id}` を展開し、"Try it out" をクリックします。
   3. `movie_id`に1(Toy StoryのID) を入力して"Execute"ボタンを押します。
   4. `Forrest Gump (1994)`, `Pulp Fiction (1994)`, `Jurassic Park (1993)` といった、Toy Storyと一緒に評価されることが多い映画がスコアと共に返ってくるはずです。


こちらのようにfastapiのdocsからAPIの返却結果を確認可能です！
![fastapiのdocs](/images/fastapi-movie-recommender/docs.png)