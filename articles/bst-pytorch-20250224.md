---
title: "Alibabaの推薦システムBehavior Sequence Transformer"
emoji: "🎴"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["推薦システム", "Pytorch", "深層学習"]
published: false
---

# はじめに
この記事では、Alibabaが提案した「**Behavior Sequence Transformer (BST)**」というモデルについて解説し、その実装をMovieLensデータセットに適用した例を紹介します。

以下は実装です。  
Google colab：  
https://colab.research.google.com/drive/1gv3jAHTLgVChAlw7JyRFM5YTGrCX1PIF?usp=sharing

Repository：  
https://github.com/rintaro121/behavior-sequence-transformer-pytorch

# **Behavior Sequence Transformer (BST)**
URL：[Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874)

## 論文概要・背景
BSTの発表はKDD’2019と若干古いですが、業界に先駆けてTransformerベースの手法を実稼働環境で検証し、A/Bテスト結果も報告したという内容になっており、先行事例として重要な位置付けの論文です。

A/Bテストでは、AlibabaのE-commerceプラットフォームで検証を行い、従来の手法（WDLやDIN）に比べてBSTが大幅な精度向上をもたらした、という結果が報告されています。  
Alibabaが持つ、数億規模のユーザ/アイテムに対しても運用可能、というのもポイントです。

## 手法
BSTは名前の通り、Transformerを利用したモデルになっています。

入力は以下の三つです  
（１）ユーザが過去クリックしたアイテムの情報  
（２）ユーザの情報  
（３）候補アイテムの情報

アイテムの情報は、アイテムID、カテゴリ情報、タグ情報などから構成され、ユーザの情報は、性別、年齢、居住地などから構成されます。  
（１）と（３）についてはTransformerによる特徴抽出を行い、（２）については埋め込み層によって特徴を得ます。  
２つの連結（concat）してMLPで処理しクリック確率を予測する、という流れで処理されます。

![bstのアーキテクチャ](/images/bst/bst.png)

# BSTの実装
次にMovieLensデータセットを使った簡単な実装について解説します。
## データの概要
MovieLensは、ユーザが映画に付けた評価データから成るデータセットになっており、以下のような情報を含んでいます。（詳しい定義は[MovieLensのREADME](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)をご覧ください。）

**User Table**
| ユーザID | 性別 | 年齢 | 職業 |
| --- | --- | --- | --- |
| 1 | 女性 | 18 | 10 |
| 2 | 男性 | 56 | 16 |

**Movie Table**
| 映画ID | タイトル | ジャンル |
| --- | --- | --- |
| 1 | Toy Story (1995) | Animation|Children's|Comedy |
| 2 | Jumanji (1995) | Adventure|Children's|Fantasy |

**Rating Table**
| ユーザID | 映画ID | 評価値 | タイムスタンプ |
| --- | --- | --- | --- |
| 1 | 1193 | 5 | 978300760 |
| 1 | 661 | 3 | 978302109 |

## **データの準備**
### 行動シーケンスの作成
ユーザの行動シーケンスは、評価をつけた時間のタイムスタンプを元に作成してます。

| ユーザID | 評価をつけた映画IDの系列 |
| --- | --- |
| 1 | [1193, 661, …] |
| 2 | [1180, 1192,…] |

これは、ユーザIDでgroupbyした後、timestampでソートすることで作成可能です。

```python
ratings = pd.read_csv(path_to_ratings_data)
ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

ratings_data = pd.DataFrame(
    data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)),
        "ratings": list(ratings_group.rating.apply(list)),
        "timestamps": list(ratings_group.unix_timestamp.apply(list)),
    }
)
```

しかし、これだとユーザ数分のシーケンスしか作れず、ユーザ数も6000程度とそこまで多くないので以下のようにData Augmentationを行っています。

Data Augmentation前

| ユーザID | 評価をつけた映画IDの系列 |
| --- | --- |
| 1 | [1,2,3,4,5,6,7,8,9,10] |
| 2 | [11,12,13,14,15,16,17,18,19,110] |

Data Augmentation後

| ユーザID | 評価をつけた映画IDの系列 |
| --- | --- |
| 1 | [1,2,3] |
| 1 | [4,5,6] |
| 1 | [7,8,9] |
| 2 | [11,12,13] |
| 2 | [14,15,16] |
| 2 | [17,18,19] |

このように、シーケンスを一定の長さ（（ここでは３））で区切ることで、データ数を増やしています

上記のような変換は、[torch.Tensor.unfold()](https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
)を使うことで簡単に作成できます。

```python
movie_ids_seq = (
    movie_id_history.ravel().unfold(0, sequence_length, window_size).to(torch.int32)
)
```
### Positive Sample・Negative Sampleの作成
BSTで行うタスクは候補アイテム（Target Item）のクリック確率の予測ですが、作成したシーケンスにはユーザが実際にクリックしたアイテムしか含まれていません。

なのでクリックしていないアイテムも負例として追加してあげる必要があります。

今回は「ユーザの行動系列に含まれてないアイテム」を取得して、負例として元のシーケンスにランダムに付与します。

なので正例・負例は以下のようになります。

| ユーザID | 評価をつけた映画IDの系列 | target Item ID |
| --- | --- | --- |
| 1 | [1,2,3] | 4（正例） |
| 1 | [4,5,6] | 1284（負例） |
| 1 | [7,8,9] | 10（正例） |
| 2 | [11,12,13] | 729（負例） |
| 2 | [14,15,16] | 4281（負例） |
| 2 | [17,18,19] | 20（正例） |

最終的に、ユーザの特徴を紐づけて学習に使用するデータセットを作成します。

つまり、「ユーザの特徴やターゲットアイテムが入力された時、ターゲットアイテムがクリックされたかを予測する」というタスクになります。

| ユーザID | 性別 | 年齢 | 職業 | 評価をつけた映画IDの系列 | target Item ID | 目的変数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 女性 | 18 | 10 | [1,2,3] | 4（正例） | 1 |
| 1 | 女性 | 18 | 10 | [4,5,6] | 1284（負例） | 0 |
| 1 | 女性 | 18 | 10 | [7,8,9] | 10（正例） | 1 |
| 2 | 男性 | 56 | 16 | [11,12,13] | 729（負例） | 0 |
| 2 | 男性 | 56 | 16 | [14,15,16] | 4281（負例） | 0 |
| 2 | 男性 | 56 | 16 | [17,18,19] | 20（正例） | 1 |

## モデル
BSTのクラスは、embedding layer・transformer layer・mlp layerから構成され、
- embedding layer：ユーザ情報、映画情報の埋め込み
- transformer layer：ユーザの行動系列を処理
- mlp layer：ユーザ情報とtransformer layerの出力から、候補アイテムのクリック確率を予測

という役割になっています。

各layerの詳細な実装は[model.py](https://github.com/rintaro121/behavior-sequence-transformer-pytorch/blob/main/src/model.py)をご覧ください。

```python
class BST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(cfg)
        self.transformer_layer = TransformerLayer(cfg)
        self.mlp_layer = MLPLayer(cfg)

    def forward(self, user_feat, seq_item, target_item):
        user_emb, seq_item_emb, target_item_emb = self.embedding_layer(
            user_feat, seq_item, target_item
        )

        transformer_output = self.transformer_layer(seq_item_emb)

        concat_feat = torch.concat(
            [user_emb, transformer_output, target_item_emb],
            dim=-1,
        )

        p_ctr = self.mlp_layer(concat_feat)
        return p_ctr
```

embedding layer、transformer layer、mlp layerの定義はそれそれ以下のようになってます。

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.user_embedding = nn.Embedding(cfg.num_user, cfg.user_emb_dim)
        self.sex_embedding = nn.Embedding(cfg.num_sex, cfg.sex_emb_dim)
        self.age_embedding = nn.Embedding(cfg.num_age_group, cfg.age_group_emb_dim)
        self.occupation_embedding = nn.Embedding(cfg.num_occupation, cfg.occupation_emb_dim)
        self.movie_embedding = nn.Embedding(cfg.num_movie, cfg.movie_emb_dim)

    def forward(self, user_feat, seq_item, target_item):
        # Get user embeddings
        user_id, sex, age, occupation = user_feat
        user_emb = self.user_embedding(user_id)
        sex_emb = self.sex_embedding(sex)
        age_emb = self.age_embedding(age)
        occupation_emb = self.occupation_embedding(occupation)

        # Get movie embedding
        seq_item_emb = self.movie_embedding(seq_item)
        target_item_emb = self.movie_embedding(target_item)

        user_feat = torch.concat([user_emb, sex_emb, age_emb, occupation_emb], dim=-1)

        return user_feat, seq_item_emb, target_item_emb


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pe = PositionalEncoding(
            d_model=cfg.d_model,
            dropout=cfg.dropout_rate,
            max_len=cfg.seq_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2
        )

    def forward(self, movie_seq_emb):
        x = self.pe(movie_seq_emb)
        enc_out = self.transformer_encoder(x)[:, -1, :]
        return enc_out


class MLPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.mlp_dim, cfg.mlp_hidden_1_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.mlp_hidden_1_dim, cfg.mlp_hidden_2_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.mlp_hidden_2_dim, cfg.mlp_hidden_3_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.mlp_hidden_3_dim, cfg.mlp_hidden_4_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.mlp(x)
        return output

```

## 損失
1/0のクリック予測なので損失はBCELossで計算しています。

```python
model = BST()
loss_fn = nn.BCELoss()

for batch in dataloader:
	user_feat, input_seq, target_item, label = batch
	output = model(user_feat, input_seq, target_item)
	output = output.squeeze(-1)
	label = label.type(torch.float32)
	
	loss = loss_fn(output, label)
```

## 結果
train/validationのlossとvalidationのaccuracyを可視化しています。

mlflowを使って、以下のコマンドから確認できます。

```bash
poetry run mlflow ui
```

![experiment results](/images/bst/metrics.png)