---
title: "推薦システムの新たなパラダイム Generative Recommendation"
emoji: "🧬"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [Pytorch, Deep Learning, 推薦システム]
published: false
---

近年Meta[^1][^2]、ByteDance[^3]、Kuaishou[^4]、Google[^5]、Alibaba[^6]といったビッグテックの企業から、Generative Recommendationに関する手法が数多く発表されています。
これは、推薦システムの問題設定を新しい形で定義するもので、RecSys、CIKM、WWWなどの学会でもチュートリアルが開かれたりとホットなトピックになっています。

この記事では、Generative Recommendationの概要からSemantic IDなどの構成要素の解説、そしてMovielensを使った実装まで紹介していこうと思います。

詳細な実装はこちらをご覧ください。
https://github.com/rintaro121/semantic-id-movielens

# 前提知識・背景
Generative Recommendation（GR）とは、「**言語モデルが次の単語を生成する要領で、次にユーザーがクリックするアイテムを、自己回帰的に生成する推薦**」のことを指します。
![generative-recommendation-illustration](/images/generative-recommendation/generative_recommendation_illustration.png)


従来の推薦は、ユーザ情報やアイテム情報、コンテキストの情報など多くの入力を、人手による特徴量設計を行ったりモデル設計を行い処理する形が主流でした。
その一方で、GRはユーザーの行動履歴を入力とし、推薦のタスクを「次のアイテムIDを生成する問題」として定式化する、という違いがあります。

## Generative Recommendationのメリット
GRの大きなメリットの一つは、推薦タスクをスケーリング則が効く形に落とし込めることです。

Metaが発表したGRに関する初期の研究[^1]では、従来の手法（DLRM）では確認できなかったScaling則が、GRベースの提案手法（HSTU）では確認できることを示しています。
![scaling](/images/generative-recommendation/scaling.png)

このスケーリングはGPT-3やLLaMa 2のパラメータ数の規模まで確認できていること、Metaが提供するプラットフォームでのオンラインA/Bテストでも主要メトリクスを改善したこと、など実環境でもGRが機能することを示しました。（この研究を皮切りに多くの企業でGRの研究が盛んになった印象があります。）

## Semantic IDを用いたGenerative Recommendation
先述の通り、GRでは次の単語を生成する形でアイテムの予測を行います。
しかし、推薦モデルが扱うアイテム数は言語モデルが扱う単語の数に比べて大幅に多い、という問題があります。
例えば、Netflixの事例ではそれぞれのモデルが扱うトークン数について、以下のような言及があります[^7]。
> *For example, in the Netflix use case, we need a 40 times larger catalog size than GPT 3’s, and our model digested 2 trillion tokens periodically compared to 500B tokens used for GPT 3.*
> 訳: *例えば、Netflixのユースケースでは、GPT-3の40倍の規模のカタログが必要であり、当社のモデルは定期的に2兆トークンを処理しましたが、GPT-3では5,000億トークンが使用されていました。*

更に、多くのプラットフォームでは新規アイテムの追加によるコールドスタート問題のため、単語のように固定IDをアイテムに割り当てることは困難です。

これらに対処するための方法が、GoogleのTIGER[^5]（NeurIPS'23）で導入された、Semantic IDという概念です。アイテムが持つコンテンツ情報（タイトルや説明などのテキスト情報やサムネイルなどの画像情報）をエンコーダで埋め込み表現にし、それを量子化することで複数のID（トークン）の組み合わせに変換します。

![rqvae](/images/generative-recommendation/rqvae.png)

最初のトークンが大まかなカテゴリ、次のトークンがより細かいカテゴリ...というように、似たアイテムほど ID の前半を共有しやすくなることが期待されます。
コンテンツ情報さえあればこのトークンは生成可能なので、新規アイテムでも既存の似たアイテムであればIDが重なることが期待され、コールドスタートへの対応も可能になります。

このSemantic IDは単語と同様に扱うことが可能なので、Transformerなどのアーキテクチャを用いて、自己回帰型の問題設定に落とし込むことが可能となります。
![tiger](/images/generative-recommendation/tiger.png)

# Generative Recommendationの実装
ここでは、MovieLensという推薦システム向けのオープンデータセットを題材として、Generative Recommendationの実装を行います。

MovieLensは映画に関するデータセットとなっており、ユーザの映画に対する評価とそのタイムスタンプをインタラクションデータとして集めたものになっています。
インタラクションのデータだけでなく、映画に関する情報（タイトルやジャンル）、ユーザに関する情報（年齢や職業）などもメタデータとして利用可能です。


今回実装するモデルのフローは大きく以下の3つです。
1. コンテンツ情報のエンコード
2. Semantic IDの生成
3. Generative Recommendationによるアイテム予測

## コンテンツ情報のエンコード
まず、映画のテキスト情報を埋め込み表現に変換します。

今回は`SentenceTransformer`が提供しているモデルを用いて、映画のタイトルとジャンル情報をエンコードします。
利用モデルは、`all-MiniLM-L6-v2`で、入力テキストを384次元のベクトルに変換しています。

例えば、トイストーリーであれば、MovieLensのデータのタイトル・ジャンルから、`Toy Story (1995), Genre: Animation|Children's|Comedy`というテキストを作成し、それをそのままベクトル表現に変換します。

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def encode_text(model_name, input_text_list):
    model = SentenceTransformer(model_name)
    return model.encode(input_text_list)

input_text_list = movie_df["input_text"].to_list()
embeddings = encode_text(MODEL_NAME, input_text_list)
# Input: "Toy Story (1995), Genre: Animation|Children's|Comedy"
# Output: [-0.043559, 0.004969, … 0.074716]
```

## RQ-VAEによるSemantic IDの作成
Semantic IDは、アイテムの情報をトークンの系列で表す表現方法です。
従来のレコメンドでは、アイテムIDから`nn.Embedding()`などで直接表現を作成していましたが、Semantic IDでは、`<sid_57><sid_247><sid_74>`のようなトークンの系列として一つのアイテムを表現します。
学習の過程で似通ったアイテムは、類似するPrefixを持つことが想定され、適切に学習されたSemantic IDは木構造のような形をとることが期待されます。

![semantic-id](/images/generative-recommendation/semantic_id.png)


Itemの情報からSemantic IDを作成するための方法としてはいくつかありますが、ここではRQ-VAE[^8]を利用します。
RQ-VAEでは、アイテムのメタデータをアイテム表現にエンコードしたのちに、そのアイテム表現に近いベクトルをcodebookの中から選択しIDを割り当てていく、という量子化を行います。割り当てられた表現のIDがSemantic IDと対応します。この処理は階層的に行われ、アイテム表現と割り当てた表現の残差に対してまた新たなCodebookのIDに割り当てる、という処理を繰り返します。


RQ-VAEの損失は以下の形で表されます。
$$
L = L_{recon} + L_{rqvae}
$$
$L_{recon}$は、入力の表現$x$と最終的に生成された量子化表現$\hat{x}$の差分がどれくらいかを測るものです。モデルの入力と出力の間の二乗誤差から計算することができます。
この損失はAutoEncoderやVAEなどでも使われるもので、一般にreconstruction loss（再構築誤差）と呼ばれます。
$$
L_{recon} = ||x-\hat{x}||^2
$$

$L_{rqvae}$は、codebook内のベクトルがエンコーダによって生成された表現（あるいは残差）とどれくらい一致するかを測るものです。

$$
L_{\text{rqvae}} := \sum_{i=0}^{m-1} \left[ \|\text{sg}[r_i] - e_{c_i}\|^2 + \beta\|r_i - \text{sg}[e_{c_i}]\|^2 \right]
$$
1つ目の項（$|\text{sg}[r_i] - e_{c_i}|^2)$）は、codebook lossと呼ばれ、エンコーダの残差とcodebookのベクトルを計算します。残差（$r_i$）にcodebookの表現を近づけることを目的とするため、残差（$r_i$）に対しては勾配を止め、codebookのベクトルのみが更新対象になります。

2つ目の項（$\beta|r_i - \text{sg}[e_{c_i}]|^2$）はcommitment lossと呼ばれ、ここではエンコーダが学習対象です。
1つ目の項と同じ値を計算していますが、こちらはcodebookのベクトルに対して勾配を止めます。
結果として、エンコーダの出力がcodebookのベクトルに近づくこと（よりコミットすること）を期待しています。
```python
codebook = self.codebooks[i]

distance = (
    residual.pow(2).sum(dim=1, keepdim=True) 
    - 2 * residual @ codebook.t() 
    + codebook.pow(2).sum(dim=1).unsqueeze(0)
)

codes = torch.argmin(distance, dim=1)
selected = codebook[codes]
quantized_sum = quantized_sum + selected

codebook_loss = F.mse_loss(selected, residual.detach())
commitment_loss = F.mse_loss(residual, selected.detach())
```
詳細なモデルや学習コードはこちらをご覧ください。

また、ここで計算したSemantic IDは全アイテムに対して一意ではないことに注意が必要です。
二つのアイテムが同じSemantic IDを共有した場合には、RQ-VAEのモデルを学習後に、追加のトークンを足す、などの方法でIDの衝突を避けることが可能です。

## Generative Recommendation
### Semantic IDの表現方法について
ここまでで、各アイテムに対してSemantic IDを割り当てることができました。
このSemantic IDを利用して、Generative Recommendationのモデルを学習します。
通常の推薦モデルでは、ユーザーの履歴から次に見るIDを直接予測します。一方で、Generative RecommendationではSemantic IDのtoken列が生成の対象になります。

例えば、ある映画が以下の Semantic ID を持っているとします。
```
[1, 1, 88, 0]
```
ただし、このままでは 1つ目のIDと2つ目のIDが両方１なので、区別がつきません。
そこで実装では、階層ごとに offset を足して、すべてのSemantic IDを一意のIDに変換しています。
```python
# special tokens
self.pad_token_id = 0         # padding用
self.bos_token_id = 1         # 系列開始を表すtoken
self.target_bos_token_id = 2  # itemの生成開始位置を表すtoken
self.eos_token_id = 3         # 系列終了を表すtoken

# Semantic ID tokens
offsets = []
offset = 4  # 0〜3はspecial tokenで利用してるので4から開始

# sid_vocab_sizes = [128, 128, 128, 93]
for vocab_size in sid_vocab_sizes:
    offsets.append(offset)
    offset += vocab_size

# offsets = [4, 132, 260, 388]
```

最終的な学習データは以下の形式になります。
実装では、履歴を`prompt_ids`、予測対象を`target_ids`として扱っています。

```python
prompt_ids = tokenizer.make_prompt_tokens(sample.history_movie_ids)
target_ids = tokenizer.make_target_tokens(sample.target_movie_id)

input_ids = torch.cat([prompt_ids, target_ids])
labels = torch.cat([
    torch.full_like(prompt_ids, -100),
    target_ids,
])
```
履歴側の`labels`を-100としており、損失を計算する際の`F.cross_entropy`で`ignore_index=-100`を指定することで、履歴のtokenに関する損失の計算を省くことができます。

## Decoder-only Transformer
今回のGRは簡単のためにDecoder-only Transformerとして実装しています。
アーキテクチャはシンプルになっており、token embeddingとposition embeddingを足し合わせたのちに、causal maskをかけてTransformerに入力しています。

```python
x = self.token_embedding(input_ids) + self.position_embedding(positions)

causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
    diagonal=1,
)

hidden = self.causal_transformer(
    x,
    mask=causal_mask,
    src_key_padding_mask=~attention_mask.bool(),
)
logits = self.lm_head(self.norm(hidden))
```
`causal_mask`を適用することで、各tokenは未来のtokenを見ることができないようになっています。
そのため、モデルは以下のように、過去から未来の方向へSemantic IDを生成する形で学習されます。
```text
history -> sid_0
history, sid_0 -> sid_1
history, sid_0, sid_1 -> sid_2
history, sid_0, sid_1, sid_2 -> sid_3
```
lossの計算については通常の言語モデルと同じように、1 tokenずつずらして計算しています。
```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]

loss = F.cross_entropy(
    shift_logits.view(-1, self.vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

## 最終的なアイテム生成
Transformerの出力とアイテムを対応づける方法は幾つ考えられますが、ここではgreedyにdecodingを行う方法を考えます。

```python
next_logits = last_token_logits(outputs["logits"], attention_mask)
next_token = next_logits.argmax(dim=1)
```

greedyな生成では、各stepで最も確率が高いtokenを1つピックアップしていく、というシンプルな方法です。

ただし、greedy decodingでは、出力される推薦は1件だけです。
他の方法としてbeam searchにより、複数の候補取得を行うことが考えられます。
ここではbeam searchを使った候補生成については触れませんが、詳しい実装については、snap-researchによるGRのフレームワーク[^9]の[こちら](https://github.com/snap-research/GRID/blob/main/src/models/modules/semantic_id/tiger_generation_model.py#L253)の実装が参考になるかと思います。

最後に手元で学習したモデルの生成結果についていくつか確認してみます。

こちらのケースでは、実際にユーザがインタラクションしたのは、ノートルダムの鐘（Hunchback of Notre Dame）です。当てることはできていませんが、ディズニー映画などを推薦することができています。
```plaintext
==== case 2 user_id=2 ==== 
[history] 
- Three Musketeers, The (1993) 
- Circle of Friends (1995) 
- Walk in the Clouds, A (1995)
- Billy Madison (1995) 
- Snow White and the Seven Dwarfs (1937) 
- Robin Hood: Men in Tights (1993) 
- Aristocats, The (1970) - Jungle Book, The (1994)

[target] Hunchback of Notre Dame, The (1996)

[top-5]
   1. Toy Story (1995) score=-4.034 sid=(103, 133, 348, 388)
   2. Forrest Gump (1994) score=-4.084 sid=(59, 202, 368, 388)
   3. Beauty and the Beast (1991) score=-4.415 sid=(103, 252, 271, 388)
   4. Willy Wonka & the Chocolate Factory (1971) score=-4.564 sid=(112, 237, 355, 388)
   5. Babe (1995) score=-4.784 sid=(112, 146, 348, 388)
```

また、以下のケースではStar Wars: Episode Iが正解で正しいエピソードは当てられていませんが、スターウォーズ作品を推薦することができています。
```plaintext
==== case 5 user_id=6 ====
[history]
  - Crow: City of Angels, The (1996)
  - Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002)
  - Speed 2: Cruise Control (1997)
  - Lara Croft: Tomb Raider (2001)
  - Rambo: First Blood Part II (1985)
  - Predator 2 (1990)
  - Dances with Wolves (1990)
  - Indiana Jones and the Temple of Doom (1984)
[target] Star Wars: Episode I - The Phantom Menace (1999)
[top-5]
   1. Star Wars: Episode VI - Return of the Jedi (1983) score=-4.608 sid=(124, 158, 318, 388)
   2. Star Wars: Episode IV - A New Hope (1977) score=-4.727 sid=(7, 232, 339, 388)
   3. Aliens (1986) score=-4.729 sid=(124, 238, 339, 388)
   4. Die Hard (1988) score=-4.764 sid=(8, 236, 273, 388)
   5. Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) score=-4.769 sid=(7, 214, 313, 388)
```

# おわりに
今回は、MovieLensを題材に、Semantic IDの作成からDecoder-only TransformerによるGenerative Recommendationの実装までを紹介しました。

実験結果を見ると、正解アイテムそのものを常に当てられるわけではないものの、ユーザーの履歴に関連する映画をある程度推薦できていることが確認できます。

今回はシンプルなDecoder-onlyモデルとgreedy decodingを用いましたが、Encoder-Decoder型のモデルに変更することや、beam searchによって複数候補を生成することで、さらなる精度改善が期待できます。


### References
[^1]: [Zhai, Jiaqi, et al. "Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations." _arXiv preprint arXiv:2402.17152_ (2024).](https://arxiv.org/abs/2402.17152)

[^2]: [Zhang, Buyun, et al. "Wukong: Towards a scaling law for large-scale recommendation." _arXiv preprint arXiv:2403.02545_ (2024).](https://arxiv.org/abs/2403.02545)

[^3]: [Lan, Yu-Ting, et al. "Next-User Retrieval: Enhancing Cold-Start Recommendations via Generative Next-User Modeling." _arXiv preprint arXiv:2506.15267_ (2025).](https://arxiv.org/abs/2506.15267)

[^4]:[ Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." _arXiv preprint arXiv:2502.18965_ (2025).](https://arxiv.org/abs/2502.18965)

[^5]: [Rajput, Shashank, et al. "Recommender systems with generative retrieval." _Advances in Neural Information Processing Systems_ 36 (2023): 10299-10315.](https://arxiv.org/abs/2305.05065)

[^6]: [Meng, Yue, et al. "A generative re-ranking model for list-level multi-objective optimization at taobao." _Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval_. 2025.](https://dl.acm.org/doi/10.1145/3726302.3731935)

[^7]: [Towards Generalizable and Efficient Large-Scale Generative Recommenders](https://netflixtechblog.medium.com/towards-generalizable-and-efficient-large-scale-generative-recommenders-a7db648aa257)

[^8]: [Lee, Doyup, et al. "Autoregressive image generation using residual quantization." _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_. 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Autoregressive_Image_Generation_Using_Residual_Quantization_CVPR_2022_paper.pdf)

[^9]: [Generative Recommendation with Semantic IDs (GRID)](https://github.com/snap-research/GRID)