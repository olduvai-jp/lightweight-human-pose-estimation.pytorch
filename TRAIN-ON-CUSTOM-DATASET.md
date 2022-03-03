# カスタムデータセットで学習

これはいくらかの :clock1: が掛かります。しかしここに :zero: から :muscle: へのガイドがあります！ この記事の終わりにはあなたは :shipit: :neckbeard: :godmode: を感じますよ, 保証します!

## 序文

### `BODY_PARTS_KPT_IDS` と `BODY_PARTS_PAF_IDS` とは何ですか?

どちらのリストも、キーポイントを人物インスタンスにグループ化することに関連している。1つ目はキーポイントのヒートマップで、各タイプの可能なキーポイント（首、左肩、右肩、左肘など）をすべて局所化し、2つ目は事前に定義されたタイプのキーポイント間の接続を予測します。

ヒートマップから、ネットワークが見つけることができたすべてのキーポイントの座標を抽出することができます。次に、これらのキーポイントを人物にグループ化する必要がある。なぜなら、我々はすでにキーポイントの座標とそのタイプを知っているので、見つかったキーポイントはすべて目的の人物に属しているからです。しかし、画像内に複数の人物が写っている場合、状況は難しくなります。この場合、どうすればよいのでしょうか？例えば、右肩のキーポイントが2つ、首のキーポイントが1つだけ見つかったとします。首のキーポイントは1つでよく、1人の人物のポーズを抽出できる可能性があります。しかし、右肩のキーポイントは2つある。我々は、1つのポーズには最大でも1つの右肩が含まれることを知っています。どちらを選ぶべきだろうか？難しいですが、ネットワークに助けてもらいましょう。

キーポイントを人物のインスタンスにまとめるために、ネットワークは各人物のキーポイント間の接続を予測するように学習します。骨格の骨のようなもの。そのため、どのキーポイント同士がつながっているかがわかれば、最初のキーポイントから順に、他のキーポイントとつながっているかどうかを確認しながら、全ポーズを読み取ることができます。1つ目のキーポイントとその隣のキーポイントの接続が確立したら、隣のキーポイントとそのキーポイントが接続しているキーポイントを探索し、キーポイントをポーズに組み立てていく、ということを繰り返しています。ネットワークが接続を予測すべきキーポイントのインデックスのペアは、まさに [`BODY_PARTS_KPT_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L5-L6) リストで定義されているものです。それでは、ポーズスキームの画像を確認してみましょう。

<p align="center">
  <img src="data/shake_it_off.jpg" />
</p>

`[1, 5]`のペアは、インデックス `1` と `5` のキーポイント間の接続、つまり首と左肩に相当することがわかります。ペア `[14, 16]` は、右目と右耳のキーポイントに対応します。これらのペアは、ネットワークがどのキーポイント間の接続を学習すべきかを知る必要があるため、学習の前に（あなたによって）定義されます。[`BODY_PARTS_PAF_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L7-L8) リストには、対応するキーポイントのペア間の接続をエンコードするネットワーク出力チャネルのインデックスが定義されています。PAF は part affinity field の略で、[original paper](https://arxiv.org/pdf/1611.08050.pdf) にある用語で、キーポイントのペア間の接続を記述しています。

### 接続するキーポイントのペアをどのように選択するか？

全てのポイントを接続するとします。この場合、「`(キーポイント数)*(キーポイント数-1)`」のキーポイントペアを持つことになります（キーポイント自身との接続はインスタンスにグループ化するために不要なのでスキップされます）。仮にキーポイント数が18で全対全接続方式を採用した場合、ネットワークは「`18 * 17 = 306`」個のキーポイント間の接続を学習する必要があります。例えば、右肘と右肩の接続を検出できなかった場合、右肘と首の接続や他のキーポイントとの接続を確認することで、右肘をポーズにグループ化することができるのです。

実際のキーポイントペアの数は、ネットワークの推論速度と精度のトレードオフになります。この研究では、19組のキーポイントのペアを用意しました。しかし、ベストプラクティスとして、残りのキーポイントと接続される特別なルートキーポイントを定義することは、より良い精度を得るために理にかなっています（上述）。通常、オクルージョンが少なく、検出が容易な最も堅牢なキーポイントが、ルートキーポイントの候補として適しています。ルートキーポイントは、グループ化を開始する最初のキーポイントとなる。人物の場合は、通常、首か骨盤（または両方、あるいはそれ以上、トレードオフの関係にあります）です。

### キーポイントペア間の接続は、ネットワークレベルでどのように実装されているのでしょうか？

キーポイントペア間の接続は、キーポイント間の単位ベクトルとして表現されます。 したがって、座標(x<sub>a</sub>, y<sub>a</sub>)の点 `a` と、座標(x<sub>b</sub>, y<sub>b</sub>)の点 `b` からなる単位ベクトル c<sub>ba</sub> は、(x<sub>b</sub>-x<sub>a</sub>, y<sub>b</sub>-y<sub>a</sub>)のように計算し、この長さを正規化したものになります. このベクトルは、ペアのキーポイント間のすべてのピクセルに含まれます。ネットワークは、2つの別々のチャンネルを予測します: 各キーポイントペアの接続ベクトルの `x` 成分と `y` 成分をそれぞれ1つずつ出力します． つまり、19組のキーポイントに対して、ネットワークは「19 * 2 = 38」チャンネルをキーポイント間の接続として予測することになります。 推論時には、キーポイントのペアから特定のタイプのキーポイントをすべて抽出し、それらのキーポイントで形成されるベクトルとネットワークで学習したベクトルを比較します。 ベクトルが一致した場合、これらのキーポイントは接続されていることになります。 対応するキーポイントのペアの接続ベクトルの `x` と `y` 成分のネットワーク出力チャンネルのインデックスが [`BODY_PARTS_PAF_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L7-L8) リストに格納されます。


### キーポイントはどのようにインスタンスに分類されるのですか？

上述したように、ネットワークはキーポイントとあらかじめ定義されたキーポイントのペア間の接続という2つのテンソルを出力します。ここでは、[`BODY_PARTS_KPT_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L5-L6)のリストから最初のペア `[1, 2]` (首と右肩) から始めることにします。行 [63-92](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L63-L92) は，キーポイントのペアのうち，1つまたは両方が欠落している場合の処理を行います．このような場合、存在するすべてのポーズインスタンス（`pose_entries`）が現在のキーポイントを含んでいるかどうかがチェックされます。したがって、ネットワークが右肩のキーポイントを見つけない場合、見つかったすべての首のキーポイントは、すでに存在するポーズに属しているかどうかをチェックし、そうでなければ、このキーポイントを持つ新しいポーズインスタンスが作成されます。

行 [94-141](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L94-L141) は、見つかったキーポイント（ペアの中から特定のタイプのもの）のうちどれが接続されているかを、それらの間を網羅的に探索し、学習した接続ベクトルがキーポイントの位置間のベクトルに一致するかどうかを確認することによって、検証します。

行 [159-193](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L159-L193) は、存在するポーズインスタンスの1つに、接続されたキーポイントを割り当てます。それが最初のキーポイントペアである場合 ([`part_id == 0`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L159)) は、両方のキーポイントを含む新しいポーズインスタンスが作成されます。 例えば、現在のキーポイントのペアが右肩と右肘の場合、右肘は、特定の座標を持つ右肩をすでに持つポーズインスタンスに割り当てられます（前のステップで首と右肩のペアで割り当てられました）。 ペアから最初のキーポイントを含むポーズインスタンスが見つからなかった場合、新しいポーズインスタンスが作成されます](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L159)。 そして、すべてのキーポイントのペアを1つずつ処理します。ご覧の通り、[`BODY_PARTS_KPT_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L5-L6) リスト内のキーポイントペアの順番（およびペア内のキーポイントの順番）がランダムな場合、すべてのキーポイントを持つインスタンスではなく、不一致のキーポイントペアから複数のポーズインスタンスが作成されることになります。そのため、キーポイントペアの順序は重要であり、ルートキーポイントはキーポイントをより強固に接続するために有用です。

ここでは人物のポーズについて説明しましたが、異なる種類のオブジェクトにも同じ考察が適用できます。

## Dataset format

The easiest way is to use annotation in [COCO](http://cocodataset.org/#format-data) format. So if you need to label dataset, consider [coco-annotator](https://github.com/jsbroks/coco-annotator) tool (possibly there are alternatives, but I am not aware of it). If there is already annotated dataset, just convert it to COCO [format](http://cocodataset.org/#format-data).

Now convert dataset from COCO format into [internal](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch#training) format:

```
python scripts/prepare_train_labels.py --labels custom_dataset_annotation.json
```

## トレーニングコードの修正

1. オリジナルのCOCOキーポイントの順序を内部的な順序に変換しています。新しいデータでの学習には必要ないので、[`_convert`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/datasets/transformations.py#L36)は削除しても大丈夫です。

2. オブジェクトの左右を適切に[swap](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/datasets/transformations.py#L252)するために、キーポイントのインデックスを修正しました。

3. グループ化するためのキーポイントのペアを定義するために、独自の [`BODY_PARTS_KPT_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/datasets/coco.py#L13) を設定します。

4. ネットワークオブジェクト](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/train.py#L26) に、キーポイントの出力チャンネル数 `num_heatmaps` を、検出するキーポイントの数 + 1(背景の数) 、キーポイント間の接続数を `num_pafs` に設定します。例えば、新しいオブジェクトが5つのキーポイントを持ち、グループ化のために4つのキーポイントのペアを定義した場合、ネットワークオブジェクトは次のように作成されます。
```
net = PoseEstimationWithMobileNet(num_refinement_stages, num_heatmaps=6, num_pafs=8)
```

`num_pafs` は 8 です．これは，各接続が，ペアのキーポイント間のベクトルの `x` と `y` 成分の 2 つの出力チャンネルとしてエンコードされているためです．

5. 適切なネットワークの推論と検証のために、新しいキーポイントのインデックスのペアを [`BODY_PARTS_KPT_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L5-L6) に、キーポイント間の接続に対応するネットワーク出力チャンネルのインデックスのペアを [`BODY_PARTS_PAF_IDS`](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/keypoints.py#L7-L8) にセットしてください。

6. スタンドアロンで検証を行う場合は、新たに学習したキーポイントの数とキーポイント間の接続数に応じて、[ネットワークオブジェクトの作成](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/val.py#L174)を変更します。

## Congratulations

おめでとうございます！これであなたもポーズ推定マスターです！:sunglasses: フォースと共にあらんことを！ :accept:
