# 深層学習の不安定性
## 使い方

### ディレクトリ構成
1. Learner
使用するモデルが入っています。
   * data.py：学習データ作成に関する関数を記述
   * net.py：ニューラルネットの構成を記述
   * train.py：学習に関する関数を記述
   * validate.py：検証に関する関数を記述
   * utils.py：実験に使用しそうな便利な関数を記述
   * config.yaml：各種実験設定を記述
2. trainer.py：学習を実際に実行するファイル


### 実行例
```command line
poetry run python trainer.py
```

