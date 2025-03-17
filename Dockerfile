FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# 作業ディレクトリを設定
WORKDIR /workspace

# タイムゾーン設定と必要なパッケージのインストール
ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN sed -i.bak -e 's|http://archive.ubuntu.com/ubuntu|http://mirror.math.princeton.edu/pub/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    tzdata \
    git \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11.7のソースコードをダウンロード
RUN wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz && \
    tar -xzf Python-3.11.7.tgz && \
    cd Python-3.11.7 && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.7 Python-3.11.7.tgz

# デフォルトでpython3.11を使えるようにシンボリックリンクを作成
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3

# Poetryをインストール
RUN curl -sSL https://install.python-poetry.org | python3 -

# Poetryのインストール後に環境変数を設定
ENV PATH="/root/.local/bin:$PATH"

# Poetryの設定を変更して仮想環境をプロジェクト内に作成
RUN poetry config virtualenvs.in-project true

# シンボリックリンクを作成
RUN ln -sf /usr/lib/x86_64-linux/libtinfo.so.6 /opt/conda/lib/libtinfo.so.6 && \
    ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so /usr/lib/libcublas.so && \
    ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11 /usr/lib/libcublas.so.11 && \
    ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11.11.3.6 /usr/lib/libcublas.so.11.11.3.6

# LD_LIBRARY_PATHを設定
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib:/opt/conda/lib:$LD_LIBRARY_PATH"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7"

# 仮想環境のパスを設定
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 現在のディレクトリ（ビルドコンテキスト）のすべてのファイルをコンテナ内のワークスペースにコピー
COPY . /workspace/

# Poetryでライブラリをインストール
RUN poetry lock
RUN poetry install

# コンテナ起動時に実行するコマンド
CMD ["bash"]
