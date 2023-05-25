# QCMaterial

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shinaoka.github.io/QCMaterial.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shinaoka.github.io/QCMaterial.jl/dev)
[![Build Status](https://github.com/sakurairihito/QCMaterialNew/workflows/CI/badge.svg)](https://github.com/sakurairihito/QCMaterialNew/actions)
[![DevNew](https://img.shields.io/badge/docs-dev-blue.svg)](https://sakurairihito.github.io/QCMaterialNew/build/index.html)

```
    ,-----.        _______   ,---.    ,---.   ____   ,---------.    .-''-.  .-------.   .-./`)    ____      .---.     ,---.   .--.    .-''-.  .--.      .--. 
  .'  .-,  '.     /   __  \  |    \  /    | .'  __ `.\          \ .'_ _   \ |  _ _   \  \ .-.') .'  __ `.   | ,_|     |    \  |  |  .'_ _   \ |  |_     |  | 
 / ,-.|  \ _ \   | ,_/  \__) |  ,  \/  ,  |/   '  \  \`--.  ,---'/ ( ` )   '| ( ' )  |  / `-' \/   '  \  \,-./  )     |  ,  \ |  | / ( ` )   '| _( )_   |  | 
;  \  '_ /  | :,-./  )       |  |\_   /|  ||___|  /  |   |   \  . (_ o _)  ||(_ o _) /   `-'`"`|___|  /  |\  '_ '`)   |  |\_ \|  |. (_ o _)  ||(_ o _)  |  | 
|  _`,/ \ _/  |\  '_ '`)     |  _( )_/ |  |   _.-`   |   :_ _:  |  (_,_)___|| (_,_).' __ .---.    _.-`   | > (_)  )   |  _( )_\  ||  (_,_)___|| (_,_) \ |  | 
: (  '\_/ \   ; > (_)  )  __ | (_ o _) |  |.'   _    |   (_I_)  '  \   .---.|  |\ \  |  ||   | .'   _    |(  .  .-'   | (_ o _)  |'  \   .---.|  |/    \|  | 
 \ `"/  \  )  \(  .  .-'_/  )|  (_,_)  |  ||  _( )_  |  (_(=)_)  \  `-'    /|  | \ `'   /|   | |  _( )_  | `-'`-'|___ |  (_,_)\  | \  `-'    /|  '  /\  `  | 
  '. \_/``"/)  )`-'`-'     / |  |      |  |\ (_ o _) /   (_I_)    \       / |  |  \    / |   | \ (_ o _) /  |        \|  |    |  |  \       / |    /  \    | 
    '-----' `-'   `._____.'  '--'      '--' '.(_,_).'    '---'     `'-..-'  ''-'   `'-'  '---'  '.(_,_).'   `--------`'--'    '--'   `'-..-'  `---'    `---`
```

# 論文タイトル (Paper Title)

このリポジトリは、「論文タイトル」の公式実装を含んでいます。

## 概要 (Overview)

ここでは、論文の主要なアイデアや目的、成果を簡潔に説明します。

## 必要なソフトウェア (Prerequisites)

- Python 3.8+
- NumPy
- PyTorch
- ...（その他の必要なライブラリやフレームワーク）

## インストール (Installation)

1. このリポジトリをクローンします：

    ```bash
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. 必要なPythonライブラリをインストールします：

    ```bash
    pip install -r requirements.txt
    ```

## 使い方 (Usage)

ここでは、モデルを訓練したり、テストデータを評価したりするための基本的なコマンドを説明します。

例：

```bash
python train.py --dataset YourDataset



## QCMaterialNewって何？
git cloneしましょう。

論文のリンクも貼る。

QCMaterialNewの論文

## Install packages

```bash
pip install -r requirements.txt
```

## Launch a notebook server
This project is activated automatically when
you launch a server in the project directory.

```bash
$ cd $(PROJECTDIR)
$ jupyter lab
```

## Run tests
```bash
$ cd $(PROJECTDIR)
$ julia --project=@. test/runtests.jl
$ mpirun -np 2 julia --project=@. test/runtests.jl  # Only if MPI is installed on your system
```

## Run a script file depending on QCMaterial
``bash
mpirun -np 2 julia --project=~/.julia/dev/QCMaterial ~/.julia/dev/QCMaterial/samples/run.jl
`` 
