
#!/usr/bin/env bash

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda create -y -n EfficientZero python=3.8
conda activate EfficientZero

conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -c conda-forge "redis-server>=7.4.1"
pip install -r requirements.txt

AutoROM --accept-license

cd ez/mcts/ctree
sh make.sh
cd ..

cd ori_ctree
sh make.sh
cd ..

cd ctree_v2
sh make.sh
cd ../../..
