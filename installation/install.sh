#!/usr/bin/zsh

mkdir -p results/log

echo 'export PYTHONPATH=${PYTHONPATH}:/workspaces/MOLT/src' >> ~/.zshrc
echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.zshrc
echo 'source /opt/conda/etc/profile.d/mamba.sh' >> ~/.zshrc

echo 'export PYTHONPATH=${PYTHONPATH}:/workspaces/MOLT/src' >> ~/.bashrc
echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'source /opt/conda/etc/profile.d/mamba.sh' >> ~/.bashrc

source /opt/conda/etc/profile.d/conda.sh
source /opt/conda/etc/profile.d/mamba.sh
mamba init

mamba env create -n molt_env --file environment.yml 
mamba activate molt_env