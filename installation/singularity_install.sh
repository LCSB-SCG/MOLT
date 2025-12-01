mkdir -p results/log
mkdir -p tmp

# add needed to immediately run the mamba commmand
export PYTHONPATH=${PYTHONPATH}:/workspaces/MOLT/src
source /opt/conda/etc/profile.d/mamba.sh
source /opt/conda/etc/profile.d/conda.sh
mamba activate molt_env