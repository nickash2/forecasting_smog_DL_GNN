#!/bin/bash
#SBATCH --job-name=gnn_runs
#SBATCH --partition gpu
#SBATCH --mem 16G
#SBATCH --nodes 1
#SBATCH --time 15:20:00
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --output=%x_%j.out

module purge
export TF_CPP_MIN_LOG_LEVEL=3   # Suppress TensorFlow warnings
source ~/forecasting_smog_DL_GNN/.venv/bin/activate
python -m src.graph_modelling.scripts.tune_models --multirun model=spatial_only_gcn,spatiotemporalattn,temporal_only_gru,temporal_only_gru_tempatn data=all_vars optuna.enabled=True

deactivate
