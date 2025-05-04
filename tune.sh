#!/bin/bash
#SBATCH --job-name=gnn_runs
#SBATCH --partition himem
#SBATCH --mem 32GB
#SBATCH --nodes 1
#SBATCH --time 2:15:20:00
#SBATCH --cpus-per-task 4
#SBATCH --output=%x_%j.out

module purge
export TF_CPP_MIN_LOG_LEVEL=3   # Suppress TensorFlow warnings
source ~/forecasting_smog_DL_GNN/.venv/bin/activate
python -m src.graph_modelling.scripts.tune_models --multirun model=spatial_only_gcn,temporal_only_gru,astgcn_like,astgcn data=all_vars optuna.enabled=True

deactivate
