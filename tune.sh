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
module load Python
source ~/forecasting_smog_DL_GNN/.venv/bin/activate
python -m src.graph_modelling.scripts.tune_models --multirun model=spatial_only_gcn,temporal_only_gcn,attention_gconvgru,batched_gconvgru_index,a3tgcn,astgcn_like,astgcn training.n_epochs=500 data=all_vars optuna.enabled=True

deactivate
