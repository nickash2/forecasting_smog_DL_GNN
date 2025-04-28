#!/bin/bash
#SBATCH --job-name=gnn_runs
#SBATCH --partition gpu
#SBATCH --mem 16G
#SBATCH --nodes 1
#SBATCH --time 05:00:00
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --output=%x_%j.out

module purge
module load Python
source ~/forecasting_smog_DL_GNN/.venv/bin/activate
python -m src.graph_modelling.scripts.compare_models --multirun model=spatial_only_gcn,temporal_only_gru,astgcn_seq2seq,batched_gconvgru_index,attention_gconvgru,astgcn_like data=all_vars

deactivate
