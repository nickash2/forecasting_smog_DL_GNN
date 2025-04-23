#!/bin/bash
#SBATCH --job-name=gnn_hydra_tuning_parallel
#SBATCH --cpus-per-task 8
#SBATCH --mem 2G
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00

# List of models to run

module purge
source ~/forecasting_smog_DL_GNN/.venv/bin/activate
cd src
python tune_and_final_hydra.py --multirun training.epochs=200 hp_tuning.n_trials=70
deactivate
