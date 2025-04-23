#!/bin/bash
#SBATCH --job-name=gnn_hydra_tuning
#SBATCH --partition gpu
#SBATCH --cpus-per-task 8
#SBATCH --mem 4G
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --time 1-00:00:00

module purge
module load Python/3.10.4-GCCcore-11.3.0-bare

source ~/forecasting_smog_DL_GNN/.venv/bin/activate
cd src
python tune_and_final_hydra.py --multirun training.epochs=200 hp_tuning.n_trials=70
       
deactivate
