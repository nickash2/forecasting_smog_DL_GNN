#!/bin/bash

# Parameters
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --error=/home4/s5185491/second-copy/src/multirun/2025-04-18/15-00-25/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=tune_and_final_hydra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home4/s5185491/second-copy/src/multirun/2025-04-18/15-00-25/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home4/s5185491/second-copy/src/multirun/2025-04-18/15-00-25/.submitit/%A_%a/%A_%a_%t_log.out --error /home4/s5185491/second-copy/src/multirun/2025-04-18/15-00-25/.submitit/%A_%a/%A_%a_%t_log.err /home4/s5185491/forecasting_smog_DL_GNN/.venv/bin/python -u -m submitit.core._submit /home4/s5185491/second-copy/src/multirun/2025-04-18/15-00-25/.submitit/%j
