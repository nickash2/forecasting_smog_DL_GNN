#!/bin/bash

# Parameters
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=4
#SBATCH --error=/home4/s5185491/third-copy/comparison_results/multirun/2025-04-27_23-16-20/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=compare_models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home4/s5185491/third-copy/comparison_results/multirun/2025-04-27_23-16-20/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=240
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home4/s5185491/third-copy/comparison_results/multirun/2025-04-27_23-16-20/.submitit/%A_%a/%A_%a_%t_log.out --error /home4/s5185491/third-copy/comparison_results/multirun/2025-04-27_23-16-20/.submitit/%A_%a/%A_%a_%t_log.err /home4/s5185491/forecasting_smog_DL_GNN/.venv/bin/python -u -m submitit.core._submit /home4/s5185491/third-copy/comparison_results/multirun/2025-04-27_23-16-20/.submitit/%j
