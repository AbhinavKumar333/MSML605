#!/bin/bash
#SBATCH --job-name=cpu_job             # job name
#SBATCH --output=logs/%x-%j.out        # stdout (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x-%j.err         # stderr
#SBATCH --partition=standard           # CPU partition
#SBATCH --time=02:00:00                # walltime (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load python/3.10.10

source /home/ak395/scratch.msml605/MSML605/env/bin/activate

cd /home/ak395/scratch.msml605/MSML605

python driver.py --hardware CPU --mode single --model simplecnn --subset True --dataset_size 50 --batch_size 32 --learning_rate 0.0005

deactivate