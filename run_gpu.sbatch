# run_gpu.sbatch
#!/bin/bash
#SBATCH --job-name=gpu_job             # job name
#SBATCH --output=logs/%x-%j.out        # stdout
#SBATCH --error=logs/%x-%j.err         # stderr
#SBATCH --partition=gpu                # GPU partition
#SBATCH --gres=gpu:a100:1              # request 1 A100 GPU
#SBATCH --time=04:00:00                # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load cuda/11.8
module load python/3.10.10

source /home/ak395/scratch.msml605/MSML605/env/bin/activate

cd /home/ak395/scratch.msml605/MSML605

nvidia-smi

python driver.py --hardware CPU --mode single --model simplecnn --subset True --dataset_size 50 --batch_size 32 --learning_rate 0.0005
