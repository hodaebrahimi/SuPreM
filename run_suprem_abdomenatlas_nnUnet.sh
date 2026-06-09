#!/bin/bash
#SBATCH --job-name=suprem_nnunet
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=168:00:00
#SBATCH --output=logs/suprem_abdomenatlas_nnUnet_%j.out
#SBATCH --error=logs/suprem_abdomenatlas_nnUnet_%j.err

# Load required modules
module load gcc opencv
module load python

# Activate virtual environment
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

# Set paths
DATA_DIR="/uhome/hoda2/projects/p60290_1/AbdomenAtlas1.1Dataset"
CHECKPOINT="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/supervised_suprem_unet_2100.pth"
SAVE_DIR="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/suprem_preds/abdomenatlas_nnUnet"

# Create logs directory
mkdir -p logs

# Run inference
cd /uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/direct_inference

PYTHONWARNINGS=ignore torchrun --nproc_per_node=8 inference_abdomenatlas_nnUnet.py \
    --data_root_path "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --backbone unet \
    --num_workers 4 \
    --overlap 0.75 \
    --sw_batch_size 8
