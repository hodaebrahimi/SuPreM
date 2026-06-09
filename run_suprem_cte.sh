#!/bin/bash
#SBATCH --job-name=suprem_cte
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/suprem_cte_%j.out
#SBATCH --error=logs/suprem_cte_%j.err

# Load required modules
module load gcc opencv
module load python

# Activate virtual environment
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

# Set paths
DATA_DIR="/uhome/hoda2/rdss/p60290_1/IBD_Data/CTEs_cd_overlap"
CHECKPOINT="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/supervised_suprem_swinunetr_2100.pth"
SAVE_DIR="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/suprem_preds/cte_cd_overlap_25organs"

# Create logs directory
mkdir -p logs

# Run inference with 1 GPU
cd /uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/direct_inference

PYTHONWARNINGS=ignore python inference_cte.py \
    --data_root_path "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --backbone swinunetr \
    --num_workers 4 \
    --overlap 0.75 \
    --sw_batch_size 8
