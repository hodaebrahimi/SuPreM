#!/bin/bash
#SBATCH --job-name=cte_nnunet
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-9%3
#SBATCH --output=logs/cte_nnunet_%A_%a.out
#SBATCH --error=logs/cte_nnunet_%A_%a.err

module load gcc opencv
module load python
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

ROOT="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"
DATA="$ROOT/cte_input"
CKPT="$ROOT/supervised_suprem_unet_2100.pth"
SAVE="$ROOT/cte_suprem_nnunet"

# 1000 CTE cases split round-robin into 10 chunks by prep_cte.py; one array task per chunk.
CASE_LIST=$(printf "%s/chunks/chunk_%02d.txt" "$DATA" "$SLURM_ARRAY_TASK_ID")

mkdir -p logs
cd "$ROOT/direct_inference"

# "nnunet" here = SuPreM's U-Net backbone (supervised_suprem_unet_2100.pth)
# No --labels_dir: the CTE cases have no intestinal-tract ground truth, so no Dice.
PYTHONWARNINGS=ignore python suprem_raos_inference.py \
    --data_root_path "$DATA" \
    --save_dir "$SAVE" \
    --checkpoint "$CKPT" \
    --backbone unet \
    --case_list "$CASE_LIST" \
    --num_workers 4 \
    --overlap 0.75 \
    --sw_batch_size 4 \
    --skip_existing
