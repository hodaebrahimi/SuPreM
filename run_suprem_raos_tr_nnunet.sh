#!/bin/bash
#SBATCH --job-name=raos_tr_nnunet
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/raos_tr_nnunet_%j.out
#SBATCH --error=logs/raos_tr_nnunet_%j.err

module load gcc opencv
module load python
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

ROOT="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"
DATA="$ROOT/raos_tr_input"
CKPT="$ROOT/supervised_suprem_unet_2100.pth"
SAVE="$ROOT/raos_tr_suprem/nnunet"
LABELS="/uhome/hoda2/rdss/p60290_1/IBD_Data/labelsTr_intestinal_tract"

mkdir -p logs
cd "$ROOT/direct_inference"

# "nnunet" here = SuPreM's U-Net backbone (supervised_suprem_unet_2100.pth)
PYTHONWARNINGS=ignore python suprem_raos_inference.py \
    --data_root_path "$DATA" \
    --save_dir "$SAVE" \
    --checkpoint "$CKPT" \
    --backbone unet \
    --case_list "$ROOT/raos_tr_suprem/test_10cases.txt" \
    --labels_dir "$LABELS" \
    --num_workers 4 \
    --overlap 0.75 \
    --sw_batch_size 4
