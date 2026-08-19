#!/bin/bash
#SBATCH --job-name=raos_eval
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/raos_eval_%j.out
#SBATCH --error=logs/raos_eval_%j.err
#
# Run a SuPreM backbone on a RAOS eval set and report intestinal-tract Dice vs GT.
#
# Usage:
#   sbatch --export=ALL,SET=val,BACKBONE=unet      run_suprem_raos_eval.sh
#   sbatch --export=ALL,SET=val,BACKBONE=swinunetr run_suprem_raos_eval.sh
#   sbatch --export=ALL,SET=ts,BACKBONE=unet       run_suprem_raos_eval.sh
#   sbatch --export=ALL,SET=ts,BACKBONE=swinunetr  run_suprem_raos_eval.sh
#
#   SET      = val | ts        (which RAOS split)
#   BACKBONE = unet | swinunetr  ("unet" == SuPreM U-Net, the "nnunet"-style backbone)

set -euo pipefail

SET="${SET:?Set SET=val or SET=ts}"
BACKBONE="${BACKBONE:?Set BACKBONE=unet or BACKBONE=swinunetr}"

module load gcc opencv
module load python
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

ROOT="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"
DATA="$ROOT/raos_${SET}_input"
LABELS="$ROOT/raos_${SET}_gt_bin"

case "$BACKBONE" in
    unet)       CKPT="$ROOT/supervised_suprem_unet_2100.pth";       SUB="nnunet" ;;
    swinunetr)  CKPT="$ROOT/supervised_suprem_swinunetr_2100.pth";  SUB="swinunetr" ;;
    *) echo "Unknown BACKBONE=$BACKBONE" >&2; exit 1 ;;
esac
SAVE="$ROOT/raos_${SET}_suprem/${SUB}"

mkdir -p logs
cd "$ROOT/direct_inference"

echo "SET=$SET  BACKBONE=$BACKBONE  ->  $SAVE"
PYTHONWARNINGS=ignore python suprem_raos_inference.py \
    --data_root_path "$DATA" \
    --save_dir "$SAVE" \
    --checkpoint "$CKPT" \
    --backbone "$BACKBONE" \
    --labels_dir "$LABELS" \
    --num_workers 4 \
    --overlap 0.75 \
    --sw_batch_size 4
