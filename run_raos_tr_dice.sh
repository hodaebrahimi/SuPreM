#!/bin/bash
#SBATCH --job-name=raos_tr_dice
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/raos_tr_dice_%j.out
#SBATCH --error=logs/raos_tr_dice_%j.err
#
# Per-class Dice (duodenum / colon / small_bowel + intestinal-tract) for both SuPreM
# backbones on the whole RAOS tr set, vs the per-organ RAOS labelsTr (9/10/11).
# Writes raos_tr_suprem/<backbone>/dice_per_organ.csv and prints a summary table.
# Submit with a dependency so it fires only after both inference jobs succeed.

module load gcc opencv
module load python
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

ROOT="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"
mkdir -p "$ROOT/logs"
cd "$ROOT"

PYTHONWARNINGS=ignore python eval_raos_dice.py --sets tr --backbones nnunet swinunetr
