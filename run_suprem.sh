#!/bin/bash
#SBATCH --job-name=suprem_inference
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/suprem_%j.out
#SBATCH --error=logs/suprem_%j.err

# Load required modules
module load gcc opencv
module load python

# Activate virtual environment
source /uhome/hoda2/projects/p60290_1/envs/suprem/bin/activate

# Set paths
datarootpath="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/raos_ts_samples"
backbone="swinunetr"
pretrainpath="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/supervised_suprem_swinunetr_2100.pth"
savepath="/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM/suprem_preds"

# Run inference
cd direct_inference/
python -W ignore inference.py --save_dir $savepath.$backbone --checkpoint $pretrainpath --data_root_path $datarootpath --backbone $backbone --store_result --suprem
