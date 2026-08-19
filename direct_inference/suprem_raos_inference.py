"""
Run SuPreM (Universal_model) inference on RAOS cases and save the intestinal-tract
organs (colon, small bowel, duodenum) for each CT scan.

This replaces an earlier broken draft of this file that loaded the SuPreM checkpoint
into a bare SwinUNETR with a randomly-initialised 19-class head and used the wrong
(TotalSegmentator) organ indices. This version mirrors the proven logic in
inference_abdomenatlas.py / inference_abdomenatlas_nnUnet.py:

  * Universal_model + word_embedding encoding (so the full checkpoint loads).
  * Correct SuPreM organ indices:  duodenum=14, colon=18, intestine/small_bowel=19.
  * sigmoid -> threshold_organ -> organ_post_process -> MONAI inverse transform.

It is backbone-agnostic: pass --backbone swinunetr (supervised_suprem_swinunetr_2100.pth)
or --backbone unet (supervised_suprem_unet_2100.pth, the "nnUnet"-style backbone).

Input layout: <data_root_path>/<case>/ct.nii.gz  (one sub-dir per case).
Output layout: <save_dir>/<case>/segmentations/{colon,small_bowel,duodenum}.nii.gz
               <save_dir>/<case>/intestinal_tract.nii.gz   (union of the three)
"""
import os
import csv
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, ToTensord,
    EnsureChannelFirstd, LoadImage, EnsureChannelFirst, ResampleToMatch,
)
from monai.data import DataLoader, Dataset, list_data_collate, MetaTensor

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.Universal_model import Universal_model
from utils.utils import (
    TEMPLATE, NUM_CLASS,
    threshold_organ, organ_post_process,
    pseudo_label_single_organ,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
torch.multiprocessing.set_sharing_strategy('file_system')

# SuPreM organ index (1-indexed) -> output filename.
# "intestine" (class 19) is the small bowel; saved as small_bowel.nii.gz per request.
INTESTINAL_ORGANS = {
    14: "duodenum",       # 14
    18: "colon",          # 18
    19: "small_bowel",    # 19 (== SuPreM "intestine")
}


def get_val_transforms(args):
    # EnsureChannelFirstd (not a Lambda channel-add hack) so the MetaTensor keeps a
    # correct affine through Orientation/Spacing/CropForeground. We map predictions
    # back to the original CT grid with ResampleToMatch using that affine, rather than
    # MONAI's Invertd (whose inverse op-stack is mangled by list_data_collate here).
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max,
                             b_min=args.b_min, b_max=args.b_max, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ])


def dice_binary(pred, gt):
    # Explicitly check for empty masks before computing Dice:
    #   both empty            -> NaN  (organ absent from FOV; excluded from the mean)
    #   exactly one empty     -> 0.0  (a miss or a false positive)
    #   both present          -> 2|p&g| / (|p|+|g|)
    pred, gt = pred.astype(bool), gt.astype(bool)
    pe, ge = not pred.any(), not gt.any()
    if pe and ge:
        return float('nan')
    if pe or ge:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum()))


def run_inference(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires a GPU.")
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}  |  backbone: {args.backbone}")

    # Collect case directories containing ct.nii.gz
    all_cases = sorted([
        d for d in os.listdir(args.data_root_path)
        if os.path.isdir(os.path.join(args.data_root_path, d))
        and os.path.exists(os.path.join(args.data_root_path, d, 'ct.nii.gz'))
    ])

    # Optionally restrict to a case list (one case id / UID per line)
    if args.case_list:
        with open(args.case_list) as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        missing = wanted - set(all_cases)
        if missing:
            print(f"WARNING: {len(missing)} listed cases not found under {args.data_root_path}:")
            for m in sorted(missing):
                print(f"  - {m}")
        all_cases = [c for c in all_cases if c in wanted]

    # Optionally skip cases already fully written (resume-friendly for the 12h SLURM limit).
    if args.skip_existing:
        def done(c):
            cd = os.path.join(args.save_dir, c)
            seg = os.path.join(cd, 'segmentations')
            return (os.path.exists(os.path.join(cd, 'intestinal_tract.nii.gz'))
                    and all(os.path.exists(os.path.join(seg, o + '.nii.gz'))
                            for o in INTESTINAL_ORGANS.values()))
        before = len(all_cases)
        all_cases = [c for c in all_cases if not done(c)]
        print(f"--skip_existing: {before - len(all_cases)} already done, {len(all_cases)} remaining")

    print(f"Processing {len(all_cases)} cases")
    if not all_cases:
        print("Nothing to process. Exiting.")
        return

    data_dicts = [{'image': os.path.join(args.data_root_path, c, 'ct.nii.gz'), 'name_img': c}
                  for c in all_cases]

    val_transforms = get_val_transforms(args)
    loader = DataLoader(Dataset(data=data_dicts, transform=val_transforms),
                        batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=list_data_collate)

    # Load model (Universal_model loads the full SuPreM checkpoint, including the head)
    print("Loading Universal_model...")
    model = Universal_model(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1, out_channels=NUM_CLASS,
        backbone=args.backbone, encoding='word_embedding',
    )
    store_dict = model.state_dict()
    store_keys = list(store_dict.keys())
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    load_values = list(checkpoint['net'].values())
    for i in range(len(store_dict)):
        store_dict[store_keys[i]] = load_values[i]
    model.load_state_dict(store_dict)
    print(f"Loaded {len(store_dict)} parameters from {os.path.basename(args.checkpoint)}")

    model.to(device)
    model.eval()
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)
    organ_list_all = TEMPLATE['assemble']  # organs 1-25, needed by organ_post_process
    dice_rows = []

    for index, batch in enumerate(tqdm(loader, desc="Inference")):
        image = batch["image"].to(device)
        name_img = batch["name_img"]
        if isinstance(name_img, list):
            name_img = name_img[0]

        image_file_path = os.path.join(args.data_root_path, name_img, 'ct.nii.gz')
        case_save_path = os.path.join(args.save_dir, name_img)
        seg_save_path = os.path.join(case_save_path, 'segmentations')
        os.makedirs(seg_save_path, exist_ok=True)

        original_nii = nib.load(image_file_path)
        affine, original_shape = original_nii.affine, original_nii.shape

        # Network-space image MetaTensor (carries the post-transform affine: RAS, target
        # spacing, crop-shifted origin) and the original CT grid as the resample target.
        net_img = batch["image"][0]                              # (1, nx, ny, nz)
        dst_grid = EnsureChannelFirst()(LoadImage()(image_file_path))  # (1, *original_shape)
        resample = ResampleToMatch(mode="nearest")

        with torch.no_grad():
            pred = sliding_window_inference(
                image, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size,
                model, overlap=args.overlap, mode='gaussian',
            )
            pred_sigmoid = F.sigmoid(pred)

        pred_hard = threshold_organ(pred_sigmoid, args).cpu()
        torch.cuda.empty_cache()

        pred_hard_post, _ = organ_post_process(pred_hard.numpy(), organ_list_all, case_save_path, args)
        pred_hard_post = torch.tensor(pred_hard_post)

        intestinal_mask = None
        for organ_idx, organ_name in INTESTINAL_ORGANS.items():
            pseudo = pseudo_label_single_organ(pred_hard_post, organ_idx, args)  # (1, 1, nx, ny, nz)
            # wrap the network-space mask with the image's affine, then resample to the CT grid
            organ_net = MetaTensor(pseudo[0].float().cpu(), affine=net_img.affine)  # (1, nx, ny, nz)
            organ_orig = resample(organ_net, dst_grid)                           # (1, *original_shape)
            organ_mask = (organ_orig[0].numpy() > 0).astype(np.uint8)
            nib.save(nib.Nifti1Image(organ_mask, affine),
                     os.path.join(seg_save_path, organ_name + '.nii.gz'))
            intestinal_mask = organ_mask if intestinal_mask is None \
                else np.logical_or(intestinal_mask, organ_mask).astype(np.uint8)

        nib.save(nib.Nifti1Image(intestinal_mask, affine),
                 os.path.join(case_save_path, 'intestinal_tract.nii.gz'))

        # Optional Dice vs binary intestinal-tract ground truth
        if args.labels_dir:
            gt_path = os.path.join(args.labels_dir, name_img + '.nii.gz')
            if os.path.exists(gt_path):
                gt = (nib.load(gt_path).get_fdata() > 0).astype(np.uint8)
                if gt.shape == intestinal_mask.shape:
                    d = dice_binary(intestinal_mask, gt)
                    dice_rows.append((name_img, d))
                    print(f"{name_img}: combined intestinal-tract Dice = {d:.4f}")
                else:
                    print(f"{name_img}: GT shape {gt.shape} != pred {intestinal_mask.shape}, skipping Dice")
            else:
                print(f"{name_img}: no GT at {gt_path}")

        print(f"[{index+1}/{len(loader)}] Saved {name_img}")

    if dice_rows:
        csv_path = os.path.join(args.save_dir, 'dice_results.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case', 'intestinal_tract_dice'])
            w.writerows(dice_rows)
        scores = [d for _, d in dice_rows if not np.isnan(d)]
        n_empty = len(dice_rows) - len(scores)
        print(f"\nMean intestinal-tract Dice: {np.mean(scores):.4f} +/- {np.std(scores):.4f} "
              f"(n={len(scores)}; {n_empty} both-empty excluded)  ->  {csv_path}")


def main():
    p = argparse.ArgumentParser(description="SuPreM inference on RAOS cases (intestinal tract organs)")
    p.add_argument('--data_root_path', required=True, help='Dir containing <case>/ct.nii.gz sub-dirs')
    p.add_argument('--save_dir', required=True, help='Output directory')
    p.add_argument('--checkpoint', required=True, help='Path to SuPreM checkpoint')
    p.add_argument('--backbone', default='swinunetr', help='swinunetr or unet')
    p.add_argument('--case_list', default=None, help='Optional txt of case ids to restrict to (one per line)')
    p.add_argument('--labels_dir', default=None, help='Optional dir of <case>.nii.gz binary GT for Dice')
    p.add_argument('--skip_existing', action='store_true', default=False,
                   help='Skip cases whose output masks are already fully written (resume)')
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--roi_x', type=int, default=96)
    p.add_argument('--roi_y', type=int, default=96)
    p.add_argument('--roi_z', type=int, default=96)
    p.add_argument('--space_x', type=float, default=1.5)
    p.add_argument('--space_y', type=float, default=1.5)
    p.add_argument('--space_z', type=float, default=1.5)
    p.add_argument('--a_min', type=float, default=-175)
    p.add_argument('--a_max', type=float, default=250)
    p.add_argument('--b_min', type=float, default=0.0)
    p.add_argument('--b_max', type=float, default=1.0)
    p.add_argument('--overlap', type=float, default=0.75)
    p.add_argument('--sw_batch_size', type=int, default=4)
    p.add_argument('--threshold_organ', default='Pancreas Tumor')
    p.add_argument('--cpu', action='store_true', default=False)
    p.add_argument('--create_dataset', action='store_true', default=False)

    run_inference(p.parse_args())


if __name__ == "__main__":
    main()
