#!/usr/bin/env python3
"""
Evaluate SuPreM RAOS predictions against RAOS ground truth.

Computes, for each case:
  * per-organ Dice  : duodenum / colon / small_bowel  (pred organ vs GT organ)
  * whole-tract Dice: intestinal_tract                (union of the three)

GT is read straight from the RAOS multi-organ label map, which is per-organ:
  duodenum = 9, colon = 10, small_bowel = 11   (intestinal tract = {9,10,11}).

Empty-mask handling (checked BEFORE computing Dice):
  both pred and GT empty  -> status 'both_empty', Dice = NaN, EXCLUDED from the mean
                             (organ genuinely absent from the scan FOV)
  GT present, pred empty  -> status 'miss',       Dice = 0.0  (model missed the organ)
  GT empty,  pred present -> status 'false_pos',  Dice = 0.0  (false positive)
  both present            -> status 'ok',         Dice = 2|p&g| / (|p|+|g|)

This reads the per-organ prediction masks the inference job already saved, so it
does NOT require re-running inference.
"""
import os
import csv
import glob
import argparse
import numpy as np
import nibabel as nib

RAOS = "/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)"
ROOT = "/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"

ORGAN_LABEL = {"duodenum": 9, "colon": 10, "small_bowel": 11}
ORGANS = list(ORGAN_LABEL)                       # column order
TRACT_LABELS = tuple(ORGAN_LABEL.values())       # (9, 10, 11)

LABEL_DIR = {"tr": "labelsTr", "val": "labelsVal", "ts": "labelsTs"}


def dice_with_status(pred, gt):
    """Return (dice_or_nan, status). NaN only for the both-empty case."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pe, ge = not pred.any(), not gt.any()
    if pe and ge:
        return float("nan"), "both_empty"
    if ge:                       # pred present, no GT -> false positive
        return 0.0, "false_pos"
    if pe:                       # GT present, pred empty -> miss
        return 0.0, "miss"
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum())), "ok"


def eval_set_backbone(set_name, backbone):
    pred_root = os.path.join(ROOT, f"raos_{set_name}_suprem", backbone)
    label_dir = os.path.join(RAOS, LABEL_DIR[set_name])
    if not os.path.isdir(pred_root):
        print(f"  [skip] no predictions at {pred_root}")
        return None

    cases = sorted(d for d in os.listdir(pred_root)
                   if os.path.isdir(os.path.join(pred_root, d, "segmentations")))
    rows = []
    for uid in cases:
        seg_dir = os.path.join(pred_root, uid, "segmentations")
        tract_path = os.path.join(pred_root, uid, "intestinal_tract.nii.gz")
        organ_paths = {o: os.path.join(seg_dir, o + ".nii.gz") for o in ORGANS}
        if not (os.path.exists(tract_path) and all(os.path.exists(p) for p in organ_paths.values())):
            continue  # case not fully written yet

        lbl_path = os.path.join(label_dir, uid + ".nii.gz")
        if not os.path.exists(lbl_path):
            print(f"    WARNING: no GT label for {uid}")
            continue
        gt_arr = np.asarray(nib.load(lbl_path).dataobj).astype(np.int16)

        row = {"case": uid}
        shape_ok = True
        for organ in ORGANS:
            pred = np.asarray(nib.load(organ_paths[organ]).dataobj)
            gt = (gt_arr == ORGAN_LABEL[organ])
            if pred.shape != gt.shape:
                row[f"{organ}_dice"], row[f"{organ}_status"] = float("nan"), "shape_mismatch"
                shape_ok = False
                continue
            d, s = dice_with_status(pred, gt)
            row[f"{organ}_dice"], row[f"{organ}_status"] = d, s

        pred_tract = np.asarray(nib.load(tract_path).dataobj)
        gt_tract = np.isin(gt_arr, TRACT_LABELS)
        if pred_tract.shape != gt_tract.shape:
            row["intestinal_dice"], row["intestinal_status"] = float("nan"), "shape_mismatch"
        else:
            d, s = dice_with_status(pred_tract, gt_tract)
            row["intestinal_dice"], row["intestinal_status"] = d, s
        rows.append(row)

    if not rows:
        print(f"  [{set_name}/{backbone}] no fully-written cases yet")
        return None

    # write per-case CSV
    fields = ["case"]
    for o in ORGANS:
        fields += [f"{o}_dice", f"{o}_status"]
    fields += ["intestinal_dice", "intestinal_status"]
    out_csv = os.path.join(pred_root, "dice_per_organ.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary: mean over cases where Dice is defined (excludes both_empty / shape_mismatch)
    print(f"  [{set_name}/{backbone}]  n_cases={len(rows)}  -> {out_csv}")
    summary = {}
    for key in ORGANS + ["intestinal"]:
        vals = [r[f"{key}_dice"] for r in rows if not np.isnan(r[f"{key}_dice"])]
        statuses = [r[f"{key}_status"] for r in rows]
        n_both_empty = statuses.count("both_empty")
        n_miss = statuses.count("miss")
        n_fp = statuses.count("false_pos")
        mean = float(np.mean(vals)) if vals else float("nan")
        std = float(np.std(vals)) if vals else float("nan")
        summary[key] = (mean, std, len(vals), n_both_empty, n_miss, n_fp)
        print(f"      {key:13s} Dice={mean:.4f} ± {std:.4f}  (n={len(vals)}"
              f"; both_empty={n_both_empty}, miss={n_miss}, false_pos={n_fp})")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["val", "ts"], choices=["tr", "val", "ts"])
    p.add_argument("--backbones", nargs="+", default=["nnunet", "swinunetr"])
    args = p.parse_args()

    all_summ = {}
    for s in args.sets:
        for b in args.backbones:
            print(f"\n=== {s} / {b} ===")
            res = eval_set_backbone(s, b)
            if res:
                all_summ[(s, b)] = res

    # compact final table
    if all_summ:
        print("\n" + "=" * 78)
        print(f"{'set/backbone':22s} {'duodenum':>10s} {'colon':>10s} {'small_bowel':>12s} {'tract':>10s}")
        print("-" * 78)
        for (s, b), summ in all_summ.items():
            print(f"{s+'/'+b:22s} "
                  f"{summ['duodenum'][0]:>10.4f} {summ['colon'][0]:>10.4f} "
                  f"{summ['small_bowel'][0]:>12.4f} {summ['intestinal'][0]:>10.4f}")
        print("=" * 78)


if __name__ == "__main__":
    main()
