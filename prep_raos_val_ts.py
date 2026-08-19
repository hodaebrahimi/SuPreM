#!/usr/bin/env python3
"""
Stage RAOS val/test sets for SuPreM inference and build the binary
intestinal-tract ground truth.

For each set (val, ts) this produces:
  raos_<set>_input/<UID>/ct.nii.gz        -> symlink to the RAOS CT image
  raos_<set>_gt_bin/<UID>.nii.gz          -> binary GT (1 where label in {9,10,11})

RAOS multi-organ label indices for the intestinal tract were verified by
intersecting labelsTr against the prebuilt labelsTr_intestinal_tract binary GT:
  9 = duodenum, 10 = colon, 11 = intestine (small bowel).
These are exactly the three organs SuPreM predicts (duodenum/colon/small_bowel).
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nib

RAOS = "/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)"
ROOT = "/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"
INTESTINAL_LABELS = (9, 10, 11)  # duodenum, colon, small bowel

SETS = {
    "val": ("imagesVal", "labelsVal", False),  # images have NO _0000 suffix
    "ts":  ("imagesTs",  "labelsTs",  True),   # images have a _0000 suffix
}


def uid_from_image(fname, has_suffix):
    base = fname[:-len(".nii.gz")]
    if has_suffix and base.endswith("_0000"):
        base = base[:-len("_0000")]
    return base


def prep_set(set_name):
    img_dir_name, lbl_dir_name, has_suffix = SETS[set_name]
    img_dir = os.path.join(RAOS, img_dir_name)
    lbl_dir = os.path.join(RAOS, lbl_dir_name)
    input_dir = os.path.join(ROOT, f"raos_{set_name}_input")
    gt_dir = os.path.join(ROOT, f"raos_{set_name}_gt_bin")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    print(f"\n[{set_name}] {len(images)} images in {img_dir}")
    staged, gt_built, missing_lbl = 0, 0, 0
    for img in images:
        uid = uid_from_image(os.path.basename(img), has_suffix)

        # stage CT as <UID>/ct.nii.gz symlink
        case_dir = os.path.join(input_dir, uid)
        os.makedirs(case_dir, exist_ok=True)
        link = os.path.join(case_dir, "ct.nii.gz")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(img), link)
        staged += 1

        # build binary GT
        lbl = os.path.join(lbl_dir, uid + ".nii.gz")
        if not os.path.exists(lbl):
            print(f"  WARNING: no label for {uid}")
            missing_lbl += 1
            continue
        nl = nib.load(lbl)
        arr = np.asarray(nl.dataobj).astype(np.int16)
        binmask = np.isin(arr, INTESTINAL_LABELS).astype(np.uint8)
        nib.save(nib.Nifti1Image(binmask, nl.affine, nl.header),
                 os.path.join(gt_dir, uid + ".nii.gz"))
        gt_built += 1

    print(f"[{set_name}] staged {staged} CTs -> {input_dir}")
    print(f"[{set_name}] built {gt_built} binary GTs -> {gt_dir}"
          + (f"  ({missing_lbl} missing labels)" if missing_lbl else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["val", "ts"], choices=["val", "ts"])
    args = p.parse_args()
    for s in args.sets:
        prep_set(s)
    print("\nDone.")
