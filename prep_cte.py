#!/usr/bin/env python3
"""
Stage the CTE (Crohn's enterography) scans for SuPreM inference.

Mirrors prep_raos_val_ts.py, minus the ground-truth step -- these cases have no
intestinal-tract GT, so inference runs without --labels_dir and no Dice is computed.

Produces:
  cte_input/<CASE>/ct.nii.gz          -> symlink to the CTE scan
  cte_input/chunks/chunk_NN.txt       -> case lists for the SLURM job array

Chunks are assigned round-robin (case i -> chunk i % n_chunks) so each array task
gets a comparable mix of volume sizes instead of one task drawing all the big ones.
"""
import os
import glob
import argparse

CTE = "/uhome/hoda2/rdss/p60290_1/IBD_Data/CTEs_cd_overlap"
ROOT = "/uhome/hoda2/projects/p60290_1/AbdomenAtlas/SuPreM"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cte_dir", default=CTE)
    p.add_argument("--input_dir", default=os.path.join(ROOT, "cte_input"))
    p.add_argument("--n_chunks", type=int, default=10)
    args = p.parse_args()

    images = sorted(glob.glob(os.path.join(args.cte_dir, "*.nii.gz")))
    print(f"{len(images)} CTE scans in {args.cte_dir}")
    if not images:
        raise SystemExit("no scans found")

    chunk_dir = os.path.join(args.input_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    cases = []
    for img in images:
        case = os.path.basename(img)[: -len(".nii.gz")]
        case_dir = os.path.join(args.input_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        link = os.path.join(case_dir, "ct.nii.gz")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(img), link)
        cases.append(case)

    print(f"staged {len(cases)} CTs -> {args.input_dir}")

    for k in range(args.n_chunks):
        members = cases[k::args.n_chunks]
        path = os.path.join(chunk_dir, f"chunk_{k:02d}.txt")
        with open(path, "w") as f:
            f.write("\n".join(members) + "\n")
        print(f"  chunk_{k:02d}.txt: {len(members)} cases")

    print("\nDone.")


if __name__ == "__main__":
    main()
