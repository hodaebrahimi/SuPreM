# RAOS-tr SuPreM inference — work log

Date: 2026-06-08

## Goal
Run the SuPreM model on the 10 RAOS *training-set* cases listed in
`raos_tr_suprem/test_10cases.txt` and save the **colon, small bowel, and duodenum**
segmentations for each CT scan under `raos_tr_suprem/`. Run **both** SuPreM backbones
(SwinUNETR and the U-Net / "nnUnet" backbone).

## Data layout discovered
- CT scans for the 10 case UIDs live at:
  `/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr/<UID>.nii.gz`
  (named `<UID>.nii.gz` — **no** `_0000` suffix).
- Binary intestinal-tract ground truth (values 0/1):
  `/uhome/hoda2/rdss/p60290_1/IBD_Data/labelsTr_intestinal_tract/<UID>.nii.gz`
- All 10 images and labels verified present.

## SuPreM organ indices (Universal_model)
`duodenum = 14`, `colon = 18`, `intestine (= small bowel) = 19`.
(NOT the TotalSegmentator `14/15/16` convention.)

## History of `direct_inference/suprem_raos_inference.py`
- **Never committed (untracked) and never run successfully.** Its default output dir was
  never created, none of its distinctive outputs (`intestinal_combined.nii.gz`,
  `small_bowel.nii.gz`) exist anywhere on disk, and no SLURM log references it.
- It was an abandoned early draft with three bugs, superseded by
  `inference_abdomenatlas.py` / `inference_abdomenatlas_nnUnet.py`:
  1. Loaded the checkpoint into a bare `SwinUNETR` with a **randomly-initialised**
     19-class output head (invalid) instead of `Universal_model` + word embeddings.
  2. Used the wrong organ indices `14/15/16` (TotalSegmentator) instead of `14/18/19`.
  3. Crude `scipy.ndimage.zoom` resampling instead of proper MONAI inverse transforms.

## What was done this session
1. Verified the 10 cases and their GT on disk.
2. Staged the data into the case-dir/`ct.nii.gz` layout the loader expects:
   `raos_tr_input/<UID>/ct.nii.gz` → symlink to the RAOS image. (Machine-specific;
   gitignored. Regenerate with the loop below.)
3. **Rewrote `direct_inference/suprem_raos_inference.py`** into a correct,
   backbone-agnostic script: `Universal_model` + word embeddings, indices `14/18/19`,
   `sigmoid → threshold_organ → organ_post_process → MONAI inverse transform`,
   `--case_list` filter, optional binary Dice vs `--labels_dir`.
4. Created two SLURM run scripts (separate output sub-dirs so they don't collide):
   - `run_suprem_raos_tr_swin.sh`   → backbone `swinunetr`, `supervised_suprem_swinunetr_2100.pth`
   - `run_suprem_raos_tr_nnunet.sh` → backbone `unet`,      `supervised_suprem_unet_2100.pth`
     ("nnunet" = SuPreM's U-Net backbone, not a separate nnU-Net install).

## How to (re)stage and run
```bash
# stage (regenerates raos_tr_input/)
SRC="/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr"
while read u; do mkdir -p raos_tr_input/$u && ln -sf "$SRC/$u.nii.gz" raos_tr_input/$u/ct.nii.gz; done < raos_tr_suprem/test_10cases.txt

# run both models
sbatch run_suprem_raos_tr_swin.sh
sbatch run_suprem_raos_tr_nnunet.sh
```

## Output layout (per model)
```
raos_tr_suprem/{swinunetr,nnunet}/<UID>/
  segmentations/colon.nii.gz
  segmentations/small_bowel.nii.gz     # SuPreM "intestine" (class 19)
  segmentations/duodenum.nii.gz
  intestinal_tract.nii.gz              # union of the three
raos_tr_suprem/{swinunetr,nnunet}/dice_results.csv   # combined-tract Dice vs binary GT
```
Note: GT is binary (0/1), so only the combined intestinal-tract Dice is meaningful —
per-organ Dice is not computable against this GT.

## Bug fixed (2026-06-08): empty duodenum masks

**Symptom:** the first run produced `duodenum.nii.gz` with 0 voxels for all 10
cases on both backbones, while colon/small_bowel were non-empty (but undersized).

**Root cause — the spatial inverse was broken, not the model.** A GPU diagnostic
showed the model predicts duodenum richly in network space (~18k–28k voxels post
connected-component). The masks died in the save path:
1. `get_val_transforms` added the channel dim with a `Lambdad` hack. Combined with
   `list_data_collate`, the MetaTensor's `applied_operations` collapsed to a single
   malformed `'list'` entry, so MONAI `Invertd` could not reverse
   Orientation/Spacing/CropForeground. It returned the **network-space** mask
   unchanged (RAS, 1.5 mm, shape e.g. 224×155×345) instead of the original CT grid
   (e.g. 343×237×173, LPS). (`inference.py` has the same latent bug but never noticed:
   it saves the network-space mask with the original affine — geometrically wrong but
   non-empty.)
2. `match_shape_to_original` then force-fit that network-space mask into the original
   shape by origin-aligned truncation. Large sprawling organs (colon/intestine) kept a
   coincidental fraction; the compact duodenum landed entirely outside the bogus
   overlap → 0 voxels.

**Fix (in `suprem_raos_inference.py`):**
- `Lambdad` → `EnsureChannelFirstd` (keeps a correct affine on the MetaTensor).
- Dropped `Invertd` + `match_shape_to_original`. Now wrap each network-space organ
  mask in a `MetaTensor` carrying the image's post-transform affine and
  `ResampleToMatch(mode="nearest")` it onto the original CT grid loaded with
  `LoadImage()/EnsureChannelFirst()`. This is affine-based, so it is geometrically
  exact and robust to MONAI's fragile inverse op-stack.

**Result:** duodenum non-empty in 10/10 cases (~14k–31k voxels) on both backbones;
masks align to the CT/GT. Combined intestinal-tract Dice:
swinunetr 0.8905 ± 0.0179, nnunet ≈ 0.89 (n=10).
