import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, ToTensord,
    EnsureChannelFirstd, Lambdad,
)
from monai.data import DataLoader, Dataset, list_data_collate, decollate_batch
from monai.transforms import Invertd, SaveImaged

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.Universal_model import Universal_model
from utils.utils import (
    TEMPLATE, NUM_CLASS, ORGAN_NAME_LOW,
    threshold_organ, organ_post_process,
    pseudo_label_single_organ, pseudo_label_all_organ,
)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

torch.multiprocessing.set_sharing_strategy('file_system')

# 25 organ names (indices 1-25 from ORGAN_NAME_LOW)
ORGAN_25 = ORGAN_NAME_LOW[:25]

# Intestinal tract: combine duodenum (organ 14), colon (organ 18), intestine/small_bowel (organ 19)
INTESTINAL_ORGAN_INDICES = [14, 18, 19]
INTESTINAL_ORGAN_NAMES = ["duodenum", "colon", "intestine"]


def get_val_transforms(args):
    return Compose([
        LoadImaged(keys=["image"]),
        Lambdad(keys=["image"], func=lambda x: x[None] if x.ndim == 3 else x),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ])


def invert_transform(name, batch, val_transforms):
    """Invert spatial transforms to map predictions back to original space."""
    post_transforms = Invertd(
        keys=name,
        transform=val_transforms,
        orig_keys="image",
        meta_keys=name + "_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )
    batch = [post_transforms(x) for x in decollate_batch(batch)]
    return batch


def run_inference(args):
    # Setup distributed if launched with torchrun
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    print(f"[Rank {local_rank}/{world_size}] GPU: {torch.cuda.get_device_name(local_rank)}")

    # Collect nii.gz files from flat directory
    all_files = sorted([
        f for f in os.listdir(args.data_root_path)
        if f.endswith('.nii.gz')
    ])
    print(f"Found {len(all_files)} nii.gz files total")

    # Split files across GPUs
    all_files = all_files[local_rank::world_size]
    print(f"[Rank {local_rank}] Processing {len(all_files)} files")

    data_dicts = []
    for f in all_files:
        case_name = f.replace('.nii.gz', '')
        data_dicts.append({
            'image': os.path.join(args.data_root_path, f),
            'name_img': case_name,
        })

    # Transforms and dataloader
    val_transforms = get_val_transforms(args)
    test_dataset = Dataset(data=data_dicts, transform=val_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=list_data_collate,
    )

    # Load model
    print("Loading Universal_model with swinunetr backbone...")
    model = Universal_model(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=NUM_CLASS,
        backbone=args.backbone,
        encoding='word_embedding',
    )

    store_dict = model.state_dict()
    store_dict_keys = list(store_dict.keys())
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    load_dict = checkpoint['net']
    load_dict_values = list(load_dict.values())

    for i in range(len(store_dict)):
        store_dict[store_dict_keys[i]] = load_dict_values[i]

    model.load_state_dict(store_dict)
    print(f"Loaded {len(store_dict)} parameters from checkpoint")

    model.to(device)
    model.eval()
    torch.backends.cudnn.benchmark = True

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    organ_list_all = TEMPLATE['assemble']  # organs 1-25

    print(f"\n[Rank {local_rank}] Segmenting {len(organ_list_all)} organs on {len(data_dicts)} cases")
    print(f"Saving to: {save_dir}\n")

    for index, batch in enumerate(tqdm(test_loader, desc=f"Rank {local_rank}")):
        image = batch["image"].to(device)
        name_img = batch["name_img"]
        if isinstance(name_img, list):
            name_img = name_img[0]

        image_file_path = os.path.join(args.data_root_path, name_img + '.nii.gz')
        case_save_path = os.path.join(save_dir, name_img)
        os.makedirs(case_save_path, exist_ok=True)
        organ_seg_save_path = os.path.join(case_save_path, 'segmentations')
        os.makedirs(organ_seg_save_path, exist_ok=True)

        affine = nib.load(image_file_path).affine

        with torch.no_grad():
            pred = sliding_window_inference(
                image,
                (args.roi_x, args.roi_y, args.roi_z),
                args.sw_batch_size, model,
                overlap=args.overlap,
                mode='gaussian',
            )
            pred_sigmoid = F.sigmoid(pred)

        pred_hard = threshold_organ(pred_sigmoid, args)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        # Post-processing
        B = pred_hard.shape[0]
        for b in range(B):
            pred_hard_post, _ = organ_post_process(
                pred_hard.numpy(), organ_list_all, case_save_path, args,
            )
            pred_hard_post = torch.tensor(pred_hard_post)

        # Save individual organ segmentations (inverted back to original space)
        for organ_index in organ_list_all:
            pseudo_label_single = pseudo_label_single_organ(pred_hard_post, organ_index, args)
            organ_name = ORGAN_NAME_LOW[organ_index - 1]
            batch[organ_name] = pseudo_label_single.cpu()
            BATCH = invert_transform(organ_name, batch, val_transforms)
            organ_invertd = np.squeeze(BATCH[0][organ_name].numpy(), axis=0)
            organ_save = nib.Nifti1Image(organ_invertd.astype(np.uint8), affine)
            nib.save(organ_save, os.path.join(organ_seg_save_path, organ_name + '.nii.gz'))

        # Save combined label map
        pseudo_label_all = pseudo_label_all_organ(pred_hard_post, args)
        batch['pseudo_label'] = pseudo_label_all.cpu()
        BATCH = invert_transform('pseudo_label', batch, val_transforms)
        pseudo_label_invertd = np.squeeze(BATCH[0]['pseudo_label'].numpy(), axis=0)
        combined_save = nib.Nifti1Image(pseudo_label_invertd.astype(np.uint8), affine)
        nib.save(combined_save, os.path.join(case_save_path, 'combined_labels.nii.gz'))

        # Save intestinal tract binary mask (duodenum + colon + intestine/small_bowel)
        intestinal_mask = np.zeros_like(pseudo_label_invertd, dtype=np.uint8)
        for organ_idx in INTESTINAL_ORGAN_INDICES:
            intestinal_mask = np.logical_or(
                intestinal_mask, pseudo_label_invertd == organ_idx
            ).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(intestinal_mask, affine),
            os.path.join(case_save_path, 'intestinal_tract.nii.gz'),
        )

        print(f"[{index+1}/{len(test_loader)}] Saved {name_img}")


def main():
    parser = argparse.ArgumentParser(description="SuPreM 25-organ inference on flat nii.gz directory")
    parser.add_argument('--data_root_path', required=True, help='Directory containing .nii.gz files')
    parser.add_argument('--save_dir', required=True, help='Output directory')
    parser.add_argument('--checkpoint', required=True, help='Path to SuPreM checkpoint')
    parser.add_argument('--backbone', default='swinunetr', help='Backbone architecture')
    parser.add_argument('--num_workers', type=int, default=4)

    # Spatial parameters
    parser.add_argument('--roi_x', type=int, default=96)
    parser.add_argument('--roi_y', type=int, default=96)
    parser.add_argument('--roi_z', type=int, default=96)
    parser.add_argument('--space_x', type=float, default=1.5)
    parser.add_argument('--space_y', type=float, default=1.5)
    parser.add_argument('--space_z', type=float, default=1.5)

    # Intensity parameters
    parser.add_argument('--a_min', type=float, default=-175)
    parser.add_argument('--a_max', type=float, default=250)
    parser.add_argument('--b_min', type=float, default=0.0)
    parser.add_argument('--b_max', type=float, default=1.0)

    # Inference parameters
    parser.add_argument('--overlap', type=float, default=0.75)
    parser.add_argument('--sw_batch_size', type=int, default=4, help='Number of sliding window patches per forward pass')
    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--cpu', action="store_true", default=False)
    parser.add_argument('--create_dataset', action="store_true", default=False)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
