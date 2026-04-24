import numpy as np
import nibabel as nib
import torch
from torch import nn
from pathlib import Path
from functools import lru_cache

from data_utils import load_mask_array
from unet3d_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "unet3d_epoch_140.pth"


@lru_cache(maxsize=1)
def load_model(checkpoint_path=None):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint_path}")

    model = Unet(4, 32, 1, 1)
    model = nn.DataParallel(model).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)['model_state']
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


# =========================
# UTILS
# =========================
def calculate_padding(dimension, divisible_by=16):
    remainder = dimension % divisible_by
    if remainder == 0:
        return 0, 0
    padding = divisible_by - remainder
    return padding // 2, padding - padding // 2


def pad_image(img, divisible_by=16):
    pads = [calculate_padding(s, divisible_by) for s in img.shape]
    padded = np.pad(img, pads, mode='constant')
    return padded, pads


def unpad_image(img, pads):
    slices = []
    for l, r in pads:
        slices.append(slice(l, -r if r != 0 else None))
    return img[tuple(slices)]


def unet_inference_batch(img_batch, mask, checkpoint_path=None):
    model = load_model(checkpoint_path)
    padded_images = []
    pads_list = []

    for img_data in img_batch:
        padded, pads = pad_image(img_data)
        padded_images.append(padded)
        pads_list.append(pads)

    x = torch.from_numpy(np.stack(padded_images)[:, np.newaxis]).float().to(device)

    with torch.inference_mode():
        pred = model(x)

    outputs = pred.squeeze(1).cpu().numpy()
    results = []

    for out, pads in zip(outputs, pads_list):
        unpadded = unpad_image(out, pads)
        unpadded[mask <= 0] = 0
        results.append(unpadded)

    return results


# =========================
# CORE FUNCTION
# =========================
def denoise_r2s_map(r2s_map, mask, checkpoint_path=None):
    return unet_inference_batch([r2s_map], mask, checkpoint_path=checkpoint_path)[0]


def run_denoiser(input_map_path, output_path, bet_mask_path=None, checkpoint_path=None):
    input_map_path = Path(input_map_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = nib.load(str(input_map_path))
    r2s_map = image.get_fdata()
    mask = load_mask_array(bet_mask_path, r2s_map.shape)
    result = denoise_r2s_map(r2s_map, mask, checkpoint_path=checkpoint_path)

    nib.save(
        nib.Nifti1Image(result, image.affine),
        str(output_path)
    )

    print(f"Saved DeepRelaxo: {output_path}")
