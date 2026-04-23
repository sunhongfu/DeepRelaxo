import numpy as np
import nibabel as nib
import torch
from torch import nn
from pathlib import Path

from UNet_upsample import Unet

CODE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = CODE_DIR / "Unet_140.pkl"
file_type = '.nii'
_UNET_MODELS = {}


# =========================
# UTILS
# =========================
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_unet_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or get_device()
    model = nn.DataParallel(Unet(4, 32, 1, 1)).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)['model_state']
    except RuntimeError as exc:
        if "failed finding central directory" in str(exc):
            raise RuntimeError(
                f"Could not read UNet checkpoint at {checkpoint_path}. "
                "The file appears to be incomplete or corrupted. Replace it with a "
                "fresh copy of Unet_140.pkl and rerun the pipeline."
            ) from exc
        raise

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def get_unet_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or get_device()
    cache_key = (str(Path(checkpoint_path).resolve()), str(device))

    if cache_key not in _UNET_MODELS:
        _UNET_MODELS[cache_key] = load_unet_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    return _UNET_MODELS[cache_key]


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


def Unet_inference(img_data, mask, inference_model=None, device=None):
    device = device or get_device()
    inference_model = inference_model or get_unet_model(device=device)

    img_data, pads = pad_image(img_data)

    x = torch.from_numpy(img_data[np.newaxis, np.newaxis]).float().to(device)

    with torch.no_grad():
        pred = inference_model(x)

    out = pred.squeeze().cpu().numpy()
    out = unpad_image(out, pads)
    out[mask <= 0] = 0

    return out


# =========================
# CORE FUNCTION
# =========================
def run_unet_on_folder(transformer_dir, root_data_dir, save_dir, checkpoint_path=CHECKPOINT_PATH):

    transformer_dir = Path(transformer_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = get_unet_model(checkpoint_path=checkpoint_path, device=device)
    mask = nib.load(str(Path(root_data_dir) / 'BET_mask.nii')).get_fdata()

    files = sorted(transformer_dir.glob('*.nii'))

    for path in files:
        data = nib.load(str(path)).get_fdata()

        result = Unet_inference(data, mask, inference_model=model, device=device)

        save_name = path.name.replace("trans", "unet")
        save_path = save_dir / save_name

        nib.save(
            nib.Nifti1Image(result, nib.load(str(path)).affine),
            str(save_path)
        )

        print(f"Saved UNet: {save_path}")
