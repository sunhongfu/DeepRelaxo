import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path

from echo_wise_loader_no_loop import Dataset_Load
from transformer import Transformer

CODE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = CODE_DIR / "checkpoint_epoch_80_Transformer.pth"
BATCH_SIZE = 50000
_TRANSFORMER_MODELS = {}


# =========================
# LOAD MODEL
# =========================
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_transformer_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = nn.DataParallel(Transformer())

    # The saved checkpoint was produced before wrapping with DataParallel.
    new_state_dict = {
        f'module.{key}': value
        for key, value in checkpoint['model_state_dict'].items()
    }
    model.load_state_dict(new_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


def get_transformer_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or get_device()
    cache_key = (str(Path(checkpoint_path).resolve()), str(device))

    if cache_key not in _TRANSFORMER_MODELS:
        _TRANSFORMER_MODELS[cache_key] = load_transformer_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    return _TRANSFORMER_MODELS[cache_key]


# =========================
# CORE FUNCTION
# =========================
def estimate_r2s(subject_path, model=None, device=None, batch_size=BATCH_SIZE):
    device = device or get_device()
    model = model or get_transformer_model(device=device)
    mask = nib.load(str(subject_path / 'BET_mask.nii')).get_fdata()

    echoNos = [2, 3, 4, 5]
    values = {}

    for i in echoNos:
        results_r2s = []

        dataset = Dataset_Load(
            sim_root_path=subject_path,
            reference_invivo_path=subject_path,
            echo_no=i
        )

        loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)

        with torch.no_grad():
            for mags, tes in loader:

                mags = torch.unsqueeze(mags, -1).to(device)
                tes = torch.unsqueeze(tes, -1).to(device)

                # normalize
                median_values, _ = torch.median(mags, dim=1, keepdim=True)
                mags = mags / median_values

                r2s = model(tes, mags)
                results_r2s.append(r2s.cpu())

        results_r2s = torch.cat(results_r2s, dim=0).squeeze(-1)

        r2s_map = torch.zeros(mask.shape)
        r2s_map[mask > 0] = results_r2s

        values[i] = torch.nan_to_num(r2s_map, nan=0.0, posinf=0.0, neginf=0.0)

    return values


def save_r2s(r2s_dict, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for echo, data in r2s_dict.items():
        filename = save_dir / f"R2s_trans_{echo}.nii"
        nib.save(nib.Nifti1Image(data.numpy(), np.eye(4)), str(filename))
        print(f"Saved Transformer: {filename}")


# =========================
# WRAPPER (IMPORTANT)
# =========================
def run_transformer(subject_dir, save_dir, checkpoint_path=CHECKPOINT_PATH):
    subject_path = Path(subject_dir)
    device = get_device()
    model = get_transformer_model(checkpoint_path=checkpoint_path, device=device)

    r2s = estimate_r2s(subject_path, model=model, device=device)
    save_r2s(r2s, save_dir)
