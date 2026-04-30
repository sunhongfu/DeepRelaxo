import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from functools import lru_cache
from data_utils import load_array_with_affine, load_mask_array
from transformer_mlp_model import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "transformer_mlp_epoch_80.pth"

BATCH_SIZE = 50000


@lru_cache(maxsize=1)
def load_model(checkpoint_path=None):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = nn.DataParallel(Transformer())

    # The checkpoint was saved without DataParallel prefixes.
    new_state_dict = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


def _resolve_output_affine(magnitude_affine, bet_mask_path=None):
    if magnitude_affine is not None:
        return magnitude_affine

    if bet_mask_path is not None:
        _, mask_affine = load_array_with_affine(bet_mask_path)
        if mask_affine is not None:
            return mask_affine

    return np.eye(4, dtype=np.float32)


def load_subject_data(magnitude_entries, bet_mask_path=None):
    magnitude_arrays = []
    first_affine = None

    for entry in magnitude_entries:
        magnitude_array, affine = load_array_with_affine(entry["path"])
        if magnitude_array.ndim != 3:
            raise ValueError(
                f"Expected a 3D magnitude volume in {entry['path']}, but got shape {magnitude_array.shape}"
            )
        magnitude_arrays.append(magnitude_array)
        if first_affine is None and affine is not None:
            first_affine = affine

    shapes = [arr.shape for arr in magnitude_arrays]
    if len(set(shapes)) > 1:
        details = "\n".join(
            f"    {Path(str(entry['path'])).name}: shape {arr.shape}"
            for entry, arr in zip(magnitude_entries, magnitude_arrays)
        )
        raise ValueError(
            "Magnitude echoes have mismatched volume dimensions — all echoes "
            "must share the same shape. Detected:\n" + details
        )

    mag_list = np.stack(magnitude_arrays, axis=-1)
    mask = load_mask_array(bet_mask_path, mag_list.shape[:3])
    masked_magnitudes = torch.from_numpy(mag_list[mask])
    return _resolve_output_affine(first_affine, bet_mask_path=bet_mask_path), mask, masked_magnitudes


def load_subject_data_4d(magnitude_4d_entry, te_values_ms, bet_mask_path=None):
    mag_list, affine = load_array_with_affine(magnitude_4d_entry["path"])

    if mag_list.ndim != 4:
        raise ValueError(
            f"Expected a 4D magnitude volume at {magnitude_4d_entry['path']}, but got shape {mag_list.shape}"
        )

    if mag_list.shape[-1] != len(te_values_ms):
        raise ValueError(
            f"4D magnitude echo dimension {mag_list.shape[-1]} does not match number of TE values {len(te_values_ms)}"
        )

    mask = load_mask_array(bet_mask_path, mag_list.shape[:3])
    masked_magnitudes = torch.from_numpy(mag_list[mask])
    return _resolve_output_affine(affine, bet_mask_path=bet_mask_path), mask, masked_magnitudes


def batched_voxel_inference(model, magnitudes, te_values_ms, batch_size):
    total_voxels = magnitudes.shape[0]
    results = torch.empty(total_voxels, dtype=torch.float32)
    echo_no = len(te_values_ms)
    tes = te_values_ms.view(1, echo_no, 1).to(device)

    with torch.inference_mode():
        for start in range(0, total_voxels, batch_size):
            end = min(start + batch_size, total_voxels)
            mags = magnitudes[start:end, :echo_no].to(device).unsqueeze(-1)

            median_values = torch.median(mags, dim=1, keepdim=True).values.clamp_min_(1e-8)
            mags = mags / median_values

            batch_tes = tes.expand(end - start, -1, -1)
            results[start:end] = model(batch_tes, mags).squeeze(-1).cpu()

    return results


# =========================
# CORE FUNCTION
# =========================
def estimate_r2s(
    magnitude_entries=None,
    magnitude_4d_path=None,
    te_values_ms=None,
    bet_mask_path=None,
    checkpoint_path=None,
    batch_size=BATCH_SIZE,
):
    model = load_model(checkpoint_path)
    te_values_ms = torch.as_tensor(te_values_ms, dtype=torch.float32)
    if (magnitude_entries is None) == (magnitude_4d_path is None):
        raise ValueError("Provide exactly one of magnitude_entries or magnitude_4d_path")

    if magnitude_entries is not None:
        affine, mask, magnitudes = load_subject_data(magnitude_entries, bet_mask_path=bet_mask_path)
    else:
        affine, mask, magnitudes = load_subject_data_4d(
            magnitude_4d_path,
            te_values_ms,
            bet_mask_path=bet_mask_path,
        )

    results_r2s = batched_voxel_inference(model, magnitudes, te_values_ms, batch_size)
    r2s_map = torch.zeros(mask.shape, dtype=torch.float32)
    r2s_map[mask > 0] = results_r2s
    r2s_map = torch.nan_to_num(r2s_map, nan=0.0, posinf=0.0, neginf=0.0)

    return r2s_map, affine


def save_r2s(r2s_map, save_dir, affine):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = save_dir / "R2s_transformer_mlp.nii"
    nib.save(nib.Nifti1Image(r2s_map.numpy(), affine), str(filename))
    print(f"Saved Estimator: {filename}")


# =========================
# WRAPPER (IMPORTANT)
# =========================
def run_estimator(
    magnitude_entries=None,
    magnitude_4d_path=None,
    te_values_ms=None,
    save_dir=None,
    bet_mask_path=None,
    checkpoint_path=None,
    batch_size=BATCH_SIZE,
):
    r2s, affine = estimate_r2s(
        magnitude_entries=magnitude_entries,
        magnitude_4d_path=magnitude_4d_path,
        te_values_ms=te_values_ms,
        bet_mask_path=bet_mask_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size
    )
    save_r2s(r2s, save_dir, affine)
