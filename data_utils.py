from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io
import torch


def _load_mat_array(path, variable=None):
    if variable is not None:
        try:
            mat_data = scipy.io.loadmat(path)
            if variable in mat_data:
                return np.asarray(mat_data[variable])
        except NotImplementedError:
            pass

        with h5py.File(path, "r") as handle:
            if variable not in handle:
                raise KeyError(f"MAT file {path} does not contain variable '{variable}'")
            return np.asarray(handle[variable])

    try:
        mat_data = scipy.io.loadmat(path)
        candidates = [
            (name, np.asarray(value))
            for name, value in mat_data.items()
            if not name.startswith("__") and np.issubdtype(np.asarray(value).dtype, np.number)
        ]
    except NotImplementedError:
        candidates = []

    if not candidates:
        with h5py.File(path, "r") as handle:
            candidates = [
                (name, np.asarray(dataset))
                for name, dataset in handle.items()
                if isinstance(dataset, h5py.Dataset) and np.issubdtype(np.asarray(dataset).dtype, np.number)
            ]

    candidates = [
        (name, np.squeeze(array))
        for name, array in candidates
        if np.asarray(array).size > 1
    ]
    candidates = [
        (name, array)
        for name, array in candidates
        if array.ndim >= 3
    ]

    if not candidates:
        raise ValueError(
            f"Could not find a usable 3D/4D numeric variable in MAT file {path}. Specify 'variable'."
        )

    if len(candidates) > 1:
        variable_names = ", ".join(name for name, _ in candidates)
        raise ValueError(
            f"MAT file {path} contains multiple usable variables ({variable_names}). Specify 'variable'."
        )

    return candidates[0][1]


def load_array_with_affine(path, variable=None):
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]

    if suffixes and suffixes[-1] == ".mat":
        return np.asarray(_load_mat_array(path, variable=variable), dtype=np.float32), None

    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32), image.affine


def load_mask_array(mask_path, reference_shape, variable=None):
    if mask_path is None:
        return np.ones(reference_shape, dtype=bool)

    mask, _ = load_array_with_affine(mask_path, variable=variable)
    mask = mask > 0

    if mask.shape != reference_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference shape {reference_shape}"
        )

    return mask


def resolve_echo_entries(echoes, base_dir):
    if not echoes:
        raise ValueError("Config must contain a non-empty 'echoes' list")

    base_dir = Path(base_dir)
    magnitude_entries = []
    te_values_ms = []

    for index, entry in enumerate(echoes, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Echo entry {index} must be an object with 'file' and 'te_ms'")

        file_value = str(entry.get("file", "")).strip()
        te_value = entry.get("te_ms")
        variable_value = entry.get("variable", entry.get("key"))

        if not file_value or te_value is None:
            raise ValueError(f"Echo entry {index} must contain both 'file' and 'te_ms'")

        magnitude_path = Path(file_value)
        if not magnitude_path.is_absolute():
            magnitude_path = base_dir / magnitude_path
        if not magnitude_path.exists():
            raise FileNotFoundError(f"Echo entry {index} references missing file: {magnitude_path}")

        try:
            te_ms = float(te_value)
        except ValueError as exc:
            raise ValueError(f"Echo entry {index} has invalid te_ms value: {te_value}") from exc

        magnitude_entries.append({"path": magnitude_path, "variable": variable_value})
        te_values_ms.append(te_ms)

    return magnitude_entries, torch.tensor(te_values_ms, dtype=torch.float32)
