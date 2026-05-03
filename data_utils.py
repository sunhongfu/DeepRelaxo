from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io
import torch


def _load_mat_array(path):
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
            f"Could not find a usable 3D/4D numeric array in MAT file {path}. "
            "Each .mat file must contain a single 3D (mask / single echo) or 4D (multi-echo) numeric variable."
        )

    if len(candidates) > 1:
        variable_names = ", ".join(name for name, _ in candidates)
        raise ValueError(
            f"MAT file {path} contains multiple numeric arrays ({variable_names}). "
            "DeepRelaxo expects each .mat file to hold exactly one variable. "
            "Re-save the file with only the relevant array."
        )

    return candidates[0][1]


def load_array_with_affine(path):
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]

    if suffixes and suffixes[-1] == ".mat":
        return np.asarray(_load_mat_array(path), dtype=np.float32), None

    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32), image.affine


def load_mask_array(mask_path, reference_shape):
    if mask_path is None:
        return np.ones(reference_shape, dtype=bool)

    mask, _ = load_array_with_affine(mask_path)
    mask = mask > 0

    if mask.shape != reference_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference shape {reference_shape}"
        )

    return mask


def _is_magnitude_dicom(ds):
    image_type = list(getattr(ds, "ImageType", []))
    if not image_type:
        return True
    markers = {str(t).upper() for t in image_type}
    if "M" in markers or "MAGNITUDE" in markers:
        return True
    if {"P", "PHASE", "I", "IMAGINARY", "R", "REAL"} & markers:
        return False
    for m in markers:
        if m.startswith("M_") or m.endswith("_M"):
            return True
        if m.startswith("P_") or m.endswith("_P"):
            return False
    return True


def _dicom_to_nifti_affine(items):
    first_ds = items[0][1]
    iop = np.array(first_ds.ImageOrientationPatient, dtype=float)
    ps = first_ds.PixelSpacing
    row_step_lps = iop[3:6] * float(ps[0])
    col_step_lps = iop[0:3] * float(ps[1])
    ipp_first = np.array(first_ds.ImagePositionPatient, dtype=float)
    if len(items) > 1:
        ipp_last = np.array(items[-1][1].ImagePositionPatient, dtype=float)
        slice_step_lps = (ipp_last - ipp_first) / (len(items) - 1)
    else:
        cross = np.cross(iop[0:3], iop[3:6])
        thickness = float(getattr(first_ds, "SliceThickness", 1.0))
        slice_step_lps = cross * thickness
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_step_lps
    affine_lps[:3, 1] = col_step_lps
    affine_lps[:3, 2] = slice_step_lps
    affine_lps[:3, 3] = ipp_first
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ affine_lps


def load_dicom_files(file_paths, output_dir):
    """Parse DICOM files from a multi-echo magnitude GRE acquisition, group by
    EchoTime, sort by slice position, and save each echo as a NIfTI in `output_dir`.

    Returns a list sorted by ascending te_ms:
        [{"te_ms": float, "nifti_path": Path, "shape": tuple}, ...]
    """
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for DICOM input. Install with: pip install pydicom"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for f in file_paths:
        f = Path(f)
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except (InvalidDicomError, OSError, EOFError, AttributeError, ValueError):
            continue
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        candidates.append((f, ds))

    if not candidates:
        raise ValueError(
            "No readable DICOM files with pixel data were found in the upload. "
            "Make sure the folder contains DICOM images (extensions like .dcm, .ima, "
            ".dicom, or no extension)."
        )

    magnitude = [(f, ds) for f, ds in candidates if _is_magnitude_dicom(ds)]
    if not magnitude:
        raise ValueError(
            "No magnitude DICOMs detected (only phase/real/imaginary found). "
            "Please provide the magnitude series of a multi-echo gradient-echo (GRE) acquisition."
        )

    studies = {getattr(ds, "StudyInstanceUID", None) for _, ds in magnitude}
    studies.discard(None)
    if len(studies) > 1:
        raise ValueError(
            f"DICOMs from {len(studies)} different studies/exams detected. "
            "Please provide only the multi-echo gradient-echo magnitude images "
            "from a single acquisition."
        )

    echo_groups = {}
    for f, ds in magnitude:
        te = getattr(ds, "EchoTime", None)
        if te is None:
            continue
        te_key = round(float(te), 4)
        echo_groups.setdefault(te_key, []).append((f, ds))

    if not echo_groups:
        raise ValueError(
            "No EchoTime tag found in any DICOM. Cannot determine multi-echo grouping."
        )

    if len(echo_groups) < 2:
        only_te = next(iter(echo_groups))
        raise ValueError(
            f"Only one unique EchoTime ({only_te:g} ms) detected. DeepRelaxo requires "
            "a multi-echo GRE acquisition with at least 2 echoes. Make sure the folder "
            "contains all echoes from one multi-echo scan."
        )

    counts = {te: len(items) for te, items in echo_groups.items()}
    if len(set(counts.values())) > 1:
        details = "\n".join(f"    TE = {te:g} ms : {n} slices" for te, n in sorted(counts.items()))
        raise ValueError(
            "Echo groups have mismatched slice counts — likely DICOMs from multiple "
            "scans are mixed. Detected:\n" + details +
            "\n\nPlease provide only the multi-echo gradient-echo magnitude images "
            "from one scan."
        )

    def sort_key(item):
        _, ds = item
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            try:
                return float(ipp[2])
            except (TypeError, ValueError, IndexError):
                pass
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            try:
                return float(sl)
            except (TypeError, ValueError):
                pass
        return int(getattr(ds, "InstanceNumber", 0))

    for te in echo_groups:
        echo_groups[te].sort(key=sort_key)

    echoes = []
    for echo_idx, te in enumerate(sorted(echo_groups), start=1):
        items = echo_groups[te]
        slices = [it[1].pixel_array for it in items]
        if len({s.shape for s in slices}) > 1:
            raise ValueError(f"Inconsistent slice shapes within echo TE = {te:g} ms")
        volume = np.stack(slices, axis=-1).astype(np.float32)
        first_ds = items[0][1]
        slope = float(getattr(first_ds, "RescaleSlope", 1.0))
        intercept = float(getattr(first_ds, "RescaleIntercept", 0.0))
        if slope != 1.0 or intercept != 0.0:
            volume = volume * slope + intercept
        affine = _dicom_to_nifti_affine(items)
        nii_path = output_dir / f"dcm_converted_to_nii_e{echo_idx}.nii"
        nib.save(nib.Nifti1Image(volume, affine), str(nii_path))
        echoes.append({
            "te_ms": float(te),
            "nifti_path": nii_path,
            "shape": tuple(volume.shape),
        })

    shapes = {e["shape"] for e in echoes}
    if len(shapes) > 1:
        details = "\n".join(f"    TE = {e['te_ms']:g} ms : shape {e['shape']}" for e in echoes)
        raise ValueError("Echoes have mismatched volume shapes:\n" + details)

    return echoes


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

        magnitude_entries.append({"path": magnitude_path})
        te_values_ms.append(te_ms)

    return magnitude_entries, torch.tensor(te_values_ms, dtype=torch.float32)
