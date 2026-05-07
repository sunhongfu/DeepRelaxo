#!/usr/bin/env python3
"""
dicom_to_nifti — convert multi-echo GRE magnitude DICOMs into NIfTI files +
a params.json that DeepRelaxo can load directly.

DICOMs are walked recursively from one or more folders, magnitude images are
filtered (using ImageType, ComplexImageComponent, and the GE private tag),
grouped by EchoTime, and slices sorted by ImagePositionPatient. The output
is one 3D NIfTI per echo plus a `params.json` containing TE values and
voxel size — everything `run_deeprelaxo_pipeline.py --from_converted` needs.

Run `python dicom_to_nifti.py --help` for usage examples.
"""

import argparse
import json
import sys
from pathlib import Path

# data_utils.py lives next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_dicom_files


_EXAMPLES = """\
examples:
  # A single folder of multi-echo magnitude DICOMs (the typical case):
  python dicom_to_nifti.py --dicom_dir /path/to/magnitude_dicoms

  # Specify the output folder (default: ./dicom_converted):
  python dicom_to_nifti.py --dicom_dir /path/to/dicoms --out_dir ./converted

  # Then run the pipeline straight from the converted folder:
  python run_deeprelaxo_pipeline.py --from_converted ./dicom_converted \\
                                    --mask BET_mask.nii

outputs (in --out_dir):
  dcm_converted_to_nii_e1.nii, dcm_converted_to_nii_e2.nii, ...
                    — one 3D magnitude NIfTI per echo (echoes ascending by TE)
  params.json       — TE list (ms), voxel size (mm), echo filenames
"""


def _walk_files(folder, parser):
    p = Path(folder)
    if not p.is_dir():
        parser.error(f"Not a directory: {folder}")
    files = [str(c) for c in p.rglob("*") if c.is_file()]
    if not files:
        parser.error(f"No files found in: {folder}")
    print(f"Found {len(files)} files in {folder}")
    return files


def _voxel_size_from_first_echo(nifti_path):
    """Read voxel size (mm) from the NIfTI header — fast, no pixel decode."""
    try:
        import nibabel as nib
        zooms = nib.load(str(nifti_path)).header.get_zooms()
        return [float(z) for z in zooms[:3]]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        prog="dicom_to_nifti.py",
        description=__doc__.strip().split("\n\n", 1)[0],
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dicom_dir", required=True, metavar="DIR",
        help="Folder of multi-echo GRE magnitude DICOMs. Walked recursively. "
             "Non-magnitude images (phase / real / imaginary) are filtered "
             "out automatically.",
    )
    parser.add_argument(
        "--out_dir", default="./dicom_converted", metavar="DIR",
        help="Where to write the converted NIfTI files and params.json "
             "(default: %(default)s).",
    )
    args = parser.parse_args()

    file_paths = _walk_files(args.dicom_dir, parser)

    print(f"\nParsing {len(file_paths)} DICOM file(s)…")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        echoes = load_dicom_files(file_paths, out_dir)
    except Exception as exc:
        parser.error(f"DICOM parsing failed: {exc}")

    te_ms = [float(e["te_ms"]) for e in echoes]
    te_ms_string = ", ".join(f"{t:g}" for t in te_ms)
    voxel = _voxel_size_from_first_echo(echoes[0]["nifti_path"])
    voxel_string = " ".join(f"{v:.4g}" for v in voxel) if voxel else ""

    params = {
        "te_ms":              te_ms,
        "te_ms_string":       te_ms_string,
        "voxel_size_mm":      voxel,
        "voxel_size_string":  voxel_string,
        "n_echoes":           len(echoes),
        "echo_files":         [Path(e["nifti_path"]).name for e in echoes],
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))

    print()
    print("=" * 60)
    print("✅ Conversion complete")
    print("=" * 60)
    print(f"Output dir   : {out_dir}")
    for i, e in enumerate(echoes, 1):
        print(f"  Echo {i}: {Path(e['nifti_path']).name}    "
              f"TE = {e['te_ms']:g} ms    shape {e['shape']}")
    print(f"Parameters   : params.json")
    print()
    print("─── Acquisition values ───")
    print(f"  Echo Times (ms)  : {te_ms_string}")
    if voxel:
        print(f"  Voxel size (mm)  : {voxel_string}")
    print("──────────────────────────")
    print()
    print("Next step:")
    print(f"  python run_deeprelaxo_pipeline.py --from_converted {out_dir} \\")
    print(f"      --mask <BET_mask.nii>")


if __name__ == "__main__":
    main()
