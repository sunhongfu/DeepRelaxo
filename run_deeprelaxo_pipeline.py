import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
from data_utils import (
    resolve_echo_entries,
)
import run_estimator_stage as estimator_stage
import run_denoiser_stage as denoiser_stage
import yaml

HF_REPO_ID = "sunhongfu/DeepRelaxo"
CHECKPOINT_FILES = [
    "transformer_mlp_epoch_80.pth",
    "unet3d_epoch_140.pth",
]
DEMO_FILES = [
    "demo/BET_mask.nii",
    "demo/mag1.nii",
    "demo/mag2.nii",
    "demo/mag3.nii",
    "demo/mag4.nii",
    "demo/mag5.nii",
    "demo/params.json",
]


def _resolve_path(config_dir, value):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def _download_files(file_list, local_subdir):
    repo_root = Path(__file__).resolve().parent
    target_dir = repo_root / local_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    for remote_path in file_list:
        local_name = Path(remote_path).name
        destination = target_dir / local_name
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type="model",
        )
        shutil.copyfile(downloaded_path, destination)
        print(f"Downloaded: {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--data_dir')
    parser.add_argument('--echo_files', nargs='+')
    parser.add_argument('--echo_4d')
    parser.add_argument(
        '--from_converted', metavar='DIR',
        help=(
            'Folder produced by `dicom_to_nifti.py` (contains '
            'dcm_converted_to_nii_e*.nii and a params.json). TE values and '
            'echo filenames are read from params.json automatically; '
            'override TEs with --te_ms if needed.'
        ),
    )
    parser.add_argument(
        '--dicom_dir',
        help=(
            'Folder of multi-echo GRE magnitude DICOMs. The folder is walked '
            'recursively, magnitude images filtered, grouped by EchoTime, '
            'sorted by ImagePositionPatient, and saved as one NIfTI per echo '
            'in <transformer_out_parent>/dicom_converted_nii/. TE values are '
            'auto-detected from headers (override with --te_ms if needed). '
            'For repeat runs over the same data, prefer running '
            '`dicom_to_nifti.py` once and using --from_converted.'
        ),
    )
    parser.add_argument('--te_ms', nargs='+', type=float)
    parser.add_argument('--mask')
    parser.add_argument('--bet_mask')
    parser.add_argument('--transformer_out')
    parser.add_argument('--deeprelaxo_out')
    parser.add_argument('--transformer_batch_size', type=int)
    parser.add_argument('--download-checkpoints', action='store_true')
    parser.add_argument('--download-demo', action='store_true')

    args = parser.parse_args()

    if args.download_checkpoints:
        _download_files(CHECKPOINT_FILES, "checkpoints")
        return

    if args.download_demo:
        _download_files(DEMO_FILES, "demo")
        return

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = yaml.safe_load(config_path.read_text())
        config_dir = config_path.parent

        required_keys = ["data_dir", "echoes"]
        if "echoes" not in config and "echo_4d" in config:
            required_keys = ["data_dir", "echo_4d", "te_ms"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Config file is missing required key(s): {missing}")

        data_dir = _resolve_path(config_dir, config["data_dir"])
        magnitude_entries = None
        magnitude_4d_path = None
        if "echoes" in config:
            magnitude_entries, te_values_ms = resolve_echo_entries(config["echoes"], base_dir=data_dir)
        else:
            echo_4d_config = config["echo_4d"]
            if isinstance(echo_4d_config, dict):
                magnitude_4d_path = {"path": _resolve_path(data_dir, echo_4d_config["file"])}
            else:
                magnitude_4d_path = {"path": _resolve_path(data_dir, echo_4d_config)}
            te_values_ms = config["te_ms"]
        transformer_out = _resolve_path(config_dir, config.get("transformer_out", "transformer_outputs"))
        deeprelaxo_out = _resolve_path(config_dir, config.get("deeprelaxo_out", "deeprelaxo_outputs"))
        bet_mask = _resolve_path(data_dir, config.get("mask", config.get("bet_mask")))
        transformer_batch_size = int(config.get("transformer_batch_size", 50000))
    else:
        # data_dir is optional in CLI mode — defaults to current working directory.
        # Relative paths in --echo_files / --echo_4d / --mask are resolved against
        # it; absolute paths bypass it.
        data_dir = Path(args.data_dir) if args.data_dir else Path.cwd()
        mask_value = args.mask if args.mask is not None else args.bet_mask
        bet_mask = _resolve_path(data_dir, mask_value)
        transformer_out = _resolve_path(Path.cwd(), args.transformer_out or "transformer_outputs")
        deeprelaxo_out = _resolve_path(Path.cwd(), args.deeprelaxo_out or "deeprelaxo_outputs")
        transformer_batch_size = int(args.transformer_batch_size or 50000)
        magnitude_entries = None
        magnitude_4d_path = None

        if sum(option is not None for option in
               [args.echo_files, args.echo_4d, args.dicom_dir,
                args.from_converted]) > 1:
            parser.error(
                "use at most one of --echo_files, --echo_4d, --dicom_dir, "
                "or --from_converted"
            )

        if args.from_converted is not None:
            cdir = _resolve_path(data_dir, args.from_converted)
            if not cdir.is_dir():
                parser.error(f"--from_converted is not a directory: {cdir}")
            params_path = cdir / "params.json"
            if not params_path.exists():
                parser.error(
                    f"params.json not found in {cdir}. Did you produce this "
                    "folder with `dicom_to_nifti.py`?"
                )
            params = json.loads(params_path.read_text())
            params_te = params.get("te_ms") or []
            mag_nii = params.get("magnitude_nifti")
            if not mag_nii or not params_te:
                parser.error(
                    f"params.json in {cdir} is missing `magnitude_nifti` / "
                    "`te_ms`. Re-run `dicom_to_nifti.py` to regenerate it."
                )
            mag_path = cdir / mag_nii
            if not mag_path.exists():
                parser.error(f"Magnitude file missing: {mag_path}")
            # The unified converter writes a single magnitude NIfTI: 3D for
            # single-echo, 4D for multi-echo. Feed it to the pipeline as 4D.
            magnitude_4d_path = {"path": mag_path}
            if args.te_ms is not None:
                if len(args.te_ms) != len(params_te):
                    parser.error(
                        f"--te_ms count ({len(args.te_ms)}) doesn't match "
                        f"echoes in params.json ({len(params_te)})."
                    )
                te_values_ms = args.te_ms
                print(f"Using user-supplied TEs (ms): {te_values_ms}")
            else:
                te_values_ms = params_te
                print(f"TEs from params.json (ms): {te_values_ms}")
        elif args.dicom_dir is not None:
            from data_utils import load_dicom_files
            dicom_path = Path(args.dicom_dir).resolve()
            if not dicom_path.is_dir():
                parser.error(f"--dicom_dir is not a directory: {dicom_path}")
            file_list = [str(p) for p in dicom_path.rglob("*") if p.is_file()]
            if not file_list:
                parser.error(f"--dicom_dir contains no files: {dicom_path}")
            nii_out_dir = transformer_out.parent / "dicom_converted_nii"
            print(f"Parsing DICOMs from {dicom_path}")
            print(f"Writing converted NIfTI files to {nii_out_dir}")
            echoes = load_dicom_files(file_list, nii_out_dir)
            print(f"Parsed {len(echoes)} echoes from DICOM:")
            for i, e in enumerate(echoes, 1):
                print(f"  Echo {i}:  {e['nifti_path'].name}  "
                      f"TE = {e['te_ms']:g} ms   shape {e['shape']}")
            magnitude_entries = [{"path": str(e["nifti_path"])} for e in echoes]
            if args.te_ms is not None:
                if len(args.te_ms) != len(echoes):
                    parser.error(
                        f"--te_ms count ({len(args.te_ms)}) doesn't match the number "
                        f"of parsed echoes ({len(echoes)})"
                    )
                te_values_ms = args.te_ms
                print(f"Using user-supplied TEs: {te_values_ms}")
            else:
                te_values_ms = [e["te_ms"] for e in echoes]
        elif args.echo_4d is not None:
            if args.te_ms is None:
                parser.error("when using --echo_4d, provide --te_ms")
            magnitude_4d_path = {"path": _resolve_path(data_dir, args.echo_4d)}
            te_values_ms = args.te_ms
        elif args.echo_files is not None:
            if args.te_ms is None:
                parser.error("when using --echo_files, provide --te_ms")
            if len(args.echo_files) != len(args.te_ms):
                parser.error("--echo_files and --te_ms must have the same length")
            echoes = [
                {"file": file_value, "te_ms": te_value}
                for file_value, te_value in zip(args.echo_files, args.te_ms)
            ]
            magnitude_entries, te_values_ms = resolve_echo_entries(echoes, base_dir=data_dir)
        else:
            parser.error(
                "when using direct CLI, provide one of: "
                "--from_converted, --dicom_dir, "
                "--echo_files with --te_ms, or --echo_4d with --te_ms"
            )

    print("\n==============================")
    print("STEP 1: ESTIMATOR")
    print("==============================")

    estimator_stage.run_estimator(
        magnitude_entries=magnitude_entries,
        magnitude_4d_path=magnitude_4d_path,
        te_values_ms=te_values_ms,
        save_dir=transformer_out,
        bet_mask_path=bet_mask,
        batch_size=transformer_batch_size
    )

    print("\n==============================")
    print("STEP 2: DENOISER")
    print("==============================")

    denoiser_stage.run_denoiser(
        input_map_path=transformer_out / "R2s_transformer_mlp.nii",
        output_path=deeprelaxo_out / "R2s_deeprelaxo.nii",
        bet_mask_path=bet_mask,
    )

    print("\n✅ DONE")


if __name__ == "__main__":
    main()
