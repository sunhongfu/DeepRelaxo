import argparse
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

        if sum(option is not None for option in [args.echo_files, args.echo_4d]) > 1:
            parser.error("use at most one of --echo_files or --echo_4d")

        if args.echo_4d is not None:
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
                "when using direct CLI, provide either --echo_files with --te_ms or --echo_4d with --te_ms"
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
