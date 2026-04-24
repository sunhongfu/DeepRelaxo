# DeepRelaxo

DeepRelaxo: Fast Mono-Exponential Magnitude Brain R2* Mapping With Reduced Echoes Using Self-Supervised Deep Learning (MRM paper).

This repository contains a PyTorch inference pipeline for the DeepRelaxo cascade: an estimator stage based on Transformer-MLP followed by a denoiser stage based on 3D-UNet.

Pretrained checkpoints and demo data are hosted on Hugging Face:
[sunhongfu/DeepRelaxo](https://huggingface.co/sunhongfu/DeepRelaxo/tree/main)

## Layout

- `run_deeprelaxo_pipeline.py` runs the estimator stage followed by the denoiser stage.
- `run_estimator_stage.py` contains estimator-stage inference utilities.
- `run_denoiser_stage.py` contains single-map denoiser inference utilities.
- `echo_voxel_dataset.py` contains the echo-wise voxel dataset loader.
- `transformer_mlp_model.py` and `unet3d_model.py` define the model architectures.

## Quick Start

### 1. Get the code

Choose one:

**Option A - Git**

```bash
git clone https://github.com/sunhongfu/DeepRelaxo.git
cd DeepRelaxo
```

**Option B - Browser / HTTP download**

1. Open your GitHub repository page.
2. Click **Code** -> **Download ZIP**.
3. Unzip the file.
4. Open a terminal in the unzipped folder.

### 2. Download checkpoints and demo data

Pretrained checkpoints and demo files are available on Hugging Face:
[https://huggingface.co/sunhongfu/DeepRelaxo/tree/main](https://huggingface.co/sunhongfu/DeepRelaxo/tree/main)

If Python is available:

```bash
python run_deeprelaxo_pipeline.py --download-checkpoints
python run_deeprelaxo_pipeline.py --download-demo
```

On Windows, if `python` is not found, try:

```powershell
py run_deeprelaxo_pipeline.py --download-checkpoints
py run_deeprelaxo_pipeline.py --download-demo
```

Required checkpoint files:

- `transformer_mlp_epoch_80.pth`
- `unet3d_epoch_140.pth`

Recommended demo files:

- `demo/params.json`
- demo magnitude files
- demo mask file

If you download files manually, place them like this:

```text
DeepRelaxo/
├── checkpoints/
│   ├── transformer_mlp_epoch_80.pth
│   └── unet3d_epoch_140.pth
├── demo/
│   ├── params.json
│   ├── BET_mask.nii
│   ├── mag1.nii
│   ├── mag2.nii
│   ├── mag3.nii
│   ├── mag4.nii
│   └── mag5.nii
├── config.yaml
├── run_deeprelaxo_pipeline.py
└── ...
```

### 3. Run the demo

```bash
python run_deeprelaxo_pipeline.py --config config.yaml
```

## Data and Checkpoints

Large data files, generated inference outputs, and model checkpoints are intentionally ignored by git.

## Running Inference

The pipeline is driven by a YAML config file. A sample is provided at `config.yaml`.

```bash
python run_deeprelaxo_pipeline.py --config config.yaml
```

The old direct CLI style also works.

Explicit direct CLI example:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_files mag1.nii mag2.nii mag3.nii \
  --te_ms 4.9 9.9 14.8 \
  --mask BET_mask.nii \
  --transformer_out transformer_outputs \
  --deeprelaxo_out deeprelaxo_outputs
```

MATLAB direct CLI example with one file per echo:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_files mag1.mat mag2.mat mag3.mat \
  --te_ms 4.9 9.9 14.8 \
  --variable magnitude \
  --mask BET_mask.nii
```

Single 4D NIfTI direct CLI example:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_4d magnitudes_4d.nii.gz \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

Single 4D MATLAB direct CLI example:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_4d magnitudes_4d.mat \
  --variable magnitudes \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

Example config:

```yaml
# DeepRelaxo configuration

data_dir: Data/your_subject
echoes:
  - file: mag_echo_01.nii
    te_ms: 4.9
  - file: mag_echo_02.nii
    te_ms: 9.9
  - file: mag_echo_03.nii
    te_ms: 14.8

mask: BET_mask.nii
transformer_out: transformer_outputs
deeprelaxo_out: deeprelaxo_outputs
transformer_batch_size: 50000
```

Or, for a single 4D magnitude NIfTI:

```yaml
data_dir: Data/your_subject
echo_4d: magnitudes_4d.nii.gz
te_ms: [4.9, 9.9, 14.8]
mask: BET_mask.nii
```

For MATLAB inputs, each echo entry may also include `variable`, and `echo_4d` may be an object:

```yaml
data_dir: Data/your_subject
echoes:
  - file: mag_echo_01.mat
    variable: magnitude
    te_ms: 4.9
  - file: mag_echo_02.mat
    variable: magnitude
    te_ms: 9.9
```

```yaml
data_dir: Data/your_subject
echo_4d:
  file: magnitudes_4d.mat
  variable: magnitudes
te_ms: [4.9, 9.9, 14.8]
```

`data_dir`, `transformer_out`, and `deeprelaxo_out` are resolved relative to the config file location.
Use either `echoes` for multiple 3D echo files, or `echo_4d` plus `te_ms` for a single 4D volume.
Each item in `echoes` must contain a magnitude `file` and its `te_ms` value; echo `file` paths are resolved relative to `data_dir`.
For `.mat` files, you may provide `variable` to select the MATLAB variable.
If no `variable` is provided and the MAT file contains exactly one usable 3D/4D numeric variable, that variable is used automatically.
If the MAT file contains more than one usable variable, the code raises an error and asks you to specify `variable`.
If using `echo_4d`, its last dimension must match the number of `te_ms` values.
Outputs from `.mat` inputs use the input NIfTI mask affine when a NIfTI mask is provided; otherwise they fall back to an identity affine.
If provided, `mask` is also resolved relative to `data_dir` when it is not an absolute path.
If `mask` is omitted, the code uses an all-ones mask with the image shape.
If `transformer_out` is omitted, the default is `transformer_outputs`.
If `deeprelaxo_out` is omitted, the default is `deeprelaxo_outputs`.
If `transformer_batch_size` is omitted, the default is `50000`.
The denoiser runs with a fixed batch size of `1`.
Checkpoint paths are fixed by the code and loaded from the repository `checkpoints/` folder:
`checkpoints/transformer_mlp_epoch_80.pth` and `checkpoints/unet3d_epoch_140.pth`.

## Demo Metadata

If you publish demo data, it is useful to include a separate `demo/params.json` file for scan metadata and dataset description.

This serves a different purpose from `config.yaml`:
- `config.yaml` is the runnable pipeline configuration
- `demo/params.json` is descriptive metadata for the demo dataset

A starter file is provided at [demo/params.json](/home/uqhsun8/codes/DeepRelaxo/demo/params.json). You can fill in the placeholder fields with the actual demo acquisition details before publishing.

Stage 2 is modular: it denoises a single voxel-wise R2* map rather than scanning a folder. This allows users to run the denoiser independently on their own R2* NIfTI maps.
