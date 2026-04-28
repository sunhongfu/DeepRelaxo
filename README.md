# DeepRelaxo

DeepRelaxo: Fast Mono-Exponential Magnitude Brain R2* Mapping With Reduced Echoes Using Self-Supervised Deep Learning (MRM paper).

This repository contains a PyTorch inference pipeline for the DeepRelaxo cascade: an estimator stage based on Transformer-MLP followed by a denoiser stage based on 3D-UNet.

Pretrained checkpoints and demo data are hosted on Hugging Face:
[sunhongfu/DeepRelaxo](https://huggingface.co/sunhongfu/DeepRelaxo/tree/main)

## Layout

- `app.py` — Gradio web app for browser-based inference.
- `run_deeprelaxo_pipeline.py` — runs the estimator stage followed by the denoiser stage.
- `run_estimator_stage.py` — estimator-stage inference utilities.
- `run_denoiser_stage.py` — single-map denoiser inference utilities.
- `transformer_mlp_model.py` and `unet3d_model.py` — model architectures.

---

## Quick Start

### 1. Get the code

**Option A — Git**

```bash
git clone https://github.com/sunhongfu/DeepRelaxo.git
cd DeepRelaxo
```

**Option B — Download ZIP**

1. Open the GitHub repository page.
2. Click **Code** → **Download ZIP**.
3. Unzip and open a terminal in the folder.

---

### 2. Set up Python environment

You need Python 3.10 or 3.11. Check your version:

```bash
python --version
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH** during installation.

**Create and activate a virtual environment:**

macOS / Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt. Run this activation command each time you open a new terminal.

---

### 3. Install PyTorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your OS and CUDA version, and copy the install command. For example:

**CUDA 12.4 (recommended if you have an NVIDIA GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**CPU only (slower, but works without a GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Download checkpoints

```bash
python run_deeprelaxo_pipeline.py --download-checkpoints
```

Checkpoints are saved to `checkpoints/` (~350 MB total). This only needs to be done once.

---

### 6. Run

Choose the web app (recommended) or the command-line interface.

---

## Web App

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Usage

1. **Upload echo files** — click *Upload Echo Files* to select magnitude images. You can repeat this to add files from different folders. Files are automatically sorted in natural numeric order (e.g. mag1, mag2, …, mag10). Confirm the processing order before running.
2. **Set echo times** — enter *First TE* and *Echo Spacing*, then click *Fill Echo Times ↓* to populate the TE field. For irregular echo spacings, type the comma-separated values directly.
3. **Brain mask** — optionally upload a BET mask. If omitted, all voxels are processed.
4. **Run Pipeline** — log output streams live. The Step 1 result (`R2s_transformer_mlp.nii`) is available to download as soon as it is ready, before Step 2 finishes.
5. **Download** — the final output (`R2s_deeprelaxo.nii`) appears in the Result panel when complete.

To try with the included demo data, first download it:

```bash
python run_deeprelaxo_pipeline.py --download-demo
```

Then click **Load Demo Data** in the app to pre-fill all inputs, and hit **Run Pipeline**.

---

## Command-Line Interface

The pipeline can also be driven from the terminal using a YAML config file or explicit arguments.

### Config file

```bash
python run_deeprelaxo_pipeline.py --config config.yaml
```

Example `config.yaml`:

```yaml
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

For a single 4D NIfTI:

```yaml
data_dir: Data/your_subject
echo_4d: magnitudes_4d.nii.gz
te_ms: [4.9, 9.9, 14.8]
mask: BET_mask.nii
```

### Direct CLI

Multiple 3D NIfTI echoes:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_files mag1.nii mag2.nii mag3.nii \
  --te_ms 4.9 9.9 14.8 \
  --mask BET_mask.nii
```

Single 4D NIfTI:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_4d magnitudes_4d.nii.gz \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

MATLAB inputs (one file per echo):

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_files mag1.mat mag2.mat mag3.mat \
  --te_ms 4.9 9.9 14.8 \
  --variable magnitude \
  --mask BET_mask.nii
```

MATLAB 4D:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_4d magnitudes_4d.mat \
  --variable magnitudes \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

For MATLAB inputs, `variable` selects the array inside the `.mat` file. If omitted and the file contains exactly one usable array, it is selected automatically.

### Config notes

- `data_dir`, `transformer_out`, and `deeprelaxo_out` are resolved relative to the config file.
- Echo file paths are resolved relative to `data_dir`.
- If `mask` is omitted, an all-ones mask is used.
- Default `transformer_batch_size` is `50000`; reduce if GPU memory is limited.
- Checkpoints are loaded from `checkpoints/transformer_mlp_epoch_80.pth` and `checkpoints/unet3d_epoch_140.pth`.

---

## Data and Checkpoints

Large files (checkpoints, demo data, inference outputs) are excluded from git. Download them with:

```bash
python run_deeprelaxo_pipeline.py --download-checkpoints
python run_deeprelaxo_pipeline.py --download-demo
```

Manual download: [https://huggingface.co/sunhongfu/DeepRelaxo/tree/main](https://huggingface.co/sunhongfu/DeepRelaxo/tree/main)

Place files as follows:

```text
DeepRelaxo/
├── checkpoints/
│   ├── transformer_mlp_epoch_80.pth
│   └── unet3d_epoch_140.pth
├── demo/
│   ├── BET_mask.nii
│   ├── mag1.nii
│   ├── mag2.nii
│   ├── mag3.nii
│   ├── mag4.nii
│   └── mag5.nii
└── ...
```
