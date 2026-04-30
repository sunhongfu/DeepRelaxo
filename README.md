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

### 2. Install dependencies

A fresh virtual environment is the recommended way — it isolates DeepRelaxo's dependencies from anything else on your system and avoids version conflicts.

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

**Install PyTorch.** Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your OS and CUDA version, and copy the install command. For example:

CUDA 12.4 (recommended if you have an NVIDIA GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CPU only (slower, but works without a GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Install remaining dependencies.** Pick one of the two options below depending on whether you want the browser-based web app:

- **Web app + Command-Line** (recommended for most users):

  ```bash
  pip install -r requirements-webapp.txt
  ```

  Adds Gradio and Matplotlib for the browser UI and slice previews.

- **Command-Line only** (lighter install, fewer dependencies, no web stack):

  ```bash
  pip install -r requirements.txt
  ```

  Skips Gradio and its ~18 transitive packages (FastAPI, Pydantic, Uvicorn, etc.). Recommended for headless servers, HPC clusters, or environments where Gradio's deps conflict with other tools.

<details>
<summary><b>Already have PyTorch installed? Click here to skip the venv setup.</b></summary>

If you have a working Python environment with PyTorch and prefer to reuse it, install one of the two requirements files above into that environment directly.

Note: this may upgrade or downgrade other packages in the environment (e.g. `pydantic`, `typing-extensions`) and can conflict with unrelated tools like TensorFlow. A clean venv avoids this. The Command-Line only install is much less likely to cause conflicts since it doesn't pull in Gradio's dependency chain.

</details>

---

### 3. Download checkpoints (and optionally demo data)

Large files (checkpoints and demo data) are excluded from git and hosted on Hugging Face: [sunhongfu/DeepRelaxo](https://huggingface.co/sunhongfu/DeepRelaxo/tree/main).

**Download checkpoints** (~350 MB, required, one-time):

```bash
python run_deeprelaxo_pipeline.py --download-checkpoints
```

**Optional — download demo data.** If you'd like to try DeepRelaxo on a provided 5-echo GRE NIfTI dataset (TE = 4.9, 9.9, 14.8, 19.8, 24.7 ms) before adapting it to your own data, also run:

```bash
python run_deeprelaxo_pipeline.py --download-demo
```

This places sample magnitude echoes and a BET mask in `demo/`. See [Run Demo Examples](#run-demo-examples) below for how to run it.

**Manual download (optional).** If the auto-download fails (e.g. behind a firewall), grab the files from the Hugging Face page above and place them as follows:

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
│   ├── mag5.nii
│   └── params.json
└── ...
```

---

### 4. Run

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

### Direct Command-Line

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
  --mask BET_mask.nii
```

MATLAB 4D:

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir Data/your_subject \
  --echo_4d magnitudes_4d.mat \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

`.mat` inputs (magnitudes or mask) must contain **a single numeric array** per file. If a `.mat` file contains multiple arrays, re-save it with only the array you want. The same applies to `.mat` masks.

### Config notes

- `data_dir`, `transformer_out`, and `deeprelaxo_out` are resolved relative to the config file.
- Echo file paths are resolved relative to `data_dir`.
- If `mask` is omitted, an all-ones mask is used.
- Default `transformer_batch_size` is `50000`; reduce if GPU memory is limited.
- Checkpoints are loaded from `checkpoints/transformer_mlp_epoch_80.pth` and `checkpoints/unet3d_epoch_140.pth`.

---

## Run Demo Examples

Once you've downloaded the demo data (see step 3 of the Quick Start), you can run it in any of the following ways. The demo is a 5-echo GRE acquisition with TE = 4.9, 9.9, 14.8, 19.8, 24.7 ms.

### Option 1 — Web app (one click)

Launch the web app:

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860), click **Load Demo Data** to pre-fill all inputs (echoes, TEs, mask), then hit **Run Pipeline**.

### Option 2 — Command-line with direct arguments

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir demo \
  --echo_files mag1.nii mag2.nii mag3.nii mag4.nii mag5.nii \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

### Option 3 — Command-line with a YAML config

Save the following as `demo_config.yaml`:

```yaml
data_dir: demo
echoes:
  - file: mag1.nii
    te_ms: 4.9
  - file: mag2.nii
    te_ms: 9.9
  - file: mag3.nii
    te_ms: 14.8
  - file: mag4.nii
    te_ms: 19.8
  - file: mag5.nii
    te_ms: 24.7
mask: BET_mask.nii
```

Then run:

```bash
python run_deeprelaxo_pipeline.py --config demo_config.yaml
```

All three options produce the same outputs: a Step 1 R2* map (`R2s_transformer_mlp.nii`) and a final denoised R2* map (`R2s_deeprelaxo.nii`).
