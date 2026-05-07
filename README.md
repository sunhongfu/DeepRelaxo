# DeepRelaxo

**[Fast Mono-Exponential Magnitude Brain R2* Mapping With Reduced Echoes Using Self-Supervised Deep Learning](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70405?af=R)**

[MRM 2025](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70405?af=R) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/DeepRelaxo/tree/main) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

This repository contains a PyTorch inference pipeline for the DeepRelaxo cascade: an estimator stage based on Transformer-MLP followed by a denoiser stage based on 3D-UNet.

**Jump to:** [Highlights](#highlights) · [Layout](#layout) · [Quick Start](#quick-start) · [DICOM → NIfTI conversion](#dicom--nifti-conversion) · [Web App](#web-app) · [Command-Line Interface](#command-line-interface) · [Run Demo Examples](#run-demo-examples)

## Highlights

- **NIfTI / MAT input** — multiple 3D echoes or a single 4D volume (`.nii`, `.nii.gz`, `.mat` v5/v7.3 all supported).
- **DICOM → NIfTI converter** — standalone `dicom_to_nifti.py` (byte-identical with the copies in iQSM and iQSM+). One folder or many (`--dicom_dir A B …`); auto-classifies by `ImageType` / `ComplexImageComponent` / GE private tag `(0043, 102f)`; writes a single NIfTI per modality (3D for single-echo, 4D for multi-echo) plus a `params.json` ready for `--from_converted`. Modality flags (`--phase_dir`, `--mag_dir`, …) are a rescue path for mis-tagged DICOMs. Mixed-modality folders are fine — DeepRelaxo only consumes the magnitude output.
- **Browser-based UI** — collapsible sections, live progress, slice slider, shape verification (echo-to-echo and mask-vs-magnitudes), brain-mask preview alongside the R2* maps, per-run "equivalent CLI command" log entry, port auto-fallback (`7860 → 7861 → …`), and SSH-aware launch (the auto-open browser step is skipped on remote hosts and a port-forward hint is printed instead).
- **Two outputs** — Step 1 (`R2s_transformer_mlp.nii`) is downloadable as soon as it's ready; Step 2 (`R2s_deeprelaxo.nii`) finalises afterwards.

## Layout

- `app.py` — Gradio web app for browser-based inference.
- `run_deeprelaxo_pipeline.py` — runs the estimator stage followed by the denoiser stage.
- `run_estimator_stage.py` — estimator-stage inference utilities.
- `run_denoiser_stage.py` — single-map denoiser inference utilities.
- `dicom_to_nifti.py` — standalone, self-contained DICOM → NIfTI converter. Byte-identical with the copies in iQSM and iQSM+; writes a single magnitude NIfTI (3D / 4D) plus a `params.json` ready for `--from_converted`.
- `data_utils.py` — NIfTI / MAT / DICOM loaders and shape utilities (the in-repo magnitude-only DICOM loader is what `run_deeprelaxo_pipeline.py --dicom_dir` uses; for new workflows prefer the standalone `dicom_to_nifti.py`).
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

## DICOM → NIfTI conversion

If your data is a folder of raw DICOMs from a multi-echo GRE acquisition, convert it once with the standalone script before running DeepRelaxo. The script is **byte-identical with the copies in iQSM and iQSM+** — independent of the downstream pipeline. Pick whichever repo's copy is closest at hand.

### Two modes

**Normal path — `--dicom_dir DIR [DIR …]`** &nbsp;·&nbsp; auto-classifies by DICOM `ImageType`, `ComplexImageComponent`, and the GE private tag `(0043, 102f)`. Use this whenever your tags are reliable. DeepRelaxo only needs magnitude, so passing extra modalities is harmless — they're written to disk but not consumed:

```bash
# Single mixed folder of DICOMs (any extension or none — .dcm, .ima, .dicom):
python dicom_to_nifti.py --dicom_dir /path/to/dicoms

# Multiple folders walked together (e.g. magnitude + phase in separate dirs):
python dicom_to_nifti.py --dicom_dir /path/to/mag /path/to/phase

# Specify the output folder (default: ./dicom_converted):
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --out_dir ./my_subject_converted
```

**Rescue path — `--mag_dir`** &nbsp;·&nbsp; for when magnitude DICOMs are mis-tagged (e.g. `ImageType = ORIGINAL/PRIMARY/OTHER` and `--dicom_dir` would route them to the wrong bucket). The script trusts the caller and force-places every file into the magnitude bucket:

```bash
python dicom_to_nifti.py --mag_dir /path/to/magnitude_dicoms
```

The `--phase_dir`, `--real_dir`, `--imag_dir` rescue flags exist too but produce phase / real / imaginary NIfTIs that DeepRelaxo doesn't consume — if you don't have any magnitude DICOMs at all, you'll have to derive magnitude yourself before running DeepRelaxo. The two modes can't be mixed.

After conversion the script writes (names depend on echo count):

```text
converted/
├── dcm_converted_magnitude[_4d].nii.gz   # 3D for single-echo, 4D for multi-echo
├── dcm_converted_phase[_4d].nii.gz       # written if phase was present (DeepRelaxo ignores it)
└── params.json
```

…and prints a copy-paste-friendly summary:

```text
─── Acquisition values (paste these into the web app) ───
  Echo Times (ms)  : 4.9, 9.9, 14.8, 19.8, 24.7
  Voxel size (mm)  : 1 1 1
  B0 (Tesla)       : 3.0
─────────────────────────────────────────────────────────
```

`params.json` carries machine-readable values (`te_ms`, `voxel_size_mm`, `b0_T`, `b0_direction`) and **copy-paste strings** (`te_ms_string`, `voxel_size_string`, `b0_direction_string`) formatted exactly the way the web app's input field expects them. Open the JSON, copy the relevant string, paste into the form. Or skip the form and use `--from_converted` (auto-fills everything from the same JSON):

```bash
# CLI — auto-loads TEs and the magnitude NIfTI path from params.json:
python run_deeprelaxo_pipeline.py --from_converted ./dicom_converted \
                                  --mask BET_mask.nii

# Web app — upload dcm_converted_magnitude[_4d].nii.gz to "MRI Magnitudes"
# (4D files are accepted as multi-echo), paste te_ms_string from
# params.json into the Echo Times field:
python app.py
```

> **`--out_dir` is overwritten in place on each run.** A single consolidated `params.json` is written. If `params.json` or any `dcm_converted_*.nii(.gz)` already exists, the script lists them and prints a clear warning before overwriting. Use a fresh `--out_dir` per subject / acquisition.

> **Why isn't this in the web app?** Browsers don't handle thousand-file folder uploads well: tempfile creation per file hits the OS open-file limit (`OSError: Too many open files`), and the "Upload N files to this site?" prompt makes some users worry their data is being uploaded to a remote server. Running the converter locally avoids both — and DICOM → NIfTI is a one-time step, so re-running the pipeline after that is fast.

---

## Web App

```bash
python app.py
```

The app picks port `7860` by default; if it's busy, it falls back automatically (`7861`, `7862`, …). Your default browser opens automatically at the dark-themed URL once the server is ready (no flash of light theme).

### Usage walk-through

The page is organised top-to-bottom; each section is a collapsible accordion.

#### 1. MRI Magnitudes

Click **Add NIfTI / MAT Magnitudes** and pick:
- multiple 3D files (one per echo) — `.nii`, `.nii.gz`, or `.mat`; or
- a single 4D volume of shape `(X, Y, Z, n_echoes)`.

Files with unsupported extensions are dropped with a Gradio warning toast. You can add files in batches.

`.mat` files are accepted in **both v5 (default `save`) and v7.3 (`save -v7.3`)** formats; each `.mat` file must contain exactly one numeric array.

> **Have raw DICOMs?** The web app deliberately doesn't accept whole DICOM folders — uploading thousands of files through a browser is unreliable (file-handle exhaustion, mid-transfer drops) and the privacy prompts confuse users. Convert them locally with [`dicom_to_nifti.py`](#dicom--nifti-conversion) instead, then upload the resulting NIfTI files here.

#### 2. Processing Order
Lists every magnitude file that will be fed to the pipeline (sorted in natural numeric order: `mag1`, `mag2`, …, `mag10`). Below the list, a one-line shape summary tells you whether all files share the same volume dimensions — and if not, lists each per-file shape so you can spot the outlier.

- Click ✕ next to any file to remove it (≥2 files only).
- Click **✕ Remove all magnitudes** to wipe the list.

#### 3. Echo Times (ms)
A single textbox accepts two equivalent formats:

- **Comma-separated values** — one per echo, irregular spacings allowed: `2.4, 3.6, 9.2, 20.8`
- **Compact `first_TE : spacing : count`** — uniform spacing only: `4.9 : 5 : 5` expands to `4.9, 9.9, 14.8, 19.8, 24.7`

#### 4. Brain Mask *(optional)*
Click **Select Brain Mask** to provide a BET (or any binary) mask. Supported: `.nii`, `.nii.gz`, `.mat` (v5 or v7.3).

After upload, the field shows: `Loaded: BET_mask.nii · Shape: 192 × 256 × 176 · dtype: uint8 · ✓ matches magnitudes`. If the mask shape doesn't match the spatial dims of your magnitudes, you get a clear warning so you can fix it before running.

Click **✕ Remove Brain Mask** to clear it. **Without a mask, all voxels are processed** — significantly slower than processing only the masked brain region.

#### 5. Hyper-parameters *(collapsed by default)*
**Voxel Batch Size** — defaults to 50,000. Reduce if you run out of GPU memory.

#### 6. Run Pipeline
Hit the green **Run Pipeline** button. Three sections appear in sequence:

- **Log** — streaming console output, including a *RUN CONFIGURATION* block that prints the equivalent command-line invocation so you can reproduce the run from a terminal.
- **Results** — `R2s_transformer_mlp.nii` (Step 1) downloadable as soon as it's ready, then `R2s_deeprelaxo.nii` (Step 2) once the denoiser finishes.
- **Visualisation** — middle-slice grayscale preview of both maps, with a **Z-slice slider** (Prev / Next buttons or drag) to scrub through the volume. Display window (`R2*` min / max in s⁻¹) is editable. If you supplied a brain mask, it's previewed below the R2* maps (sharing the same slice slider) so you can spot orientation mismatches at a glance.

GPU memory is released between runs, so you can upload a new dataset and re-run without restarting the page.

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

From a folder produced by the [standalone `dicom_to_nifti.py`](#dicom--nifti-conversion) — TE values and the magnitude NIfTI path are read from `params.json`:

```bash
python run_deeprelaxo_pipeline.py \
  --from_converted Data/your_subject/dicom_converted \
  --mask BET_mask.nii
```

`--from_converted` reads `dcm_converted_magnitude[_4d].nii.gz` (3D for single-echo, 4D for multi-echo) and the TE list from `params.json`. This is the recommended path.

DICOM folder, one-shot (skips the standalone-conversion step — convenient if you don't plan to re-run on the same data):

```bash
python run_deeprelaxo_pipeline.py \
  --dicom_dir Data/your_subject_dicoms \
  --mask BET_mask.nii
```

The folder is walked recursively, magnitude images are filtered, grouped by `EchoTime`, sorted by `ImagePositionPatient`, and saved as one NIfTI per echo (`dcm_converted_to_nii_e1.nii`, `dcm_converted_to_nii_e2.nii`, …) inside `<transformer_out_parent>/dicom_converted_nii/`. TE values come from headers — pass `--te_ms` only to override. This is the legacy in-repo magnitude-only loader; for repeat runs over the same dataset, prefer the standalone `dicom_to_nifti.py` + `--from_converted` (output schema is unified and `params.json` carries copy-paste-ready strings).

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
- Echo file and mask paths are resolved relative to `data_dir`. **Absolute paths are kept verbatim** (so you can mix files from different folders).
- `--data_dir` is **optional in direct CLI mode** — defaults to the current working directory if omitted.
- If `mask` is omitted, an all-ones mask is used.
- Default `transformer_batch_size` is `50000`; reduce if GPU memory is limited.
- Checkpoints are loaded from `checkpoints/transformer_mlp_epoch_80.pth` and `checkpoints/unet3d_epoch_140.pth`.

---

## Run Demo Examples

Once you've downloaded the demo data (see step 3 of the Quick Start), the easiest way to verify your install is to run the demo from the command line. The demo is a 5-echo GRE acquisition with TE = 4.9, 9.9, 14.8, 19.8, 24.7 ms.

### Option 1 — Command-line with direct arguments

```bash
python run_deeprelaxo_pipeline.py \
  --data_dir demo \
  --echo_files mag1.nii mag2.nii mag3.nii mag4.nii mag5.nii \
  --te_ms 4.9 9.9 14.8 19.8 24.7 \
  --mask BET_mask.nii
```

### Option 2 — Command-line with a YAML config

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

### Option 3 — Run the demo through the web app

Launch the web app:

```bash
python app.py
```

Under **MRI Magnitudes**, click *Add NIfTI / MAT Magnitudes* and pick all five `demo/mag*.nii` files. In **Echo Times**, paste `4.9, 9.9, 14.8, 19.8, 24.7` (or the compact form `4.9 : 5 : 5`). In **Brain Mask**, click *Select Brain Mask* and pick `demo/BET_mask.nii`. Hit **Run Pipeline**.

All three options produce the same outputs: a Step 1 R2* map (`R2s_transformer_mlp.nii`) and a final denoised R2* map (`R2s_deeprelaxo.nii`).
