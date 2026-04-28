import re
import sys
import queue
import shutil
import tempfile
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import run_estimator_stage as estimator_stage
import run_denoiser_stage as denoiser_stage

_pipeline_lock = threading.Lock()


class _QueueWriter:
    def __init__(self, q, orig):
        self._q = q
        self._orig = orig

    def write(self, text):
        if text.strip():
            self._q.put(text.rstrip())
        if self._orig:
            self._orig.write(text)
        return len(text)

    def flush(self):
        if self._orig:
            self._orig.flush()

    def isatty(self):
        return False


def _natural_key(f):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", _to_path(f).name)]


def _to_path(f):
    if f is None:
        return None
    if isinstance(f, str):
        return Path(f)
    if hasattr(f, "path"):
        return Path(f.path)
    if hasattr(f, "name"):
        return Path(f.name)
    return Path(str(f))


def _make_slice_image(nii_path):
    img = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.float32)
    mid = data.shape[2] // 2
    sl = np.rot90(data[:, :, mid])
    nonzero = sl[sl > 0]
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size else float(sl.max())
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(sl, cmap="hot", vmin=0, vmax=vmax, aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", pad_inches=0, dpi=150, facecolor="black")
    plt.close(fig)
    return tmp.name


def _run_thread(job, work_dir, mode, echo_paths, te_list, mask_path, batch_size, mat_variable=None):
    log_q = job["log_queue"]
    orig = sys.stdout
    sys.stdout = _QueueWriter(log_q, orig)
    try:
        with _pipeline_lock:
            job["status"] = "running"
            transformer_out = work_dir / "transformer_outputs"
            deeprelaxo_out = work_dir / "deeprelaxo_outputs"

            print("============================")
            print("STEP 1: ESTIMATOR")
            print("============================")

            if mode == "4d":
                estimator_stage.run_estimator(
                    magnitude_4d_path={"path": echo_paths[0], "variable": mat_variable},
                    te_values_ms=te_list,
                    save_dir=transformer_out,
                    bet_mask_path=mask_path,
                    batch_size=batch_size,
                )
            else:
                estimator_stage.run_estimator(
                    magnitude_entries=[{"path": p, "variable": mat_variable} for p in echo_paths],
                    te_values_ms=te_list,
                    save_dir=transformer_out,
                    bet_mask_path=mask_path,
                    batch_size=batch_size,
                )

            job["step1_result"] = str(transformer_out / "R2s_transformer_mlp.nii")
            job["step1_image"] = _make_slice_image(job["step1_result"])

            print("\n============================")
            print("STEP 2: DENOISER")
            print("============================")

            result_path = deeprelaxo_out / "R2s_deeprelaxo.nii"
            denoiser_stage.run_denoiser(
                input_map_path=transformer_out / "R2s_transformer_mlp.nii",
                output_path=result_path,
                bet_mask_path=mask_path,
            )

            print("\n✅ Pipeline complete!")
            job["status"] = "done"
            job["result_path"] = str(result_path)
            job["result_image"] = _make_slice_image(str(result_path))
    except Exception as exc:
        import traceback
        print(f"\n❌ Error: {exc}")
        print(traceback.format_exc())
        job["status"] = "error"
    finally:
        sys.stdout = orig
        log_q.put(None)


def _stream_job(job):
    log = ""
    while True:
        msg = job["log_queue"].get()
        if msg is None:
            break
        log += msg + "\n"
        yield log, job.get("step1_result"), job.get("step1_image"), job.get("result_image")
    yield (
        log,
        job.get("result_path") or job.get("step1_result"),
        job.get("step1_image"),
        job.get("result_image"),
    )


def run_pipeline(mode, echo_files_3d, echo_file_4d, te_ms_str, mask_file, batch_size, mat_variable):
    te_list = [float(t.strip()) for t in te_ms_str.split(",") if t.strip()]
    if not te_list:
        yield "❌ Enter echo times (ms)", None
        return

    work_dir = Path(tempfile.mkdtemp(prefix="deeprelaxo_"))
    echo_paths = []

    if mode == "Multiple 3D Echoes":
        if not echo_files_3d:
            yield "❌ Upload echo files", None
            return
        files = echo_files_3d if isinstance(echo_files_3d, list) else [echo_files_3d]
        files = sorted(files, key=_natural_key)
        if len(files) != len(te_list):
            yield f"❌ {len(files)} files but {len(te_list)} TE values — counts must match", None
            return
        for f in files:
            src = _to_path(f)
            dst = work_dir / src.name
            shutil.copy(src, dst)
            echo_paths.append(dst)
        pipeline_mode = "3d"
    else:
        if not echo_file_4d:
            yield "❌ Upload a 4D file", None
            return
        src = _to_path(echo_file_4d)
        dst = work_dir / src.name
        shutil.copy(src, dst)
        echo_paths.append(dst)
        pipeline_mode = "4d"

    mask_path = None
    if mask_file:
        src = _to_path(mask_file)
        dst = work_dir / src.name
        shutil.copy(src, dst)
        mask_path = dst

    log_q: queue.Queue = queue.Queue()
    job = {"status": "queued", "log_queue": log_q, "result_path": None}
    threading.Thread(
        target=_run_thread,
        args=(job, work_dir, pipeline_mode, echo_paths, te_list, mask_path, int(batch_size), mat_variable or None),
        daemon=True,
    ).start()

    yield from _stream_job(job)



def toggle_mode(mode):
    is_3d = mode == "Multiple 3D Echoes"
    return gr.update(visible=is_3d), gr.update(visible=not is_3d)


with gr.Blocks(title="DeepRelaxo") as app:
    gr.Markdown("# DeepRelaxo\nDeep learning brain R2* mapping with reduced echoes")

    accumulated = gr.State([])

    with gr.Row(equal_height=False):

        # ── Left column: inputs ──────────────────────────────────────
        with gr.Column(scale=4):
            mode = gr.Radio(
                ["Multiple 3D Echoes", "Single 4D Volume"],
                value="Multiple 3D Echoes",
                label="Input Mode",
            )

            with gr.Group(visible=True) as group_3d:
                with gr.Row():
                    upload_btn = gr.UploadButton(
                        "Upload Echo Files",
                        file_count="multiple",
                        variant="secondary",
                    )
                    clear_btn = gr.Button("Clear All", variant="stop")
                sorted_order = gr.Textbox(
                    label="Processing order — confirm before running",
                    interactive=False,
                    placeholder="Upload files to see sorted order",
                )

            with gr.Group(visible=False) as group_4d:
                echo_file_4d = gr.File(
                    file_count="single",
                    label="4D Volume File",
                )

            with gr.Row():
                first_te = gr.Number(label="First TE (ms)", precision=2)
                echo_spacing = gr.Number(label="Echo Spacing (ms)", precision=2)
                n_echoes = gr.Number(label="Number of Echoes (from files)", precision=0, interactive=False)
                fill_te_btn = gr.Button("Fill Echo Times ↓", variant="secondary")

            te_ms = gr.Textbox(
                label="Echo Times (ms)",
                placeholder="e.g. 4.9, 9.9, 14.8, 19.8, 24.7",
                info="Fill using the fields above (equal spacing), or type values directly for irregular echo times.",
            )

            mask_file = gr.File(
                file_count="single",
                label="Brain Mask (optional — if omitted, all voxels are processed)",
            )

            with gr.Row():
                batch_size = gr.Number(
                    value=50000,
                    label="Voxel Batch Size",
                    info="Reduce if you run out of GPU memory",
                )
                mat_variable = gr.Textbox(
                    label="MATLAB variable name (optional)",
                    placeholder="e.g. magnitude",
                    info="Only needed for .mat files with multiple variables. Leave blank for NIfTI or single-variable .mat files.",
                )

            with gr.Row():
                run_btn = gr.Button("Run Pipeline", variant="primary", scale=2)
                demo_btn = gr.Button("Load Demo Data", scale=1)

        # ── Right column: log + result ───────────────────────────────
        with gr.Column(scale=5):
            log_out = gr.Textbox(label="Log", lines=18, max_lines=18, interactive=False)
            result_file = gr.File(label="Result")
            with gr.Row():
                img_step1 = gr.Image(label="Step 1 — Transformer-MLP (middle slice)", show_download_button=False)
                img_step2 = gr.Image(label="Step 2 — DeepRelaxo (middle slice)", show_download_button=False)

    def _format_order(paths):
        if not paths:
            return ""
        sorted_paths = sorted(paths, key=lambda p: _natural_key(p))
        return "\n".join(f"{i+1}. {Path(p).name}" for i, p in enumerate(sorted_paths))

    def add_files(new_files, current):
        if not new_files:
            return current, _format_order(current), len(current) or None
        existing_names = {Path(p).name for p in current}
        new_paths = [
            str(_to_path(f)) for f in (new_files if isinstance(new_files, list) else [new_files])
            if _to_path(f).name not in existing_names
        ]
        updated = current + new_paths
        return updated, _format_order(updated), len(updated) or None

    def clear_files():
        return [], "", None

    def compute_te_list(first_te, spacing, n):
        if not first_te or not spacing or not n:
            return gr.update()
        tes = [round(first_te + i * spacing, 4) for i in range(int(n))]
        return ", ".join(f"{t:g}" for t in tes)

    def load_demo():
        demo_dir = REPO_ROOT / "demo"
        if not (demo_dir / "mag1.nii").exists():
            return [], "❌ Demo data not found.\nRun: python run_deeprelaxo_pipeline.py --download-demo", None, "", None
        work_dir = Path(tempfile.mkdtemp(prefix="deeprelaxo_demo_"))
        paths = []
        for i in range(1, 6):
            dst = work_dir / f"mag{i}.nii"
            shutil.copy(demo_dir / f"mag{i}.nii", dst)
            paths.append(str(dst))
        mask_path = None
        if (demo_dir / "BET_mask.nii").exists():
            dst = work_dir / "BET_mask.nii"
            shutil.copy(demo_dir / "BET_mask.nii", dst)
            mask_path = str(dst)
        return paths, _format_order(paths), len(paths), 4.9, 5.0, "4.9, 9.9, 14.8, 19.8, 24.7", mask_path

    mode.change(toggle_mode, inputs=mode, outputs=[group_3d, group_4d])
    upload_btn.upload(add_files, inputs=[upload_btn, accumulated], outputs=[accumulated, sorted_order, n_echoes])
    clear_btn.click(clear_files, outputs=[accumulated, sorted_order, n_echoes])
    fill_te_btn.click(compute_te_list, inputs=[first_te, echo_spacing, n_echoes], outputs=te_ms)
    demo_btn.click(load_demo, outputs=[accumulated, sorted_order, n_echoes, first_te, echo_spacing, te_ms, mask_file])

    run_btn.click(
        run_pipeline,
        inputs=[mode, accumulated, echo_file_4d, te_ms, mask_file, batch_size, mat_variable],
        outputs=[log_out, result_file, img_step1, img_step2],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
