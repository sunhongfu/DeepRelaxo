import re
import sys
import queue
import shutil
import tempfile
import threading
from functools import lru_cache
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


@lru_cache(maxsize=8)
def _volume_array(nii_path):
    return nib.load(str(nii_path)).get_fdata().astype(np.float32)


def _make_slice_image(nii_path, slice_idx=None, vmin=0, vmax=100):
    data = _volume_array(str(nii_path))
    depth = data.shape[2]
    if slice_idx is None:
        slice_idx = depth // 2
    slice_idx = max(0, min(int(slice_idx), depth - 1))
    sl = np.rot90(data[:, :, slice_idx])
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(sl, cmap="gray", vmin=float(vmin), vmax=float(vmax), aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", pad_inches=0, dpi=150, facecolor="black")
    plt.close(fig)
    return tmp.name


def _print_run_config(work_dir, mode, echo_paths, te_list, mask_path, batch_size):
    print("============================")
    print("RUN CONFIGURATION")
    print("============================")
    if mode == "4d":
        print(f"Input mode      : Single 4D volume")
        print(f"Echo file       : {Path(echo_paths[0]).name}")
    else:
        print(f"Input mode      : Multiple 3D echoes ({len(echo_paths)} files)")
        print("Echo files (in processing order):")
        for i, (p, te) in enumerate(zip(echo_paths, te_list), 1):
            print(f"  {i}. {Path(p).name}    TE = {te} ms")
    print(f"TE values (ms)  : {', '.join(str(t) for t in te_list)}")
    print(f"Brain mask      : {Path(mask_path).name if mask_path else '(none — all voxels processed)'}")
    print(f"Batch size      : {batch_size}")
    print(f"Staging dir     : {work_dir}")
    print()
    print("Equivalent command-line invocation:")
    cmd = ["python run_deeprelaxo_pipeline.py"]
    cmd.append(f"--data_dir {work_dir}")
    if mode == "4d":
        cmd.append(f"--echo_4d {Path(echo_paths[0]).name}")
    else:
        cmd.append("--echo_files " + " ".join(Path(p).name for p in echo_paths))
    cmd.append("--te_ms " + " ".join(str(t) for t in te_list))
    if mask_path:
        cmd.append(f"--mask {Path(mask_path).name}")
    cmd.append(f"--transformer_batch_size {batch_size}")
    print("  " + " \\\n    ".join(cmd))
    print()


def _run_thread(job, work_dir, mode, echo_paths, te_list, mask_path, batch_size, vmin=0, vmax=100):
    log_q = job["log_queue"]
    orig = sys.stdout
    sys.stdout = _QueueWriter(log_q, orig)
    try:
        with _pipeline_lock:
            job["status"] = "running"
            transformer_out = work_dir / "transformer_outputs"
            deeprelaxo_out = work_dir / "deeprelaxo_outputs"

            _print_run_config(work_dir, mode, echo_paths, te_list, mask_path, batch_size)

            print("============================")
            print("STEP 1: ESTIMATOR")
            print("============================")

            if mode == "4d":
                estimator_stage.run_estimator(
                    magnitude_4d_path={"path": echo_paths[0]},
                    te_values_ms=te_list,
                    save_dir=transformer_out,
                    bet_mask_path=mask_path,
                    batch_size=batch_size,
                )
            else:
                estimator_stage.run_estimator(
                    magnitude_entries=[{"path": p} for p in echo_paths],
                    te_values_ms=te_list,
                    save_dir=transformer_out,
                    bet_mask_path=mask_path,
                    batch_size=batch_size,
                )

            job["step1_result"] = str(transformer_out / "R2s_transformer_mlp.nii")
            job["step1_image"] = _make_slice_image(job["step1_result"], vmin=vmin, vmax=vmax)
            job["depth"] = _volume_array(job["step1_result"]).shape[2]

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
            job["result_image"] = _make_slice_image(str(result_path), vmin=vmin, vmax=vmax)
    except Exception as exc:
        import traceback
        print(f"\n❌ Error: {exc}")
        print(traceback.format_exc())
        job["status"] = "error"
    finally:
        sys.stdout = orig
        log_q.put(None)


def _result_files(job):
    files = [p for p in (job.get("step1_result"), job.get("result_path")) if p]
    return files or None


def _state_and_slider_update(job):
    state = (job.get("step1_result"), job.get("result_path"))
    depth = job.get("depth")
    if depth and not job.get("_slider_init"):
        job["_slider_init"] = True
        slider = gr.update(visible=True, minimum=0, maximum=depth - 1,
                           value=depth // 2, interactive=True)
    else:
        slider = gr.update()
    return state, slider


def _stream_job(job):
    log = ""
    while True:
        msg = job["log_queue"].get()
        if msg is None:
            break
        log += msg + "\n"
        state, slider = _state_and_slider_update(job)
        yield (log, _result_files(job), job.get("step1_image"),
               job.get("result_image"), state, slider)
    state, slider = _state_and_slider_update(job)
    yield (log, _result_files(job), job.get("step1_image"),
           job.get("result_image"), state, slider)


def run_pipeline(echo_files, te_ms_str, mask_file, batch_size, vmin, vmax):
    _noop = (None, None, None, (None, None), gr.update())
    te_list = [float(t.strip()) for t in te_ms_str.split(",") if t.strip()]
    if not te_list:
        yield ("❌ Enter echo times (ms)", *_noop)
        return

    if not echo_files:
        yield ("❌ Upload magnitude file(s) — multiple 3D echoes or a single 4D volume", *_noop)
        return

    files = echo_files if isinstance(echo_files, list) else [echo_files]
    files = sorted(files, key=_natural_key)
    missing = [str(_to_path(f)) for f in files if not (_to_path(f) and _to_path(f).exists())]
    if missing:
        yield (
            "❌ Some uploaded files no longer exist on disk (likely cleared by the browser/server). "
            "Click 'Clear All' and re-upload:\n  " + "\n  ".join(missing),
            *_noop,
        )
        return

    work_dir = Path(tempfile.mkdtemp(prefix="deeprelaxo_"))
    echo_paths = []
    for f in files:
        src = _to_path(f)
        dst = work_dir / src.name
        shutil.copy(src, dst)
        echo_paths.append(dst)

    if len(echo_paths) == 1:
        pipeline_mode = "4d"
    else:
        pipeline_mode = "3d"
        if len(echo_paths) != len(te_list):
            yield (f"❌ {len(echo_paths)} echo files but {len(te_list)} TE values — counts must match", *_noop)
            return

    mask_path = None
    if mask_file:
        src = _to_path(mask_file)
        if src is not None and src.exists():
            dst = work_dir / src.name
            shutil.copy(src, dst)
            mask_path = dst

    log_q: queue.Queue = queue.Queue()
    job = {"status": "queued", "log_queue": log_q, "result_path": None}
    threading.Thread(
        target=_run_thread,
        args=(job, work_dir, pipeline_mode, echo_paths, te_list, mask_path, int(batch_size), float(vmin), float(vmax)),
        daemon=True,
    ).start()

    yield from _stream_job(job)



def _detect_echoes(paths):
    if not paths:
        return None
    if len(paths) == 1:
        p = Path(paths[0])
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            try:
                shape = nib.load(str(p)).shape
                if len(shape) == 4:
                    return int(shape[-1])
            except Exception:
                pass
        return None
    return len(paths)


CUSTOM_CSS = """
/* Section titles — coloured for quick scanning */
.gradio-container h3 {
    padding-left: 12px !important;
    color: #1d4ed8 !important;
    border-left: 4px solid #1d4ed8 !important;
    margin-left: 4px !important;
}
.dark .gradio-container h3 {
    color: #60a5fa !important;
    border-left-color: #60a5fa !important;
}

/* Section panels — thicker, more visible borders */
.dr-section {
    margin-bottom: 16px !important;
    border: 2px solid #4b5563 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
}
.dark .dr-section {
    border-color: #9ca3af !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.30) !important;
}

/* Make secondary buttons look clearly clickable instead of blending in (light mode) */
button.secondary {
    background: #f3f4f6 !important;
    border: 1px solid #9ca3af !important;
    color: #111827 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
    transition: background 0.12s ease-in-out, border-color 0.12s ease-in-out !important;
}
button.secondary:hover {
    background: #e5e7eb !important;
    border-color: #6b7280 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.10) !important;
}
button.secondary:active {
    background: #d1d5db !important;
}

/* Dark mode variant — explicit colors so text stays readable */
.dark button.secondary {
    background: #374151 !important;
    border: 1px solid #6b7280 !important;
    color: #f3f4f6 !important;
}
.dark button.secondary:hover {
    background: #4b5563 !important;
    border-color: #9ca3af !important;
    color: #ffffff !important;
}
.dark button.secondary:active {
    background: #1f2937 !important;
}
"""


FORCE_DARK_JS = """
<script>
  (function() {
    try {
      var url = new URL(window.location.href);
      if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.replace(url.toString());
      }
    } catch (e) {}
  })();
</script>
"""


with gr.Blocks(title="DeepRelaxo") as app:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>{FORCE_DARK_JS}", padding=False)
    gr.Markdown(
        "# DeepRelaxo — Deep learning brain R2* mapping with reduced echoes\n"
        "<span style='font-size: 1.2em'>"
        "🐙 [GitHub (web app docs)](https://github.com/sunhongfu/DeepRelaxo#web-app)"
        " &nbsp;·&nbsp; "
        "📄 [Paper (MRM)](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70405?af=R)"
        " &nbsp;·&nbsp; "
        "🌐 [Hongfu Sun](https://sunhongfu.github.io)"
        "</span>"
    )

    accumulated = gr.State([])

    with gr.Row(equal_height=False):

        # ── Left column: inputs ──────────────────────────────────────
        with gr.Column(scale=4):
            # ── 1. Magnitudes ──────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Magnitudes  *(multiple 3D echoes OR a single 4D volume; .nii / .nii.gz / .mat)*")
                magnitudes_input = gr.File(
                    file_count="multiple",
                    show_label=False,
                    height=180,
                )
                sorted_order = gr.Textbox(
                    label="Processing order — confirm before running; rename your files if they aren't sorted correctly",
                    interactive=False,
                    placeholder="Upload files to see sorted order",
                    lines=5,
                    max_lines=15,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear All", variant="stop")
                    demo_btn = gr.Button("Load Demo Data", variant="secondary")

            # ── 2. Echo Times ──────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Echo Times")
                with gr.Row():
                    first_te = gr.Number(label="First TE (ms)", precision=2)
                    echo_spacing = gr.Number(label="Echo Spacing (ms)", precision=2)
                    n_echoes = gr.Number(label="Number of Echoes", precision=0, interactive=True)
                fill_te_btn = gr.Button("Compute full train of echo times ↓", variant="secondary")
                te_ms = gr.Textbox(
                    label="Echo Times (ms)",
                    placeholder="e.g. 4.9, 9.9, 14.8, 19.8, 24.7",
                    info="Fill using the fields above (equal spacing), or type values directly for irregular echo times.",
                )

            # ── 3. Brain Mask ──────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Brain Mask  *(optional — if omitted, all voxels are processed)*")
                mask_file = gr.File(
                    file_count="single",
                    show_label=False,
                )

            # ── 4. Hyper-parameters ───────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Hyper-parameters")
                batch_size = gr.Number(
                    value=50000,
                    label="Voxel Batch Size (50,000)",
                    info="Reduce if you run out of GPU memory",
                )

            run_btn = gr.Button("Run Pipeline", variant="primary")

        # ── Right column: log + result ───────────────────────────────
        with gr.Column(scale=5):
            # ── 5. Log ─────────────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Log")
                log_out = gr.Textbox(show_label=False, lines=8, max_lines=20, interactive=False, autoscroll=True)

            # ── 6. Results ─────────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Results  *(click the file size on the right to download)*")
                result_file = gr.File(show_label=False, file_count="multiple")

            # ── 7. Visualisation ───────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Visualisation")
                with gr.Row():
                    img_step1 = gr.Image(
                        label="Step 1 — Transformer-MLP",
                        show_download_button=False,
                        show_fullscreen_button=False,
                        height=420,
                    )
                    img_step2 = gr.Image(
                        label="Step 2 — DeepRelaxo",
                        show_download_button=False,
                        show_fullscreen_button=False,
                        height=420,
                    )
                with gr.Row(equal_height=True):
                    prev_btn = gr.Button("◀ Prev", scale=1)
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        label="Slice (Z)",
                        show_label=False,
                        container=False,
                        interactive=True,
                        scale=8,
                    )
                    next_btn = gr.Button("Next ▶", scale=1)
                with gr.Row():
                    vmin_input = gr.Number(value=0, label="Display window min (R2*, s⁻¹)", precision=2)
                    vmax_input = gr.Number(value=100, label="Display window max (R2*, s⁻¹)", precision=2)
            output_state = gr.State((None, None))

    def _format_order(paths):
        if not paths:
            return ""
        sorted_paths = sorted(paths, key=lambda p: _natural_key(p))
        entries = [f"{i+1}. {Path(p).name}" for i, p in enumerate(sorted_paths)]
        n_cols = 4 if len(entries) > 5 else 1
        col_width = max(len(e) for e in entries) + 3
        rows = []
        for i in range(0, len(entries), n_cols):
            chunk = entries[i:i + n_cols]
            rows.append("".join(e.ljust(col_width) for e in chunk).rstrip())
        return "\n".join(rows)

    def add_files(new_files, current):
        if not new_files:
            return current, _format_order(current), _detect_echoes(current), None
        files = new_files if isinstance(new_files, list) else [new_files]
        new_paths = [str(_to_path(f)) for f in files]
        new_names = {Path(p).name for p in new_paths}
        kept = [p for p in current if Path(p).name not in new_names]
        updated = kept + new_paths
        return updated, _format_order(updated), _detect_echoes(updated), None

    def clear_files():
        return [], "", None, None

    def compute_te_list(first_te, spacing, n):
        if not first_te or not spacing or not n:
            return gr.update()
        tes = [round(first_te + i * spacing, 4) for i in range(int(n))]
        return ", ".join(f"{t:g}" for t in tes)

    def load_demo():
        demo_dir = REPO_ROOT / "demo"
        if not (demo_dir / "mag1.nii").exists():
            return [], "❌ Demo data not found.\nRun: python run_deeprelaxo_pipeline.py --download-demo", None, "", None
        paths = [str(demo_dir / f"mag{i}.nii") for i in range(1, 6)]
        mask = demo_dir / "BET_mask.nii"
        mask_path = str(mask) if mask.exists() else None
        return paths, _format_order(paths), len(paths), 4.9, 5.0, "4.9, 9.9, 14.8, 19.8, 24.7", mask_path

    magnitudes_input.upload(
        add_files,
        inputs=[magnitudes_input, accumulated],
        outputs=[accumulated, sorted_order, n_echoes, magnitudes_input],
    )
    clear_btn.click(clear_files, outputs=[accumulated, sorted_order, n_echoes, magnitudes_input])
    fill_te_btn.click(compute_te_list, inputs=[first_te, echo_spacing, n_echoes], outputs=te_ms)
    demo_btn.click(load_demo, outputs=[accumulated, sorted_order, n_echoes, first_te, echo_spacing, te_ms, mask_file])

    run_btn.click(
        run_pipeline,
        inputs=[accumulated, te_ms, mask_file, batch_size, vmin_input, vmax_input],
        outputs=[log_out, result_file, img_step1, img_step2, output_state, slice_slider],
    )

    def render_slice(state, idx, vmin, vmax):
        p1, p2 = state if state else (None, None)
        img1 = _make_slice_image(p1, idx, vmin, vmax) if p1 else None
        img2 = _make_slice_image(p2, idx, vmin, vmax) if p2 else None
        return img1, img2

    def step_slice(current, state, delta):
        if state is None or state[0] is None:
            return gr.update()
        depth = _volume_array(state[0]).shape[2]
        return max(0, min(int(current) + delta, depth - 1))

    prev_btn.click(lambda c, s: step_slice(c, s, -1), inputs=[slice_slider, output_state], outputs=slice_slider)
    next_btn.click(lambda c, s: step_slice(c, s, +1), inputs=[slice_slider, output_state], outputs=slice_slider)

    slice_slider.change(
        render_slice,
        inputs=[output_state, slice_slider, vmin_input, vmax_input],
        outputs=[img_step1, img_step2],
    )
    vmin_input.change(
        render_slice,
        inputs=[output_state, slice_slider, vmin_input, vmax_input],
        outputs=[img_step1, img_step2],
    )
    vmax_input.change(
        render_slice,
        inputs=[output_state, slice_slider, vmin_input, vmax_input],
        outputs=[img_step1, img_step2],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
