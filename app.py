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


_RED_WAIT = (
    "<span style='color: #dc2626; font-weight: 700; font-size: 1.05em;'>{msg}</span>"
)


def _parse_te_input(s):
    """Accept either 'TE1, TE2, TE3, ...' or compact 'first:spacing:count'."""
    s = (s or "").strip()
    if not s:
        return []
    if ":" in s and "," not in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "TE compact form must be 'first_TE : spacing : count' "
                f"(got {len(parts)} parts in '{s}')"
            )
        try:
            first = float(parts[0])
            spacing = float(parts[1])
            n = int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid number in TE compact form '{s}'. "
                "Use 'first_TE : spacing : count' (e.g. '4.9 : 5 : 5')."
            ) from exc
        if n < 1:
            raise ValueError(f"TE count must be ≥ 1 (got {n})")
        return [round(first + i * spacing, 6) for i in range(n)]
    return [float(t.strip()) for t in s.split(",") if t.strip()]


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


@lru_cache(maxsize=128)
def _file_shape(path_str):
    """Return the volume shape of a NIfTI/MAT file, or None on failure.

    NIfTI shape is read from the header (lazy, fast). MAT requires loading
    the array. Cached so repeated UI renders don't re-read the file.
    """
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return tuple(nib.load(str(p)).shape)
        from data_utils import load_array_with_affine
        arr, _ = load_array_with_affine(p)
        return tuple(arr.shape)
    except Exception:
        return None


def _shape_summary(paths):
    """Markdown summary that shows per-file shapes so the user can verify each one.

    - 0 files → empty
    - 1 file → "Shape: A × B × C"
    - many files → header line (matched / mismatched / unreadable) + per-file list
    """
    if not paths:
        return ""
    paths = list(paths)
    items = [(Path(p).name, _file_shape(p)) for p in paths]
    n = len(paths)
    unreadable = [name for name, s in items if s is None]
    unique_shapes = {s for _, s in items if s is not None}

    def _fmt(s):
        return " × ".join(str(d) for d in s)

    if n == 1:
        name, shape = items[0]
        if shape:
            return f"&nbsp;&nbsp;**Shape:** {_fmt(shape)}"
        return f"&nbsp;&nbsp;`{name}` *(shape could not be read)*"

    # Multi-file: always list per-file shapes for transparency.
    if not unreadable and len(unique_shapes) == 1:
        header = (
            f"**{n} files** &nbsp;·&nbsp; "
            f"all matching shape **{_fmt(next(iter(unique_shapes)))}** ✓"
        )
    elif len(unique_shapes) > 1:
        header = (
            f"**{n} files** &nbsp;·&nbsp; "
            f"⚠️ **mismatched shapes** — all files must share the same volume dimensions"
        )
    else:
        header = (
            f"**{n} files** &nbsp;·&nbsp; "
            f"⚠️ {len(unreadable)} could not be read"
        )
    lines = [header, ""]
    for name, s in items:
        lines.append(f"- `{name}` — {_fmt(s) if s else '*(could not read)*'}")
    return "\n".join(lines)


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


def _print_run_config(work_dir, mode, echo_paths, te_list, mask_path, batch_size,
                      orig_echo_paths=None, orig_mask_path=None):
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
    print(f"Working dir     : {work_dir}")
    # The pipeline copies all inputs into work_dir, so the equivalent CLI
    # invocation uses --data_dir = work_dir + bare filenames (resolved
    # relative to data_dir by run_deeprelaxo_pipeline.py).
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
    _sep = "-" * 56
    print()
    print(_sep)
    print("  Equivalent command-line invocation:")
    print(_sep)
    print("  " + " \\\n      ".join(cmd))
    print(_sep)
    print()


def _run_thread(job, work_dir, mode, echo_paths, te_list, mask_path, batch_size, vmin=0, vmax=100,
                orig_echo_paths=None, orig_mask_path=None):
    log_q = job["log_queue"]
    orig = sys.stdout
    sys.stdout = _QueueWriter(log_q, orig)
    try:
        with _pipeline_lock:
            job["status"] = "running"
            transformer_out = work_dir / "transformer_outputs"
            deeprelaxo_out = work_dir / "deeprelaxo_outputs"

            _print_run_config(work_dir, mode, echo_paths, te_list, mask_path, batch_size,
                              orig_echo_paths=orig_echo_paths, orig_mask_path=orig_mask_path)

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
        # Defensive GPU cleanup — both stages already release on success,
        # but if something raised partway we still want a clean slate so
        # the next run doesn't see accumulated CUDA memory.
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
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


def _visibility_updates(job):
    return (
        gr.update(visible=True),  # log_group — always visible while running
        gr.update(visible=bool(_result_files(job))),  # results_group
        gr.update(visible=bool(job.get("step1_image"))),  # viz_group
    )


def _result_info_md(job):
    files = _result_files(job)
    if not files:
        return ""
    return _shape_summary(files)


def _stream_job(job):
    log = ""
    while True:
        msg = job["log_queue"].get()
        if msg is None:
            break
        log += msg + "\n"
        state, slider = _state_and_slider_update(job)
        log_v, res_v, viz_v = _visibility_updates(job)
        yield (log, _result_files(job), _result_info_md(job),
               job.get("step1_image"), job.get("result_image"),
               state, slider, log_v, res_v, viz_v)
    state, slider = _state_and_slider_update(job)
    log_v, res_v, viz_v = _visibility_updates(job)
    yield (log, _result_files(job), _result_info_md(job),
           job.get("step1_image"), job.get("result_image"),
           state, slider, log_v, res_v, viz_v)


def run_pipeline(echo_files, te_ms_str, mask_file, batch_size, vmin, vmax):
    _noop = (
        None,           # result_file
        "",             # result_info
        None,           # img_step1
        None,           # img_step2
        (None, None),   # output_state
        gr.update(),    # slice_slider
        gr.update(visible=True),   # log_group — show error message
        gr.update(visible=False),  # results_group
        gr.update(visible=False),  # viz_group
    )
    try:
        te_list = _parse_te_input(te_ms_str)
    except ValueError as exc:
        yield (f"❌ {exc}", *_noop)
        return
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

    orig_echo_paths = [str(_to_path(f)) for f in files]
    _omask = _to_path(mask_file)
    orig_mask_path = str(_omask) if _omask and _omask.exists() else None

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
        args=(job, work_dir, pipeline_mode, echo_paths, te_list, mask_path, int(batch_size), float(vmin), float(vmax),
              orig_echo_paths, orig_mask_path),
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
/* ── Global type scale — bump body / labels / info text for comfort ── */
.gradio-container {
    --text-xxs: 12px;
    --text-xs: 13px;
    --text-sm: 15px;
    --text-md: 17px;
    --text-lg: 19px;
    --text-xl: 22px;
    --text-xxl: 26px;
    --block-info-text-size: 15px;
    --block-label-text-size: 16px;
    --block-title-text-size: 18px;
    --section-text-size: 17px;
    font-size: 17px !important;
    line-height: 1.55 !important;
}
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .markdown,
.gradio-container .markdown p {
    font-size: 17px !important;
    line-height: 1.6 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    font-size: 16px !important;
}
.gradio-container button {
    font-size: 16px !important;
}

/* Section titles — coloured for quick scanning */
.gradio-container h3 {
    font-size: 1.4rem !important;
    padding: 4px 0 6px 14px !important;
    color: #1d4ed8 !important;
    border-left: 5px solid #1d4ed8 !important;
    margin: 8px 0 14px 4px !important;
}
.dark .gradio-container h3 {
    color: #60a5fa !important;
    border-left-color: #60a5fa !important;
}

/* Accordion label — match the h3 section heading style */
.dr-accordion > .label-wrap,
.dr-accordion > div > .label-wrap,
.dr-accordion button.label-wrap,
.dr-accordion > .label-wrap span,
.dr-accordion button.label-wrap span {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #1d4ed8 !important;
}
.dr-accordion > .label-wrap,
.dr-accordion > div > .label-wrap,
.dr-accordion button.label-wrap {
    border-left: 5px solid #1d4ed8 !important;
    padding: 6px 0 6px 14px !important;
    margin: 4px 0 8px 4px !important;
    background: transparent !important;
}
.dark .dr-accordion > .label-wrap,
.dark .dr-accordion > div > .label-wrap,
.dark .dr-accordion button.label-wrap,
.dark .dr-accordion > .label-wrap span,
.dark .dr-accordion button.label-wrap span {
    color: #60a5fa !important;
    border-left-color: #60a5fa !important;
}

/* Processing-Order — hide the upload drop zone (we don't accept uploads here)
   and the top-level "clear all" toolbar X overlay. The per-file X
   (.label-clear-button) sits inside .file-preview-holder for 2+ files; for
   the 1-file case it isn't rendered at all by Gradio, so we provide an
   explicit Clear Processing Order button below instead. */
#dr-sorted-files .upload-container,
#dr-sorted-files .wrap.svelte-12ioyct,
#dr-sorted-files .upload-button,
#dr-sorted-files button:has(svg.feather-upload),
#dr-sorted-files .icon-button-wrapper.top-panel {
    display: none !important;
}
/* Brain Mask — hide the native per-file X (label-clear-button) and the
   top-level toolbar X. The explicit "✕ Remove Brain Mask" button below is
   the canonical removal path; the native X was overlaying the download icon. */
#dr-mask-file .label-clear-button,
#dr-mask-file .icon-button-wrapper.top-panel {
    display: none !important;
}
/* Explicit Remove buttons (Processing Order, Brain Mask) — full-width */
#dr-clear-order-btn,
#dr-clear-order-btn button,
#dr-mask-clear-btn,
#dr-mask-clear-btn button {
    width: 100% !important;
    margin-top: 4px !important;
}

/* Accordion expanded content — visually match the rest of the dr-sections.
   Gradio gives the open panel its own background; force it transparent so
   the parent dr-section's color/border carry through. */
.dr-accordion,
.dr-accordion > div,
.dr-accordion > div > div,
.dr-accordion .open,
.dr-accordion .accordion-content,
.dr-accordion > div:not(.label-wrap) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* But keep the outer dr-section frame visible */
.dr-section.dr-accordion {
    background: var(--block-background-fill) !important;
    border: 2px solid #4b5563 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
}
.dark .dr-section.dr-accordion {
    border-color: #9ca3af !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.30) !important;
}

/* Section panels — more breathing room, thicker visible borders */
.dr-section {
    margin-bottom: 24px !important;
    padding: 16px 20px !important;
    border: 2px solid #4b5563 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
}
.dark .dr-section {
    border-color: #9ca3af !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.30) !important;
}

/* Markdown content inside sections — indent paragraphs / lists from the edge */
.dr-section .prose,
.dr-section .markdown,
.dr-section .gradio-markdown {
    padding: 0 6px !important;
}
.dr-section .prose p,
.dr-section .markdown p {
    margin: 6px 0 8px 0 !important;
    padding-left: 4px !important;
}
.dr-section .prose ul,
.dr-section .prose ol,
.dr-section .markdown ul,
.dr-section .markdown ol {
    padding-left: 28px !important;
    margin: 6px 0 8px 0 !important;
}
.dr-section .prose li,
.dr-section .markdown li {
    margin: 3px 0 !important;
}

/* Pull-quote / italic asides ("*Auto-filled when …*") */
.dr-section em {
    color: #6b7280 !important;
}
.dark .dr-section em {
    color: #9ca3af !important;
}

/* Tab body content — match section padding */
.dr-input-tabs .tab-nav + div,
.dr-input-tabs > div > div:not(.tab-nav) {
    padding: 14px 18px !important;
}

/* Run Pipeline — green CTA, large, full-width */
#dr-run-btn,
#dr-run-btn button {
    background: #16a34a !important;
    background-image: linear-gradient(180deg, #22c55e, #15803d) !important;
    color: #ffffff !important;
    border-color: #15803d !important;
    font-size: 18px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    width: 100% !important;
}
#dr-run-btn:hover,
#dr-run-btn button:hover {
    background: #15803d !important;
    background-image: linear-gradient(180deg, #16a34a, #14532d) !important;
    border-color: #14532d !important;
}
#dr-run-btn:active,
#dr-run-btn button:active {
    background: #14532d !important;
}


/* Input-method tabs — bigger and more comfortable */
.dr-input-tabs button[role="tab"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 20px 36px !important;
    border: 2px solid #9ca3af !important;
    border-bottom: 2px solid #4b5563 !important;
    border-radius: 10px 10px 0 0 !important;
    background: #e5e7eb !important;
    color: #374151 !important;
    margin-right: 8px !important;
    opacity: 1 !important;
    box-shadow: 0 -1px 0 rgba(0, 0, 0, 0.08) inset !important;
    transition: all 0.15s ease !important;
}
.dr-input-tabs button[role="tab"]:hover {
    background: #d1d5db !important;
    color: #111827 !important;
    border-color: #6b7280 !important;
}
.dr-input-tabs button[role="tab"][aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border-color: #1d4ed8 !important;
    border-bottom: 2px solid #1d4ed8 !important;
    box-shadow: 0 -3px 8px rgba(29, 78, 216, 0.35) !important;
    transform: translateY(-1px) !important;
}
.dark .dr-input-tabs button[role="tab"] {
    background: #374151 !important;
    color: #e5e7eb !important;
    border-color: #6b7280 !important;
    border-bottom-color: #9ca3af !important;
    box-shadow: 0 -1px 0 rgba(255, 255, 255, 0.08) inset !important;
}
.dark .dr-input-tabs button[role="tab"]:hover {
    background: #4b5563 !important;
    color: #ffffff !important;
    border-color: #9ca3af !important;
}
.dark .dr-input-tabs button[role="tab"][aria-selected="true"] {
    background: #60a5fa !important;
    color: #0b1220 !important;
    border-color: #60a5fa !important;
    border-bottom-color: #60a5fa !important;
    box-shadow: 0 -3px 8px rgba(96, 165, 250, 0.45) !important;
}
/* Suppress Gradio's default orange accent underline on the selected tab */
.dr-input-tabs button[role="tab"]::after,
.dr-input-tabs button[role="tab"]::before,
.dr-input-tabs button[role="tab"][aria-selected="true"]::after,
.dr-input-tabs button[role="tab"][aria-selected="true"]::before {
    display: none !important;
    content: none !important;
    background: none !important;
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


with gr.Blocks(title="DeepRelaxo", analytics_enabled=False) as app:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>", padding=False)
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

    with gr.Column():
        # ── 1. MRI Magnitudes (tabs) ──────────────────────────
        with gr.Accordion(
            "MRI Magnitudes", open=True,
            elem_classes=["dr-section", "dr-accordion"],
        ):
            gr.Markdown("Multi-echo GRE magnitude images. Pick one input method below.")
            with gr.Tabs(elem_classes="dr-input-tabs"):
                with gr.Tab("📁 DICOM Folder  (recommended)") as tab_dicom:
                    gr.Markdown(
                        "Pick the folder of multi-echo GRE **magnitude** DICOMs — echoes, "
                        "TE values, and slice ordering are read from headers automatically."
                    )
                    dicom_input = gr.UploadButton(
                        "📁  Select DICOM Folder",
                        file_count="directory",
                        variant="primary",
                    )
                    dicom_info = gr.Markdown("")

                with gr.Tab("📄 NIfTI / MAT files  (advanced)") as tab_nifti:
                    gr.Markdown(
                        "Pick pre-converted magnitudes — multiple 3D echoes (one file each) "
                        "or a single 4D volume. Supported: `.nii`, `.nii.gz`, `.mat`. "
                        "**You'll also need to enter Echo Times below.**"
                    )
                    magnitudes_input = gr.UploadButton(
                        "📄  Add NIfTI / MAT Magnitudes",
                        file_count="multiple",
                        file_types=[".nii", ".nii.gz", ".gz", ".mat"],
                        variant="primary",
                    )
                    magnitudes_info = gr.Markdown("")

        # ── 2. Processing Order ───────────────────────────────
        with gr.Accordion(
            "Processing Order", open=False,
            elem_classes=["dr-section", "dr-accordion"],
        ) as order_group:
            gr.Markdown(
                "Confirm the order before running. If the order looks wrong, "
                "rename your files (natural numeric sort: `mag1`, `mag2`, …, `mag10`)."
            )
            sorted_files = gr.File(
                file_count="multiple",
                show_label=False,
                interactive=True,
                height=180,
                elem_id="dr-sorted-files",
            )
            sorted_info = gr.Markdown("")
            # Gradio's gr.File(file_count="multiple") doesn't render a per-file X
            # when only one file is present. This explicit button is the escape
            # hatch — also useful for clearing 2+ files in one click.
            clear_order_btn = gr.Button(
                "✕  Remove all magnitudes",
                variant="stop",
                visible=False,
                elem_id="dr-clear-order-btn",
            )

        # ── 3. Echo Times ─────────────────────────────────────
        with gr.Accordion(
            "Echo Times (ms)", open=True,
            elem_classes=["dr-section", "dr-accordion"],
        ):
            gr.Markdown(
                "Two accepted formats:\n"
                "- Comma-separated values — one per echo, irregular spacings allowed "
                "(e.g. `2.4, 3.6, 9.2, 20.8`).\n"
                "- Compact `first_TE : spacing : count` for uniform spacing "
                "(e.g. `3.5 : 4.4 : 8` → `3.5, 7.9, 12.3, 16.7, 21.1, 25.5, 29.9, 34.3`).\n\n"
                "*Auto-filled when you use the DICOM Folder tab.*"
            )
            te_ms = gr.Textbox(
                show_label=False,
                placeholder="e.g.  2.4, 3.6, 9.2, 20.8    or compact:  3.5 : 4.4 : 8",
            )

        # ── 4. Brain Mask ─────────────────────────────────────
        with gr.Accordion(
            "Brain Mask (optional)", open=False,
            elem_classes=["dr-section", "dr-accordion"],
        ) as mask_group:
            gr.Markdown(
                "Optional — if omitted, **all voxels are processed**, which is "
                "significantly slower than processing only the masked brain-tissue region.\n\n"
                "Supported: `.nii`, `.nii.gz`, `.mat`"
            )
            mask_button = gr.UploadButton(
                "🧠  Select Brain Mask",
                file_count="single",
                file_types=[".nii", ".nii.gz", ".gz", ".mat"],
                variant="primary",
            )
            # Hidden file display — appears with per-file X + download icon
            # only after a mask is uploaded (mirrors Processing Order).
            mask_file = gr.File(
                file_count="single",
                show_label=False,
                interactive=True,
                visible=False,
                elem_id="dr-mask-file",
                height=70,
            )
            mask_info = gr.Markdown("")
            # Explicit Remove button — same pattern as Clear Processing Order.
            # Visible only when a mask is loaded.
            mask_clear_btn = gr.Button(
                "✕  Remove Brain Mask",
                variant="stop",
                visible=False,
                elem_id="dr-mask-clear-btn",
            )

        # ── 5. Hyper-parameters (collapsed by default) ────────
        with gr.Accordion(
            "Hyper-parameters",
            open=False,
            elem_classes=["dr-section", "dr-accordion"],
        ):
            batch_size = gr.Number(
                value=50000,
                label="Voxel Batch Size (50,000)",
                info="Reduce if you run out of GPU memory",
            )

        run_btn = gr.Button("Run Pipeline", variant="primary", elem_id="dr-run-btn")

        # ── 5. Log ─────────────────────────────────────────────
        with gr.Accordion(
            "Log", open=True, visible=False,
            elem_classes=["dr-section", "dr-accordion"],
        ) as log_group:
            log_out = gr.Textbox(show_label=False, lines=8, max_lines=20, interactive=False, autoscroll=True)

        # ── 6. Visualisation ───────────────────────────────────
        with gr.Accordion(
            "Visualisation", open=True, visible=False,
            elem_classes=["dr-section", "dr-accordion"],
        ) as viz_group:
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

        # ── 7. Results ─────────────────────────────────────────
        with gr.Accordion(
            "Results", open=True, visible=False,
            elem_classes=["dr-section", "dr-accordion"],
        ) as results_group:
            gr.Markdown("Click the file size on the right to download.")
            result_file = gr.File(show_label=False, file_count="multiple")
            result_info = gr.Markdown("")

    def _sort_paths(paths):
        return sorted(paths, key=lambda p: _natural_key(p))

    def _clear_btn_update(count):
        return gr.update(visible=count >= 1)

    def add_files(new_files, current, progress=gr.Progress()):
        if not new_files:
            srt = _sort_paths(current) if current else []
            return (current, srt or None, _shape_summary(srt), None, gr.update(), "",
                    _clear_btn_update(len(srt)))
        files = new_files if isinstance(new_files, list) else [new_files]
        progress(0.1, desc=f"Reading {len(files)} uploaded file(s)…")
        accepted_exts = (".nii", ".nii.gz", ".mat")
        new_paths = []
        rejected = []
        for f in files:
            p = _to_path(f)
            if p is None:
                continue
            name = p.name.lower()
            if any(name.endswith(ext) for ext in accepted_exts):
                new_paths.append(str(p))
            else:
                rejected.append(p.name)
        if rejected:
            gr.Warning(
                "Ignored unsupported file(s): "
                + ", ".join(rejected)
                + ". Only .nii, .nii.gz and .mat files are accepted."
            )
        if not new_paths:
            srt = _sort_paths(current) if current else []
            return (
                current, srt or None, _shape_summary(srt), None, gr.update(),
                f"⚠️ No supported files in this upload (rejected: {', '.join(rejected)})." if rejected else "",
                _clear_btn_update(len(srt)),
            )
        # Drop any DICOM-converted leftovers from the accumulated list — uploading
        # via the NIfTI / MAT tab is treated as switching input source, not as
        # appending to a previous DICOM run.
        current = [p for p in current if not Path(p).name.startswith("dcm_converted_to_nii_e")]
        progress(0.5, desc=f"Merging into list (existing {len(current)} + new {len(new_paths)})…")
        new_names = {Path(p).name for p in new_paths}
        kept = [p for p in current if Path(p).name not in new_names]
        updated = kept + new_paths
        srt = _sort_paths(updated)
        progress(0.85, desc=f"Computing shape summary across {len(srt)} files…")
        summary = _shape_summary(srt)
        progress(1.0, desc="Done")
        added_names = [Path(p).name for p in new_paths]
        if len(added_names) == 1:
            status = f"✅ Added file: `{added_names[0]}`"
        else:
            status = (
                f"✅ Added {len(added_names)} files:\n\n"
                + "\n".join(f"- `{n}`" for n in added_names)
            )
        return (updated, srt or None, summary, None, gr.update(open=True), status,
                _clear_btn_update(len(srt)))

    def parse_dicom(files, progress=gr.Progress()):
        # Outputs: accumulated, sorted_files, sorted_info, te_ms,
        # magnitudes_input, dicom_input, dicom_info, order_group, clear_order_btn
        if not files:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                None,
                "",
                gr.update(),
                gr.update(),
            )
        raw_list = files if isinstance(files, list) else [files]

        progress(0.05, desc=f"Reading {len(raw_list)} upload entry(ies)…")

        # Expand any directories that arrived as a single entry.
        file_paths = []
        for f in raw_list:
            p = _to_path(f)
            if p is None:
                continue
            if p.is_dir():
                file_paths.extend(str(c) for c in p.rglob("*") if c.is_file())
            elif p.exists():
                file_paths.append(str(p))

        if not file_paths:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                None,
                "❌ No readable files were found in the upload.",
                gr.update(),
                gr.update(),
            )

        if len(file_paths) == 1:
            name = Path(file_paths[0]).name
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                None,
                (
                    f"⚠️ Only one file was uploaded (`{name}`).\n\n"
                    f"DeepRelaxo needs the **entire folder** of multi-echo DICOMs "
                    f"(≥ 2 echoes). It looks like you navigated into the folder and "
                    f"selected a single file by mistake.\n\n"
                    f"**How to fix:** in the OS folder dialog, click the folder name "
                    f"**once** in the file list (or in the side pane) and confirm — "
                    f"don't enter the folder and click a file inside."
                ),
                gr.update(),
                gr.update(),
            )

        progress(0.25, desc=f"Parsing {len(file_paths)} DICOM files…")

        work_dir = Path(tempfile.mkdtemp(prefix="deeprelaxo_dicom_"))
        try:
            from data_utils import load_dicom_files
            echoes = load_dicom_files(file_paths, work_dir)
        except Exception as exc:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                None,
                f"❌ DICOM parsing failed:\n{exc}",
                gr.update(),
                gr.update(),
            )

        progress(0.95, desc="Building summary…")

        nifti_paths = [str(e["nifti_path"]) for e in echoes]
        te_str = ", ".join(f"{e['te_ms']:g}" for e in echoes)
        # Use <pre> so the column-aligned layout survives Markdown rendering
        # (otherwise newlines and indentation get collapsed).
        name_w = max(len(Path(e["nifti_path"]).name) for e in echoes) + 2
        body_lines = [f"✅ Parsed {len(echoes)} echoes from DICOM:"]
        for i, e in enumerate(echoes, 1):
            fname = Path(e["nifti_path"]).name
            body_lines.append(
                f"  Echo {i}:  {fname.ljust(name_w)}TE = {e['te_ms']:g} ms     shape {e['shape']}"
            )
        body_lines.append("")
        body_lines.append(f"NIfTI files written to: {work_dir}")
        info = (
            "<pre style='font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; "
            "white-space: pre; margin: 0; line-height: 1.5;'>"
            + "\n".join(body_lines)
            + "</pre>"
        )
        srt = _sort_paths(nifti_paths)
        progress(1.0, desc="Done")
        return (
            nifti_paths,                # accumulated state
            srt,                         # sorted_files
            _shape_summary(srt),         # sorted_info
            te_str,                      # te_ms
            None,                        # magnitudes_input — clear
            None,                        # dicom_input — clear
            info,                        # dicom_info
            gr.update(open=True),        # order_group — auto-expand
            _clear_btn_update(len(srt)), # clear_order_btn — show
        )

    def show_mask_info(mask, accumulated_paths):
        if mask is None:
            return ""
        path = _to_path(mask)
        if path is None or not path.exists():
            return ""
        try:
            from data_utils import load_array_with_affine
            arr, _ = load_array_with_affine(path)
            mask_shape = tuple(arr.shape)
            shape_str = " × ".join(str(s) for s in mask_shape)
            base = (
                f"&nbsp;&nbsp;**Loaded:** `{path.name}` &nbsp;·&nbsp; "
                f"**Shape:** {shape_str} &nbsp;·&nbsp; **dtype:** `{arr.dtype}`"
            )
        except Exception as exc:
            return f"⚠️ Could not read mask: {exc}"

        # Compare against the magnitudes' spatial shape (first 3 dims for 4D).
        if not accumulated_paths:
            return base + " &nbsp;·&nbsp; *(load magnitudes to verify shape match)*"
        mag_spatials = set()
        for p in accumulated_paths:
            s = _file_shape(p)
            if s and len(s) >= 3:
                mag_spatials.add(tuple(s[:3]))
        if not mag_spatials:
            return base
        if len(mag_spatials) > 1:
            return base + " &nbsp;·&nbsp; ⚠️ magnitudes have mismatched shapes — cannot compare"
        expected = next(iter(mag_spatials))
        mask_spatial = mask_shape[:3] if len(mask_shape) >= 3 else mask_shape
        if mask_spatial == expected:
            return base + " &nbsp;·&nbsp; ✓ **matches magnitudes**"
        exp_str = " × ".join(str(s) for s in expected)
        return (
            base + f" &nbsp;·&nbsp; ⚠️ **does not match magnitudes** "
            f"(expected {exp_str})"
        )

    def on_mask_upload(uploaded, accumulated_paths, progress=gr.Progress()):
        if uploaded is None:
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        progress(0.3, desc="Reading mask file…")
        info = show_mask_info(uploaded, accumulated_paths)
        progress(1.0, desc="Done")
        return gr.update(value=uploaded, visible=True), info, gr.update(visible=True)

    # Surface a red "waiting" message immediately on click so remote users
    # see something during the slow network upload phase. Note: if the user
    # cancels the OS file picker, the message will linger until the next
    # interaction — Gradio gives no cancel event we can hook.
    mask_button.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for mask selection / upload — info will appear once the file transfer completes…"
        ),
        outputs=mask_info,
    )
    mask_button.upload(
        on_mask_upload,
        inputs=[mask_button, accumulated],
        outputs=[mask_file, mask_info, mask_clear_btn],
    )

    # When the value is cleared (X click or programmatic clear),
    # hide the file display + clear button + wipe the info line.
    def on_mask_change(value):
        if value is None:
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        return gr.update(), gr.update(), gr.update()

    mask_file.change(
        on_mask_change,
        inputs=mask_file,
        outputs=[mask_file, mask_info, mask_clear_btn],
    )
    # Also explicitly handle the per-file X click via the .delete event,
    # since some Gradio versions don't fire .change for that gesture.
    mask_file.delete(
        lambda: (gr.update(value=None, visible=False), "", gr.update(visible=False)),
        outputs=[mask_file, mask_info, mask_clear_btn],
    )

    # Explicit Remove button — same pattern as Clear Processing Order.
    mask_clear_btn.click(
        lambda: (gr.update(value=None, visible=False), "", gr.update(visible=False)),
        outputs=[mask_file, mask_info, mask_clear_btn],
    )

    def sync_after_remove(visible_files):
        """User clicked X on a file row (only fires for ≥2-file display)."""
        files = (visible_files if isinstance(visible_files, list)
                 else ([] if visible_files is None else [visible_files]))
        files = [f for f in files if f is not None]
        if not files:
            return [], None, "", gr.update(), _clear_btn_update(0)
        paths = [str(_to_path(f)) for f in files]
        return paths, paths, _shape_summary(paths), gr.update(open=True), _clear_btn_update(len(paths))

    sorted_files.change(
        sync_after_remove,
        inputs=[sorted_files],
        outputs=[accumulated, sorted_files, sorted_info, order_group, clear_order_btn],
    ).then(show_mask_info, inputs=[mask_file, accumulated], outputs=mask_info)
    sorted_files.delete(
        sync_after_remove,
        inputs=[sorted_files],
        outputs=[accumulated, sorted_files, sorted_info, order_group, clear_order_btn],
    ).then(show_mask_info, inputs=[mask_file, accumulated], outputs=mask_info)

    # Explicit "Clear Processing Order" button — needed because gr.File
    # doesn't render a per-file X for the single-file display state.
    def on_clear_order():
        return [], None, "", gr.update(), _clear_btn_update(0)

    clear_order_btn.click(
        on_clear_order,
        outputs=[accumulated, sorted_files, sorted_info, order_group, clear_order_btn],
    ).then(show_mask_info, inputs=[mask_file, accumulated], outputs=mask_info)

    magnitudes_input.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for file selection / upload — Processing Order will populate once the file transfer completes…"
        ),
        outputs=magnitudes_info,
    )
    magnitudes_input.upload(
        add_files,
        inputs=[magnitudes_input, accumulated],
        outputs=[accumulated, sorted_files, sorted_info, magnitudes_input,
                 order_group, magnitudes_info, clear_order_btn],
    ).then(show_mask_info, inputs=[mask_file, accumulated], outputs=mask_info)
    dicom_input.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for folder selection / upload — parsing will start once the file transfer completes…"
        ),
        outputs=dicom_info,
    )
    dicom_input.upload(
        parse_dicom,
        inputs=[dicom_input],
        outputs=[accumulated, sorted_files, sorted_info, te_ms,
                 magnitudes_input, dicom_input, dicom_info, order_group, clear_order_btn],
    ).then(show_mask_info, inputs=[mask_file, accumulated], outputs=mask_info)

    run_btn.click(
        run_pipeline,
        inputs=[accumulated, te_ms, mask_file, batch_size, vmin_input, vmax_input],
        outputs=[log_out, result_file, result_info, img_step1, img_step2,
                 output_state, slice_slider, log_group, results_group, viz_group],
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

def _find_free_port(preferred=7860, max_tries=20, host="127.0.0.1"):
    import socket
    for offset in range(max_tries):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in range {preferred}–{preferred + max_tries - 1}"
    )


if __name__ == "__main__":
    import os
    host = "127.0.0.1"
    port = _find_free_port(7860, host=host)
    if port != 7860:
        print(f"⚠️  Port 7860 is in use — falling back to {port}")

    url = f"http://{host}:{port}/"
    is_remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
                     or os.environ.get("SSH_TTY"))
    if is_remote:
        print()
        print("=" * 60)
        print("Running over SSH — auto-open skipped.")
        print(f"Open this URL in your local browser:\n  {url}")
        print("If the host isn't reachable from your laptop, forward the port:")
        print(f"  ssh -L {port}:127.0.0.1:{port} <user>@<host>")
        print("=" * 60)
        print()
    else:
        # Poll the port so the browser doesn't arrive before Gradio is listening.
        import webbrowser, socket, threading, time

        def _open_when_ready():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.3)
                    if s.connect_ex((host, port)) == 0:
                        try:
                            webbrowser.open(url)
                        except Exception:
                            pass
                        return
                time.sleep(0.2)

        threading.Thread(target=_open_when_ready, daemon=True).start()
    app.launch(server_name=host, server_port=port, max_file_size="5gb")
