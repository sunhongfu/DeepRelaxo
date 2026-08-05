# Standalone Docker image for DeepRelaxo -- lets any platform (Windows, macOS, Linux) run
# the full pipeline, including automatic bet2 brain extraction. vendor/bet2/ is a
# Linux/x86_64 binary that can't execute natively on Windows or macOS -- Docker Desktop
# runs a real Linux VM on both platforms, so a container gives bet2 (and everything else)
# a genuine Linux environment regardless of host OS.
#
# Build (from this repo's root; requires network access -- checkpoints are downloaded
# during the build so the image is immediately usable with no first-run wait):
#   docker build --platform linux/amd64 -t deeprelaxo .
#
# --platform linux/amd64 is required explicitly on Apple Silicon hosts -- python:3.11-slim
# publishes a native arm64 manifest, so without this flag Docker silently builds for arm64
# and the GPU-enabled torch wheel below fails to resolve.
#
# Run (web app, default) -- open http://localhost:7860 in your browser:
#   docker run --rm -p 7860:7860 deeprelaxo
#
# With GPU acceleration (needs the NVIDIA Container Toolkit on the host; CPU-only works
# too, just slower, no changes needed):
#   docker run --rm --gpus all -p 7860:7860 deeprelaxo
#
# CLI instead of the web app, e.g. against a folder of NIfTIs already on the host:
#   docker run --rm -v /path/to/data:/data deeprelaxo \
#       python3 run_deeprelaxo_pipeline.py --from_converted /data/converted \
#       --transformer_out /data/transformer_out --deeprelaxo_out /data/deeprelaxo_out
FROM python:3.11-slim

WORKDIR /app

COPY . .

# GPU-enabled torch first (before the numpy-dependent packages requirements-webapp.txt
# pulls in) so pip's resolver picks numpy-2-compatible builds of everything else from the
# start -- same ordering rationale as or_qsm's own docker/qsm.dockerfile. cu124 wheels
# bundle their own CUDA/cuDNN/cuBLAS runtime; only the host's driver is needed at runtime
# via --gpus all, and the same wheel runs CPU-only with no GPU passed at all.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements-webapp.txt

# Pretrained checkpoints (https://huggingface.co/sunhongfu/DeepRelaxo), baked in at build
# time so the image is immediately usable -- no first-run download/wait.
RUN python run_deeprelaxo_pipeline.py --download-checkpoints

# bet2 arrived via `COPY . .` above (vendor/bet2/, ~118MB -- see vendor/bet2/README.md for
# provenance/license); bet2_utils.py finds it automatically relative to this file, no
# extra ENV needed.

# 0.0.0.0 so `-p 7860:7860` can actually reach the server -- 127.0.0.1 (app.py's native
# desktop default) is the container's own loopback, unreachable from the host.
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python3", "app.py"]
