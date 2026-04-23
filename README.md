# DeepRelaxo

DeepRelaxo: Fast Mono-Exponential Magnitude Brain R2* Mapping With Reduced Echoes Using Self-Supervised Deep Learning (MRM paper).

This repository contains a PyTorch inference pipeline for Transformer-based R2* estimation followed by DeepRelaxo UNet refinement.

## Layout

- `run_full_pipeline.py` runs the Transformer step followed by the UNet step.
- `transformer_sim.py` contains Transformer inference utilities.
- `test_prisma_Unet.py` contains UNet inference utilities.
- `echo_wise_loader_no_loop.py` contains the echo-wise dataset loader.
- `transformer.py` and `UNet_upsample.py` define the model architectures.

## Data and Checkpoints

Large data files, generated inference outputs, and model checkpoints are intentionally ignored by git. Keep the required local files in place when running the pipeline:

- `checkpoint_epoch_80_Transformer.pth`
- `Unet_140.pkl`
- subject data under `Data/`
