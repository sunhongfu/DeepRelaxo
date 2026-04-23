# DeepRelaxo Code

PyTorch inference pipeline for Transformer-based R2* estimation followed by DeepRelaxo UNet refinement.

## Layout

- `Code/run_full_pipeline.py` runs the Transformer step followed by the UNet step.
- `Code/transformer_sim.py` contains Transformer inference utilities.
- `Code/test_prisma_Unet.py` contains UNet inference utilities.
- `Code/echo_wise_loader_no_loop.py` contains the echo-wise dataset loader.
- `Code/transformer.py` and `Code/UNet_upsample.py` define the model architectures.

## Data and Checkpoints

Large data files, generated inference outputs, and model checkpoints are intentionally ignored by git. Keep the required local files in place when running the pipeline:

- `Code/checkpoint_epoch_80_Transformer.pth`
- `Code/Unet_140.pkl`
- subject data under `Data/`
