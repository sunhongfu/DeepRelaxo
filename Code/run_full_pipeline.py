import argparse
from pathlib import Path

import transformer_sim as transformer_script
import test_prisma_Unet as unet_script


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--transformer_out', required=True)
    parser.add_argument('--unet_out', required=True)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    transformer_out = Path(args.transformer_out)
    unet_out = Path(args.unet_out)

    print("\n==============================")
    print("STEP 1: TRANSFORMER")
    print("==============================")

    transformer_script.run_transformer(
        subject_dir=data_dir,
        save_dir=transformer_out
    )

    print("\n==============================")
    print("STEP 2: DeepRelaxo")
    print("==============================")

    unet_script.run_unet_on_folder(
        transformer_dir=transformer_out,
        root_data_dir=data_dir,
        save_dir=unet_out
    )

    print("\n✅ DONE")


if __name__ == "__main__":
    main()