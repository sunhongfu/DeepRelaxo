import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import torch.nn as nn

from data_utils import load_mask_array, resolve_te_values_ms


class Dataset_Load(nn.Module):

    def __init__(
        self,
        sim_root_path,
        reference_invivo_path,
        echo_no,
        te_values_ms=None,
        first_te_ms=None,
        echo_spacing_ms=None,
        bet_mask_path=None,
    ):
        #  Dataset_Load(sim_root_path= subject_path_sim ,reference_invivo_path = subject_path1_invivo, echo_no= i)

        super(Dataset_Load, self).__init__()

        self.entries = []
        self.echo_no = echo_no
        self.TEs = resolve_te_values_ms(
            te_values=te_values_ms,
            first_te_ms=first_te_ms,
            echo_spacing_ms=echo_spacing_ms,
            echo_count=echo_no,
        ).tolist()
        self.max_len = 10
        # import pdb; pdb.set_trace()
        # Load magnitude data for all echoes
        mag_list = np.stack([nib.load(str(sim_root_path / f'mag{i}.nii')).get_fdata()  for i in range(1, len(self.TEs) + 1)],
                            axis=-1)
        mask_path = Path(bet_mask_path) if bet_mask_path else None
        mask = load_mask_array(mask_path, mag_list.shape[:3])

        # Create a boolean mask where both masked_r2s and masked_amp are greater than 0
        valid_mask = mask

        # Get the indices of valid voxels
        valid_indices = np.where(valid_mask)

        # Select valid magnitudes for the valid voxels
        self.magnitudes = mag_list[valid_indices]  # Shape will be (n_valid_voxels, 8)

    def __getitem__(self, index):
        # Extract the magnitudes for the given index
        mags = torch.tensor(self.magnitudes[index, :self.echo_no])
        #import pdb;  pdb.set_trace()
        # Extract the corresponding echo times (TEs)
        TEs = torch.tensor(self.TEs[0:self.echo_no])

        return mags, TEs

    def __len__(self):
        return len(self.magnitudes)
