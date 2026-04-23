import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset


class Dataset_Load(Dataset):

    def __init__(self, sim_root_path, reference_invivo_path, echo_no):
        super().__init__()

        sim_root_path = Path(sim_root_path)
        reference_invivo_path = Path(reference_invivo_path)
        self.echo_no = echo_no
        self.TEs = [0.0049 * 1000, 0.0099 * 1000, 0.0148 * 1000, 0.0198 * 1000, 0.0247 * 1000]

        mag_list = np.stack([
            nib.load(str(sim_root_path / f'mag{i}.nii')).get_fdata()
            for i in range(1, 6)
        ], axis=-1)

        mask = nib.load(str(reference_invivo_path / 'BET_mask.nii')).get_fdata()

        valid_mask = mask > 0
        valid_indices = np.where(valid_mask)

        self.magnitudes = mag_list[valid_indices]

    def __getitem__(self, index):
        mags = torch.tensor(self.magnitudes[index, :self.echo_no], dtype=torch.float32)
        TEs = torch.tensor(self.TEs[0:self.echo_no], dtype=torch.float32)

        return mags, TEs

    def __len__(self):
        return len(self.magnitudes)
