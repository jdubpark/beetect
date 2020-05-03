from pathlib import Path

import torch
from torch.utils.data import Dataset

class BeeDataset(Dataset):
    def __init__(self):
        """
        Args:

        """
        self.root_dir

    def __len__(self):
        return len(self.)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        return
