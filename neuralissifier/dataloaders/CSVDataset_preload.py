import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dynamic_brainage.dataloaders.CSVDataset import CSVDataset
import pandas as pd
import mat73
import scipy.io
import time
import copy
import tqdm


def get_ram_subset(dataset, idx):
    kwargs = copy.deepcopy(dataset.kwargs)
    kwargs['idx'] = idx
    dataset = RAMDataset(**kwargs)
    return dataset


class RAMDataset(Dataset):
    def __init__(self, *args, num_subj_to_prefetch=64, **kwargs):
        self.csvdata = CSVDataset(*args, **kwargs)
        for key in dir(self.csvdata):
            if key[0] != "_":
                setattr(self, key, getattr(self.csvdata, key))
        self.full_data = []
        self.full_label = []
        self.full_index = []
        dl = torch.utils.data.DataLoader(
            self.csvdata, batch_size=32, num_workers=12, prefetch_factor=2)
        self.full_data = None
        memory_used = 0.
        self.prefetched = []
        for i, batch in tqdm.tqdm(enumerate(dl), total=min(len(dl),num_subj_to_prefetch)):
            data, label, index = batch
            if self.full_data is None:
                self.full_data = data
            else:
                self.full_data = torch.cat([self.full_data, data], dim=0)
            self.prefetched.append(i)
            self.full_label.append(label)
            self.full_index.append(index)
       
        #self.full_data = torch.cat(self.full_data, 0)
        self.full_label = torch.cat(self.full_label, 0)
        self.full_index = torch.cat(self.full_index, 0)

    def __len__(self):
        return len(self.full_data)

    def __getitem__(self, k):
        return self.full_data[k], self.full_label[k], self.full_index[k]
      


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_dataset = RAMDataset(N_subs=99999999999, sequential=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for batch_i, (fnc, age) in enumerate(test_dataloader):
        print("Loaded batch %d with FNC shape %s, and average age %.2f" %
              (batch_i, str(fnc.shape), age.mean().item()))
