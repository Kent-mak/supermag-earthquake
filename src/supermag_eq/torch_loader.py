import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
from glob import glob

class GeomagDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Directory containing .nc files.
            chunk_size (int): Number of samples to load at a time.
        """
        self.files = sorted(glob(os.path.join(data_dir, "*.nc")))
        self.ds = xr.open_mfdataset(
            self.files,
            engine="netcdf4",
            combine="nested",  # Stack files without checking coordinate alignment
            concat_dim="sample",  # Ensures only sample dimension is concatenated
            join="override",  # Ignores mismatched time coordinates
            chunks={"sample": 2055, "time": 10080}  # Ensures time remains fixed
        )

        print(self.ds)
        

    def __len__(self):
        return self.ds.sizes["sample"]  # Total number of samples across all files

    def __getitem__(self, idx):
        # Read a single sample (or small chunk)
        sample = self.ds.isel(sample=idx).load()  # Load only 1 sample at a time

        # Extract data variables
        data_vars = ["dbn_nez", "dbe_nez", "dbz_nez"]

        # Convert to NumPy arrays
        data = np.stack([sample[var].values for var in data_vars])  # Shape: (3, 10080)
        label = sample["labels"].values  

        # Convert to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # Assuming classification

        return data, label
     
def get_dataloader(data_dir: str, batch_size: int, num_workers: int=4):

    dataset = GeomagDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Adjust based on memory availability
        shuffle=True,  # Randomize order of samples
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True  # Optimize for GPU training
    )

    return dataloader


if __name__ == "__main__":
    directory = r"D:\earthquake-prediction\data\dataset\supermag"
    dataset = GeomagDataset(data_dir=directory)
    labels = dataset.ds["labels"].values
    counts = np.bincount(labels)
    for label, count in enumerate(counts):
        print(f"Class {label}: {count} samples")
   
    data, label = dataset[0]

    print("Data shape:", data.shape)  # Expected shape: (3, 10080) or (features,)
    print(data)
    print("Label:", label)

    # dataloader = get_dataloader(directory, 32, num_workers=16)

    # for batch in dataloader:
    #     data, labels = batch
    #     print("Batch Data Shape:", data.shape)  # Expected: (batch_size, 3, 10080)
    #     print("Batch Labels Shape:", labels.shape)  # Expected: (batch_size, 10080)
    #     break  # Stop after first batch