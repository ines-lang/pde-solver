import h5py
import numpy as np
from pathlib import Path

# Define the HDF5 file name
dataset = "dataset.h5"
dataset_path = Path(dataset)

# Initialize accumulators
global_sum = 0.0
global_squared_sum = 0.0
global_min = np.inf
global_max = -np.inf
total_elements = 0

# Process datasets
with h5py.File(dataset_path, "r") as h5file:
    for name in h5file:
        data = h5file[name][()]
        global_sum += np.sum(data)
        global_squared_sum += np.sum(data ** 2)
        global_min = min(global_min, np.min(data))
        global_max = max(global_max, np.max(data))
        total_elements += data.size

# Compute statistics
mean = global_sum / total_elements
variance = (global_squared_sum / total_elements) - mean ** 2
std = np.sqrt(variance)

# Print results
print("Dataset Global Statistics")
print(f"  Mean: {mean:.6f}")
print(f"  Std : {std:.6f}")
print(f"  Min : {global_min:.6f}")
print(f"  Max : {global_max:.6f}")
