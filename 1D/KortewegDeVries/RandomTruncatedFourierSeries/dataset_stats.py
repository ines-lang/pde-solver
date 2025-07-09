import h5py
import numpy as np
from pathlib import Path

# Define the HDF5 file name
dataset = "dataset.h5"
dataset_path = Path(dataset)


# Initialize accumulators for Welford's method
mean = 0.0
M2 = 0.0
global_min = np.inf
global_max = -np.inf
total_elements = 0

# Process datasets
with h5py.File(dataset_path, "r") as h5file:
    for name in h5file:
        data = h5file[name][()].astype(np.float64) # new incorporation
        
        # Check for NaN and Inf
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        data_min = np.min(data)
        data_max = np.max(data)

        print(f"[{name}] min={data_min:.3e}, max={data_max:.3e}, has_nan={has_nan}, has_inf={has_inf}")

        if has_nan or has_inf:
            print(f"[Warning] Dataset '{name}' contains NaN or Inf. Skipping it.")
            continue  # Skip this dataset to avoid breaking the stats

        # Flatten for easier processing
        data_flat = data.flatten()
        n = data_flat.size
        # Update global min/max
        global_min = min(global_min, np.min(data_flat))
        global_max = max(global_max, np.max(data_flat))
        # Update Welford's algorithm
        total_elements += n
        delta = data_flat - mean
        mean += np.sum(delta) / total_elements
        delta2 = data_flat - mean
        M2 += np.sum(delta * delta2)

# Compute statistics
variance = M2 / total_elements
std = np.sqrt(variance)

# Print results
print("Dataset Global Statistics")
print(f"  Mean: {mean:.6f}")
print(f"  Std : {std:.6f}")
print(f"  Min : {global_min:.6f}")
print(f"  Max : {global_max:.6f}")
