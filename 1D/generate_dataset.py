import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import random
import numpy as np

from stepper import generate_dataset

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinskyConservative' (ks)

ic : str
    Initial condition function. Options: 'sine_ic_2d', 'gaussian_ic_2d', 'random_ic_2d', 'RandomTruncatedFourierSeries'

bc : callable
    Boundary condition. (Unused As JAX computes spatial derivatives using the Fast Fourier Transform (FFT)
    which assumes periodicity, kept for consistency)

num_spatial_dims : int
    Number of spatial dimensions. 1 for this plot as we are using a 1D PDE

x_domain_extent : float
    Spatial domain extent (length of the domain in each spatial dimension)

num_points : int
    Number of points in each spatial dimension (spatial resolution)

dt_save : float
    Time step size used by the numerical solver to integrate the PDE.

t_end : float
    Final simulation time

save_freq : int
    Save data every this many steps

nu : float
    Viscosity (used for Burgers and KS equations)

simulations : int
    Number of simulations to run

plotted_sim : int
    Number of simulations to plot

plot_sim : bool
    Whether to plot the simulations or not

stats : bool
    Whether to compute statistics on the dataset or not

seed : int
    Random seed for reproducibility
"""

pde = "KortewegDeVries" # options: 'KuramotoSivashinskyConservative', 'Burgers', 'KortewegDeVries'
num_spatial_dims = 1 
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries', 'GaussianRandomField'
bc = None

x_domain_extent = 64.0
num_points = 100 
dt_save = 0.01
t_end = 100.0 
save_freq = 1 

nu = 0.1  # For Burgers and KortewegDeVries equations

simulations = 2
plotted_sim = 1
plot_sim = True
stats = None
seed = 42 

# =========================================
# GENERATE AND SAVE DATASET
# =========================================
seed_list = list(range(simulations)) 

all_trajectories = generate_dataset(
    pde=pde,
    num_spatial_dims=num_spatial_dims,
    ic=ic,
    bc=bc,
    x_domain_extent=x_domain_extent,
    num_points=num_points,  
    dt_save=dt_save,
    t_end=t_end,
    nu=nu,
    save_freq=save_freq,
    seed_list=seed_list)
all_trajectories = jnp.stack(all_trajectories)
print("Original shape:", all_trajectories.shape)

#  Directory dependant on the pde and initial condition
base_dir = os.path.join(pde, ic)
os.makedirs(base_dir, exist_ok=True)
file_name = "dataset.h5"
data_path = os.path.join(base_dir, file_name)
plots_path = os.path.join(base_dir, "plots")
os.makedirs(plots_path, exist_ok=True)

# Create the h5py file and save the dataset
with h5py.File(data_path, "w") as h5file:
    for sim_idx in range(len(seed_list)):
        seed = seed_list[sim_idx]
        u_xt = all_trajectories[sim_idx, :, :]  # 0 is valid as we have only one channel
        dataset_name = 'velocity_{:03d}'.format(seed)
        h5file.create_dataset(dataset_name, data=u_xt)  
    print(f"File created at {data_path}")
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

# ========================
# STADISTICS
# ========================
if stats:
    # Detect number of channels from first loaded dataset
    data = all_trajectories
    num_channels = data.shape[2] # Channels are in the third position

    # Initialize accumulators (outside the loop)
    means = np.zeros(num_channels, dtype=np.float64)
    M2s   = np.zeros(num_channels, dtype=np.float64)
    mins  = np.full(num_channels, np.inf, dtype=np.float64)
    maxs  = np.full(num_channels, -np.inf, dtype=np.float64)
    counts = np.zeros(num_channels, dtype=np.int64)

    # Iterate over simulations
    for sim_idx in range(data.shape[0]):
        sim_data = data[sim_idx]  # shape: (T, X, channels)

        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        # Uncomment for debugging
        # print(f"min={np.min(data):.3e}, max={np.max(data):.3e}, has_nan={has_nan}, has_inf={has_inf}")
        if has_nan or has_inf:
            print(f"[Warning] Dataset contains NaN or Inf. Skipping it.")
            continue

        for c in range(num_channels):
            channel_data = sim_data[..., c].ravel()
            n = channel_data.size

            # Update min/max
            mins[c] = min(mins[c], np.min(channel_data))
            maxs[c] = max(maxs[c], np.max(channel_data))

            # Welford update
            counts[c] += n
            delta = channel_data - means[c]
            means[c] += np.sum(delta) / counts[c]
            delta2 = channel_data - means[c]
            M2s[c] += np.sum(delta * delta2)

    # Final stats
    stds = np.sqrt(M2s / counts)

    print("Dataset Global Statistics (Per Channel)")
    for c in range(num_channels):
        print(f"Channel {c}: Mean={means[c]:.6f}, Std={stds[c]:.6f}, "
            f"Min={mins[c]:.6f}, Max={maxs[c]:.6f}")
    
# ========================
# PLOT 1D ANIMATIONS
# ========================
if plot_sim:
    random.seed(seed)
    selected_simulations = random.sample(range(len(seed_list)), plotted_sim)
    for n_sim in selected_simulations:
        seed = seed_list[n_sim]
        plt.imshow(all_trajectories[n_sim, :, 0, :].T, # first simulation, channel 0
                aspect='auto', cmap='RdBu', vmin=-4, vmax=4, origin="lower") # changed the values due to running datatset_stats.py on the dataset
        plt.xlabel("Time")
        plt.ylabel("Space")
        plt.title(f"{ic} - seed {seed}")
        plt.show()
        plt.savefig(os.path.join(plots_path, f"seed_{seed:02d}.png"))
        plt.close()