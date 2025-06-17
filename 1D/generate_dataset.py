import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
import random

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

dt : float
    Time step size

t_end : float
    Final simulation time

save_freq : int
    Save data every this many steps

nu : float
    Viscosity (used only for Burgers)

simulations : int
    Number of simulations to run

plotted_sim : int
    Number of simulations to plot
"""

pde = "KuramotoSivashinskyConservative" # options: KuramotoSivashinskyConservative (ks)
num_spatial_dims = 1 
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries', 'GaussianRandomField'
bc = None

x_domain_extent = 100.0
num_points = 200 
dt = 0.1
t_end = 1000.0 
save_freq = 1 
simulations = 50
plotted_sim = 10

# For Burgers equation, set viscosity
nu = 0.1

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
    dt=dt,
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
        dataset_name = f'velocity_{seed:03d}'
        h5file.create_dataset(dataset_name, data=u_xt)  
    print(f"File created at {data_path}")
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

# ========================
# Plot
# ========================
random.seed(time.time())
selected_simulations = random.sample(range(len(seed_list)), plotted_sim)
for n_sim in selected_simulations:
    seed = seed_list[n_sim]
    plt.imshow(all_trajectories[n_sim, :, 0, :].T, # first simulation, channel 0
            aspect='auto', cmap='RdBu', vmin=-2, vmax=2, origin="lower")
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.title(f"{ic} - seed {seed}")
    plt.show()
    plt.savefig(os.path.join(plots_path, f"seed_{seed}.png"))
    plt.close()