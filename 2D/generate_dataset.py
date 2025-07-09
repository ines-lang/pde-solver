import os
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import time
import random
import matplotlib.animation as animation

from stepper import generate_dataset

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinsky' (ks), 'Burgers'

ic : str
    Initial condition function. Options: 'RandomTruncatedFourierSeries'

bc : callable
    Boundary condition. (Unused As JAX computes spatial derivatives using the Fast Fourier Transform (FFT)
    which assumes periodicity, kept for consistency)

num_spatial_dims : int
    Number of spatial dimensions. 1 for this plot as we are using a 1D PDE

domain_extent : float
    Spatial domain extent (length of the domain in each spatial dimension)
    x_domain_extent and y_domain_extent must be the same for the solver to work.

num_points : int
    Number of points in each spatial dimension (spatial resolution)

dt_solver : float
    Time step size used by the numerical solver to integrate the PDE.

t_end : float
    Final simulation time

save_freq : int
    Save data every this many steps

nu : float
    Viscosity (used only for Burgers)

Re : float
    Reynolds number (used only for Kolmogorov)

simulations : int
    Number of simulations to run

plotted_sim : int
    Number of simulations to plot
"""

pde = "Kolmogorov" # options: KuramotoSivashinsky (ks), Burgers, Kolmogorov
num_spatial_dims = 2
ic = "RandomSpectralVorticityField" # options: 'RandomTruncatedFourierSeries'
bc = None

x_domain_extent = 100.0
y_domain_extent = 100.0 
dt_solver = 0.001
t_end = 100.0 
save_freq = 100
simulations = 1
plotted_sim = 1

# For Burgers equation, set viscosity
nu = 0.1
# For Kolmogorov equation, set Reynolds number
Re = 250

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
    y_domain_extent=y_domain_extent,
    num_points=num_points,  
    dt_solver=dt_solver,
    t_end=t_end,
    nu=nu,
    Re=Re,
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
        sim_idx = int(sim_idx)  # Ensure sim_idx is an integer
        seed = seed_list[sim_idx]
        u_xt = all_trajectories[sim_idx, :, :, :]
        dataset_name = f'velocity_{seed:03d}'
        h5file.create_dataset(dataset_name, data=u_xt)  
    print(f"Dataset saved at {data_path}")
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

# ========================
# Plot 2D animation
# ========================
random.seed(time.time())
selected_simulations = random.sample(range(len(seed_list)), plotted_sim)

for n_sim in selected_simulations:
    seed = seed_list[n_sim]
    u_xt = all_trajectories[n_sim]

    num_channels = u_xt.shape[1]  # Number of channels (e.g., 1 for u, 2 for u and v in Burgers)
    
    for ch in range(num_channels):  # loop over each channel
        u_component = u_xt[:, ch]  # shape: (T, H, W)

        extent = (
            0, x_domain_extent,
            0, y_domain_extent)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(u_component[0].T, cmap='RdBu', origin='lower', extent=extent,
                    vmin=-2, vmax=2, aspect='auto')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("u(x, t)")

        title = ax.set_title(f"{ic} - seed {seed} - channel {ch} - t = 0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Animation update function
        def update(t_idx):
            frame = u_component[t_idx].T  # transpose so x is horizontal and y vertical
            im.set_array(frame)
            title.set_text(f"{ic} - seed {seed} - channel {ch} - t = {t_idx}")
            return im, title
        
        skip = 2  # or compute it based on desired duration
        frames = range(0, u_component.shape[0], skip)
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)

        # Save the MP4
        video_path = os.path.join(plots_path, f"evolution_seed_{seed}_channel_{ch}.mp4")
        ani.save(video_path, writer='ffmpeg', fps=60)
        print(f"Animation saved at {video_path} for seed {seed}, channel {ch}")

        plt.close(fig)