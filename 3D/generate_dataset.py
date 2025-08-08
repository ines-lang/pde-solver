import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import time
import random
import matplotlib.animation as animation
import imageio.v3 as imageio
from mpl_toolkits.mplot3d import Axes3D

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
    x_domain_extent, y_domain_extent and z_domain_extent must be the same for the solver to work.

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

simulations : int
    Number of simulations to run

plotted_sim : int
    Number of simulations to plot
"""

pde = "KuramotoSivashinsky" # options: 'KuramotoSivashinsky', 'Burgers', 'KortewegDeVries'
num_spatial_dims = 3
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries'
bc = None

x_domain_extent = 100.0
y_domain_extent = 100.0 
z_domain_extent = 100.0 
num_points = 100
dt_solver = 0.0001
t_end = 100.0 
save_freq = 1
simulations = 5
plotted_sim = 5

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
    dt_solver=dt_solver,
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
plots_3d_path = os.path.join(base_dir, "plots_3d")
os.makedirs(plots_path, exist_ok=True)

# Create the h5py file and save the dataset
with h5py.File(data_path, "w") as h5file:
    for sim_idx in range(len(seed_list)):
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
# Plot render center slices
# ========================
random.seed(time.time())
selected_simulations = random.sample(range(len(seed_list)), plotted_sim)

for n_sim in selected_simulations:
    seed = seed_list[n_sim]
    u_xt = all_trajectories[n_sim]  # shape: (T, C, X, Y, Z)

    num_channels = u_xt.shape[1]
    
    for ch in range(num_channels):
        u_component = u_xt[:, ch]  # shape: (T, X, Y, Z)
        T, X, Y, Z = u_component.shape

        # Get central slice in Z direction
        center_z = Z // 2
        slice_sequence = u_component[:, :, :, center_z]  # shape: (T, X, Y)

        extent = (0, x_domain_extent, 0, y_domain_extent)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(slice_sequence[0].T, cmap='RdBu', origin='lower',
                       extent=extent, vmin=-15, vmax=15, aspect='auto')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("u(x, y, z_center)")
        title = ax.set_title(f"{ic} - seed {seed} - channel {ch} - t = 0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(t_idx):
            im.set_array(slice_sequence[t_idx].T)
            title.set_text(f"{ic} - seed {seed} - channel {ch} - t = {t_idx}")
            return im, title

        skip = 2
        frames = range(0, T, skip)
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)

        video_path = os.path.join(plots_path, f"center_slice_seed_{seed}_channel_{ch}.mp4")
        ani.save(video_path, writer='ffmpeg', fps=10)
        print(f"Animation saved at {video_path}")

        plt.close(fig)

# # ========================
# # Plot 3D animation (time-intensive)
# # ========================
# random.seed(time.time())
# selected_simulations = random.sample(range(len(seed_list)), plotted_sim)

# for n_sim in selected_simulations:
#     seed = seed_list[n_sim]
#     u_xt = all_trajectories[n_sim]

#     num_channels = u_xt.shape[1]  # Number of channels (e.g., 1 for u, 2 for u and v in Burgers)
    
#     for ch in range(num_channels):  # loop over each channel
#         u_component = u_xt[:, ch]  # shape: (T, H, W)

#         extent = (
#             0, x_domain_extent,
#             0, y_domain_extent,
#             0, z_domain_extent)
        
#          # Compute global vmin/vmax for normalization (you can also set fixed values)
#         vmin = -2
#         vmax = 2
        
#         frame_dir = os.path.join(plots_3d_path, f"frames_seed_{seed}_channel{ch}")
#         os.makedirs(frame_dir, exist_ok=True)
#         frame_paths = []

#         print(f"Rendering 3D frames for seed {seed}, channel {ch}...")
        
#         skip = 2
#         frames = range(0, u_component.shape[0], skip)

#         for t in frames:
#             volume = u_component[t]  # (X, Y, Z)
#             normed = (volume - vmax) / (vmax - vmin)

#             cmap = plt.cm.RdBu
#             colors = cmap(normed)  # Shape: (X, Y, Z, 4)
#             filled = np.ones_like(volume, dtype=bool)

#             fig = plt.figure(figsize=(6, 6))
#             ax = fig.add_subplot(111, projection='3d')
#             ax.voxels(filled, facecolors=colors, edgecolor='none')

#             ax.set_title(f"{ic} – Seed {seed} – Ch {ch} – t = {t}")
#             ax.set_xlabel("X")
#             ax.set_ylabel("Y")
#             ax.set_zlabel("Z")
#             ax.view_init(elev=20, azim=30)

#             frame_path = os.path.join(frame_dir, f"frame_{t:03}.png")
#             plt.savefig(frame_path, bbox_inches='tight')
#             frame_paths.append(frame_path)
#             plt.close(fig)

#         # Make MP4 animation
#         video_path = os.path.join(plots_path, f"evolution3D_seed_{seed}_channel_{ch}.mp4")
#         writer = imageio.get_writer(video_path, fps=10)
#         for fpath in frame_paths:
#             frame = imageio.imread(fpath)
#             if frame.ndim == 2:
#                 frame = np.stack([frame]*3, axis=-1)
#             elif frame.shape[2] == 4:
#                 frame = frame[:, :, :3]
#             writer.append_data(frame)
#         writer.close()

#         print(f"3D animation saved at {video_path}")