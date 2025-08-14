import os
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
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

dt_save : float
    Time step size used by the numerical solver to integrate the PDE.

t_end : float
    Final simulation time

save_freq : int
    Save data every this many steps

nu : float
    Viscosity (used for Burgers and KS equations)

Re : float
    Reynolds number (used only for Kolmogorov)

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

pde = "KortewegDeVries" # options: 'KuramotoSivashinsky', 'Burgers', 'Kolmogorov', 'KortewegDeVries'
num_spatial_dims = 2
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries', 'RandomSpectralVorticityField'
bc = None

x_domain_extent = 100.0
y_domain_extent = 100.0 
num_points = 100
dt_save = 0.0001
t_end = 500.0 
save_freq = 1

nu = 0.1  # For Burgers and KortewegDeVries equations
Re = 250  # For Kolmogorov equation

simulations = 5
plotted_sim = 1
plot_sim = None
stats = True
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
# with h5py.File(data_path, "w") as h5file:
#     for sim_idx in range(len(seed_list)):
#         sim_idx = int(sim_idx)  # Ensure sim_idx is an integer
#         seed = seed_list[sim_idx]
#         u_xt = all_trajectories[sim_idx, :, :, :]
#         dataset_name = 'velocity_{:03d}'.format(seed)
#         h5file.create_dataset(dataset_name, data=u_xt)  
#     print(f"Dataset saved at {data_path}")
#     def print_structure(name, obj):
#         if isinstance(obj, h5py.Group):
#             print(f"Group: {name}")
#         elif isinstance(obj, h5py.Dataset):
#             print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

with h5py.File(data_path, "w") as h5file:
    for seed, u_xt in zip(seed_list, all_trajectories):
        h5file.create_dataset(
            f"velocity_{seed:03d}",
            data=u_xt.astype(np.float32),
            compression="gzip", compression_opts=4,
            chunks=True
        )
print(f"Dataset saved at {data_path}")

# Structure print phase
with h5py.File(data_path, "r") as h5file:
    h5file.visititems(print_structure)

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
# PLOT 2D ANIMATIONS
# ========================
if plot_sim:
    random.seed(seed)
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
                        vmin=-7, vmax=7, aspect='auto')
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
            ani.save(video_path, writer='ffmpeg', fps=20)
            print(f"Animation saved at {video_path} for seed {seed:02d}, channel {ch}")

            plt.close(fig)