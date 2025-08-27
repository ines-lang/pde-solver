import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
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

pde = "KortewegDeVries" # options: 'KuramotoSivashinsky', 'Burgers', 'KortewegDeVries'
num_spatial_dims = 3
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries'
bc = None

x_domain_extent = 32.0
y_domain_extent = 32.0 
z_domain_extent = 32.0 
num_points = 64
dt_save = 0.001 # integrator step
t_end = 1.0 # final physical time
save_freq = 10  # save every X integrator steps

''' What it implies:
Total steps: n_steps = t_end / dt_save
Output interval: dt_output = dt_save * save_freq
Total saved frames: n_saved = n_steps / save_freq + 1
'''

nu = [0, 0.01]  # For Burgers and KortewegDeVries equations

simulations = 2
plotted_sim = 1
plot_sim = False
stats = True
seed = 42

# =========================================
# GENERATE DATASET
# =========================================
seed_list = list(range(simulations)) 

all_trajectories, ic_hashes, trajectory_nus = generate_dataset(
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
    seed_list=seed_list,
    seed=seed)

all_trajectories = np.stack(all_trajectories) # changed it from jnp to np to use cpu instead of gpu
print("Original shape:", all_trajectories.shape)

# =========================================
# SAVE DATASET
# =========================================

#  Directory dependant on the pde and initial condition
base_dir = os.path.join(pde, ic)
os.makedirs(base_dir, exist_ok=True)
file_name = "dataset.h5"
data_path = os.path.join(base_dir, file_name)
plots_path = os.path.join(base_dir, "plots")
os.makedirs(plots_path, exist_ok=True)

# Create the h5py file and save the dataset with groups

# Function to determine group name based on PDE and its parameters
def get_group_name(pde, seed, nu_val=None, ic_val=None):
    if pde == "Burgers":
        return f"nu_{nu_val:.3f}"
    elif pde == "KuramotoSivashinsky":
        return f"nu_{nu_val:.3f}"
    elif pde == "KortewegDeVries":
        return f"ic_{ic_val}"
    else:
        return f"seed_{seed:03d}"

with h5py.File(data_path, "w") as h5file:
    idx = 0

    if pde == "KortewegDeVries":
        for seed in seed_list:
            ic_val = ic_hashes[idx] if "ic_hashes" in locals() and idx < len(ic_hashes) else None
            group_name = get_group_name(
                pde,
                seed,
                ic_val=ic_val,   # use ic_hash for KdV
            )
            u_xt = all_trajectories[idx]
            grp = h5file.create_group(group_name)
            grp.create_dataset(f"velocity_seed_{seed:03d}", data=u_xt)
            idx += 1

    else:  # Burgers or Kuramoto-Sivashinsky // change to elif if adding more pdes
        for nu_val in nu:
            for seed in seed_list:
                group_name = get_group_name(
                    pde,
                    seed,
                    nu_val=nu_val,
                )
                u_xt = all_trajectories[idx]
                if group_name in h5file:
                    grp = h5file[group_name]
                else:
                    grp = h5file.create_group(group_name)
                grp.create_dataset(f"velocity_seed_{seed:03d}", data=u_xt)
                idx += 1

    print(f"File created at {data_path}")


    # Optional: print structure
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
    
    h5file.visititems(print_structure)

# ========================
# STADISTICS
# ========================
if stats:
    # Detect number of channels from first loaded dataset
    data = all_trajectories
    print("Original shape:", data.shape)
    num_channels = data.shape[2] # Channels are in the third position
    print("Detected num_channels:", num_channels)

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
# PLOT RENDER CENTER SLIDES
# ========================

sim_names = []
if pde in ["Burgers", "KuramotoSivashinsky"]:
    for nu_val in nu:
        for s in seed_list:
            sim_names.append(get_group_name(pde, s, nu_val=nu_val))
elif pde == "KuramotoSivashinskyConservative":
    for s in seed_list:
        sim_names.append(get_group_name(pde, s))
elif pde == "KortewegDeVries":
    for ic_val in ic_hashes:
        for s in seed_list:
            sim_names.append(get_group_name(pde, s, ic_val=ic_val))
else:
    for s in seed_list:
        sim_names.append(get_group_name(pde, s))

if plot_sim:
    random.seed(seed)
    selected_simulations = random.sample(range(len(sim_names)), plotted_sim)

    for n_sim in selected_simulations:
        sim_name = sim_names[n_sim]
        
        parts = sim_name.split("_")
        nu_val = float(parts[1])
        seed_val = int(parts[3])

         # pull the trajectory for this sim
        u_xt = all_trajectories[n_sim]  # expected shape: (C, T, X) or (C, T, X, Y, Z)
        
        for c in range(num_channels):
            u_component = u_xt[c]  # shape: (T, X, Y, Z)
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
            title = ax.set_title(f"{ic} - seed {seed} - channel {c} - t = 0")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            def update(t_idx):
                im.set_array(slice_sequence[t_idx].T)
                title.set_text(f"{ic} - seed {seed} - channel {c} - t = {t_idx}")
                return im, title

            skip = 2
            frames = range(0, T, skip)
            ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)

            video_path = os.path.join(plots_path, f"center_slice_seed_{seed}_channel_{c}.mp4")
            ani.save(video_path, writer='ffmpeg', fps=10)
            print(f"Animation saved at {video_path}")

            plt.close(fig)

# # ========================
# # PLOT 3D ANIMATIONS (TIME INTENSIVE)
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