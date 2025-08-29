import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import random
import matplotlib.animation as animation

from collections.abc import Iterable

from stepper import generate_dataset

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinsky' (ks), 'Burgers'

ic : str
    Initial condition function. Options: 'RandomTruncatedFourierSeries', 'RandomSpectralVorticityField',
    For Kolmogorov and Gray Scott you dont don’t need to pass an initial condition as it uses a fixed one.
    'RandomTruncatedFourierSeries' is used for Burgers, KS and KdV equations.

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

pde = "GrayScott" # options: 'KuramotoSivashinsky', 'Burgers', 'Kolmogorov', 'KortewegDeVries', 'GrayScott'
num_spatial_dims = 2
ic = "" # options: 'RandomTruncatedFourierSeries', 'RandomSpectralVorticityField', 
bc = None

x_domain_extent = 2.5
y_domain_extent = 2.5
num_points = 128
dt_save = 1
t_end = 5000.0 
save_freq = 50

''' What it implies:
Total steps: n_steps = t_end / dt_save
Output interval: dt_output = dt_save * save_freq
Total saved frames: n_saved = n_steps / save_freq + 1
'''

nu = [0, 0.01, 0.1, 0.5]  # For Burgers and KortewegDeVries equations
# todo implemnet Re as nu 
Re = 250  # For Kolmogorov equation

simulations = 1
plotted_sim = 1
plot_sim = True
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
    Re=Re,
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
def get_group_name(pde, seed, nu_val=None, ic_val=None, Re=None, feed=None, kill=None):
    if pde == "Burgers":
        return f"nu_{nu_val:.3f}"
    elif pde == "KuramotoSivashinsky":  # with viscosity
        return f"nu_{nu_val:.3f}"
    elif pde == "KortewegDeVries":
        return f"ic_{ic_val}"
    elif pde == "Kolmogorov":
        return f"Re_{Re:.3f}" # add Re_val when introducing a list
    elif pde == "GrayScott":
        return f"feed_{feed:.3f}_kill_{kill:.3f}"
    else:
        return f"seed_{seed:03d}"

with h5py.File(data_path, "w") as h5file:
    idx = 0

    if pde == "Kolmogorov":
        group_name = get_group_name(pde, seed_list[0], Re=Re)
        if group_name in h5file:
            grp = h5file[group_name]
        else:
            grp = h5file.create_group(group_name)

        for seed in seed_list:
            u_xt = all_trajectories[idx]
            grp.create_dataset(f"velocity_seed{seed:03d}", data=u_xt)
            idx += 1

    elif pde == "KortewegDeVries":
        for seed in seed_list:
            ic_val = ic_hashes[idx] if "ic_hashes" in locals() and idx < len(ic_hashes) else None
            group_name = get_group_name(
                pde,
                seed,
                ic_val=ic_val,   # use ic_hash for KdV
            )
            u_xt = all_trajectories[idx]
            grp = h5file.create_group(group_name)
            grp.create_dataset(f"velocity_seed{seed:03d}", data=u_xt)
            idx += 1

    elif pde in ["Burgers", "KuramotoSivashinsky"]:
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
                grp.create_dataset(f"velocity_seed{seed:03d}", data=u_xt)
                idx += 1
    
    if pde == "GrayScott":
        # Build the group name including feed & kill parameters
        feed_rate = 0.04
        kill_rate = 0.06

        group_name = get_group_name(
            pde,
            seed_list[0],
            feed=feed_rate,
            kill=kill_rate,
        )

        # Create or reuse group in HDF5
        if group_name in h5file:
            grp = h5file[group_name]
        else:
            grp = h5file.create_group(group_name)

        # Store each trajectory under its own seed name
        for seed in seed_list:
            u_xt = all_trajectories[idx]  # shape (T, C, X, Y)
            grp.create_dataset(f"state_seed{seed:03d}", data=u_xt)
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
        sim_data = data[sim_idx]  # shape: (C, T, X, Y)

        has_nan = np.isnan(sim_data).any()
        has_inf = np.isinf(sim_data).any()
        # Uncomment for debugging
        # print(f"min={np.min(data):.3e}, max={np.max(data):.3e}, has_nan={has_nan}, has_inf={has_inf}")
        if has_nan or has_inf:
            print(f"[Warning] Dataset contains NaN or Inf. Skipping it.")
            continue

        for c in range(num_channels):
            channel_data = sim_data[c].ravel()  # flatten (T, X, Y)
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

# ========================
# 1. Helper functions
# ========================

def get_sim_metadata(pde, n_sim, seed_list, nu=None, ic_hashes=None):
    """Return seed_val, nu_val, ic_val for the given PDE and simulation index."""
    if pde in ["Burgers", "KuramotoSivashinsky"]:
        nu_val = nu[n_sim % len(nu)]
        seed_val = seed_list[n_sim % len(seed_list)]
        ic_val = None
    elif pde == "KortewegDeVries":
        ic_val = ic_hashes[n_sim % len(ic_hashes)]
        seed_val = seed_list[n_sim % len(seed_list)]
        nu_val = None
    elif pde == "Kolmogorov":
        seed_val = seed_list[n_sim % len(seed_list)]
        nu_val = None
        ic_val = None
    else:
        seed_val = seed_list[n_sim % len(seed_list)]
        nu_val = None
        ic_val = None
    return seed_val, nu_val, ic_val


def create_animation(u_component, plots_path, seed_val, channel, ic="IC", nu_val=None,
                     x_extent=1.0, y_extent=1.0, duration_sec=5, cmap='viridis', vmin=None, vmax=None): # viridis for gray scott
    """
    Create and save a 2D animation for a single channel of a PDE trajectory.

    Parameters
    ----------
    u_component : ndarray
        Trajectory of shape (T, H, W)
    plots_path : str
        Folder where to save the animation
    seed_val : int
        Seed value
    channel : int
        Channel index
    ic : str
        Initial condition name
    nu_val : float or None
        Viscosity or parameter for PDE
    x_extent : float
        X-axis domain size
    y_extent : float
        Y-axis domain size
    duration_sec : float
        Duration of the final video in seconds
    cmap : str
        Colormap
    vmin, vmax : float
        Color scaling
    """
    nu_str = f"{nu_val:.3f}" if nu_val is not None else "NA"

    fig, ax = plt.subplots(figsize=(8, 8))
    init_frame = u_component[0]  # no transpose
    im = ax.imshow(init_frame, origin='lower', extent=(0, x_extent, 0, y_extent),
                   aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("u(x, t)")
    title = ax.set_title(f"{ic} - nu={nu_str} - seed {seed_val:03d} - channel {channel} - t=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Determine frames and fps, ensure full coverage
    n_total_frames = u_component.shape[0]
    fps = 20
    frames = np.linspace(0, n_total_frames - 1, num=int(duration_sec * fps), dtype=int)

    def update(t_idx):
        frame = u_component[t_idx]
        im.set_array(frame)
        title.set_text(f"{ic} - nu={nu_str} - seed {seed_val:03d} - channel {channel} - t={t_idx}")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)

    out_video = os.path.join(plots_path, f"seed_{seed_val:03d}_channel_{channel}_nu_{nu_str}.mp4")
    fallback_png = os.path.join(plots_path, f"seed_{seed_val:03d}_channel_{channel}_nu_{nu_str}_frame0.png")

    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(out_video, writer=writer)
        print(f"Saved animation: {out_video}")
    except Exception as e:
        print(f"[Error] saving animation: {e}")
        # fallback
        plt.imsave(fallback_png, init_frame, origin='lower')
        print(f"Saved fallback frame: {fallback_png}")
    finally:
        plt.close(fig)

# ========================
# 2. Main loop
# ========================

if plot_sim:
    random.seed(seed)
    selected_simulations = random.sample(range(len(all_trajectories)), plotted_sim)

    for n_sim in selected_simulations:
        u_xt = all_trajectories[n_sim]  # shape: (T, C, X, Y)
        num_channels = u_xt.shape[1]

        # Use metadata function
        seed_val, nu_val, ic_val = get_sim_metadata(pde, n_sim, seed_list, nu, ic_hashes)
        
        for c in range(num_channels):
            # Correct slicing: take all time steps for channel c
            u_component = u_xt[:, c, :, :]  # shape (T, X, Y)
            
            # Handle vmin/vmax safely
            vmin_val = mins[c % len(mins)]
            vmax_val = maxs[c % len(maxs)]
            
            create_animation(
                u_component,
                plots_path,
                seed_val,
                c,
                ic=ic,
                nu_val=nu_val,
                x_extent=x_domain_extent,
                y_extent=y_domain_extent,
                duration_sec=10,
                vmin=vmin_val,
                vmax=vmax_val
            )