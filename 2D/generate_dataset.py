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

num_spatial_dims = 2

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinsky' (ks), 'Burgers', 'Kolmogorov', 'KortewegDeVries' (KdV), 'GrayScott', 'FisherKPP'

ic : str
    Initial condition function. Proposed options:
    - For Burgers, KS and KdV: 'RandomTruncatedFourierSeries'
    - For Kolmogorov: 'SpectralFlow'
    - For FisherKPP: 'ClampedFourier'
    - For Gray-Scott: 'RandomGaussianBlobs'
    - For Swift-Hohenberg: 'RandomTruncatedFourierSeries', 'GaussianRandomField' or 'DifffusedNoise'.

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

feed_rate : float
    Feed rate (used only for Gray-Scott). Also called f in literature.

kill_rate : float
    Kill rate (used only for Gray-Scott). Also called k in literature.

reactivity : float
    Reactivity (used only for Fisher-KPP equation). Also called r in literature.

critical_wavenumber : float 
    Critical wavenumber for Swift-Hohenberg equation. Also called k.

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

pde = "GrayScott" # options: 'KuramotoSivashinsky', 'Burgers', 'Kolmogorov', 'KortewegDeVries', 'GrayScott', 'FisherKPP', 'SwiftHohenberg'
ic = "RandomTruncatedFourierSeries" # options: see description above
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

nu = [0, 0.00001, 0.01]  # For Burgers, KortewegDeVries, FisherKPP and SwiftHohenberg equations
# todo implement Re as nu 
Re = 250  # For Kolmogorov equation
reactivity = 0.6 # for FisherKPP and SwiftHohenberg
critical_wavenumber = 1.0 # critical wavenumber for SwiftHohenberg

# For Gray Scott:
feed_rate = 0.028
kill_rate = 0.056

simulations = 10
plotted_sim = 5
plot_sim = True
stats = True
seed = 42

# Define PDE-to-colormap mapping
pde_cmaps = {
    "GrayScott": "viridis",
    "FisherKPP": "vidiris",
    "SwiftHohenberg": "viridis",
    "Burgers": "RdBu",
    "KuramotoSivashinsky": "RdBu",
    "KortewegDeVries": "RdBu",
    "Kolmogorov": "inferno",
}
default_cmap = "viridis"


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
    feed_rate=feed_rate,
    kill_rate=kill_rate,
    reactivity=reactivity,
    critical_wavenumber=critical_wavenumber,
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
def get_group_name(pde, seed, nu_val=None, ic_val=None, Re=None, feed_rate=None, kill_rate=None, reactivity=None, critical_wavenumber=None):
    if pde == "Burgers":
        return f"nu_{nu_val:.3f}"
    elif pde == "KuramotoSivashinsky":  # with viscosity
        return f"nu_{nu_val:.3f}"
    elif pde == "KortewegDeVries":
        return f"ic_{ic_val}"
    elif pde == "Kolmogorov":
        return f"Re_{Re:.3f}" # add Re_val when introducing a list
    elif pde == "GrayScott":
        return f"feed_{feed_rate:.3f}_kill_{kill_rate:.3f}"
    elif pde == "FisherKPP":
        return f"nu_{nu_val:.3f}_reactivity_{reactivity:.3f}"
    elif pde == "SwiftHohenberg":
        return f"reactivity_{reactivity:.3f}_k_{critical_wavenumber:.3f}"
    else:
        return f"seed_{seed:03d}"

with h5py.File(data_path, "w") as h5file:
    idx = 0

    if pde == "Kolmogorov":
        group_name = get_group_name(pde, seed_list[0], Re=Re)
        grp = h5file.require_group(group_name)   # cleaner than manual if/else
        for seed in seed_list:
            u_xt = all_trajectories[idx]
            ds_name = f"velocity_seed{seed:03d}"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=u_xt)
            idx += 1

    elif pde == "KortewegDeVries":
        for seed in seed_list:
            ic_val = ic_hashes[idx] if idx < len(ic_hashes) else None
            group_name = get_group_name(pde, seed, ic_val=ic_val)
            grp = h5file.require_group(group_name)

            u_xt = all_trajectories[idx]
            ds_name = f"velocity_seed{seed:03d}"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=u_xt)
            idx += 1

    elif pde in ["Burgers", "KuramotoSivashinsky"]:
        for nu_val in nu:
            group_name = get_group_name(pde, 0, nu_val=nu_val)  # only use nu for group
            grp = h5file.require_group(group_name)
            for seed in seed_list:
                u_xt = all_trajectories[idx]
                ds_name = f"velocity_seed{seed:03d}"
                if ds_name in grp:
                    del grp[ds_name]
                grp.create_dataset(ds_name, data=u_xt)
                idx += 1

    elif pde == "GrayScott":
        group_name = get_group_name(pde, 0, feed_rate=feed_rate, kill_rate=kill_rate)
        grp = h5file.require_group(group_name)
        for seed in seed_list:
            u_xt = all_trajectories[idx]
            ds_name = f"state_seed{seed:03d}"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=u_xt)
            idx += 1

    elif pde == "FisherKPP":
        for nu_val in nu:
            group_name = get_group_name(pde, 0, nu_val=nu_val, reactivity=reactivity)
            grp = h5file.require_group(group_name)
            for seed in seed_list:
                u_xt = all_trajectories[idx]
                ds_name = f"state_seed{seed:03d}"
                if ds_name in grp:
                    del grp[ds_name]
                grp.create_dataset(ds_name, data=u_xt)
                idx += 1

    elif pde == "SwiftHohenberg":
        group_name = get_group_name(
            pde, 0,
            reactivity=reactivity,
            critical_wavenumber=critical_wavenumber,
        )
        grp = h5file.require_group(group_name)
        for seed in seed_list:
            u_xt = all_trajectories[idx]
            ds_name = f"velocity_seed{seed:03d}"
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=u_xt)
            idx += 1

    print(f"File created at {data_path}")

    # Optional: print structure
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {obj.name.split('/')[-1]}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {obj.name.split('/')[-1]} - Shape: {obj.shape}, Dtype: {obj.dtype}")

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
    if pde in ["Burgers", "KuramotoSivashinsky", "FisherKPP"]:
        nu_val = nu[n_sim % len(nu)]
        seed_val = seed_list[n_sim % len(seed_list)]
        ic_val = None
    elif pde == "KortewegDeVries":
        ic_val = ic_hashes[n_sim % len(ic_hashes)]
        seed_val = seed_list[n_sim % len(seed_list)]
        nu_val = None
    else:
        seed_val = seed_list[n_sim % len(seed_list)]
        nu_val = None
        ic_val = None
    return seed_val, nu_val, ic_val

def create_animation(
    u_component,
    plots_path,
    seed_val,
    channel,
    ic="IC",
    nu_val=None,
    x_extent=1.0,
    y_extent=1.0,
    dt_output=1.0,       # physical time step between saved frames
    mode="physical",     # "physical" or "fixed"
    duration_sec=None,   # used only if mode="fixed"
    fps=20,
    cmap="viridis",
    vmin=None,
    vmax=None,
):
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
        Diffusivity/parameter
    dt_output : float
        Physical time between stored frames
    mode : str
        "physical" → video matches physical simulation time
        "fixed" → video has fixed duration (duration_sec)
    duration_sec : float or None
        Fixed video length in seconds (only if mode="fixed")
    fps : int
        Frames per second
    """
    nu_str = f"{nu_val:.3f}" if nu_val is not None else "NA"

    fig, ax = plt.subplots(figsize=(8, 8))
    init_frame = u_component[0]
    im = ax.imshow(init_frame, origin="lower", extent=(0, x_extent, 0, y_extent),
                   aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("u(x, t)")
    title = ax.set_title(f"{ic} - nu={nu_str} - seed {seed_val:03d} - channel {channel} - t=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    n_total_frames = u_component.shape[0]
    sim_duration = (n_total_frames - 1) * dt_output

    if mode == "physical":
        frames = range(n_total_frames)
        print(f"[Video] Physical mode: sim_duration={sim_duration:.2f}, frames={n_total_frames}")
    elif mode == "fixed":
        if duration_sec is None:
            raise ValueError("duration_sec must be provided for mode='fixed'")
        n_video_frames = int(fps * duration_sec)
        frames = np.linspace(0, n_total_frames - 1, n_video_frames, dtype=int)
        print(f"[Video] Fixed mode: duration_sec={duration_sec:.2f}, frames={n_video_frames}")
    else:
        raise ValueError("mode must be 'physical' or 'fixed'")

    def update(t_idx):
        frame = u_component[t_idx]
        im.set_array(frame)
        t_phys = t_idx * dt_output
        title.set_text(
            f"{ic} - nu={nu_str} - seed {seed_val:03d} - channel {channel} - t={t_phys:.2f}"
        )
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)

    out_video = os.path.join(
        plots_path, f"seed_{seed_val:03d}_channel_{channel}_nu_{nu_str}.mp4"
    )

    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(out_video, writer=writer)
        print(f"Saved animation: {out_video}")
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
            
            cmap_val = pde_cmaps.get(pde, default_cmap)

            create_animation(
                u_component,
                plots_path,
                seed_val,
                c,
                ic=ic,
                nu_val=nu_val,
                x_extent=x_domain_extent,
                y_extent=y_domain_extent,
                dt_output=dt_save*save_freq,
                mode="fixed",      # or "fixed"
                duration_sec=10,      # only used if mode="fixed"
                fps=20,
                vmin=vmin_val,
                vmax=vmax_val,
                cmap=cmap_val,
            )