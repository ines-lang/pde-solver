import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import random
import numpy as np

from stepper import generate_dataset

num_spatial_dims = 1 

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinskyConservative', 'KuramotoSivashinsky', 'Burgers', 'KortewegDeVries'

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

pde = "FisherKPP" # options: 'KuramotoSivashinskyConservative', 'KuramotoSivashinsky' (adds viscosity with nu), 'Burgers', 'KortewegDeVries'
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries', 'GaussianRandomField'
bc = None

x_domain_extent = 1.0
num_points = 2048
dt_save = 0.005
t_end = 50.0
save_freq = 1

''' What it implies:
Total steps: n_steps = t_end / dt_save
Output interval: dt_output = dt_save * save_freq
Total saved frames: n_saved = n_steps / save_freq + 1
'''

nu = [0, 0.00001, 0.01]  # For Burgers, KortewegDeVries, FisherKPP and SwiftHohenberg equations
reactivity = 10 # for FisherKPP and SwiftHohenberg
critical_wavenumber = 1.0 # critical wavenumber for SwiftHohenberg

# For Gray Scott:
feed_rate = 0.028
kill_rate = 0.056

simulations = 2
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
def get_group_name(pde, seed, nu_val=None, ic_val=None, feed_rate=None, kill_rate=None, reactivity=None, critical_wavenumber=None):
    if pde == "Burgers":
        return f"nu_{nu_val:.3f}"
    elif pde == "KuramotoSivashinskyConservative":
        return f"seed_{seed:03d}"  # no viscosity
    elif pde == "KuramotoSivashinsky":  # with viscosity
        return f"nu_{nu_val:.3f}"
    elif pde == "KortewegDeVries":
        return f"ic_{ic_val}"
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
   
    if pde == "KortewegDeVries":
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
    num_channels = data.shape[1]
    print("Detected num_channels:", num_channels)

    # Initialize accumulators (outside the loop)
    means = np.zeros(num_channels, dtype=np.float64)
    M2s   = np.zeros(num_channels, dtype=np.float64)
    mins  = np.full(num_channels, np.inf, dtype=np.float64)
    maxs  = np.full(num_channels, -np.inf, dtype=np.float64)
    counts = np.zeros(num_channels, dtype=np.int64)

    # Iterate over simulations
    for sim_idx in range(data.shape[0]):
        sim_data = data[sim_idx]  # shape: (C, T, X)

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
elif pde == "GrayScott":
    for s in seed_list:
        sim_names.append(get_group_name(pde, s, feed_rate=feed_rate, kill_rate=kill_rate))
elif pde == "FisherKPP":
    for nu_val in nu:
        for s in seed_list:
            sim_names.append(get_group_name(pde, s, nu_val=nu_val, reactivity=reactivity))
elif pde == "SwiftHohenberg":
    for s in seed_list:
        sim_names.append(get_group_name(pde, s, reactivity=reactivity, critical_wavenumber=critical_wavenumber))
else:
    for s in seed_list:
        sim_names.append(get_group_name(pde, s))

if plot_sim:
    random.seed(seed)
    selected_simulations = random.sample(range(len(sim_names)), plotted_sim)
    
    for n_sim in selected_simulations:
        sim_name = sim_names[n_sim]

        for c in range(num_channels):
            plt.imshow(
                all_trajectories[n_sim, c, :, :].T,
                aspect='auto',
                cmap='viridis',
                vmin=mins[c],
                vmax=maxs[c],
                origin="lower"
            )
            plt.xlabel("Time")
            plt.ylabel("Space")
            plt.title(f"{ic} - {sim_name} - channel {c}")
            plt.savefig(os.path.join(
                plots_path, f"{sim_name}_channel_{c}.png"))
            plt.show()
            plt.close()