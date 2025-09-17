import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import random
import numpy as np
import argparse
import json
import time

from stepper import generate_dataset

num_spatial_dims = 1 

# =========================================
# USER INPUTS (PDE parameters). Default manual values (will be used if no config is passed)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinskyConservative', 'KuramotoSivashinsky', 'Burgers', 'KortewegDeVries'

ic : str
    Initial condition function. Options:
    - For Burgers, KS and KdV: 'RandomTruncatedFourierSeries' works good but others are also possible.
    - For Gray-Scott: 'RandomGaussianBlobs' is implemented and shows good results.
    - For FisherKPP: 'ClampedFourier' is implemented and shows good results.
    - For Swift-Hohenberg: 'RandomTruncatedFourierSeries', 'GaussianRandomField' and 'DifffusedNoise' are implemented.

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

pde = "FisherKPP" # options: 'KuramotoSivashinskyConservative', 'KuramotoSivashinsky' (adds viscosity with nu), 'Burgers', 'KortewegDeVries', 'GrayScott', 'FisherKPP', 'SwiftHohenberg'
ic = "RandomTruncatedFourierSeries" # options: see description above
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

nu = [0, 0.00001, 0.01]           # for Burgers, KdV, FisherKPP, SwiftHohenberg
reactivity = 10                   # for FisherKPP, SwiftHohenberg
critical_wavenumber = 1.0         # for SwiftHohenberg
feed_rate = 0.028                 # for GrayScott
kill_rate = 0.056                 # for GrayScott

simulations = 2
plotted_sim = 1
plot_sim = True
stats = True
seed = 42 

# Define PDE-to-colormap mapping
pde_cmaps = {
    "GrayScott": "viridis",
    "FisherKPP": "viridis",
    "SwiftHohenberg": "viridis",
    "Burgers": "RdBu",
    "KuramotoSivashinsky": "RdBu",
    "KuramotoSivashinskyConservative": "RdBu",
    "KortewegDeVries": "RdBu",
}
default_cmap = "viridis"

# =======================================================
# Argument parser
# =======================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate PDE simulation datasets (1D).")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    return parser.parse_args()


# =======================================================
# Config loader
# =======================================================
def load_config(args):
    """Load config file if provided, otherwise return defaults."""
    global_vars = dict(
        num_spatial_dims=num_spatial_dims,
        pde=pde,
        ic=ic,
        bc=bc,
        x_domain_extent=x_domain_extent,
        num_points=num_points,
        dt_save=dt_save,
        t_end=t_end,
        save_freq=save_freq,
        nu=nu,
        reactivity=reactivity,
        critical_wavenumber=critical_wavenumber,
        feed_rate=feed_rate,
        kill_rate=kill_rate,
        simulations=simulations,
        plotted_sim=plotted_sim,
        plot_sim=plot_sim,
        stats=stats,
        seed=seed,
    )

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        if "generate_dataset_py" in config:
            config = config["generate_dataset_py"]

        for k, v in config.items():
            if k in global_vars:
                if isinstance(global_vars[k], bool):
                    if isinstance(v, str):
                        v = v.lower() == "true"
                    else:
                        v = bool(v)
                elif isinstance(global_vars[k], int):
                    v = int(v)
                elif isinstance(global_vars[k], float):
                    v = float(v)
                elif isinstance(global_vars[k], list):
                    v = [float(x) for x in v]
                global_vars[k] = v

    return global_vars

# =======================================================
# Group naming
# =======================================================
def get_group_name(cfg, seed, ic_val=None, nu_val=None):
    pde = cfg["pde"]

    if pde == "Burgers":
        return f"nu{nu_val:.3f}"
    elif pde == "KuramotoSivashinskyConservative":
        return f"seed{seed:03d}"
    elif pde == "KuramotoSivashinsky":
        return f"nu{nu_val:.3f}"
    elif pde == "KortewegDeVries":
        return f"ic{ic_val}"
    elif pde == "GrayScott":
        return f"feed{cfg['feed_rate']:.3f}_kill{cfg['kill_rate']:.3f}"
    elif pde == "FisherKPP":
        return f"nu{nu_val:.3f}_reactivity{cfg['reactivity']:.3f}"
    elif pde == "SwiftHohenberg":
        return f"reactivity{cfg['reactivity']:.3f}_k{cfg['critical_wavenumber']:.3f}"
    else:
        return f"seed{seed:03d}"

# =======================================================
# Main
# =======================================================
if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    print("Running with configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    seed_list = list(range(cfg["simulations"]))

    start_time = time.time()
    all_trajectories, ic_hashes, trajectory_nus = generate_dataset(
        pde=cfg["pde"],
        num_spatial_dims=cfg["num_spatial_dims"],
        ic=cfg["ic"],
        bc=cfg["bc"],
        x_domain_extent=cfg["x_domain_extent"],
        num_points=cfg["num_points"],
        dt_save=cfg["dt_save"],
        t_end=cfg["t_end"],
        nu=cfg["nu"],
        feed_rate=cfg["feed_rate"],
        kill_rate=cfg["kill_rate"],
        reactivity=cfg["reactivity"],
        critical_wavenumber=cfg["critical_wavenumber"],
        save_freq=cfg["save_freq"],
        seed_list=seed_list,
        seed=cfg["seed"],
    )
    all_trajectories = np.stack(all_trajectories)
    print("Original shape:", all_trajectories.shape)

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # ===================================================
    # SAVE DATASET
    # ===================================================
    base_dir = os.path.join(cfg["pde"], cfg["ic"])
    os.makedirs(base_dir, exist_ok=True)
    data_path = os.path.join(base_dir, "dataset.h5")
    plots_path = os.path.join(base_dir, "plots")
    os.makedirs(plots_path, exist_ok=True)

    with h5py.File(data_path, "w") as h5file:
        idx = 0
        if cfg["pde"] == "KortewegDeVries":
            for seed in seed_list:
                ic_val = ic_hashes[idx] if idx < len(ic_hashes) else None
                group_name = get_group_name(cfg, seed, ic_val=ic_val)
                grp = h5file.require_group(group_name)
                ds_name = f"velocity_seed{seed:03d}"
                if ds_name in grp: del grp[ds_name]
                grp.create_dataset(ds_name, data=all_trajectories[idx])
                idx += 1

        elif cfg["pde"] in ["Burgers", "KuramotoSivashinsky", "FisherKPP"]:
            for nu_val in cfg["nu"]:
                group_name = get_group_name(cfg, 0, nu_val=nu_val)
                grp = h5file.require_group(group_name)
                for seed in seed_list:
                    ds_name = f"state_seed{seed:03d}" if cfg["pde"] == "FisherKPP" else f"velocity_seed{seed:03d}"
                    if ds_name in grp: del grp[ds_name]
                    grp.create_dataset(ds_name, data=all_trajectories[idx])
                    idx += 1

        elif cfg["pde"] == "GrayScott":
            group_name = get_group_name(cfg, 0)
            grp = h5file.require_group(group_name)
            for seed in seed_list:
                ds_name = f"state_seed{seed:03d}"
                if ds_name in grp: del grp[ds_name]
                grp.create_dataset(ds_name, data=all_trajectories[idx])
                idx += 1

        elif cfg["pde"] == "SwiftHohenberg":
            group_name = get_group_name(cfg, 0)
            grp = h5file.require_group(group_name)
            for seed in seed_list:
                ds_name = f"velocity_seed{seed:03d}"
                if ds_name in grp: del grp[ds_name]
                grp.create_dataset(ds_name, data=all_trajectories[idx])
                idx += 1

        print(f"File created at {data_path}")

        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {obj.name.split('/')[-1]}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {obj.name.split('/')[-1]} - Shape: {obj.shape}, Dtype: {obj.dtype}")
        h5file.visititems(print_structure)

    # ===================================================
    # STATS
    # ===================================================
    if cfg["stats"]:
        data = all_trajectories
        print("Original shape:", data.shape)
        num_channels = data.shape[1]
        print("Detected num_channels:", num_channels)

        means = np.zeros(num_channels, dtype=np.float64)
        M2s   = np.zeros(num_channels, dtype=np.float64)
        mins  = np.full(num_channels, np.inf, dtype=np.float64)
        maxs  = np.full(num_channels, -np.inf, dtype=np.float64)
        counts = np.zeros(num_channels, dtype=np.int64)

        for sim_idx in range(data.shape[0]):
            sim_data = data[sim_idx]  # (C, T, X)
            if np.isnan(sim_data).any() or np.isinf(sim_data).any():
                print("[Warning] NaN/Inf detected, skipping simulation.")
                continue
            for c in range(num_channels):
                channel_data = sim_data[c, :, :].ravel()
                n = channel_data.size
                mins[c] = min(mins[c], np.min(channel_data))
                maxs[c] = max(maxs[c], np.max(channel_data))
                counts[c] += n
                delta = channel_data - means[c]
                means[c] += np.sum(delta) / counts[c]
                delta2 = channel_data - means[c]
                M2s[c] += np.sum(delta * delta2)

        stds = np.sqrt(M2s / counts)
        print("Dataset Global Statistics (Per Channel)")
        for c in range(num_channels):
            print(f"Channel {c}: Mean={means[c]:.6f}, Std={stds[c]:.6f}, "
                  f"Min={mins[c]:.6f}, Max={maxs[c]:.6f}")

    # ===================================================
    # PLOT
    # ===================================================
    sim_names = []
    if cfg["pde"] in ["Burgers", "KuramotoSivashinsky", "FisherKPP"]:
        for nu_val in cfg["nu"]:
            for s in seed_list:
                sim_names.append(get_group_name(cfg, s, nu_val=nu_val))
    elif cfg["pde"] in ["KuramotoSivashinskyConservative"]:
        for s in seed_list:
            sim_names.append(get_group_name(cfg, s))
    elif cfg["pde"] == "KortewegDeVries":
        for ic_val in ic_hashes:
            for s in seed_list:
                sim_names.append(get_group_name(cfg, s, ic_val=ic_val))
    elif cfg["pde"] == "GrayScott":
        for s in seed_list:
            sim_names.append(get_group_name(cfg, s))
    elif cfg["pde"] == "SwiftHohenberg":
        for s in seed_list:
            sim_names.append(get_group_name(cfg, s))
    else:
        for s in seed_list:
            sim_names.append(get_group_name(cfg, s))

    if cfg["plot_sim"]:
        random.seed(cfg["seed"])
        selected_simulations = random.sample(range(all_trajectories.shape[0]), cfg["plotted_sim"])
        for n_sim in selected_simulations:
            # make sure sim_name list matches all_trajectories length, otherwise fallback
            sim_name = sim_names[n_sim] if n_sim < len(sim_names) else f"sim_{n_sim:03d}"
            for c in range(num_channels):
                cmap_val = pde_cmaps.get(cfg["pde"], default_cmap)
                plt.imshow(all_trajectories[n_sim, c, :, :].T,  # (X, T)
                           aspect='auto',
                           cmap=cmap_val,
                           vmin=mins[c], vmax=maxs[c],
                           origin="lower")
                plt.xlabel("Space")
                plt.ylabel("Time")
                plt.title(f"{cfg['ic']} - {sim_name} - channel {c}")
                plt.savefig(os.path.join(plots_path, f"{sim_name}_channel_{c}.png"))
                plt.close()
                print(f"Plot saved: {os.path.join(plots_path, f'{sim_name}_channel_{c}.png')}")