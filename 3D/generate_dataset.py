import os
import h5py
import exponax as ex
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import random
import matplotlib.animation as animation
import imageio.v3 as imageio
import argparse
import json
import time

from mpl_toolkits.mplot3d import Axes3D

from stepper import generate_dataset

num_spatial_dims = 3

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'KuramotoSivashinsky' (ks), 'Burgers'

ic : str
    Initial condition function. Proposed options:
    - For Burgers, KS and KdV: 'RandomTruncatedFourierSeries' is implemented but others are also possible.
    
bc : callable
    Boundary condition. (Unused As JAX computes spatial derivatives using the Fast Fourier Transform (FFT)
    which assumes periodicity, kept for consistency)

num_spatial_dims : int
    Number of spatial dimensions. 1 for this plot as we are using a 1D PDE

domain_extent : float
    Spatial domain extent (length of the domain in each spatial dimension)
    x_domain_extent, y_domain_extent and z_domain_extent must be the same for the solver to work.
    In reality y_domain_extent and z_domain_extent is unused.

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

pde = "KortewegDeVries" # options: 'KuramotoSivashinsky', 'Burgers', 'KortewegDeVries', 'GrayScott', 'FisherKPP', 'SwiftHohenberg'
ic = "RandomTruncatedFourierSeries" # options: see description above
bc = None

x_domain_extent = 32.0
y_domain_extent = 32.0 # In reality it is unused, it is the same as x_domain_extent
z_domain_extent = 32.0 # In reality it is unused, it is the same as x_domain_extent
num_points = 64
dt_save = 0.001 # integrator step
t_end = 1.0 # final physical time
save_freq = 10  # save every X integrator steps

''' What it implies:
Total steps: n_steps = t_end / dt_save
Output interval: dt_output = dt_save * save_freq
Total saved frames: n_saved = n_steps / save_freq + 1
'''

nu = [0, 0.01]  # For Burgers, KortewegDeVries equations

simulations = 2
plotted_sim = 1
plot_sim = False
stats = True
seed = 42

# Define PDE-to-colormap mapping. Change here if wanted
pde_cmaps = {
    "Burgers": "RdBu",
    "KuramotoSivashinsky": "RdBu",
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
    """Load config file if provided, otherwise return globals (manual defaults)."""
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
    elif pde == "KuramotoSivashinsky":
        return f"nu{nu_val:.3f}"
    elif pde == "KortewegDeVries":
        return f"ic{ic_val}"
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

    # =========================================
    # GENERATE DATASET
    # =========================================

    # Run dataset generation
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
        save_freq=cfg["save_freq"],
        seed_list=seed_list,
        seed=cfg["seed"],
    )
    all_trajectories = np.stack(all_trajectories)
    print("Original shape:", all_trajectories.shape)

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # =========================================
    # SAVE DATASET
    # =========================================

    #  Directory dependant on the pde and initial condition
    base_dir = os.path.join(cfg["pde"], cfg["ic"])
    os.makedirs(base_dir, exist_ok=True)
    file_name = "dataset.h5"
    data_path = os.path.join(base_dir, file_name)
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

        elif cfg["pde"] in ["Burgers", "KuramotoSivashinsky", "NavierStokesVorticity", "AllenCahn"]:
            for nu_val in cfg["nu"]:
                group_name = get_group_name(cfg, 0, nu_val=nu_val)
                grp = h5file.require_group(group_name)
                for seed in seed_list:
                    ds_name = f"state_seed{seed:03d}" if cfg["pde"] == "FisherKPP" else f"velocity_seed{seed:03d}"
                    if ds_name in grp: del grp[ds_name]
                    grp.create_dataset(ds_name, data=all_trajectories[idx])
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
    if cfg["stats"]:
        # Detect number of channels from first loaded dataset
        data = all_trajectories
        print("Original shape:", data.shape)
        num_channels = data.shape[1] # Channels are in the second position
        print("Detected num_channels:", num_channels)

        # Initialize accumulators (outside the loop)
        means = np.zeros(num_channels, dtype=np.float64)
        M2s   = np.zeros(num_channels, dtype=np.float64)
        mins  = np.full(num_channels, np.inf, dtype=np.float64)
        maxs  = np.full(num_channels, -np.inf, dtype=np.float64)
        counts = np.zeros(num_channels, dtype=np.int64)

        # Iterate over simulations
        for sim_idx in range(data.shape[0]):
            sim_data = data[sim_idx]  # shape: (T, C, X)

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
    if cfg["pde"] in ["Burgers", "KuramotoSivashinsky"]:
        for nu_val in cfg["nu"]:
            for s in seed_list:
                sim_names.append(get_group_name(cfg, s, nu_val=nu_val))
    elif cfg["pde"] == "KuramotoSivashinskyConservative":
        for s in seed_list:
            sim_names.append(get_group_name(cfg, s))
    elif cfg["pde"] == "KortewegDeVries":
        for ic_val in ic_hashes:
            for s in seed_list:
                sim_names.append(get_group_name(cfg, s, ic_val=ic_val))
    else:
        for s in seed_list:
            sim_names.append(get_group_name(cfg, s))

    if cfg["plot_sim"]:
        random.seed(cfg["seed"])
        selected_simulations = random.sample(range(len(sim_names)), cfg["plotted_sim"])

        for n_sim in selected_simulations:
            sim_name = sim_names[n_sim]

            # pull the trajectory for this sim
            u_xt = all_trajectories[n_sim]  # shape: (C, T, X, Y, Z)
            num_channels = u_xt.shape[0]

            for c in range(num_channels):
                u_component = u_xt[c]  # shape: (T, X, Y, Z)
                T, X, Y, Z = u_component.shape

                # Take central slice along Z
                center_z = Z // 2
                slice_sequence = u_component[:, :, :, center_z]  # (T, X, Y)

                extent = (0, cfg["x_domain_extent"], 0, cfg["x_domain_extent"])

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(slice_sequence[0].T, cmap='RdBu', origin='lower',
                            extent=extent, vmin=-15, vmax=15, aspect='auto')

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("u(x, y, z_center)")
                title = ax.set_title(f"{cfg['ic']} - {sim_name} - channel {c} - t=0")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

                def update(t_idx):
                    im.set_array(slice_sequence[t_idx].T)
                    title.set_text(f"{cfg['ic']} - {sim_name} - channel {c} - t={t_idx}")
                    return im, title

                skip = 2
                frames = range(0, T, skip)
                ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)

                video_path = os.path.join(plots_path, f"{sim_name}_channel_{c}.mp4")
                ani.save(video_path, writer='ffmpeg', fps=10)
                print(f"Animation saved at {video_path}")

                plt.close(fig)
