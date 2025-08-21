import os
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

pde = "KuramotoSivashinsky" # options: 'KuramotoSivashinsky', 'Burgers', 'Kolmogorov', 'KortewegDeVries'
num_spatial_dims = 2
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries', 'RandomSpectralVorticityField'
bc = None

x_domain_extent = 64.0
y_domain_extent = 64.0 
num_points = 100
dt_save = 0.0001
t_end = 500.0 
save_freq = 1

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

all_trajectories = jnp.stack(all_trajectories)
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
        return f"nu_{nu_val:.3f}_seed_{seed:03d}"
    elif pde == "KuramotoSivashinskyConservative":
        return f"seed_{seed:03d}"  # no viscosity
    elif pde == "KuramotoSivashinsky":  # with viscosity
        return f"nu_{nu_val:.3f}_seed_{seed:03d}"
    elif pde == "KortewegDeVries":
        return f"ic_{ic_val}_seed_{seed:03d}"
    elif pde == "Kolmogorov":
        return f"Re_{Re:.3f}_seed_{seed:03d}" # check if :.3f
    else:
        return f"seed_{seed:03d}"

# Save to HDF5
with h5py.File(data_path, "w") as h5file:
    idx = 0

    if pde in ["Burgers", "KuramotoSivashinsky"]:
        for nu_val in nu:   # only iterate nu if PDE actually has it
            for sim_idx, seed in enumerate(seed_list):
                group_name = get_group_name(pde, seed, nu_val=nu_val)
                grp = h5file.create_group(group_name)
                u_xt = all_trajectories[idx]
                grp.create_dataset(f'velocity_{idx:03d}', data=u_xt)
                idx += 1

    elif pde == "KuramotoSivashinskyConservative":
        for sim_idx, seed in enumerate(seed_list):
            group_name = get_group_name(pde, seed)
            grp = h5file.create_group(group_name)
            u_xt = all_trajectories[idx]
            grp.create_dataset(f'velocity_{idx:03d}', data=u_xt)
            idx += 1

    elif pde == "KortewegDeVries":
        for ic_val in ic_hashes:
            for sim_idx, seed in enumerate(seed_list):
                group_name = get_group_name(pde, seed, ic_val=ic_val)
                grp = h5file.create_group(group_name)
                u_xt = all_trajectories[idx]
                grp.create_dataset(f'velocity_{idx:03d}', data=u_xt)
                idx += 1

    else:
        for sim_idx, seed in enumerate(seed_list):
            group_name = get_group_name(pde, seed)
            grp = h5file.create_group(group_name)
            u_xt = all_trajectories[idx]
            grp.create_dataset(f'velocity_{idx:03d}', data=u_xt)
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
        u_xt = all_trajectories[n_sim]  # expected shape: (C, T, X) or (C, T, X, Y)

        for c in range(num_channels):  # loop over each channel
            u_component = u_xt[c]  # shape: (T, H, W)

            extent = (
                0, x_domain_extent,
                0, y_domain_extent)

            fig, ax = plt.subplots(figsize=(10, 10))
            init_frame = u_component[0].T  # transpose so indexing matches extent (Y,X) display
            im = ax.imshow(
                    init_frame,
                    origin='lower',
                    extent=extent,
                    aspect='auto',
                    cmap='RdBu',
                    vmin=mins[c],
                    vmax=maxs[c],
            )

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("u(x, t)")

            # create title object so update can change it
            title = ax.set_title(f"{ic} - nu={nu_val:.3f} - seed {seed_val:03d} - channel {c} - t=0")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # Use closure factory to capture current variables (avoids late-binding bug)
            def make_update(u_comp, im_obj, title_obj, seed_v, ch):
                def update(t_idx):
                    frame = u_comp[t_idx].T
                    im_obj.set_array(frame)
                    title_obj.set_text(f"{ic} - nu={nu_val:.3f} - seed {seed_v:03d} - channel {ch} - t={t_idx}")
                    return (im_obj, title_obj)
                return update

            update_fn = make_update(u_component, im, title, seed_val, c)

            # choose frames with skipping if sequence is long
            n_frames = u_component.shape[0]
            skip = 1 if n_frames <= 400 else max(1, n_frames // 400)
            frames = range(0, n_frames, skip)

            ani = animation.FuncAnimation(fig, update_fn, frames=frames, blit=False)

            # save animation (use FFMpegWriter). Ensure ffmpeg is installed.
            out_video = os.path.join(plots_path, f"nu_{nu_val:.3f}_seed_{seed_val:03d}_channel_{c}.mp4")
            try:
                writer = animation.FFMpegWriter(fps=10)
                ani.save(out_video, writer=writer)
                print(f"Saved animation: {out_video}")
            except Exception as e:
                print(f"[Error] saving animation for {sim_name} channel {c}: {e}")
                # fallback: save first frame as png
                fallback_png = os.path.join(plots_path, f"nu_{nu_val:.3f}_seed_{seed_val:03d}_channel_{c}_frame0.png")
                plt.imsave(fallback_png, init_frame, origin='lower')
                print(f"Saved fallback frame: {fallback_png}")

            plt.close(fig)