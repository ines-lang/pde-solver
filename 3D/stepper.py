import jax
import jax.numpy as jnp
import numpy as np
from typing import List
import sys
import hashlib

import exponax as ex
from exponax.stepper import KuramotoSivashinsky
from exponax.ic import *


def generate_dataset(pde: str,
                      ic: str, 
                      bc: callable, 
                      num_spatial_dims: int,
                      x_domain_extent: float,
                      num_points: int,
                      dt_save: float, 
                      t_end: float,
                      save_freq: int, 
                      nu: float,
                      feed_rate: float,
                      kill_rate: float,
                      reactivity: float,
                      critical_wavenumber: float,
                      seed_list:List,
                      seed: int):
    
    all_trajectories = []
    ic_hashes = []
    trajectory_nus = []

    if pde == "KuramotoSivashinsky":

        for nu_val in nu:
            ks_class = getattr(ex.stepper, pde)
            ks_stepper = ks_class(
                num_spatial_dims=num_spatial_dims, 
                domain_extent=x_domain_extent, 
                num_points=num_points, 
                dt=dt_save,
                second_order_scale = 1.0 - float(nu_val)
                )
            
            for seed in seed_list:
                key = jax.random.PRNGKey(seed)
                ic_class = getattr(ex.ic, ic)
                ic_instance = ic_class(num_spatial_dims=num_spatial_dims, cutoff=5)
                # --- Generate 3 ICs per seed ---
                keys = jax.random.split(key, 3)
                u_list = [ic_instance(num_points=num_points, key=k) for k in keys]
                # Remove leading channel dim if present
                for i in range(3):
                    if u_list[i].ndim == 3 and u_list[i].shape[0] == 1:
                        u_list[i] = u_list[i][0]

                u_0 = jnp.stack(u_list)  # shape: (3, X, Y, Z)
                ic_hashes.append(hash(u_0.tobytes()))
                
                # Rollout
                trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)

                trajectory_nus.append(nu_val)
        
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X, Z)
        print("Shape after stacking (Should be (N, T_sampled, C, X, Y, Z)):", all_trajectories.shape)
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "Burgers":

        burgers_class = getattr(ex.stepper, pde)

        for nu_val in nu:
            burgers_stepper = burgers_class(
                num_spatial_dims=num_spatial_dims, 
                domain_extent=x_domain_extent, 
                num_points=num_points, 
                dt=dt_save,
                diffusivity=nu_val  # Use nu_val for viscosity
                )
    
            for seed in seed_list:
                key = jax.random.PRNGKey(seed)
                ic_class = getattr(ex.ic, ic)
                common_kwargs = {
                    "num_spatial_dims": num_spatial_dims,
                }
                # Add class-specific arguments if applicable
                if ic == "RandomTruncatedFourierSeries":
                    common_kwargs["cutoff"] = 5
                try:
                    ic_instance = ic_class(**common_kwargs)
                    # --- Generate 3 ICs per seed ---
                    keys = jax.random.split(key, 3)
                    u_list = [ic_instance(num_points=num_points, key=k) for k in keys]
                    for i in range(3):
                        if u_list[i].ndim == 3 and u_list[i].shape[0] == 1:
                            u_list[i] = u_list[i][0]

                    u_0 = jnp.stack(u_list)  # shape: (3, X, Y, Z)
                    ic_hashes.append(hash(u_0.tobytes()))
                    
                    # Rollout
                    trajectories = np.array(ex.rollout(burgers_stepper, t_end, include_init=True)(u_0)) # added numpy to move rollout to CPU immediately
                    sampled_traj = trajectories[::save_freq]
                    # Debugging block
                    # print(f"seed {seed} → traj shape: {sampled_traj.shape}")
                    # print(f"min: {np.nanmin(sampled_traj):.3f}, max: {np.nanmax(sampled_traj):.3f}")
                    
                    # Check for NaNs/Infs
                    if np.isnan(sampled_traj).any() or np.isinf(sampled_traj).any():
                        raise ValueError("Trajectory contains NaNs or Infs")
                    all_trajectories.append(sampled_traj)
                    trajectory_nus.append(nu_val)  # save the nu
                except Exception as e:
                    print(f"Failed to generate trajectory for seed {seed}: {e}")
                    continue  # Skip this seed and move on
                # print(f"Generated {len(all_trajectories)} valid trajectories out of {len(seed_list)} seeds.")

            if len(all_trajectories) == 0:
                raise RuntimeError("No valid trajectories were generated.")
            
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X, Y, Z)
        print("Shape after stacking (Should be (N, T_sampled, C, X, Y, Z)):", all_trajectories.shape)

        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "KortewegDeVries":

        kdv_class = getattr(ex.stepper, pde)
        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save, # integrator dt
            single_channel=True,  # make it scalar: only 1 channel, no 1 channel per IC: ensures it expects (1, X, Y, Z)
            conservative=True,          # <— helps stability
            hyper_diffusivity=0.1,     # try 0.05 → 0.1 if still unstable
            order=2,                    # can try 3 or 4 for accuracy
        )       
        
        def ic_hash(u_0, length=8):
            arr = np.asarray(jax.device_get(u_0))  # host, contiguous
            return hashlib.sha256(arr.tobytes()).hexdigest()[:length]
        
        ic_class = getattr(ex.ic, ic)
        common_kwargs = {"num_spatial_dims": num_spatial_dims}
        if ic == "RandomTruncatedFourierSeries":
                common_kwargs["cutoff"] = 5
        ic_instance = ic_class(**common_kwargs)

        # Rollout function expects NUMBER OF STEPS, not time
        n_steps = int(np.ceil(t_end / dt_save))
        rollout_fn = ex.rollout(kdv_stepper, n_steps, include_init=True)

        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, 3)

            # Generate 3 ICs for this seed (each as scalar field with channel axis)
            u_list = []
            for k in keys:
                u = ic_instance(num_points=num_points, key=k)  # (X,Y,Z) or (1,X,Y,Z)
                if u.ndim == 4 and u.shape[0] == 1:
                    # already has (1, X, Y, Z)
                    u_c = u
                elif u.ndim == 3:
                    # add channel axis -> (1, X, Y, Z)
                    u_c = u[jnp.newaxis, ...]
                else:
                    raise ValueError(f"IC has unexpected shape {u.shape}; expected (X,Y,Z) or (1,X,Y,Z)")
                u_list.append(u_c)
            
            # Hash the batch of ICs for this seed (stack only for hashing)
            u_stack_for_hash = jnp.stack(u_list, axis=0)  # (3, 1, X, Y, Z)
            ic_hashes.append(ic_hash(u_stack_for_hash))

            # Roll out each IC separately and subsample in time
            for i, u0_single in enumerate(u_list):
                try: 
                    # ---- Roll out full trajectory on device ----
                    traj = rollout_fn(u0_single)                  # (T_int, 1, X, Y, Z) or (T_int+1, ...)

                    # ---- Build explicit save indices ----
                    # Include step 0; include last step even if not a multiple of save_freq
                    T_int = traj.shape[0]                         # includes init if include_init=True
                    save_idx = jnp.arange(0, T_int, save_freq)
                    if save_idx[-1] != T_int - 1:
                        save_idx = jnp.concatenate([save_idx, jnp.array([T_int - 1])])

                    # ---- Subsample in time ----
                    sampled = traj[save_idx]                      # (T_saved, 1, X, Y, Z)

                    # ---- Transfer to host and append ----
                    sampled_np = np.asarray(jax.device_get(sampled))

                    # Skip if any NaN/Inf (like your Burgers code)
                    if not np.isfinite(sampled_np).all():
                        raise ValueError("Trajectory contains NaNs or Infs")

                    # Optional debug
                    # print(f"seed {seed} ic {i} -> sampled shape {sampled_np.shape}, "
                    #       f"min={sampled_np.min():.3g}, max={sampled_np.max():.3g}")

                    # --- Sanity checks (keep while debugging) ---
                    assert sampled_np.ndim == 5, f"Expected 5D (T,1,X,Y,Z), got {sampled_np.shape}"
                    assert sampled_np.shape[1] == 1, f"Channel axis is not 1: {sampled_np.shape}"

                    all_trajectories.append(sampled_np)
                    trajectory_nus.append(f"seed{seed:03d}_ic{i}")
                
                except Exception as e:
                    print(f"[Skip] seed {seed} ic {i}: {e}")
                    continue

        # Final stacking: (N_runs, T_sampled, 1, X, Y, Z)
        if len(all_trajectories) == 0:
            print("[Warning] No valid trajectories collected.")
        else:
            all_trajectories = np.stack(all_trajectories, axis=0)
            print("Stacked shape (N_runs, T_sampled, 1, X, Y, Z):", all_trajectories.shape)

        return all_trajectories, ic_hashes, trajectory_nus
    
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")