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

                for u in u_list:
                    # Ensure (1, X, Y, Z)
                    if u.ndim == 3:
                        u = u[jnp.newaxis, ...]
                    elif u.ndim == 4 and u.shape[0] == 1:
                        pass
                    else:
                        raise ValueError(f"Unexpected IC shape {u.shape}")

                    ic_hashes.append(hash(np.asarray(jax.device_get(u)).tobytes()))

                    # Rollout one IC
                    traj = ex.rollout(ks_stepper, t_end, include_init=True)(u)  # (T, 1, X, Y, Z)
                    sampled_traj = traj[::save_freq]
                    all_trajectories.append(np.asarray(jax.device_get(sampled_traj)))

                    trajectory_nus.append(nu_val)
        
        if len(all_trajectories) == 0:
            raise RuntimeError("No valid KS trajectories generated.")

        # (N, T, C, X, Y, Z) → (N, C, T, X, Y, Z)
        all_trajectories = np.stack(all_trajectories)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4, 5))
        print("Shape after stacking:", all_trajectories.shape)

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

            ic_class = getattr(ex.ic, ic)
            common_kwargs = {"num_spatial_dims": num_spatial_dims}
            if ic == "RandomTruncatedFourierSeries":
                common_kwargs["cutoff"] = 5
            ic_instance = ic_class(**common_kwargs)
            
            for seed in seed_list:
                key = jax.random.PRNGKey(seed)
                keys = jax.random.split(key, 3) # 3 ICs per seed
                
                for k in keys:
                    try:
                        # --- Build vector IC with num_spatial_dims channels ---
                        comp_keys = jax.random.split(k, num_spatial_dims)
                        comps = []
                        for ck in comp_keys:
                            u_scalar = ic_instance(num_points=num_points, key=ck)  # (X,Y,Z) or (1,X,Y,Z)
                            if u_scalar.ndim == 4 and u_scalar.shape[0] == 1:
                                u_scalar = u_scalar[0]      # strip leading channel
                            elif u_scalar.ndim == 3:
                                u_scalar = u_scalar
                            else:
                                raise ValueError(f"Unexpected IC shape {u_scalar.shape}")
                            comps.append(u_scalar)

                        u0 = jnp.stack(comps, axis=0)  # (C=num_spatial_dims, X, Y, Z)

                        # Hash IC
                        ic_hashes.append(hash(np.asarray(jax.device_get(u0)).tobytes()))

                        # --- Rollout ---
                        traj = ex.rollout(burgers_stepper, t_end, include_init=True)(u0)  # (T, C, X, Y, Z)
                        traj = np.asarray(jax.device_get(traj))

                        # Subsample in time
                        sampled_traj = traj[::save_freq]  # (T_s, C, X, Y, Z)

                        if not np.isfinite(sampled_traj).all():
                            raise ValueError("Trajectory contains NaNs or Infs")

                        all_trajectories.append(sampled_traj)
                        trajectory_nus.append(nu_val)

                    except Exception as e:
                        print(f"[Skip] seed {seed}: {e}")
                        continue

        if len(all_trajectories) == 0:
            raise RuntimeError("No valid trajectories were generated.")

        # Stack into (N, T, C, X, Y, Z) and transpose to (N, C, T, X, Y, Z)
        all_trajectories = np.stack(all_trajectories)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4, 5))
        print("Shape after stacking (Should be (N, C, T, X, Y, Z)):", all_trajectories.shape)

        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "KortewegDeVries":
        kdv_class = getattr(ex.stepper, pde)

        # Decoupled timesteps
        dt_int = 1e-3                                   # << integrator dt (small, stable)
        n_steps = int(np.ceil(t_end / dt_int))          # total integrator steps
        save_stride = max(1, int(round(dt_save / dt_int)))  # how many steps between saves

        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims,
            domain_extent=x_domain_extent,
            num_points=num_points,
            dt=dt_int,                # << integrator dt, not dt_save
            single_channel=True,
            conservative=True,
            hyper_diffusivity=0.2,    # add some damping in 3D
            order=2,
        )

        rollout_fn = ex.rollout(kdv_stepper, n_steps, include_init=True)

        # ---- IC generator ----
        ic_class = getattr(ex.ic, ic)
        common_kwargs = {"num_spatial_dims": num_spatial_dims}
        if ic == "RandomTruncatedFourierSeries":
            common_kwargs["cutoff"] = 5
        ic_instance = ic_class(**common_kwargs)

        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, 3)

            for i, k in enumerate(keys):
                u = ic_instance(num_points=num_points, key=k)
                if u.ndim == 3:
                    u = u[jnp.newaxis, ...]  # (1, X, Y, Z)
                elif not (u.ndim == 4 and u.shape[0] == 1):
                    raise ValueError(f"Unexpected IC shape {u.shape}")

                # normalize to avoid blow-ups
                u = u / (jnp.std(u) + 1e-8)
                u = jnp.clip(u, -2.0, 2.0)

                try:
                    traj = rollout_fn(u)        # (T_int+1, 1, X, Y, Z)
                    T_int = traj.shape[0]

                    # pick frames at save_stride
                    save_idx = jnp.arange(0, T_int, save_stride)
                    if save_idx[-1] != T_int - 1:
                        save_idx = jnp.concatenate([save_idx, jnp.array([T_int - 1])])

                    sampled = traj[save_idx]    # (T_saved, 1, X, Y, Z)
                    sampled_np = np.asarray(jax.device_get(sampled))

                    if not np.isfinite(sampled_np).all():
                        raise ValueError("Trajectory contains NaNs or Infs")

                    all_trajectories.append(sampled_np)
                    trajectory_nus.append(f"seed{seed:03d}_ic{i}")

                except Exception as e:
                    print(f"[Skip] seed {seed} ic {i}: {e}")
                    continue

        if len(all_trajectories) == 0:
            print("[Warning] No valid trajectories collected.")
            return np.empty((0, 1, 0, num_points, num_points, num_points)), ic_hashes, trajectory_nus

        all_trajectories = np.stack(all_trajectories, axis=0)                 # (N, T, 1, X, Y, Z)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4, 5)) # (N, 1, T, X, Y, Z)
        
        return all_trajectories, ic_hashes, trajectory_nus
 
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")