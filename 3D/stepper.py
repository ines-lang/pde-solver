import jax
import jax.numpy as jnp
import numpy as np
from typing import List

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
                      seed_list:List):
    
    if pde == "KuramotoSivashinsky":
        ks_class = getattr(ex.stepper, pde)
        ks_stepper = ks_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent, 
            num_points=num_points, 
            dt=dt_save,
            )
        all_trajectories = []
        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            ic_class = getattr(ex.ic, ic)
            u_0 = ic_class(
                num_spatial_dims=num_spatial_dims, cutoff=5,
            )(num_points=num_points, key=key)
            trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
        all_trajectories = jnp.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    
    elif pde == "Burgers":
        burgers_class = getattr(ex.stepper, pde)
        burgers_stepper = burgers_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent, 
            num_points=num_points, 
            dt=dt_save,
            )
        all_trajectories = []
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
                # Generate a single initial condition for all 3 velocity component
                u_scalar = ic_instance(num_points=num_points, key=key)  # (1, X, Y, Z)
                if u_scalar.shape[0] == 1:
                    u_scalar = u_scalar[0]  # → (X, Y, Z)

                # Tile to (3, X, Y, Z)
                u_0 = jnp.tile(u_scalar[None, ...], (3, 1, 1, 1))

                trajectories = np.array(ex.rollout(burgers_stepper, t_end, include_init=True)(u_0)) # added numpy to move rollout to CPU immediately
                sampled_traj = trajectories[::save_freq]
                # Debugging block
                print(f"seed {seed} → traj shape: {sampled_traj.shape}")
                print(f"min: {np.nanmin(sampled_traj):.3f}, max: {np.nanmax(sampled_traj):.3f}")
                if np.isnan(sampled_traj).any() or np.isinf(sampled_traj).any():
                    raise ValueError("Trajectory contains NaNs or Infs")
                all_trajectories.append(sampled_traj)
            except Exception as e:
                print(f"Failed to generate trajectory for seed {seed}: {e}")
                continue  # Skip this seed and move on
            print(f"Generated {len(all_trajectories)} valid trajectories out of {len(seed_list)} seeds.")

        if len(all_trajectories) == 0:
            raise RuntimeError("No valid trajectories were generated.")
        
        # using numpy
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    
    elif pde == "KortewegDeVries":
        kdv_class = getattr(ex.stepper, pde)
        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save,
            )
        all_trajectories = []
        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            ic_class = getattr(ex.ic, ic)
            common_kwargs = {
                "num_spatial_dims": num_spatial_dims,
            }
            # Add class-specific arguments if applicable
            if ic == "RandomTruncatedFourierSeries":
                common_kwargs["cutoff"] = 5
            
            ic_instance = ic_class(**common_kwargs)
            # Generate the initial condition with additional parameters
            key1, key2 = jax.random.split(key)

            u_0_1 = ic_instance(num_points=num_points, key=key1)
            u_0_2 = ic_instance(num_points=num_points, key=key2)

            # Remove leading channel dim if present: (1, 200, 200) → (200, 200)
            if u_0_1.ndim == 3 and u_0_1.shape[0] == 1:
                u_0_1 = u_0_1[0]
                u_0_2 = u_0_2[0]

            # Stack into batch: (2, 200, 200)
            u_0 = jnp.stack([u_0_1, u_0_2])

            trajectories = ex.rollout(kdv_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")