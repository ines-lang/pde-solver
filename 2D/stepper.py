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
                      y_domain_extent: float,
                      num_points: int,
                      dt_solver: float, 
                      t_end: float,
                      save_freq: int, 
                      nu: float,
                      seed_list:List):
    
    if pde == "KuramotoSivashinsky":
        ks_class = getattr(ex.stepper, pde)
        ks_stepper = ks_class(
            num_spatial_dims=num_spatial_dims, domain_extent=x_domain_extent, # cuidado con el dominio
            num_points=num_points, dt=dt_solver,
            )
        all_trajectories = []
        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            ic_class = getattr(ex.ic, ic)
            u_0 = ic_class(
                num_spatial_dims=num_spatial_dims, cutoff=5, #only first 5 fourier modes used
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
            domain_extent=x_domain_extent, # cuidado con el dominio
            num_points=num_points, 
            dt=dt_solver,
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

            trajectories = ex.rollout(burgers_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
            # using numpyyy todooo
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")