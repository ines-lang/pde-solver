import jax
import jax.numpy as jnp
from typing import List

import exponax as ex
from exponax.stepper import KuramotoSivashinskyConservative
from exponax.ic import *


def generate_dataset(pde: str,
                      ic: str, 
                      bc: callable, 
                      num_spatial_dims: int,
                      x_domain_extent: float,
                      num_points: int,
                      dt_solver: float, 
                      t_end: float,
                      save_freq: int, 
                      nu: float,
                      seed_list:List):
    
    if pde == "KuramotoSivashinskyConservative":
        ks_class = getattr(ex.stepper, pde)
        ks_stepper = ks_class(
            num_spatial_dims=num_spatial_dims, domain_extent=x_domain_extent,
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
            num_spatial_dims=num_spatial_dims, domain_extent=x_domain_extent,
            num_points=num_points, dt=dt_solver,
            )
        all_trajectories = []
        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            # Dynamically get the initial condition class from the module
            ic_class = getattr(ex.ic, ic)
            # Define common keyword arguments for all initial condition classes
            common_kwargs = {
                "num_spatial_dims": num_spatial_dims,
            }
            # Add class-specific arguments if applicable
            if ic == "RandomTruncatedFourierSeries":
                common_kwargs["cutoff"] = 5  
            # Instantiate the initial condition generator
            ic_instance = ic_class(**common_kwargs)
            # Generate the initial condition with additional parameters
            u_0 = ic_instance(num_points=num_points, key=key)
            
            trajectories = ex.rollout(burgers_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
        all_trajectories = jnp.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    elif pde == "KortewegDeVries":
        kdv_class = getattr(ex.stepper, pde)
        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims, domain_extent=x_domain_extent,
            num_points=num_points, dt=dt_solver,
            )
        all_trajectories = []
        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            # Dynamically get the initial condition class from the module
            ic_class = getattr(ex.ic, ic)
            # Define common keyword arguments for all initial condition classes
            common_kwargs = {
                "num_spatial_dims": num_spatial_dims,
            }
            # Add class-specific arguments if applicable
            if ic == "RandomTruncatedFourierSeries":
                common_kwargs["cutoff"] = 5  
            # Instantiate the initial condition generator
            ic_instance = ic_class(**common_kwargs)
            # Generate the initial condition with additional parameters
            u_0 = ic_instance(num_points=num_points, key=key)
            
            trajectories = ex.rollout(kdv_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
        all_trajectories = jnp.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")