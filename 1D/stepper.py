import jax
import jax.numpy as jnp
import numpy as np
from typing import List
import hashlib

import exponax as ex
from exponax.stepper import KuramotoSivashinskyConservative
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
                      nu: list,
                      seed_list:List,
                      seed: int):
    
    if pde == "KuramotoSivashinskyConservative":
        ks_class = getattr(ex.stepper, pde)
        ks_stepper = ks_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save
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
            # u_0_squeeze = jnp.squeeze(u_0)  # removes all singleton axes = jnp.squeeze(u_0)  # removes all singleton axes
            # Compute and store hash for grouping
            trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
        all_trajectories = jnp.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)

        ic_hashes = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy hashes for each trajectory for consistency
        trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    if pde == "KuramotoSivashinsky":
        all_trajectories = []
        all_ic_hashes = []

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
                ic_generator = ic_class(
                    num_spatial_dims=num_spatial_dims,
                    cutoff=5)
                # generate IC
                u_0 = ic_generator(
                    key=key,
                    num_points=num_points)
                # store hash for reproducibility
                all_ic_hashes.append(hash(u_0.tobytes()))
                # rollout
                trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)
            
        all_trajectories = np.stack(all_trajectories)  # (N_seeds, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N_seeds, C, T, X)

        trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency

        return all_trajectories, all_ic_hashes, trajectory_nus
    
    elif pde == "Burgers":
        burgers_class = getattr(ex.stepper, pde)

        all_trajectories = []
        trajectory_nus = []  # NEW: track which nu was used

        for nu_val in nu:
            burgers_stepper = burgers_class(
                num_spatial_dims=num_spatial_dims,
                domain_extent=x_domain_extent,
                num_points=num_points,
                dt=dt_save,
                diffusivity=nu_val
            )
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
                
                 # Remove leading channel dim if present
                if u_0.ndim == 3 and u_0.shape[0] == 1:
                    u_0 = u_0[0]
                
                # Stack into batch: (1, 200)
                u_0 = jnp.expand_dims(u_0, axis=0)  # (1, 200)
                
                # Rollout
                trajectories = ex.rollout(burgers_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)
                trajectory_nus.append(nu_val)  # save the nu

        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)
        
        ic_hashes = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy hashes for each trajectory for consistency
        trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "KortewegDeVries":
        kdv_class = getattr(ex.stepper, pde)
        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save,
            )
        
        def ic_hash(u_0, length=8):
            full_hash = hashlib.sha256(u_0.tobytes()).hexdigest()
            return full_hash[:length]  # Return first 'length' characters of the hash
        
        all_trajectories = []
        ic_hashes = []  # store hash per trajectory
        
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

            # Remove leading channel dim if present
            if u_0.ndim == 3 and u_0.shape[0] == 1:
                u_0 = u_0[0]

            # Stack into batch: (1, 200)
            u_0 = jnp.expand_dims(u_0, axis=0)  # (1, 200)
            
            # Compute and store hash for grouping
            ic_hashes.append(ic_hash(u_0))  # removes all singleton axes))

            trajectories = ex.rollout(kdv_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
            
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)
        
        trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")
