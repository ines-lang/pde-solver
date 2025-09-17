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
                      feed_rate: float,
                      kill_rate: float,
                      reactivity: float,
                      critical_wavenumber: float,
                      seed_list:List,
                      seed: int):
    
    all_trajectories = []
    trajectory_nus = []
    ic_hashes = []

    if pde == "KuramotoSivashinskyConservative":

        for nu_val in nu:
            ks_class = getattr(ex.stepper, pde)
            ks_stepper = ks_class(
                num_spatial_dims=num_spatial_dims, 
                domain_extent=x_domain_extent,
                num_points=num_points, 
                dt=dt_save
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
                # u_0_squeeze = jnp.squeeze(u_0)  # removes all singleton axes = jnp.squeeze(u_0)  # removes all singleton axes
                # Compute and store hash for grouping
                trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)
                trajectory_nus.append(nu_val)          # keep viscosity
                ic_hashes.append(f"sim_{len(ic_hashes)}")  # dummy id
                
        all_trajectories = jnp.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)

        return all_trajectories, ic_hashes, trajectory_nus
    
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
                ic_generator = ic_class(
                    num_spatial_dims=num_spatial_dims,
                    cutoff=5)
                # generate IC
                u_0 = ic_generator(
                    key=key,
                    num_points=num_points)

                # rollout
                trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)
                trajectory_nus.append(nu_val)          # keep viscosity
                ic_hashes.append(f"sim_{len(ic_hashes)}")  # dummy id

        all_trajectories = np.stack(all_trajectories)  # (N_seeds, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N_seeds, C, T, X)

        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "Burgers":
        
        burgers_class = getattr(ex.stepper, pde)

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
                
                # Normalize shape → always (C, X)
                if u_0.ndim == 1:                   # (X,)
                    u_0 = u_0[jnp.newaxis, :]       # -> (1, X)
                elif u_0.ndim == 2:                  # already (1, X)
                    pass
                else:
                    raise ValueError(f"Unexpected IC shape {u_0.shape}, expected (1, X)")
                
                # Rollout
                trajectories = ex.rollout(burgers_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)

                trajectory_nus.append(nu_val)  # save the nu
                ic_hashes.append(f"sim_{len(ic_hashes)}")  # dummy id

        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)
        
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

            # Normalize shape → always (C, X)
            if u_0.ndim == 1:                   # (X,)
                u_0 = u_0[jnp.newaxis, :]       # -> (1, X)
            elif u_0.ndim == 2:                  # already (1, X)
                pass
            else:
                raise ValueError(f"Unexpected IC shape {u_0.shape}, expected (1, X)")
            
            # Compute and store hash for grouping
            ic_hashes.append(ic_hash(u_0))  # removes all singleton axes))

            # Rollout
            trajectories = ex.rollout(kdv_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)

            trajectory_nus.append(f"sim_{len(trajectory_nus)}")  # dummy id

        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        # Move channel dimension to position 1
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "GrayScott":

        gray_scott_class = getattr(ex.stepper.reaction, pde)
        gray_scott_stepper = gray_scott_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save,
            feed_rate=feed_rate,
            kill_rate=kill_rate,
            )
        
        for seed in seed_list:
            # Generate initial condition
            key = jax.random.PRNGKey(seed)
            # IC: random Gaussian blobs
            # Check IC
            if ic != "RandomGaussianBlobs":
                raise ValueError(f"IC '{ic}' not implemented for PDE 'GrayScott'. Use 'RandomGaussianBlobs'.")
            v_gen = ex.ic.RandomGaussianBlobs(
                num_spatial_dims=num_spatial_dims,
                domain_extent=x_domain_extent,
                num_blobs=4,
                position_range=(0.2, 0.8),  # if config != "gs-kappa" else (0.4, 0.6),
                variance_range=(0.005, 0.01),
            )
            # Actually sample the field
            v_field = v_gen(num_points=num_points, key=key)   # shape (1, X, Y)
            # Build u field as complement
            u_field = 1.0 - v_field                           # shape (1, X, Y)
            # Concatenate to get initial state
            u_0 = jnp.concatenate([u_field, v_field], axis=0)  # shape (2, X, Y)

            trajectories = ex.rollout(gray_scott_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
            
            trajectory_nus.append(f"sim_{len(trajectory_nus)}")  # dummy id
            
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)

        trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
        ic_hashes = [f"sim_{i}" for i in range(len(all_trajectories))]

        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "FisherKPP":
        
        for nu_val in nu:

            fisher_class = getattr(ex.stepper.reaction, pde)
            fisher_stepper = fisher_class(
                num_spatial_dims=num_spatial_dims, 
                domain_extent=x_domain_extent,
                num_points=num_points, 
                dt=dt_save,
                diffusivity=nu_val,
                reactivity=reactivity,
                )
            
            for seed in seed_list:
                key = jax.random.PRNGKey(seed)
                # --- Base IC: Random truncated Fourier series ---
                # Check IC
                if ic != "ClampedFourier":
                    raise ValueError(f"IC '{ic}' not implemented for PDE 'FisherKPP'. Use 'ClampedFourier'.")
                
                base_ic = ex.ic.RandomTruncatedFourierSeries(
                    num_spatial_dims=num_spatial_dims, cutoff=5
                )

                # --- Clamp to [0, 1] for Fisher-KPP ---
                ic_gen = ex.ic.ClampingICGenerator(base_ic, limits=(0.0, 1.0))

                # --- Generate the initial condition ---
                u_0 = ic_gen(num_points=num_points, key=key)  # shape (1, X, Y)

                # --- Rollout ---
                trajectories = ex.rollout(fisher_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq] # deleted
                all_trajectories.append(sampled_traj)
                
                trajectory_nus.append(nu_val)  # save the nu
                ic_hashes.append(f"sim_{len(ic_hashes)}")  # dummy id
                
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)

        return all_trajectories, ic_hashes, trajectory_nus

    elif pde == "SwiftHohenberg":
        
        swift_class = getattr(ex.stepper.reaction, pde)
        swift_stepper =swift_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save,
            reactivity=reactivity, # r
            critical_number=critical_wavenumber, # k
            polynomial_coefficients=(0.0, 0.0, 1.0, -1.0),  # u² - u³
            )
        
        for seed in seed_list:
            # Generate initial condition
            key = jax.random.PRNGKey(seed)
            
            # --- Initial condition ---
            if ic == "RandomTruncatedFourierSeries":
                base_ic = ex.ic.RandomTruncatedFourierSeries(num_spatial_dims=num_spatial_dims, cutoff=5)
            elif ic == "GaussianRandomField":
                base_ic = ex.ic.GaussianRandomField(num_spatial_dims=num_spatial_dims)
            elif ic == "DiffusedNoise":
                base_ic = ex.ic.DiffusedNoise(num_spatial_dims=num_spatial_dims, intensity=1e-3)
            else:
                raise ValueError(f"IC '{ic}' not implemented for PDE 'SwiftHohenberg'. Use 'RandomTruncatedFourierSeries', 'GaussianRandomField' or 'DiffusedNoise' instead .")

            ic_gen = ex.ic.ScaledICGenerator(base_ic, scale=0.1)  # small perturbation

            u_0 = ic_gen(num_points=num_points, key=key)  # shape (1, X, Y)

            # Rollout
            trajectories = ex.rollout(swift_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)

            trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
            ic_hashes = [f"sim_{i}" for i in range(len(all_trajectories))]
            
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        all_trajectories = np.moveaxis(all_trajectories, -2, 1)  # (N, C, T, X)
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")