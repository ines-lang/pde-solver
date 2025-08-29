import jax
import jax.numpy as jnp
import numpy as np
from typing import List
import sys
import hashlib

import exponax as ex
from exponax.stepper import KuramotoSivashinsky
from exponax.ic import *

sys.path.append("/local/disk1/stu_isuarez/pde-solver/2D/kolmogorov-flow/Controlling_Kolmogorov-Flow") 
from kolmogorov_flow.Controlling_Kolmogorov_Flow.solvers import transient
from kolmogorov_flow.Controlling_Kolmogorov_Flow.equations.flow import FlowConfig 
import kolmogorov_flow.Controlling_Kolmogorov_Flow.equations.base as base
import kolmogorov_flow.Controlling_Kolmogorov_Flow.equations.utils as utils 

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
                     Re: float,
                     seed_list:List,
                     seed: int):

    all_trajectories = []
    trajectory_nus = []
    ic_hashes = []
    
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
                u_0 = ic_class(
                    num_spatial_dims=num_spatial_dims, 
                    cutoff=5, #only first 5 fourier modes used
                )(num_points=num_points, key=key)

                # rollout
                trajectories = ex.rollout(ks_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)

                trajectory_nus.append(nu_val)          # keep viscosity
                ic_hashes.append(f"sim_{len(ic_hashes)}")  # dummy id

        print("Shape before stacking (Should be (N, T_sampled, C, X, Y)):", jnp.array(all_trajectories).shape)
        
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X, Y)
        
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

                trajectory_nus.append(nu_val)  # save the nu
                ic_hashes.append(f"sim_{len(ic_hashes)}")  # dummy id

        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X Y)
        
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
            
            # Compute and store hash
            ic_hashes.append(ic_hash(u_0)) # it needs to be added

            trajectories = ex.rollout(kdv_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)

            trajectory_nus.append(f"sim_{len(trajectory_nus)}")  # dummy id

        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X, Y)

        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "Kolmogorov":
        
        dt = dt_save
        end_time = t_end
        total_steps = int(end_time / dt) 
        step_to_save = save_freq

        for seed in seed_list:
            flow = FlowConfig(grid_size=(num_points, num_points))
            print("grid_size (should output (100, 100)) :", flow.grid_size)
            flow.Re = Re
            flow.k = 4
            # Initialize state in Fourier space
            omega_0 = flow.initialize_state()

            # Setup PDE equation and time stepper
            equation = base.PseudoSpectralNavierStokes2D(flow)
            step_fn = transient.RK4_CN(equation, dt)
            # Run simulation and recover real-space vorticity
            _, trajectory_real = transient.iterative_func(
                step_fn, omega_0, total_steps, step_to_save, ignore_intermediate_steps=True)
            # solving the save_freq issue
            trajectory_real = np.array(jax.device_get(trajectory_real))  # transfer entire array at once
            trajectory_real = trajectory_real[::save_freq]  # then slice in NumPy (fast)
            print("trajectory shape (should be (T_sampled, 100, 100)):", trajectory_real.shape)

            trajectory = np.array(trajectory_real)  # shape: (T_sampled, X, Y) and changed from jnp to np
            trajectory = np.expand_dims(trajectory, axis=1)  # add channel dimension
            all_trajectories.append(trajectory) 
            print("Shape before stacking (Should be (T_sampled, 1, X, Y)):", trajectory.shape)

            ic_hashes.append(f"sim_{len(ic_hashes)}")  
            trajectory_nus.append(Re)                  # keep Reynolds
        
        # Convert all toa NumPyrray before stacking
        all_trajectories = [np.array(jax.device_get(traj)) for traj in all_trajectories]
        all_trajectories = np.stack(all_trajectories)
        print(type(all_trajectories))  
        print(all_trajectories.shape)  # (N, T, 1, X, Y)

        return all_trajectories, ic_hashes, trajectory_nus

    elif pde == "GrayScott":

        gray_scott_class = getattr(ex.stepper.reaction, pde)
        gray_scott_stepper = gray_scott_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save,
            feed_rate=0.028,
            kill_rate=0.056,
            )
        
        for seed in seed_list:
            # Generate initial condition
            key = jax.random.PRNGKey(seed)

            # IC: random Gaussian blobs
            v_gen = ex.ic.RandomGaussianBlobs(
                num_spatial_dims=2,
                domain_extent=2.5,
                num_blobs=4,
                position_range=(0.2, 0.8),  # if config != "gs-kappa" else (0.4, 0.6),
                variance_range=(0.005, 0.01),
            )

            # # u = 1 - v
            # # u_gen = ex.ic.ScaledICGenerator(v_gen, scale=-1.0)  # u = -v
            # # u_gen = ex.ic.ShiftedICGenerator(u_gen, shift=1.0) # then u = 1 - v
            # u_gen = 1.0 - v_gen  # equivalent to above two lines

            # multi_ic_gen = ex.ic.RandomMultiChannelICGenerator((u_gen, v_gen))

            # u_0 = multi_ic_gen(num_points=num_points, key=key)  # shape (2, 128, 128)

            # Actually sample the field
            v_field = v_gen(num_points=num_points, key=key)   # shape (1, X, Y)

            # Build u field as complement
            u_field = 1.0 - v_field                           # shape (1, X, Y)

            # Concatenate to get initial state
            u_0 = jnp.concatenate([u_field, v_field], axis=0)  # shape (2, X, Y)

            trajectories = ex.rollout(gray_scott_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
            
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        
        trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
        ic_hashes = [f"sim_{i}" for i in range(len(all_trajectories))]

        return all_trajectories, ic_hashes, trajectory_nus
    
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")