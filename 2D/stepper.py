import jax
import jax.numpy as jnp
import numpy as np
from typing import List
import sys

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
                     dt_solver: float, 
                     t_end: float,
                     save_freq: int, 
                     nu: float,
                     Re: float,
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
            domain_extent=x_domain_extent,
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
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    
    elif pde == "KortewegDeVries":
        kdv_class = getattr(ex.stepper, pde)
        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
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

            trajectories = ex.rollout(kdv_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        return all_trajectories
    
    elif pde == "Kolmogorov":
        dt = dt_solver
        end_time = t_end
        total_steps = int(end_time / dt) 
        step_to_save = save_freq

        all_trajectories = []

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
        # Convert all toa NumPyrray before stacking
        all_trajectories = [np.array(jax.device_get(traj)) for traj in all_trajectories]
        all_trajectories = np.stack(all_trajectories)
        print(type(all_trajectories))  
        print(all_trajectories.shape)  # (N, T, 1, X, Y)

        return all_trajectories
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")