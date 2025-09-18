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
                     feed_rate: float,
                     kill_rate: float,
                     reactivity: float,
                     critical_wavenumber: float,
                     vorticity_convection_scale: float,
                     drag: float,
                     gamma: float,
                     c1: float,
                     c3: float,
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
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4)) # (N, C, T, X, Y)
        
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
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "KortewegDeVries":

        kdv_class = getattr(ex.stepper, pde)
        kdv_stepper = kdv_class(
            num_spatial_dims=num_spatial_dims, 
            domain_extent=x_domain_extent,
            num_points=num_points, 
            dt=dt_save,
            single_channel=True,            # expect (1, X)
            conservative=True,              # helps stability
            order=2,                        # can
            )
        
        def ic_hash(u_0, length=8):
            full_hash = hashlib.sha256(u_0.tobytes()).hexdigest()
            return full_hash[:length]  # Return first 'length' characters of the hash
        
        # ---- IC generator ----
        ic_class = getattr(ex.ic, ic)
        common_kwargs = {"num_spatial_dims": 2}
        if ic == "RandomTruncatedFourierSeries":
            common_kwargs["cutoff"] = 5
        ic_instance = ic_class(**common_kwargs)

        # ---- Rollout expects NUMBER OF STEPS (like your 3D block) ----
        n_steps = int(np.ceil(t_end / dt_save))
        rollout_fn = ex.rollout(kdv_stepper, n_steps, include_init=True)  # (T_int, 1, X, Y)

        for seed in seed_list:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, 3)  # 3 ICs per seed, as in your 3D pattern

            # Generate ICs; normalize to (1, X, Y)
            u_list = []
            for k in keys:
                u = ic_instance(num_points=num_points, key=k)  # returns (X, Y) or (1, X, Y)
                if u.ndim == 3 and u.shape[0] == 1:
                    u_c = u
                elif u.ndim == 2:
                    u_c = u[jnp.newaxis, ...]  # add channel axis -> (1, X, Y)
                else:
                    raise ValueError(f"IC has unexpected shape {u.shape}; expected (X,Y) or (1,X,Y)")
                u_list.append(u_c)

            # Hash the batch of ICs for this seed (stack only for hashing)
            u_stack_for_hash = jnp.stack(u_list, axis=0)  # (3, 1, X, Y)
            ic_hashes.append(ic_hash(u_stack_for_hash))

            # Roll out each IC separately and subsample in time
            for i, u0_single in enumerate(u_list):
                try:
                    traj = rollout_fn(u0_single)                    # (T_int, 1, X, Y); includes init
                    T_int = traj.shape[0]

                    # Build explicit save indices; include last frame even if not multiple of save_freq
                    save_idx = jnp.arange(0, T_int, save_freq)
                    if save_idx[-1] != T_int - 1:
                        save_idx = jnp.concatenate([save_idx, jnp.array([T_int - 1])])

                    # Subsample on device, then move to host
                    sampled = traj[save_idx]                        # (T_saved, 1, X, Y)
                    sampled_np = np.asarray(jax.device_get(sampled))

                    # Guard against bad trajectories
                    if not np.isfinite(sampled_np).all():
                        raise ValueError("Trajectory contains NaNs or Infs")

                    # Sanity check
                    assert sampled_np.ndim == 4 and sampled_np.shape[1] == 1, \
                        f"Expected (T,1,X,Y); got {sampled_np.shape}"

                    all_trajectories.append(sampled_np)             # (T_saved, 1, X, Y)
                    trajectory_nus.append(f"seed{seed:03d}_ic{i}")  # tag per-run

                except Exception as e:
                    print(f"[Skip] seed {seed} ic {i}: {e}")
                    continue

        # Final stacking: (N_runs, T_saved, 1, X, Y) → (N_runs, 1, T_saved, X, Y)
        if len(all_trajectories) == 0:
            print("[Warning] No valid trajectories collected.")
            return np.empty((0, 1, 0, num_points, num_points)), ic_hashes, trajectory_nus

        all_trajectories = np.stack(all_trajectories, axis=0)                 # (N, T, 1, X, Y)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))    # (N, 1, T, X, Y)

        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "Kolmogorov":
        # Check IC
        if ic != "SpectralFlow":
            raise ValueError(f"IC '{ic}' not implemented for PDE 'Kolmogorov'. Use 'SpectralFlow'.")

        dt = dt_save
        end_time = t_end
        total_steps = int(end_time / dt) 
        step_to_save = save_freq

        for seed in seed_list:
            flow = FlowConfig(grid_size=(num_points, num_points)) # spatial resolution (Nx, Ny)
            print("grid_size:", flow.grid_size)
            flow.Re = Re
            flow.k = 4 # forcing wavenumber
            # Initialize state in Fourier space
            omega_0 = flow.initialize_state()
            # Setup PDE equation and time stepper
            equation = base.PseudoSpectralNavierStokes2D(flow)
            step_fn = transient.RK4_CN(equation, dt)
            # Run simulation and recover real-space vorticity
            _, trajectory_real = transient.iterative_func(
                step_fn, omega_0, total_steps, step_to_save, ignore_intermediate_steps=True
            )

            # Move once to CPU
            trajectory_real = np.array(jax.device_get(trajectory_real))  # shape: (T_saved, X, Y)

            # Add channel dimension
            trajectory = np.expand_dims(trajectory_real, axis=1)         # (T_saved, 1, X, Y)
            all_trajectories.append(trajectory)
            print("Shape before stacking (Should be (T_saved, 1, X, Y)):", trajectory.shape)

            ic_hashes.append(f"sim_{len(ic_hashes)}")  
            trajectory_nus.append(Re)    # keep Reynolds, dummy id
        
        # Stack and reorder
        all_trajectories = np.stack(all_trajectories)                 # (N, T, 1, X, Y)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, 1, T, X, Y)

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
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)
        
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
                
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X, Y)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)
           
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

            # ---- Rollout ----
            trajectories = ex.rollout(swift_stepper, t_end, include_init=True)(u_0)
            sampled_traj = trajectories[::save_freq]
            all_trajectories.append(sampled_traj)

            trajectory_nus = [f"sim_{i}" for i in range(len(all_trajectories))] # dummy variable for each trajectory for consistency
            ic_hashes = [f"sim_{i}" for i in range(len(all_trajectories))]
            
        all_trajectories = np.stack(all_trajectories)  # shape: (N, T_sampled, C, X)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)
        
        return all_trajectories, ic_hashes, trajectory_nus
    
    elif pde == "NavierStokesVorticity":
        ns_class = getattr(ex.stepper, pde)

        for nu_val in nu:
            ns_stepper = ns_class(
                num_spatial_dims=num_spatial_dims,
                domain_extent=x_domain_extent,
                num_points=num_points,
                dt=dt_save,
                diffusivity=nu_val,
                vorticity_convection_scale=vorticity_convection_scale,  
                drag=drag, 
                dealiasing_fraction=2/3,
            )

            for seed in seed_list:
                key = jax.random.PRNGKey(seed)

                # Initial condition: similar to other PDEs, use Fourier ICs
                if ic == "RandomTruncatedFourierSeries":
                    ic_class = ex.ic.RandomTruncatedFourierSeries(
                        num_spatial_dims=num_spatial_dims, 
                        cutoff=5
                    )
                    u_0 = ic_class(num_points=num_points, key=key)
                else:
                    raise ValueError(
                        f"IC '{ic}' not implemented for PDE 'NavierStokesVorticity'. "
                        f"Use 'RandomTruncatedFourierSeries'."
                    )

                # Rollout
                trajectories = ex.rollout(ns_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)

                trajectory_nus.append(nu_val)
                ic_hashes.append(f"sim_{len(ic_hashes)}")

        all_trajectories = np.stack(all_trajectories)  # (N, T_sampled, C, X, Y)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)

        return all_trajectories, ic_hashes, trajectory_nus

    elif pde == "AllenCahn":
        ac_class = getattr(ex.stepper.reaction, pde)

        for nu_val in nu:
            ac_stepper = ac_class(
                num_spatial_dims=num_spatial_dims,
                domain_extent=x_domain_extent,
                num_points=num_points,
                dt=dt_save,
                diffusivity=nu_val,               # ν
                first_order_coefficient=c1,      # c1
                third_order_coefficient=c3,     # c3 (double well)
                dealiasing_fraction=1/2,          # cubic nonlinearity
                order=2,
            )

            for seed in seed_list:
                key = jax.random.PRNGKey(seed)

                # IC: small-amplitude smooth field in [-~0.5, ~0.5]
                if ic == "RandomTruncatedFourierSeries":
                    base_ic = ex.ic.RandomTruncatedFourierSeries(
                        num_spatial_dims=num_spatial_dims, cutoff=10
                    )
                    ic_gen = ex.ic.ScaledICGenerator(base_ic, scale=0.5)
                else:
                    raise ValueError(
                        f"IC '{ic}' not implemented for PDE 'AllenCahn'. "
                        f"Use 'RandomTruncatedFourierSeries'."
                    )

                u_0 = ic_gen(num_points=num_points, key=key)  # (1, X, Y)

                trajectories = ex.rollout(ac_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)

                trajectory_nus.append(nu_val)
                ic_hashes.append(f"sim_{len(ic_hashes)}")

        all_trajectories = np.stack(all_trajectories)  # (N, T, C, X, Y)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)

        return all_trajectories, ic_hashes, trajectory_nus


    elif pde == "CahnHilliard":
        ch_class = getattr(ex.stepper.reaction, pde)

        for nu_val in nu:
            ch_stepper = ch_class(
                num_spatial_dims=num_spatial_dims,
                domain_extent=x_domain_extent,
                num_points=num_points,
                dt=dt_save,
                diffusivity=nu_val,           # ν (mobility-like prefactor here)
                gamma=gamma,                  # γ (interfacial energy scale)
                first_order_coefficient=c1,   # c1
                third_order_coefficient=c3,   # c3
                dealiasing_fraction=1/2,
                order=2,
            )

            for seed in seed_list:
                key = jax.random.PRNGKey(seed)

                # IC: small, zero-mean field (mass conservation is important)
                if ic == "RandomTruncatedFourierSeries":
                    base_ic = ex.ic.RandomTruncatedFourierSeries(
                        num_spatial_dims=num_spatial_dims, cutoff=5
                    )
                    ic_gen = ex.ic.ScaledICGenerator(base_ic, scale=0.1)
                else:
                    raise ValueError(
                        f"IC '{ic}' not implemented for PDE 'AllenCahn'. "
                        f"Use 'RandomTruncatedFourierSeries'."
                    )

                u_0 = ic_gen(num_points=num_points, key=key)  # (1, X, Y)

                # Enforce zero mean so phase fraction is balanced
                u_0 = u_0 - jnp.mean(u_0)

                trajectories = ex.rollout(ch_stepper, t_end, include_init=True)(u_0)
                sampled_traj = trajectories[::save_freq]
                all_trajectories.append(sampled_traj)

                trajectory_nus.append(nu_val)
                ic_hashes.append(f"sim_{len(ic_hashes)}")

        all_trajectories = np.stack(all_trajectories)  # (N, T, C, X, Y)
        all_trajectories = np.transpose(all_trajectories, (0, 2, 1, 3, 4))  # (N, C, T, X, Y)

        return all_trajectories, ic_hashes, trajectory_nus

    else:
        raise ValueError(f"PDE '{pde}' not implemented.")