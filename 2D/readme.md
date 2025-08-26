# 2D
- **generate_dataset.py** must be runned from 2D folder to correctly create the output directories.
- /local/disk1/stu_isuarez/pde-solver/2D/kolmogorov_flow has the solver used form kolmogorov. Is has been obtained from https://github.com/smokbel/Controlling-Kolmogorov-Flow, with minor modifications regarding:
    - Real-space Output Conversion: The solver was adapted to return the vorticity field in physical space rather than in Fourier space. This was achieved by applying jnp.fft.irfftn with the correct shape recovery inside the time-stepping loop. This change ensures that output trajectories can be directly used for dataset generation and machine learning applications.
    - Explicit Domain Shape Control: The solver now infers the spatial shape of the grid (Nx, Ny) from the initial condition in Fourier space and uses it to configure the inverse FFT operations consistently across all components (velocity, gradients, advection terms). This prevents shape mismatches.
- KS has one channel and Burgers two (for u and v)
- Doesn´t work good yet for big numbers of simulations (order of magnitude where it works: 10). Exhausted.
- Shape of all_trajectories is (N, T_sampled, C, X). It includes simulations.
- In KdV there are 2 channels reffering to the 2 ICs created.

TODO:
- try adding another IC
- Plots still has unfixed values of -2 and 2. Added at the metadata until solved.
- Kolmogorov doesnt work now

# Pseudo-Spectral Solver for the 2D Kolmogorov Flow

This repository implements a pseudo-spectral solver for the two-dimensional incompressible Navier–Stokes equations in vorticity formulation, with periodic boundary conditions and Kolmogorov-type forcing. The solver is designed for high-fidelity numerical simulations and data generation for learning-based flow control and reduced-order modeling.

## Requirements

Install the project from: https://github.com/smokbel/Controlling-Kolmogorov-Flow and add it to the 2D folder.

## Flow Configuration

The solver uses a `FlowConfig` class to define simulation parameters:

- `flow.Re`: Reynolds number  
- `flow.grid_size`: spatial resolution (Nx, Ny)  
- `flow.k`: forcing wavenumber  
- `flow.dt`: time step (external)

## Initial Conditions

Initial vorticity is generated using:

```python
omega_0 = flow.initialize_state()
```

This produces a divergence-free random field in spectral space. The field is suitable for isotropic decaying turbulence and Kolmogorov flow regimes.

Although currently only one initial condition is implemented, compatibility with the dataset generation script requires:

```python
ic = "RandomSpectralVorticityField"
```

This identifier is purely descriptive and does not influence the actual initialization, which always calls `flow.initialize_state()` internally.

## Time Integration

The system is evolved in time using a fourth-order implicit–explicit Runge–Kutta scheme (IMEX-RK4), where the viscous term is integrated implicitly and the nonlinear advection term explicitly. The solver interface:

```python
step_fn = RK4_CN(equation, dt)
_, trajectory = iterative_func(step_fn, omega_0, total_steps, step_to_save)
```

Simulation output consists of a trajectory of vorticity fields sampled at uniform intervals.
