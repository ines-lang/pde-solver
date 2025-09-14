# 2D
**generate_dataset.py** must be runned from 2D folder to correctly create the output directories.
The data that is obtained as an output can be found here: https://huggingface.co/datasets/isuarez/pde_collection.

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

