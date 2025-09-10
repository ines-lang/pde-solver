# Synthetic PDE Dataset Generator

## Description

This project enables the generation of synthetic datasets for one-, two-, and three-dimensional partial differential equations (PDEs) with **fully customizable parameters**. The implementation covers a wide range of classical PDEs that exhibit turbulence, pattern formation, chaos, shocks, and reaction–diffusion dynamics.

The solver is built on [**Exponax**](https://github.com/exponax/exponax) (JAX-based spectral solvers) and extended with a specialized implementation of the **2D Kolmogorov flow** solver ([Controlling-Kolmogorov-Flow](https://github.com/smokbel/Controlling-Kolmogorov-Flow)).

By generating **high-resolution trajectories** of these PDEs, the tool provides a foundation for research in:
- Scientific machine learning  
- Surrogate modeling and forecasting  
- Operator learning  
- Data-driven analysis of nonlinear dynamics  

> GitHub repository: [https://github.com/ines-lang/pde-solver](https://github.com/ines-lang/pde-solver)

---

## Supported PDEs

The following PDEs are currently supported:

- **Burgers equation** – nonlinear advection + viscosity (turbulence/shocks)  
- **Kuramoto–Sivashinsky equation** – chaotic and unstable dynamics  
- **Korteweg–de Vries equation** – soliton propagation  
- **Kolmogorov flow (2D)** – spectral Navier–Stokes model with forcing  
- **Gray–Scott model** – reaction–diffusion patterns (spots, stripes)  
- **Fisher–KPP equation** – traveling wave fronts and population dynamics  
- **Swift–Hohenberg equation** – pattern formation near criticality  
- **Navier–Stokes (vorticity form)** – 2D turbulence with convection and drag  
- **Allen–Cahn equation** – phase separation with double-well potential  
- **Cahn–Hilliard equation** – phase separation with mass conservation  

Each PDE comes with **compatible initial conditions (ICs)** such as:
- `RandomTruncatedFourierSeries`  
- `SpectralFlow` (Kolmogorov)  
- `RandomGaussianBlobs` (Gray–Scott)  
- `ClampedFourier` (Fisher–KPP)  
- `GaussianRandomField`, `DiffusedNoise` (Swift–Hohenberg)  

---

## Data Organization

Running `generate_dataset.py` automatically creates a directory structure organized by PDE and initial condition (IC). The hierarchy looks like this:

```
2D/
├── dataset_generator.py
├── stepper.py
├── pde/
│   └── ic/
│       ├── dataset.h5
│       ├── metadata.json
│       ├── dataset_stats.py
│       └── plots/
│           └── seed_000_channel_0.mp4 
```

- **`dataset.h5`**  
  The main simulation output. Each group corresponds to a PDE parameterization (e.g. `nu_0.010`, `Re_250.000`, `feed_0.028_kill_0.056`). Inside each group, trajectories are saved as datasets (`velocity_seed000`, `state_seed001`, etc.), with shape:
  
  (T, C, H[, W[, D]])
  
  where `T` = time steps, `C` = channels, `H/W/D` = spatial dimensions.

- **`plots/`**  
Contains `.png` snapshots and `.mp4` animations of some simulations, grouped by channel. This allows quick inspection of qualitative dynamics.

- **`metadata.json`**  
A structured record of all parameters, solver details, initial condition generation, and global statistics. This ensures full reproducibility and makes dataset sharing easier.

---

## Output Format

Simulation trajectories are stored as HDF5 datasets with the structure:

B × T × C × H [× W [× D]]


- **B**: number of simulations (batch size)  
- **T**: number of saved time steps  
- **C**: number of channels (scalar field(s) per PDE)  
- **H/W/D**: spatial dimensions  

Each dataset is grouped by PDE parameters (e.g., viscosity, Reynolds number, reaction rates). Example group names:

- Burgers: `nu_0.010`  
- Kolmogorov: `Re_250.000`  
- Gray–Scott: `feed_0.028_kill_0.056`  
- Cahn–Hilliard: `nu_1.0e-02_gamma_1.0e-03`  # TODO: check if it is with e notation

---

## File Structure

To generate datasets, you need a working directory organized by **dimension** (e.g., `1D/`, `2D/`, or `3D/`). Inside each dimension folder, place the following components:

- **`generate_dataset.py`**  
  Main script and user-facing entry point. It parses parameters, launches simulations, saves results, and produces plots and statistics.

- **`stepper.py`**  
  Defines the PDE-specific steppers, initial conditions, and rollout logic for each supported equation.

- **Solver dependencies**  
  The code relies on two external solver libraries:
  - [**Exponax**](https://github.com/exponax/exponax): a JAX-based pseudo-spectral solver framework.  
  - [**Controlling-Kolmogorov-Flow**](https://github.com/smokbel/Controlling-Kolmogorov-Flow): specialized 2D Kolmogorov flow solver.

### Example layout for 2D simulations

```
2D/
├── generate_dataset.py
├── stepper.py
├── exponax/                 # Exponax solver code
└── kolmogorov_flow/         # Kolmogorov flow solver code
```

---

## Notes and Recommendations

- **Boundary conditions**: No explicit boundary conditions are needed, as the Exponax solver is implemented in JAX and uses the Fast Fourier Transform (FFT) internally. For consistency, set the boundary condition parameter to 'None'.
- **Performance**: simulations are accelerated with JAX; NumPy is used for CPU-side dataset assembly.  
- **3D support**: implemented for Burgers and KS; visualization utilities are still under development.  

---

## Data Availability

Public datasets generated with this code are available on Hugging Face:  
👉 [isuarez/pde_collection](https://huggingface.co/datasets/isuarez/pde_collection)

---

## License

This project is released under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).  
You are free to use, modify, and distribute the code with appropriate attribution.