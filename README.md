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
- **Kuramoto–Sivashinsky (KS)** – chaotic and unstable dynamics (1D/2D); **also includes the conservative KS variant in 1D**.
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
├── generate_dataset.py
├── stepper.py
├── pde/
│   └── ic/
│       ├── dataset.h5
│       ├── metadata.json
│       └── plots/
│           └── seed_000_channel_0.mp4 
```

- **`dataset.h5`**  
  The main simulation output. Each group corresponds to a PDE parameterization (e.g. `nu_0.010`, `Re_250.000`, `feed_0.028_kill_0.056`). Inside each group, trajectories are saved as datasets (`velocity_seed000`, `state_seed001`, etc.), with shape:
  
  (C, T, H[, W[, D]])
  
  where `C` = channels, `T` = time steps, `H/W/D` = spatial dimensions.

- **`plots/`**  
Contains `.png` snapshots and `.mp4` animations of some simulations, grouped by channel. This allows quick inspection of qualitative dynamics.

- **`metadata.json`**  
A structured record of all parameters, solver details, initial condition generation, and global statistics. This ensures full reproducibility and makes dataset sharing easier.

---

## Output Format

Simulation trajectories are stored as HDF5 datasets with the structure:

B × C × T × H [× W [× D]]


- **B**: number of simulations (batch size)    
- **C**: number of channels (scalar field(s) per PDE)  
- **T**: number of saved time steps
- **H/W/D**: spatial dimensions  

Each dataset is grouped by PDE parameters (e.g., viscosity, Reynolds number, reaction rates). Example group names:

- Burgers: `nu_0.010`  
- Kolmogorov: `Re_250.000`  
- Gray–Scott: `feed_0.028_kill_0.056`  
- Cahn–Hilliard: `nu_1.0e-02_gamma_1.0e-03`  # TODO: check if it is with e notation

---

## Running the Generator

The entry point is `generate_dataset.py`, which supports two modes:

1. **Default parameters**  
   Run without arguments to use the defaults defined in the script:
   ```bash
   python generate_dataset.py
   ```

2. **Custom configuration**  
   Provide a JSON file with your parameters (faster):
   ```bash
   python generate_dataset.py --config metadata.json
   ```
   Example `metadata.json` files are provided in the repository for each PDE and dimension.  
   For more details on repository layout and code structure, see the section *File Structure*.

   The following parameters can be specified through the argparse configuration:

   - `num_spatial_dims`, `pde`, `ic`, `bc`  
   - `x_domain_extent`, `num_points`, `dt_save`, `t_end`, `save_freq`  
   - `nu`, `Re`, `reactivity`, `critical_wavenumber`  
   - `feed_rate`, `kill_rate`, `vorticity_convection_scale`, `drag`  
   - `gamma`, `c1`, `c3`  
   - `simulations`, `plotted_sim`, `plot_sim`, `stats`, `seed`  

   Note: in 1D and 3D configurations, some parameters may not apply.

   For example, some variables recorded in the metadata cannot be changed directly through it, such as the `cmap` or animation-related settings (e.g., whether the visualization is `physical`, `fixed`, etc.).

In both modes, the script generates:

- `dataset.h5` — simulation trajectories  
- `metadata.json` — full record of parameters and solver details  
- `plots/` — PNG/MP4 visualizations of selected runs

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
  - [**Controlling-Kolmogorov-Flow**](https://github.com/smokbel/Controlling-Kolmogorov-Flow): only used in `/2D`.

### Example layout for 2D simulations

```
2D/
├── generate_dataset.py
├── stepper.py
├── exponax/    
│   ├── etdrk/
│   ├── ic/
│   ├── metrics/
│   └── ...        # other submodules
└── kolmogorov_flow/
    └── Controlling_Kolmogorov_Flow/
        ├── equations/
        ├── gym/
        └── ...        # other submodules

```
Where folders containing the solvers are saved at the same level as the python running files.

#### Install

```bash
# Activate your environment
conda activate jaxfluids  # or: conda env create -f environment.yml && conda activate pde-datasets

# Install Exponax
pip install "git+https://github.com/Ceyron/exponax.git"

# Clone Kolmogorov Flow into the expected path
git clone --depth 1 https://github.com/smokbel/Controlling-Kolmogorov-Flow.git \
  2D/kolmogorov_flow/Controlling_Kolmogorov-Flow
```
---

## Notes and Recommendations

- **Boundary conditions**: No explicit boundary conditions are needed, as the Exponax solver is implemented in JAX and uses the Fast Fourier Transform (FFT) internally. For consistency, set the boundary condition parameter to 'None'.
- Although the user interface allows the definition of `y_domain_extent` and `z_domain_extent` for 2D and 3D cases, the solvers internally rely only on `x_domain_extent`, which is treated as the full spatial extent of the domain. As a result, the effective domain size is always determined by `x_domain_extent`, and editing `y_domain_extent` or `z_domain_extent` has no effect on the simulation and it is not really necesarily.
- **Performance**: simulations are accelerated with JAX; NumPy is used for CPU-side dataset assembly.  
- Each simulation is initialized with a user-defined seed from the `seed_list`. Reproducibility is guaranteed only if the same `seed_list` is used across runs. By default, the list starts from `seed=42`, but this can be customized by the user.

- **3D support**: implemented for Burgers and KS; visualization utilities are still under development. 
- **Visualizations in 3D: CLI**:  The `viz_dataset.py` tool (in `3D/`) provides a command-line interface that can render time-series from HDF5 into PNG frames and MP4 videos.

---

<!-- ## Data Availability

Public datasets generated with this code are available on Hugging Face:  
👉 [isuarez/pde_collection](https://huggingface.co/datasets/isuarez/pde_collection)

--- -->

## License

This project is released under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
