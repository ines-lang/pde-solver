# Synthetic PDE Dataset Generator

## Description

This project enables the generation of synthetic datasets for one-, two-, and three-dimensional partial differential equations (PDEs) using fully customizable parameters. Currently, for all tre-dimensions, it supports two foundational PDEs: the **Burgers equation** and the **Kuramoto–Sivashinsky (KS) equation**, both of which are known for exhibiting rich spatiotemporal behaviors such as chaos, shock development, and pattern formation.

The **Burgers equation** is a prototypical model for studying turbulence, nonlinear wave propagation, and shock dynamics. It also serves as a simplified analog of the Navier–Stokes equations, making it a valuable testbed for numerical methods. The **Kuramoto–Sivashinsky equation**, on the other hand, captures key features of chaotic and unstable dynamics, making it ideal for benchmarking surrogate models and forecasting algorithms.

For 2d it also solves the PDE Kolmogorov Equations that are a set of partial differential equations that play a crucial role in the field of stochastic processes. The solver used in this case was: https://github.com/smokbel/Controlling-Kolmogorov-Flow.

By generating high-resolution datasets from these equations, this tool provides a foundation for research in scientific machine learning, data-driven modeling, and numerical analysis of nonlinear systems.

> GitHub repository: [https://github.com/ines-lang/pde-solver](https://github.com/ines-lang/pde-solver)

---
## Data

The data that is obtained as an output can be found here: https://huggingface.co/datasets/isuarez/pde_collection.

## File Structure

The simulation pipeline is organized as follows:

- **`generate_dataset.py`** – Main script; handles user input and coordinates execution.
- **`stepper.py`** – Parses parameters and initializes the PDE simulation.

When executed, the code creates a directory structure based on the selected PDE and initial condition. Example output structure:
Output structure:
```
1D/
├── dataset_generator.py
├── stepper.py
├── pde/
│   └── ic/
│       ├── dataset.h5
│       ├── metadata.json
│       ├── dataset_stats.py
│       └── plots/
│           └── seed_i.png  # for 1D
```

- Simulation trajectories are stored in an HDF5 file (`dataset.h5`) using keys like `velocity_000`, `velocity_001`, ..., `velocity_NNN`.
- Visualization outputs are saved in the `plots/` folder.
- Summary statistics (min, max, mean, std) are generated using `dataset_stats.py`.
- All dataset generation parameters are stored in `metadata.json`.

The generic shape of the dataset is in the format: **B × T × C × H [× W [× D]]**, where:
- **B**: number of simulations (batch size),
- **T**: number of time steps,
- **C**: number of channels (e.g. velocity components),
- **H/W/D**: spatial dimensions.

---

## Notes and Recommendations

- The initial condition **`RandomTruncatedFourierSeries`** is currently the most stable and representative choice for generating robust training data.
- No explicit boundary conditions are needed, as the Exponax solver is implemented in **JAX** and uses the **Fast Fourier Transform (FFT)** internally. For consistency, set the boundary condition parameter to `'None'`.

---

## License

This project is released under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). You are free to use, modify, and distribute the code with appropriate attribution.
