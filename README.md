# dataset_generator
This project solves PDEs in 1D/2D/3D using differents solvers such as Exponax.
It organizes results by PDE and initial condition, datasets `.h5` and visualizations.

Output (and complete) structure example:
```
1D/
    dataset_generator.py
    stepper.py
    pde/
        ic/
            dataset.h5
            metadata.json
            plots/
                seed_i.png # for 1d
```
For the code to work, it is necessary to clone the Exponax repository and have the folder downloaded at the same level as the dataset_generator.