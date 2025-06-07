This project solves PDEs like Burgers in 1D/2D/3D using Exponax.
It organizes results by PDE and initial condition, outputs `.h5` datasets and visualizations.

Output structure example:
```
data/
  burgers/
    sine/
      dataset.h5
      plots/
        apebench_view.png   # if 1D
        animation.mp4       # if 2D or 3D
```