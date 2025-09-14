# 1D

To generate datasets in one dimension, run **`generate_dataset.py`** from within the `1D/` folder.  
This ensures that the output directories are created correctly.

The resulting datasets can be found at:  
[https://huggingface.co/datasets/isuarez/pde_collection](https://huggingface.co/datasets/isuarez/pde_collection)

All the variables used to obtain the dataset.h5 file can be found in the metadata.json file.

In KS, the first 200-500 steps of the trajectory will be the transitional phase, after that the chaotic attractor is reached.

Carefull: KdV solution explodes from some simulations. dataset_stats.py has been corrected to ignore this simulations when running the mean and std of the data. But user of the data must take it into account.