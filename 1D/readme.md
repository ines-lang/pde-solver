# 1D
This folder contains the code necessary to run the generation of data.
The data that is obtained as an output can be found here: https://huggingface.co/datasets/isuarez/pde_collection.
All the variables used to obtain the dataset.h5 file can be found in the metadata.json file.
In KS, the first 200-500 steps of the trajectory will be the transitional phase, after that the chaotic attractor is reached.
For now, plots have u_min=-2 and u_max=2.ç
Carefull: KdV solution explodes from some simulations. dataset_stats.py has been corrected to ignore this simulations when running the mean and std of the data. But user of the data must take it into account.

# TODO:
- Add the graphic legend of velocity to check the values when the velocity ranges. Add to the code in generate_dataset and run everthing again.

- channel to the plots name