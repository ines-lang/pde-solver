# 1D
This folder contains the code necessary to run the generation of data.
The data that is obtained as an output can be found here: https://huggingface.co/datasets/isuarez/pde_collection.
All the variables used to obtain the dataset.h5 file can be found in the metadata.json file.
In KS, the first 200-500 steps of the trajectory will be the transitional phase, after that the chaotic attractor is reached.
For now, plots have u_min=-2 and u_max=2.