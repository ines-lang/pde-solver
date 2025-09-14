# 3D

**generate_dataset.py** must be runned from 2D folder to correctly create the output directories.
The data that is obtained as an output can be found here: https://huggingface.co/datasets/isuarez/pde_collection.

Add to Kdv metadata: 
- single_channel=True  # make it scalar: only 1 channel,no 1 channel per IC: ensures it expects (1, X, Y, Z)
- conservative=True,          # <— helps stability
- hyper_diffusivity=0.05,     # try 0.05 → 0.1 if still unstable
- order=2

Other things in Kdv:
- new IC hash
- 3 ICs
-  Compute n_steps separetely in the stepper as int(np.ceil(t_end / dt_save)). Which also changes the       rollout_fn = ex.rollout(kdv_stepper, n_steps, include_init=True). Talk about it in the thesis paper.
