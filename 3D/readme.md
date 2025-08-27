# 3D
this folder is for 3d datasets

dt cannot be too big or it explodes

Add to Kdv metadata: 
- single_channel=True  # make it scalar: only 1 channel,no 1 channel per IC: ensures it expects (1, X, Y, Z)
- conservative=True,          # <— helps stability
- hyper_diffusivity=0.05,     # try 0.05 → 0.1 if still unstable
- order=2

Other things in Kdv:
- new IC hash
- 3 ICs
-  Compute n_steps separetely in the stepper as int(np.ceil(t_end / dt_save)). Which also changes the       rollout_fn = ex.rollout(kdv_stepper, n_steps, include_init=True). Talk about it in the thesis paper.

KdV gives results for: 

pde = "KortewegDeVries" # options: 'KuramotoSivashinsky', 'Burgers', 'KortewegDeVries'
num_spatial_dims = 3
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries'
bc = None

x_domain_extent = 32.0
y_domain_extent = 32.0 
z_domain_extent = 32.0 
num_points = 64
dt_save = 0.001 # integrator step
t_end = 1.0 # final physical time
save_freq = 10  # save every X integrator steps

It gives (as 27/08):

Stacked shape (N_runs, T_sampled, 1, X, Y, Z): (6, 101, 1, 64, 64, 64)
Original shape: (6, 101, 1, 64, 64, 64)
File created at KortewegDeVries/RandomTruncatedFourierSeries/dataset.h5
Group: ic_053971ca
  Dataset: ic_053971ca/velocity_seed000 - Shape: (101, 1, 64, 64, 64), Dtype: float32
Group: ic_c1f859bd
  Dataset: ic_c1f859bd/velocity_seed001 - Shape: (101, 1, 64, 64, 64), Dtype: float32
Original shape: (6, 101, 1, 64, 64, 64)
Detected num_channels: 1
Dataset Global Statistics (Per Channel)
Channel 0: Mean=-0.080291, Std=2.057481, Min=-30.742065, Max=22.626003
            