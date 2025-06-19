KS has one channel and Burgers two (for u and v)
Doesn´t work good yet for big numbers of simulations (order of magnitude where it works: 10). Exhausted.
Plots still has unfixed values of -2 and 2. Added at the metadata until solved.
Shape of all_trajectories is (N, T_sampled, C, X). It includes simulations

TODO:
- domain_extent is only obtained from x_domain. y_domain is not used. try what happens if they are removed.
- copy 3d initial condition structure in the stepper and addapt to 2d and see if it works. it seems more clean.
- solve numpy problems when trajectories array is constructed.
- try adding another IC.
- Plotting: explain skip and frames variables.