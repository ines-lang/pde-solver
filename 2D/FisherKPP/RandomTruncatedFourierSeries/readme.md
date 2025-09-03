what i have tried in 1d (i dont remeber what i have wrote in 2d but it is kind of the same as grayscott): 
x_domain_extent = 1.0
num_points = 2048
dt_save = 0.005
t_end = 50.0
save_freq = 1

nu = [0, 0.00001, 0.01]  # For Burgers, KortewegDeVries, FisherKPP and SwiftHohenberg equations
reactivity = 10 # for FisherKPP and SwiftHohenberg
critical_wavenumber = 1.0 # critical wavenumber for SwiftHohenberg
