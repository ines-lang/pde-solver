pde = "SwiftHohenberg" # options: 'KuramotoSivashinsky', 'Burgers', 'Kolmogorov', 'KortewegDeVries', 'GrayScott', 'FisherKPP', 'SwiftHohenberg'
ic = "RandomTruncatedFourierSeries" # options: 'RandomTruncatedFourierSeries', 'RandomSpectralVorticityField', 
bc = None

x_domain_extent = 20 * np.pi
y_domain_extent = 20 * np.pi
num_points = 2048
dt_save = 0.5
t_end = 20.0 
save_freq = 5

''' What it implies:
Total steps: n_steps = t_end / dt_save
Output interval: dt_output = dt_save * save_freq
Total saved frames: n_saved = n_steps / save_freq + 1
'''

nu = [0, 0.00001, 0.01]  # For Burgers, KortewegDeVries, FisherKPP and SwiftHohenberg equations
# todo implement Re as nu 
Re = 250  # For Kolmogorov equation
reactivity = 0.6 # for FisherKPP and SwiftHohenberg
critical_wavenumber = 1.0 # critical wavenumber for SwiftHohenberg

# For Gray Scott:
feed_rate = 0.028
kill_rate = 0.056

simulations = 5
plotted_sim = 2
plot_sim = True
stats = True
seed = 42
