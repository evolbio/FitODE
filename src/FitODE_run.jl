using FitODE, FitODE_settings, FitODE_plots

# Loading imported modules can take a couple of minutes

# First call to a function causes compiling and optimization, which causes a delay
# Subsequent call to a function with different argument types triggers new compile

# Read comments and code in FitODE.jl and FitODE_settings.jl

# One can initialize and then modify settings as follows
# S = default_node()	# default settings for NODE
# S = Settings(S; layer_size=50, activate=3 [, ADD OTHER OPTIONS AS NEEDED])
# S = default_ode()		# default settings for ODE
# S = Settings(S; opt_dummy_u0 = true [, ADD OTHER OPTIONS AS NEEDED])
# S = Settings(S; n=5, nsqr=25)
# See docs for Parameters.jl package

# Use single struct S to hold all settings, access by S.var_name
# Provides simple way to access and save settings via one variable
# Redefinitions in struct S in FitODE_settings.jl required to run on new system

# To reset rand seed: setup S, then reset_rseed(S, rseed).

########################### Fitting full time series ##########################

S = default_ode();

# S = default_node();
# S = reset_rseed(S, 0xe5b0652b110a89b1);

# can reset individual fields in settings
# S = Settings(S; n=3, opt_dummy_u0 = true)
# S = Settings(S; n=4, max_it = 500)	# n=4 needs more iterates
# if resetting particular fields in S, should rerun calculated fields by doing
# S = recalc_settings(S)

# L is struct that includes u0, ode_data, tsteps, see struct loss_args for other parts
p_opt1,L = fit_diffeq(S)

# bfgs sometimes fails, if so then use p_opt1
# see definition of refine_fit() for other options to refine fit
p_opt2 = refine_fit_bfgs(p_opt1,S,L)

loss1, _, _, pred1 = loss(p_opt1,S,L);		# use if p_opt2 fails or p_opt1 of interest
loss2, _, _, pred2 = loss(p_opt2,S,L);

use_2 = true;	# set to false if p_opt2 fails

p, loss_v, pred = use_2 ? (p_opt2, loss2, pred2) : (p_opt1, loss1, pred1);

# if gradient is of interest
grad = calc_gradient(p,S,L)
gnorm = sqrt(sum(abs2, grad))

# save results

save_data(p, S, L, loss_v, pred; file=S.out_file)

#############################  Plotting  ###################################

using FitODE_plots

# If S is current and will be used
# dt = load_data(S.out_file);

# Or for saved outputs from prior runs
proj_output = "/Users/steve/sim/zzOtherLang/julia/FitODE/output/";
file = "ode-n4-1.jld2"; 		# fill this in with desired file name
dt = load_data(proj_output * file);

# Or for any file path		
# dt = load_data("file_path");

# use keys(dt) for list of vars in named tuple
# dt.keyname 		# returns particular key, e.g., dt.pred

# various plots

# compare original data to smoothed target data for fitting
plot_data_orig_smooth(dt.S)		# requires rereading data from disk, a bit slower
plot_data_orig_smooth(dt.L.ode_data, dt.L.tsteps, dt.L.ode_data_orig) # a bit faster

# compare predicted values to smoothed data
plot_target_pred(dt.L.tsteps, dt.L.ode_data, dt.pred)
plot_target_pred(dt.L.tsteps, dt.L.ode_data, dt.pred; show_lines=true)
plot_target_pred(dt.L.tsteps, dt.L.ode_data, dt.pred; show_lines=true,
					num_dim=size(dt.pred,1))

plot_phase(dt.L.ode_data, dt.pred)

################### Approx Bayes, split training and prediction ##################

using FitODE_bayes, Plots, StatsPlots

# If reloading data needed
proj_output = "/Users/steve/sim/zzOtherLang/julia/FitODE/output/";
file = "ode-n4-1.jld2"; 		# fill this in with desired file base name
dt = load_data(proj_output * file);

# for NODE or with ODE with n>=4, try lower a, such as 2e-3 or 1e-3 or lower
# experiment with SGLD parameters, see pSGLD struct in FitODE_bayes
B = pSGLD(warmup=5000, sample=10000, a=2e-3)

losses, parameters, ks, ks_times = psgld_sample(dt.p, dt.S, dt.L, B)

bfile = proj_output * "bayes-" * file;
save_bayes(B, losses, parameters, ks, ks_times; file=bfile);
bt = load_bayes(bfile);

# look at decay of epsilon over time
plot_sgld_epsilon(15000; a=bt.B.a, b=bt.B.a, g=bt.B.g)

# plot loss values over time to look for convergence
plot_moving_ave(bt.losses, 300)
plot_autocorr(bt.losses, 1:20)		# autocorrelation over given range

# compare density of losses to examine convergence of loss posterior distn
plot_loss_bayes(bt.losses; skip_frac=0.0, ks_intervals=10)

# parameters
pr = p_matrix(bt.parameters);		# matrix rows for time and cols for parameter values
pts = p_ts(bt.parameters,8);		# time series for 8th parameter, change index as needed
density(pts)						# approx posterior density plot

# autocorr
autoc = auto_matrix(bt.parameters, 1:30);	# row for parameter and col for autocorr vals
plot_autocorr(pts, 1:50)			# autocorrelation plot for ts in in pts, range of lags
plot(autoc[8,:])					# another way to get autocorr plot for 8th parameter
plot_autocorr_hist(bt.parameters,10)	# distn for 10th lag over all parameters

# trajectories sampled from posterior parameter distn

plot_traj_bayes(bt.parameters,dt.L,dt.S; samples=20)

