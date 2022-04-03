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

# Load data saved to file, use only one of following		
dt = load_data("file_path");
# Or if S is current and will be used
dt = load_data(S.out_file);

# use keys(dt) for list of vars in named tuple, for example
# dt.pred 				# returns prediction data 

# various plots

# compare original data to smoothed target data for fitting
plot_data_orig_smooth(dt.S)		# requires rereading data from disk, a bit slower
plot_data_orig_smooth(dt.L.ode_data, dt.L.tsteps, dt.L.ode_data_orig) # a bit faster

# compare predicted values to smoothed data
plot_target_pred(dt.L.tsteps, dt.L.ode_data, dt.pred)
plot_target_pred(dt.L.tsteps, dt.L.ode_data, dt.pred; show_lines=true)
plot_target_pred(dt.L.tsteps, dt.L.ode_data, dt.pred; show_lines=true,
					num_dim=size(dt.pred,1))

# phase plot, needs fixing if n > 3
plot_phase(dt.L.ode_data, dt.pred)

################### Quasi-Bayes, split training and prediction ##################

using FitODE_bayes

# dt=load_data("/Users/steve/sim/zzOtherLang/julia/FitODE/output/ode-n3-1.jld2");

losses, parameters, ks, ks_times = psgld_sample(dt.p, dt.S, dt.L);





