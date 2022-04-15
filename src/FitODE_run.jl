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

# train on part of data: if end is 90, 60/90 => 60, 75/90 = 75, etc.
# S = Settings(S; train_frac = 60/90);	

# if resetting particular fields in S, should rerun calculated fields by doing
# S = recalc_settings(S);

# L is struct that includes u0, ode_data, tsteps, see struct loss_args for other parts
# A is struct that includes tsteps_all and prob_all, used if S.train_frac < 1 that
# splits data into initial training period and later period used to compare w/prediction
p_opt1,L,A = fit_diffeq(S)

# If using a subset of data for training, then need L_all with full time period for all data
# L always refers to training period, which may or may not be all time steps
L_all = (S.train_frac < 1) ? make_loss_args_all(L, A) : L;

p_opt2 = refine_fit_bfgs(p_opt1,S,L)

# run bfgs a second time if desired
# p_opt2 = refine_fit_bfgs(p_opt2,S,L)

# bfgs sometimes fails, if so then use p_opt1 or try repeating refine_fit
# p_opt2 = refine_fit(p_opt1,S,L)
# and again if needed
# p_opt2 = refine_fit(p_opt2,S,L)

# see definition of refine_fit() for other options to refine fit
# alternatively, may be options for ode solver and tolerances that would allow bfgs

loss1, _, _, pred1 = loss(p_opt1,S,L);		# use if p_opt2 fails or p_opt1 of interest
loss2, _, _, pred2 = loss(p_opt2,S,L);

use_2 = true;	# set to false if p_opt2 fails

p, loss_v, pred = use_2 ? (p_opt2, loss2, pred2) : (p_opt1, loss1, pred1);

# if gradient is of interest
grad = calc_gradient(p,S,L)
gnorm = sqrt(sum(abs2, grad))

# save results
save_data(p, S, L, L_all, loss_v, pred; file=S.out_file)

# test loading
dt_test = load_data(S.out_file);
keys(dt_test)

# If OK, then move out_file to standard location and naming for runs

# To use following steps, move saved out_file to proj_output using 
# example in following steps

# earlier version of program did not define S.train_frac or save L_all
# check with @isdefined before using any steps that require these vars

#############################  Plotting  ###################################

using FitODE_plots, Plots, Measures

# If S is current and will be used
# dt = load_data(S.out_file);

# Or for saved outputs from prior runs
proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/FitODE/output/";
train_time = "all";						# e.g., "all", "60", "75", etc
train = "train_" * train_time * "/"; 	# directory for training period
file = "node-n2-1.jld2"; 				# fill this in with desired file name
dt = load_data(proj_output * train * file);

# Or for any file path		
# dt = load_data("file_path");

# use keys(dt) for list of vars in named tuple
# dt.keyname 		# returns particular key, e.g., dt.pred

# various plots

# compare original data to smoothed target data for fitting
plot_data_orig_smooth(dt.S)		# requires rereading data from disk, a bit slower
plot_data_orig_smooth(dt.L.ode_data, dt.L_all.tsteps, dt.L.ode_data_orig) # a bit faster

# compare predicted values to smoothed data, use_all for plot beyond training period
plot_target_pred(dt)
plot_target_pred(dt; show_lines=true)
plot_target_pred(dt; show_lines=true, num_dim=size(dt.pred,1))
plot_target_pred(dt; show_lines=true, use_all=false)	# show training period only

plot_phase(dt; use_all=true)

#########################  Plot multiple target_pred runs  #######################
using FitODE_plots, Plots, Measures

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/FitODE/output/";
train_time = "all";						# e.g., "all", "60", "75", etc
train = "train_" * train_time * "/"; 	# directory for training period

ode = ["ode-n" * string(i) * "-1.jld2" for i in 2:4];
node = ["n" * x for x in ode];
# interleave vectors a and b by [a b]'[:], but for strings, use permutedims
files = [proj_output * train * file for file in permutedims([ode node])[:]];

plts = []
for f in files
	dt = load_data(f)
	push!(plts, plot_target_pred(dt; show_lines=true, num_dim=size(dt.pred,1),
				target_labels=("","")))
end
plt = plot(plts..., size=(1200,1800), layout=grid(3,2,heights=[2/9,3/9,4/9]),
		linewidth=3, top_margin=-10mm, bottom_margin=8mm, left_margin=3mm)

Plots.pdf(plt, "/Users/steve/Desktop/dynamics.pdf")

# Bug in Julia Plots.jl misplaces some of the plot titles, which I fixed by hand

############################  Plot multiple phase runs  ##########################
using FitODE_plots, Plots, Measures

# compare phase plots for ODE and NODE for n=3
proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/FitODE/output/";
train_time = "all";						
train = "train_" * train_time * "/";
ofile = "ode-n3-1.jld2";
nfile = "n" * ofile;
dt_o = load_data(proj_output * train * ofile);
dt_n = load_data(proj_output * train * nfile);

plts = [];
push!(plts, plot_phase(dt_o; use_all=true));
push!(plts, plot_phase(dt_n; use_all=true));

plt = plot(plts..., size=(950,800), layout=(1,2))

Plots.pdf(plt, "/Users/steve/Desktop/phase.pdf")

################### Approx Bayes, split training and prediction ##################

using FitODE, FitODE_bayes, FitODE_settings, Plots, StatsPlots

# If reloading data needed
proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/FitODE/output/";
train_time = "60";						# e.g., "all", "60", "75", etc
train = "train_" * train_time * "/"; 	# directory for training period
file = "ode-n2-1.jld2"; 				# fill this in with desired file name
bfile = proj_output * train * "bayes-" * file;
dfile = proj_output * train * file;

dt = load_data(dfile);					# check loaded data vars with keys(dt)

# If calculating approx bayes posterior, start here
# If loading previous calculations, skip to load_bayes()

# for NODE or with ODE with n>=4, try lower a, such as 2e-3 or 1e-3 or lower
# experiment with SGLD parameters, see pSGLD struct in FitODE_bayes
# If first call to psgld_sample gives large loss, may be that gradient
# is small causing large stochastic term, try using pre_λ=1e-1 or other values
B = pSGLD(warmup=5000, sample=10000, a=5e-4, pre_λ=1e-8)

losses, parameters, ks, ks_times = psgld_sample(dt.p, dt.S, dt.L, B)
save_bayes(B, losses, parameters, ks, ks_times; file=bfile);

# If loading previous results from psgld_sample(), skip previous three steps
bt = load_bayes(bfile);					# check loaded data vars with keys(bt)

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

plot_traj_bayes(bt.parameters,dt; samples=20)

#########################  Plot multiple Bayes runs  #############################
using FitODE_bayes, FitODE_plots, Plots, Measures

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/FitODE/output/";
train_time = "60";						# e.g., "all", "60", "75", etc
train = "train_" * train_time * "/"; 	# directory for training period

ode = ["ode-n" * string(i) * "-1.jld2" for i in 2:4];
node = ["n" * x for x in ode];
# interleave vectors a and b by [a b]'[:], but for strings, use permutedims
dfiles = [proj_output * train * file for file in permutedims([ode node])[:]];
bfiles = [proj_output * train * "bayes-" * file for file in permutedims([ode node])[:]];
files = [(dfiles[i],bfiles[i]) for i in 1:length(dfiles)];

plts = []
for ftuple in files
	dt = load_data(ftuple[1])
	bt = load_bayes(ftuple[2])
	push!(plts, plot_traj_bayes(bt.parameters,dt; samples=30,labels=false,
					multi=true, title=true, limits=true))
end
plt = plot(plts..., size=(900,1000), layout=grid(3,2),top_margin=-9mm, bottom_margin=6mm)

Plots.pdf(plt, "/Users/steve/Desktop/bayes.pdf")

# Bug in Julia Plots.jl prints one plot title incorrectly, which I fixed by hand