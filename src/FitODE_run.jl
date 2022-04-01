using FitODE_settings, FitODE, JLD2

# Loading imported modules can take a couple of minutes

# First call to a function causes compiling and optimization, which causes a delay
# Subsequent call to a function with different argument types triggers new compile

# Read comments and code in FitODE.jl and FitODE_settings.jl

# One can initialize and then modify settings as follows
# S = default_node()	# default settings for NODE
# S = Settings(S; layer_size=50, activate=3 [, ADD OTHER OPTIONS AS NEEDED])
# S = default_ode()		# default settings for ODE
# S = Settings(S; opt_dummy_u0 = true [, ADD OTHER OPTIONS AS NEEDED])
# See docs for Parameters.jl package

# Use single struct S to hold all settings, access by S.var_name
# Provides simple way to access and save settings via one variable
# Redefinitions in struct S in FitODE_settings.jl required to run on new system

# To reset rand seed: setup S, then reset_rseed(S, rseed).

########################### Fitting full time series ##########################

S = default_ode();
S = reset_rseed(S, 0xe5b0652b110a89b1)

# L is struct that includes u0, ode_data, tsteps, see struct loss_args for other parts
p_opt1,L = fit_diffeq(S)

# bfgs sometimes fails, if so then use p_opt1
# see definition of refine_fit() for other options to refine fit
p_opt2=refine_fit_bfgs(p_opt1,S,L)

loss1, _, _, pred1 = loss(p_opt1,S,L);		# use if p_opt2 fails or of interest
loss2, _, _, pred2 = loss(p_opt2,S,L);

use_2 = true;	# set to false if p_opt2 fails

p, loss_v, pred = use_2 ? (p_opt2, loss2, pred2) : (p_opt1, loss1, pred1);

# if gradient is of interest
grad = calc_gradient(p,S,L)
gnorm = sqrt(sum(abs2, grad))

# save results

save_data(p, S, L, loss_v, pred; file=S.out_file)

# To view data saved to file:		
# dt = load(S.out_file) # or load("file_path")
# dt["pred"] # for prediction data

# various plots


################### Quasi-Bayes, split training and prediction ##################


###################################################################

# final plot with third dimension and lines, could add additional dims as needed
# by altering callback() code
third = if S.n >= 3 true else false end

callback(p2,loss2,pred2,prob,u0,ww; show_lines=true, show_third=third)

callback(p3,loss3,pred3,prob,u0,ww; show_lines=true, show_third=third)
