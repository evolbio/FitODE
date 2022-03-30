# reduce imports to ones actually needed
using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, 
		Statistics, Distributions, JLD2
using lynx_hare, lynx_hare_settings

# Read comments and code in lynx_hare.jl and lynx_hare_settings.jl
# See lynx_hare_settings for initial setup, some examples:
# One can initialize and then modify settings as follows
# S = default_node()	# default settings for NODE
# S = Settings(S; layer_size=50, activate=3)
# S = default_ode()		# default settings for ODE
# S = Settings(S; opt_dummy_u0 = true)
# See docs for Parameters.jl package

# Use single struct S to hold all settings, access by S.var_name
# Provides simple way to save settings by saving this one variable

# To reset rand seed, make or renew assignment to S, see lynx_hare_settings.jl

S = default_ode()

ode_data, u0, tspan, tsteps, ode_data_orig = lynx_hare.read_data(S);
dudt, ode!, predict = lynx_hare.setup_diffeq_func(S);

beta_a = 1:1:S.wt_steps
if !S.use_node p_init = 0.1*rand(S.nsqr + S.n) end; # n^2 matrix plus n for individual growth
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(S.wt_base^beta_a[i], tsteps; trunc=S.wt_trunc)
	last_time = tsteps[length(w[1,:])]
	ts = tsteps[tsteps .<= last_time]
	# for ODE, will redefine u0 and p in call to solve, so just need right sizes here
	prob = S.use_node ?
				NeuralODE(dudt, (0.0,last_time), S.solver, saveat = ts, 
					reltol = S.rtol, abstol = S.atol) :
				ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0,
					(0.0,last_time), p_init, saveat = ts, reltol = S.rtol, abstol = S.atol)
	# On first time through loop, set up params p for optimization. Following loop
	# turns use the parameters returned from sciml_train(), which are in result.u
	if (i == 1)
		p = S.use_node ? prob.p : p_init
		if S.opt_dummy_u0 p = vcat(randn(S.n-2),p) end
	else
		p = result.u
	end
	result = DiffEqFlux.sciml_train(p -> loss(p,u0,w,prob), p,
					ADAM(S.adm_learn); cb = callback, maxiters=S.max_it)
end

# do additional optimization round with equal weights at all points
# if using large tolerances in initial fit, try reducing tols here
# to get better approach to local minimum
prob = S.use_node ?
			NeuralODE(dudt, tspan, saveat = tsteps, 
				reltol = S.rtol, abstol = S.atol) :
			ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0, tspan,
				p_init, saveat = tsteps, reltol = S.rtol, abstol = S.atol)

ww = ones(2,length(tsteps))
p1 = result.u
lossval = loss(p1,u0,ww,prob);
loss1 = lossval[1]
pred1 = lossval[2]

# Might need more iterates and small adm_learn here if BFGS() below unstable and fails
# Using small adm_learn will allow better convergence to local optimum
result2 = DiffEqFlux.sciml_train(p -> loss(p,u0,ww,prob), result.u,
			ADAM(S.adm_learn); cb = callback, maxiters=S.max_it)

p2 = result2.u
lossval = loss(p2,u0,ww,prob);
loss2 = lossval[1]
pred2 = lossval[2]

# check grad if of interest
grad = gradient(p->loss(p,u0,ww,prob)[1], p2);

# final plot with third dimension and lines, could add additional dims as needed
# by altering callback() code
third = if S.n >= 3 true else false end

callback(p2,loss2,pred2,prob,u0,ww; show_lines=true, show_third=third)

# result3 may throw an error because of instability in BFGS, if
# so then skip this block, purpose of BFGS is to track gradient
# locally to move toward local minimum, BFGS is better at doing
# that than ADAM
result3 = DiffEqFlux.sciml_train(p -> loss(p,u0,ww,prob),p2,BFGS(),
			cb = callback, maxiters=S.max_it)

p3 = result3.u
lossval = loss(p3,u0,ww,prob);
loss3 = lossval[1]
pred3 = lossval[2]

callback(p3,loss3,pred3,prob,u0,ww; show_lines=true, show_third=third)
# save output if result3 via BFGS() successful, otherwise skip to next line
jldsave(S.out_file; S, rseed, p1, loss1, pred1, p2, loss2, pred2, p3, loss3, pred3)
# end skip for result3

# If BFGS and result3 failed, then use this to save output
jldsave(S.out_file; S, rseed, p1, loss1, pred1, p2, loss2, pred2)

# Also, could do fit back to ode_data_orig after fitting to splines or gaussian filter

# Could add code for pruning model (regularization) by adding costs to parameters
# and so reducing model size, perhaps searching for minimally sufficient model

# To view data saved to file:		
# dt = load(S.out_file)
# dt["pred1"] # for prediction output for first set
