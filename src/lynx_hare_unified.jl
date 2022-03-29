using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics, Distributions, JLD2, Dates, Random, Printf

# Combines ODE and NODE into single code base, with options to switch
# between ODE and NODE. Also provides switch to allow fitting of of initial
# conditions for extra dummy variable dimensions, ie, more variables in
# dynamics than in data, option opt_dummy_u0

# It may be that optimizing dummy vars sets the value when fitting for
# the initial timesteps and then gets fixed there, causing trap
# in local minimum. With random initial value (not optimized), sometimes
# better or worse but over repeated trials may be better than optimizing
# on first values.

# For lynx and hare data, start by log transform. Thus, linear (affine) ODE
# with "bias" term can be interpreted as LV ODE with extra squared term for
# each "species". Data for n=2 2D lynx and hare fit better by using 3 or
# dimensions, ie, with extra 1 or 2 dummy variables. For NODE, no simple
# interpretation of terms. For NODE, can also use dummy variables, typically
# n=3 or 4 gives better fit than n=2. Perhaps 2D data sit in 3D manifold??

# Occasionally system will lock into local minimum that is clearly not
# a good fit. Rerun program, which seeds with different initial
# parameters and will typically avoid the same local minimum.

# Goal is to find one good fit. So possible need to do a few
# runs with different initial parameter seeds is not a difficulty.
# Program runs reasonably fast on desktop computer, depending on
# hyerparameters.

# If trouble converging, try varying adm_learn, the learning rate, the 
# solver tolerances, and the solver algorithm. If instabilities, lower
# tolerances and use "stiff" ODE solver

now_name = Dates.format(now(),"yyyymmdd_HHMMSS")
proj_dir = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare"

###################### Start of settings section ######################

# Use single named tuple S to hold all settings, access by S.var

S = (

use_node = false,	# switch between NODE and ODE
layer_size = 20,	# size of layers for NODE

# number of variables to track in (N)ODE, first two are hare and lynx
n = 3, 					# must be >= 2, number dummy variables is n-2
opt_dummy_u0 = true,	# optimize dummy init values instead of using rand values

# Larger tolerances are faster but errors make gradient descent more challenging
# However, fit is sensitive to tolerances, seems NODE may benefit from fluctuations
# with larger tolerances whereas ODE needs smoother gradient from smaller tolerances??
rtol = 1e-10,		# relative tolerance for solver, ODE -> ~1e-10, NODE -> ~1e-2 or -3
atol = 1e-12,		# absolute tolerance for solver, ODE -> ~1e-12, NODE -> ~1e-3 or -4
adm_learn = 0.0005,	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 200,		# max iterates for each incremental learning step
print_grad = true,	# show gradient on terminal, requires significant overhead

start_time = now_name,
csv_file = "$proj_dir/input/lynx_hare_data.csv",
out_file = "/Users/steve/Desktop/" * now_name * ".jld2",

git_vers = chomp(read(`git -C $proj_dir rev-parse --short HEAD`,String)),

# if true, program generates new random seed, otherwise uses rand_seed
# these defaults can be overridden by args to set_rand_seed()
generate_rand_seed = true,
rand_seed = 0x0861a3ea66cd3e9a,

# Training done iteratively, starting with first part of time series,
# then adding additional time series steps and continuing the fit
# process. For each step, the time points are weighted according
# to a 1 - cdf(Beta distribution). To move the weights to the right
# (later times), the first parameter of the beta distribution is
# increased repeatedly over i = 1..wt_steps, with the parameter
# equal to wt_base^i.

# Smaller values of wt_base move the weighting increments at a 
# slower pace and require more increments and longer run time, 
# but may gain by avoiding the common local minimum close to 
# a simple regression line through the fluctuating time series.

# wt_steps is smallest integer such that wt_base^wt_steps >=500.
wt_base = 1.1,		# good default is 1.1
wt_trunc = 1e-2,	# truncation for weights

# would be worthwhile to experiment with various solvers
# see https://diffeq.sciml.ai/stable/solvers/ode_solve/

# ODE solver, Tsit5() for nonstiff and fastest, but may be unstable.
# Alternatively use stiff solver TRBDF2(), slower but more stable.
# For smaller tolerances, if unstable try Rodas4P().
# Likely it is oscillatory dynamics that cause the difficulty.

# Might need higer adm_learn parameter with stiff solvers, which are
# more likely to get trapped in local minimum. 
# Maybe gradient through solvers differs ??
# Or maybe the stiff solvers provide less error fluctuation
# and so need greater learning momentum to shake out of local minima ??

solver = Rodas4P(), 

# Activation function:
# tanh seems to give good fit, perhaps best fit, and maybe overfit
# however, gradient does not decline near best fit, may limit methods that
# require small gradient near fixed point; identity fit not as good but
# gradient declines properly near local optimum with BFGS()

activate = 1, # use one of 1 => identity, 2 => tanh, 3 => sigmoid, 4 => swish

use_splines = true,
# if using splines, increase data to pts per year by interpolation
pts = 2,
use_gauss_filter = true, # smooth the splines by gauss filtering
filter_sd = 1.2
)# end of named tuple of settings

###################### End of settings values ######################

wt_steps = Int(ceil(log(500)/log(S.wt_base)))
nsqr = S.n*S.n

###################### End of setting setup #######################

function set_rand_seed(gen_rand_seed=S.generate_rand_seed, new_seed_val=S.rand_seed)
	global rseed = gen_rand_seed ? rand(UInt) : new_seed_val
	Random.seed!(rseed)
	println("Random seed = ", rseed)
end

df = CSV.read(S.csv_file, DataFrame);
ode_data = permutedims(Array{Float64}(df[:,2:3]));

# take log and then normalize by average
ode_data = log.(ode_data)
ode_data = ode_data .- mean(ode_data)

datasize = length(ode_data[1,:]) # Number of time points
tspan = (0.0f0, Float64(datasize-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split to equal steps

# fit spline to data and use fitted data to train
if S.use_splines
	using Interpolations
	ode_data_orig = ode_data	# store original data
	hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
	lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
	tsteps = range(tspan[1], tspan[2], length = 1+S.pts*(datasize-1)) # Split to equal steps
	if S.use_gauss_filter
		using QuadGK
		conv(y, spline, sd) = 
			quadgk(x -> spline(x) * pdf(Normal(0,sd),y-x), tspan[1], tspan[2])[1]
		ode_data = vcat([conv(y,hspline,S.filter_sd) for y=tsteps]',
							[conv(y,lspline,S.filter_sd) for y=tsteps]');
	else
		ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
	end
	#plot(tsteps,ode_data[1,:]); plot!(tsteps,ode_data[2,:])
end

u0 = ode_data[:,1] # Initial condition, first time point in data
if (S.opt_dummy_u0 == false) u0 = vcat(u0,randn(Float64,S.n-2)) end

activate = if (S.activate == 1) identity elseif (S.activate == 2) tanh
					elseif (S.activate == 3) sigmoid else swish end

# If optimizing initial conditions for dummy dimensions, then use parameters p 
# starting at initial condition u0, ie, u0 for dummy dimensions are first entries
# of p
if S.use_node
	dudt = FastChain(FastDense(S.n, S.layer_size, activate), FastDense(S.layer_size, S.n))
	if S.opt_dummy_u0
		function predict(p, prob, u_init)
		  u_new = vcat(u_init,p[1:S.n-2])
		  Array(prob(u_new, p[S.n-1:end]))
		end
	else
		predict(p, prob, u_init) = Array(prob(u_init, p))
	end
else # ode instead of node, resove u_init and p in prob arg
	function ode!(du, u, p, t)
		s = reshape(p[1:nsqr], S.n, S.n)
		du .= activate.(s*u .- p[nsqr+1:end])
	end
	if S.opt_dummy_u0
		predict(p, prob, u_init) =
			solve(prob, S.solver, u0=vcat(u_init,p[1:S.n-2]), p=p[S.n-1:end])
	else
		predict(p, prob, u_init) = solve(prob, S.solver, p=p)
	end
end

function callback(p, l, pred, prob, u_init, w;
						doplot = true, show_lines = false, show_third = false)
  if (S.print_grad)
  	grad = gradient(p->loss(p,u_init,w,prob)[1], p)[1]
  	gnorm = sqrt(sum(abs2, grad))
  	println(@sprintf("%5.3e; %5.3e", l, gnorm))
  else
  	display(l)
  end
  # plot current prediction against data
  len = length(pred[1,:])
  ts = tsteps[1:len]
  ysize = if show_third 1200 else 800 end
  panels = if show_third 3 else 2 end
  plt = plot(size=(600,ysize), layout=(panels,1))
  plot_type! = if show_lines plot! else scatter! end
  plot_type!(ts, ode_data[1,1:len], label = "hare", subplot=1)
  plot_type!(plt, ts, pred[1,:], label = "pred", subplot=1)
  plot_type!(ts, ode_data[2,1:len], label = "lynx", subplot=2)
  plot_type!(plt, ts, pred[2,:], label = "pred", subplot=2)
  if show_third
  	plot_type!(plt, ts, pred[3,:], label = "3rdD", subplot=3)
  end
  if doplot
    display(plot(plt))
  end
  return false
end

function loss(p, u_init, w, prob)
	pred_all = predict(p, prob, u_init)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred_all, prob, u_init, w
end

function weights(a; b=10, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

beta_a = 1:1:wt_steps
set_rand_seed();
if !S.use_node p_init = 0.1*rand(nsqr + S.n) end; # n^2 matrix plus n for individual growth
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(S.wt_base^beta_a[i]; trunc=S.wt_trunc)
	last_time = tsteps[length(w[1,:])]
	ts = tsteps[tsteps .<= last_time]
	# for ODE, will redefine u0 and p in call to solve, so just need right sizes here
	prob = S.use_node ?
				NeuralODE(dudt, (0.0,last_time), S.solver, saveat = ts, 
					reltol = S.rtol, abstol = S.atol) :
				ODEProblem(ode!, u0, (0.0,last_time), p_init,
					saveat = ts, reltol = S.rtol, abstol = S.atol)
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
prob = S.use_node ?
			NeuralODE(dudt, tspan, saveat = tsteps, 
				reltol = S.rtol, abstol = S.atol) :
			ODEProblem(ode!, u0, tspan, p_init,
				saveat = tsteps, reltol = S.rtol, abstol = S.atol)

ww = ones(2,length(tsteps))
p1 = result.u
lossval = loss(p1,u0,ww,prob);
loss1 = lossval[1]
pred1 = lossval[2]

result2 = DiffEqFlux.sciml_train(p -> loss(p,u0,ww,prob), result.u,
			ADAM(S.adm_learn); cb = callback, maxiters=S.max_it)

p2 = result2.u
lossval = loss(p2,u0,ww,prob);
loss2 = lossval[1]
pred2 = lossval[2]

# check grad if of interest
grad = gradient(p->loss(p,u0,ww,prob)[1], p2);

# final plot with third dimension and lines
third = if S.n >= 3 true else false end

callback(p2,loss2,pred2,prob,u0,ww; show_lines=true, show_third=third)

result3 = DiffEqFlux.sciml_train(p -> loss(p,u0,ww,prob),p2,BFGS(),
			cb = callback, maxiters=S.max_it)

p3 = result3.u
lossval = loss(p3,u0,ww,prob);
loss3 = lossval[1]
pred3 = lossval[2]

# result3 may throw an error because of instability in BFGS, if
# so then skip this
callback(p3,loss3,pred3,prob,u0,ww; show_lines=true, show_third=third)

jldsave(S.out_file; S, rseed, p1, loss1, pred1, p2, loss2, pred2, p3, loss3, pred3)

# Also, could do fit back to ode_data_orig after fitting to splines or gaussian filter

# Could add code for pruning model (regularization) by adding costs to parameters
# and so reducing model size, perhaps searching for minimally sufficient model
			
# dt = load(S.out_file)
# dt["pred1"] # for prediction data for first set
