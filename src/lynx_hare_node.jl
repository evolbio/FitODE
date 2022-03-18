using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics, Distributions, JLD2, Dates

# Data for 2D hare and lynx. n=3 works well, perhaps 2D data sit in
# 3D manifold?? n=4 gives a better fit. Maybe higher n would be 
# better, but whether that is valuable depends on goal of fitting.

# Occasionally system will lock into local minimum that is clearly not
# a good fit. Rerun program, which seeds with different initial
# parameters and will typically avoid the same local minimum.

# Goal is to find one good fit. So possible need to do a few
# runs with different initial parameter seeds is not a difficulty.
# Program runs reasonable fast on desktop computer, depending
# on parameters usually about 30 minutes per run on my desktop.

# If trouble converging, try varying adm_learn, the learning rate.

# Some initial parameters lead to instabilities in solving ODE. Either
# restart to get different initial seeding parameters or change solver
# to stiff ODE algorithm, see solver variable below.

# number of variables to track in NODE, first two are hare and lynx
n = 3 				# must be >= 2
activation = tanh 	# activation function for first layer of NN
layer_size = 20		# nodes in each layer of NN
wt_trunc = 1e-2		# truncation for weights
rtol = 1e-2			# relative tolerance for ODE solver
atol = 1e-3			# absolute tolerance for ODE solver
adm_learn = 0.0005	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 500		# max iterates for each incremental learning step
csv_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/input/lynx_hare_data.csv"
out_file = "/Users/steve/Desktop/output.jld2"

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

wt_base = 1.1		# good default is 1.1
wt_steps = Int(ceil(log(500)/log(wt_base)))

# ODE solver, Tsit5() for nonstiff and fastest, but may be unstable.
# Alternatively use stiff solver TRBDF2(), slower but more stable.
# For smaller tolerances, if unstable try KenCarp4().
# In this case, likely oscillatory dynamics cause the difficulty.

# Might need higer adm_learn parameter with stiff solvers, which are
# more likely to get trapped in local minimum. Not sure why. 
# Maybe gradient through solvers differ significantly ??
# Or maybe the stiff solvers provide less error fluctuation
# and so need greater learning momentum to shake out of local minima ??

solver = TRBDF2()

use_splines = true
# if using splines, increase data to pts per year by interpolation
pts = 2
 
df = CSV.read(csv_file, DataFrame);
ode_data = permutedims(Array{Float32}(df[:,2:3]));

# take log and then normalize by average
ode_data = log.(ode_data)
ode_data = ode_data .- mean(ode_data)

datasize = length(ode_data[1,:]) # Number of time points
tspan = (0.0f0, Float32(datasize-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split to equal steps

# fit spline to data and use fitted data to train
if use_splines
	using Interpolations
	hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
	lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
	#plot((0:0.1:90),hspline.(0:0.1:90)); plot!((0:0.1:90),lspline.(0:0.1:90))
	tsteps = range(tspan[1], tspan[2], length = 1+pts*(datasize-1)) # Split to equal steps
	ode_data_orig = ode_data	# store original data
	ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
end

u0 = ode_data[:,1] # Initial condition, first time point in data
# add additional initial values for dummy dimensions in predict_neuralode()

# Make a neural net with a NeuralODE layer
dudt2 = FastChain(FastDense(n, layer_size, activation), FastDense(layer_size, n))

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
# u0 for dummy dimensions are first entries of p, allows optimization of dummy u0
function predict_neuralode(p, prob)
  u_init = vcat(u0,p[1:n-2])
  Array(prob(u_init, p[n-1:end]))
end

callback = function (p, l, pred; doplot = true, show_lines = false, show_third = false)
  display(l)
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

function loss(p, prob, w)
	pred_all = predict_neuralode(p, prob)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred_all
end

function weights(a; b=10, trunc=1e-4) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

# Use Beta cdf weights for iterative fitting. Fits earlier parts of time
# series first with declining weights for later data points, then 
# keep fitted parameters and redo, slightly increasing weights for
# later time points. Using smaller value for x in x^beta_a[i] and
# greater top number in iterate range will improve fitting but will
# require more computation.

beta_a = 1:1:wt_steps
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(wt_base^beta_a[i]; trunc=wt_trunc)
	last_time = tsteps[length(w[1,:])]
	prob = NeuralODE(dudt2, (0.0,last_time), solver,
					saveat = tsteps[tsteps .<= last_time],reltol = rtol, abstol = atol)
	# increase p length by adding u0 for dummy dimensions
	p = if (i == 1) vcat(randn(n-2),prob.p) else result.u end
	result = DiffEqFlux.sciml_train(p -> loss(p,prob,w), p,
					ADAM(adm_learn); cb = callback, maxiters=max_it)
end

# do additional optimization round with equal weights at all points
prob = NeuralODE(dudt2, tspan, solver, saveat = tsteps, reltol = rtol, abstol = atol)
ww = ones(2,length(tsteps))
p1 = result.u
lossval = loss(p1,prob,ww)
loss1 = lossval[1]
pred1 = lossval[2]

result2 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww), result.u, ADAM(adm_learn);
			cb = callback, maxiters=max_it)

p2 = result2.u
lossval = loss(p2,prob,ww)
loss2 = lossval[1]
pred2 = lossval[2]

result3 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww),p2,BFGS(),
			cb = callback, maxiters=max_it)

p3 = result3.u
lossval = loss(p3,prob,ww)
loss3 = lossval[1]
pred3 = lossval[2]

# final plot with third dimension and lines
third = if n >= 3 true else false end
callback(p3,loss3,pred3; show_lines=true, show_third=third)

# out_file = Dates.format(now(),"yyyymmdd_HHMM") * ".jld2"
jldsave(out_file; p1, loss1, pred1, p2, loss2, pred2, p3, loss3, pred3)

# Also, could do fit back to ode_data_orig after fitting to splines
			
# dt = load(out_file)
# dt["pred1"] # for prediction data for first set
