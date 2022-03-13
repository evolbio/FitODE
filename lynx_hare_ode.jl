using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics, Distributions

# LV ODE with extra squared term for each "species". Data for 2D hare
# and lynx. Better fit using n=3 or 4. For n=5 seems to be too high for
# easy fit. NODE with n=3 fits better.

# number of variables to track in ODE, first two are hare and lynx
n = 3 # must be >= 2
nsqr = n*n
wt_trunc = 1e-2		# truncation for weights
rtol = 1e-2			# relative tolerance for ODE solver
atol = 1e-3			# absolute tolerance for ODE solver
adm_learn = 0.0002	# Adam learning rate
max_it = 500		# max iterates for each incremental learning step

use_splines = true
# if using splines, increase in data pts per year by interpolation
pts = 2

df = CSV.read("/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/lynx_hare_data.csv",
					DataFrame);
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
	ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
end

u0 = ode_data[:,1] # Initial condition, first time point in data
u0 = vcat(u0,randn(Float32,n-2))

swish(x) = x ./ (exp.(-x) .+ 1.0)
sigmoid(x) = 1.0 ./ (exp.(-x) .+ 1.0)

# tanh seems to work best
function ode!(du, u, p, t)
	s = reshape(p[1:nsqr], n, n)
	du .= tanh.(s*u .- p[nsqr+1:end])
	#du .= sigmoid(s*u .- p[nsqr+1:end])
	#du .= swish(s*u .- p[nsqr+1:end])
	#du .= s*u .- p[nsqr+1:end]
end

callback = function (p, l, pred; doplot = true)
  display(l)
  # plot current prediction against data
  len = length(pred[1,:])
  ts = tsteps[1:len]
  plt = plot(size=(600,800), layout=(2,1))
  scatter!(ts, ode_data[1,1:len], label = "hare", subplot=1)
  scatter!(plt, ts, pred[1,:], label = "pred", subplot=1)
  scatter!(ts, ode_data[2,1:len], label = "lynx", subplot=2)
  scatter!(plt, ts, pred[2,:], label = "pred", subplot=2)
  if doplot
    display(plot(plt))
  end
  return false
end

function weights(a; b=10, trunc=1e-4) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

function loss(p, prob, w)
	# First rows are hare & lynx, others dummies
	pred = solve(prob, p=p)[1:2,:] 
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred
end

# Use Beta cdf weights for iterative fitting. Fits earlier parts of time
# series first with declining weights for later data points, then 
# keep fitted parameters and redo, slightly increasing weights for
# later time points.

p = 0.1*rand(nsqr + n);	# n^2 matrix plus vector n for individual growth
beta_a = 1:1:51
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(1.1^beta_a[i]; trunc=wt_trunc)
	last_time = tsteps[length(w[1,:])]
	prob = ODEProblem(ode!, u0, tspan, p, saveat = tsteps[tsteps .<= last_time],
					reltol = rtol, abstol = atol)
	p = if (i == 1) prob.p else result.u end
	result = DiffEqFlux.sciml_train(p -> loss(p,prob,w), p,
					ADAM(adm_learn); cb = callback, maxiters=max_it)
end

# do additional optimization round with equal weights at all points
prob = ODEProblem(ode!, u0, tspan, p, saveat = tsteps, reltol = rtol, abstol = atol)
ww = ones(2,length(tsteps))
# change p to result.u
result2 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww), result.u, ADAM(adm_learn);
			cb = callback, maxiters=max_it)

# save variable values
# using JDL2
# @save "filename" variable
# @load "filename" # lists variables
# @load "filename" var_name1 var_name2 ...

