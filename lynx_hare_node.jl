using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics, Distributions

# number of variables to track in ODE, first two are hare and lynx
n = 4 # must be >= 2
use_splines = true
# points per year
pts = if use_splines 2 else 1 end
 
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
u0 = vcat(u0,rand(Float32,n-2)*30)

# Make a neural net with a NeuralODE layer
dudt2 = FastChain(FastDense(n, 50, tanh), FastDense(50, n))
#prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
function predict_neuralode(p, prob)
  Array(prob(u0, p))
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

function loss(p, prob, w)
	pred = predict_neuralode(p, prob)[1:2,:] # First rows are hare & lynx, others dummies
	# bigger exponent gives steeper dropoff at end
	pred_length = length(pred[1,:])
# 	println(pred_length); println(length(w[1,:]))
# 	println(tsteps[length(w[1,:])]); println()
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred
end

function weights(a; b=10, trunc=1e-4) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

beta_a = 1:1:51
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(1.1^beta_a[i]; trunc=1e-2)
	last_time = tsteps[length(w[1,:])]
	prob = NeuralODE(dudt2, (0.0,last_time), # name particular ODE solver 
					saveat = tsteps[tsteps .<= last_time])
	p = if (i == 1) prob.p else result.u end
	result = DiffEqFlux.sciml_train(p -> loss(p,prob,w), p,
					ADAM(0.002); cb = callback, maxiters=500)
end

# do additional BFGS round
prob = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
ww = ones(2,length(tsteps))
result = DiffEqFlux.sciml_train(p -> loss(p,prob,ww), result.u,
					BFGS(); cb = callback, maxiters=500)

