using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics

# number of variables to track in ODE, first two are hare and lynx
n = 10 # must be >= 2
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

function loss_neuralode(p, i, prob, incr)
	pred = predict_neuralode(p,prob)[1:2,:] # First rows  hare & lynx, others are dummies
	pred_length = length(pred[1,:])
	incr_steps = Int(incr*pts)
	old_length = pred_length-incr_steps
	pred_old = @view pred[:,1:old_length]
	pred_new = @view pred[:,1+old_length:pred_length]
	data_old = @view ode_data[:,1:old_length]
	data_new = @view ode_data[:,1+old_length:pred_length]
	loss_old = sum(abs2, data_old .- pred_old)
	loss_new = sum(abs2, data_new .- pred_new)
	loss = 10.0 * loss_old + loss_new
	return loss, pred, i
end


# function loss_neuralode(p, i, prob, incr)
# 	pred = predict_neuralode(p, prob)[1:2,:] # First rows are hare & lynx, others dummies
# 	# bigger exponent gives steeper dropoff at end
# 	pred_length = length(pred[1,:])
# 	w = [0.01 + 1-(x/pred_length)^3 for x = 1:pred_length]'
# 	weight = vcat(w,w)
# 	loss = sum(abs2, weight .* (ode_data[:,1:pred_length] .- pred))
# 	return loss, pred, i
# end

callback = function (p, l, pred, i; doplot = true)
  display(l)
  # plot current prediction against data
  ts = tsteps[tsteps .<= i]
  plt = plot(size=(600,800), layout=(2,1))
  scatter!(ts, ode_data[1,:], label = "hare", subplot=1)
  scatter!(plt, ts, pred[1,:], label = "pred", subplot=1)
  scatter!(ts, ode_data[2,:], label = "lynx", subplot=2)
  scatter!(plt, ts, pred[2,:], label = "pred", subplot=2)
  if doplot
    display(plot(plt))
  end
  return false
end

# Input parameters are prob_neuralode.p, return parameters in result.u
#i = 30
#result = DiffEqFlux.sciml_train(p -> loss_neuralode(p,i), prob_neuralode.p, cb = callback)

incr = 30.0
for i in incr:incr:90.0
	global result
	println(i)
	prob = NeuralODE(dudt2, (0.0,i), Tsit5(), saveat = tsteps[tsteps .<= i])
	p = if (i == incr) prob.p else result.u end
	result = DiffEqFlux.sciml_train(p -> loss_neuralode(p,i,prob,incr), p,
					ADAM(0.02); cb = callback, maxiters=300)
end
