using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames, Statistics

use_splines = true

# number of variables to track in ODE, first two are hare and lynx
n = 5 # must be >= 2
nsqr = n*n
# n^2 matrix for pairwise interactions plus vector n for individual growth
p = 0.1*rand(nsqr + n);

df = CSV.read("/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/lynx_hare_data.csv",
					DataFrame);
ode_data = permutedims(Array{Float32}(df[:,2:3]));
# take log and then normalize by average
ode_data = log.(ode_data)
ode_data = ode_data .- mean(ode_data)

datasize = length(ode_data[1,:]) # Number of time points
tspan = (0.0f0, Float32(datasize-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = 2*datasize) # Split time range into equal steps

# spline
if use_splines
	using Interpolations
	hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
	lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
	#plot((0:0.1:90),hspline.(0:0.1:90)); plot!((0:0.1:90),lspline.(0:0.1:90))
	ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
end

u0 = ode_data[:,1] # Initial condition, first time point in data
u0 = vcat(u0,randn(Float32,n-2))

swish(x) = x ./ (exp.(-x) .+ 1.0)
sigmoid(x) = 1.0 ./ (exp.(-x) .+ 1.0)

# input vector length S
function ode!(du, u, p, t)
	w = reshape(p[1:nsqr], n, n)
	du .= sigmoid(w*u .- p[nsqr+1:end])
	#du .= swish(w*u .- p[nsqr+1:end])
	#du .= swish(w*u) .- p[nsqr+1:end]
end

callback = function (p, l, pred, i; doplot = true)
  display(l)
  # plot current prediction against data
  ts = tsteps[tsteps .<= i]
  plt = scatter(ts, ode_data[1,:], label = "data")
  scatter!(plt, ts, pred[1,:], label = "prediction")
  if doplot
    display(plot(plt))
  end
  return false
end

# try training in parts, following https://diffeqflux.sciml.ai/dev/examples/local_minima/
function loss(p, i, prob)
  #pred = solve(prob, Tsit5(), p=p, saveat = tsteps[tsteps .<= i], maxiters=100000)
  # see https://diffeqflux.sciml.ai/dev/ControllingAdjoints/ for sensealg
  pred = solve(prob, p=p, saveat = tsteps[tsteps .<= i], maxiters=300000,
  				abstol=1e-4, reltol=1e-2)
  # use first two variables to match lynx & hare data
  # other variables are there to help the fitting
  loss = sum(abs2, pred[1:2,:] .- ode_data[:,1:length(pred[1,:])])
  return loss, pred, i
end

#for i in 9.0:9.0:18.0
p = 0.1*rand(nsqr + n);
for i in 3.0:3.0:90.0
	println(i)
	#display(p)
	prob = ODEProblem(ode!, u0, (0.0,i), p)
	result = DiffEqFlux.sciml_train(p -> loss(p, i, prob), p, ADAM(0.02),
		cb = callback, maxiters = 1000, abstol=1e-4, reltol=1e-2)
	p = result.u
end
