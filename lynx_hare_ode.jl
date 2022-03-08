using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames, Statistics
# using Interpolations

# number of variables to track in ODE, first two are hare and lynx
n = 5 # must be >= 2
nsqr = n*n
# n^2 matrix for pairwise interactions plus vector n for individual growth
p = 0.1*rand(nsqr + n);

df = CSV.read("/Users/steve/Desktop/current/hare_lynx_data.csv", DataFrame);
ode_data = permutedims(Array{Float32}(df[:,2:3]));
# take log and then normalize by average
ode_data = log.(ode_data)
ode_data = ode_data .- mean(ode_data)

u0 = ode_data[:,1] # Initial condition, first time point in data
u0 = vcat(u0,rand(Float32,n-2)*30)

datasize = length(ode_data[1,:]) # Number of time points
tspan = (0.0f0, Float32(datasize-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps

swish(x) = x ./ (exp.(-x) .+ 1.0)

# input vector length S
function ode!(du, u, p, t)
	w = reshape(p[1:nsqr], n, n)
	#du .= swish(w*u .- p[nsqr+1:end])
	du .= swish(w*u) .- p[nsqr+1:end]
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
  pred = solve(prob, p=p, saveat = tsteps[tsteps .<= i], maxiters=100000)
  # use first two variables to match lynx & hare data
  # other variables are there to help the fitting
  loss = sum(abs2, pred[1:2,:] .- ode_data[:,1:length(pred[1,:])])
  return loss, pred, i
end

#for i in 9.0:9.0:18.0
for i in 6.0:6.0:90.0
	println(i)
	#display(p)
	prob = ODEProblem(ode!, u0, (0.0,i), p)
	result = DiffEqFlux.sciml_train(p -> loss(p, i, prob), p, ADAM(0.02),
		cb = callback, maxiters = 2000)
	p = result.u
end

##############################################################
# train in one round over all time

# prob = ODEProblem(ode!, u0, tspan, p)

# function loss(p)
#   pred = solve(prob, Tsit5(), p=p, saveat = tsteps, maxiters=4000)
#   # use first two variables to match lynx & hare data
#   # other variables are there to help the fitting
#   loss = sum(abs2, pred[1:2,:] .- ode_data)
#   return loss, pred
# end

# result = DiffEqFlux.sciml_train(p -> loss(w(p,n)), p; cb = callback)

