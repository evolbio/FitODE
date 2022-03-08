using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames

# number of variables to track in ODE, first two are hare and lynx
n = 10 # must be >= 2

df = CSV.read("/Users/steve/Desktop/current/hare_lynx_data.csv", DataFrame);
ode_data = permutedims(Array{Float32}(df[:,2:3]));

u0 = ode_data[:,1] # Initial condition, first time point in data
u0 = vcat(u0,rand(Float32,n-2)*30)

datasize = length(ode_data[1,:]) # Number of time points
tspan = (0.0f0, Float32(datasize-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps

# Make a neural net with a NeuralODE layer
dudt2 = FastChain(FastDense(n, 50, tanh), FastDense(50, n))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
	pred = predict_neuralode(p)[1:2,:] # First two rows are lynx & hare, others are dummies
	loss = sum(abs2, ode_data .- pred) # Just sum of squared error
	return loss, pred
end

callback = function (p, l, pred; doplot = true)
  display(l)
  # plot current prediction against data
  plt = scatter(tsteps, ode_data[1,:], label = "data")
  scatter!(plt, tsteps, pred[1,:], label = "prediction")
  if doplot
    display(plot(plt))
  end
  return false
end

# Input parameters are prob_neuralode.p, return parameters in result.u
result = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, cb = callback)

