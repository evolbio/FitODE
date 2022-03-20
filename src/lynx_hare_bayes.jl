using DiffEqFlux, DifferentialEquations, Flux, Plots, StatsPlots
using CSV, DataFrames, Statistics, JLD2

n = 3 				# must be >= 2
activation = tanh 	# activation function for first layer of NN
layer_size = 20		# nodes in each layer of NN
rtol = 1e-2			# relative tolerance for ODE solver
atol = 1e-3			# absolute tolerance for ODE solver
csv_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/input/lynx_hare_data.csv"
out_file = "/Users/steve/Desktop/bayes.jld2"
solver = TRBDF2()

train_end_time = 90.0

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

train_end_index = findall(x->x==train_end_time, tsteps)[1]
y_train = ode_data[1:2,1:train_end_index]

dudt = FastChain(FastDense(n, layer_size, activation), FastDense(layer_size, n))
prob_node = NeuralODE(dudt, tspan, solver, saveat = tsteps,reltol = rtol, abstol = atol)
train_prob = NeuralODE(dudt, (0., train_end_time), solver, saveat = tsteps[1:train_end_index],
				reltol = rtol, abstol = atol)

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
function predict(p, prob, u_init)
  Array(prob(u_init, p))
end

function loss(p, prob, u_init)
	pred = predict(p, prob, u_init)[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	loss = sum(abs2, ode_data[:,1:pred_length] .- pred)
	return loss
end

sgld(∇L, θᵢ, t, a = 2.5e-3, b = 0.05, γ = 0.35) = begin
    ϵ = a*(b + t)^-γ
    η = ϵ.*randn(size(θᵢ))
    Δθᵢ = .5ϵ*∇L + η
    θᵢ .-= Δθᵢ
end

parameters = []
losses = Float64[]
grad_norm = Float64[]

# read parameters
in_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/output/n3-3.jld2"
dt = load(in_file)
θ = dt["p2"] # normally, θ = dt["p3"], but p3 may not work for some inputs
if (length(θ) != length(train_prob.p)) error("Mismatch between input param length and model") end

# get u0, first column of predicted values is u0 
u0 = dt["pred3"][:,1]
if (length(u0) != n) error("Dimension mismatch between n variable and input") end
println("Loss in input = ", dt["loss3"])

println(loss(θ, train_prob, u0))
#@time for t in 1:45000
for t in 44500:45000
	if t % 100 == 0 println("t = " , t) end
	grad = gradient(p -> loss(p, train_prob, u0), θ)[1]
	sgld(grad, θ, t)
	tmp = deepcopy(θ)
	curr_loss = loss(θ, train_prob, u0)
	append!(losses, curr_loss)
	append!(grad_norm, sum(abs2, grad))
	append!(parameters, [tmp])
	println(curr_loss)
end

plot(losses, yscale = :log10)
plot(grad_norm, yscale =:log10)