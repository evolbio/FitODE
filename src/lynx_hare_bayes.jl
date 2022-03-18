using DiffEqFlux, DifferentialEquations, GalacticOptim, Flux, Plots, StatsPlots
using CSV, DataFrames, Statistics

n = 3 				# must be >= 2
activation = tanh 	# activation function for first layer of NN
layer_size = 20		# nodes in each layer of NN
rtol = 1e-2			# relative tolerance for ODE solver
atol = 1e-3			# absolute tolerance for ODE solver
adm_learn = 0.0005	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 500		# max iterates for each incremental learning step
csv_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/input/lynx_hare_data.csv"
out_file = "/Users/steve/Desktop/output.jld2"
train_end_time = 10.0

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

train_end_index = findall(x->x==train_end_time, tsteps)[1]
y_train = ode_data[1:2,1:train_end_index]

dudt = FastChain(FastDense(n, layer_size, activation), FastDense(layer_size, n))
prob_node = NeuralODE(dudt, tspan, solver, saveat = tsteps)
train_prob = NeuralODE(dudt, (0., train_end_time), solver, saveat = tsteps[1:train_end_index])

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
# u0 for dummy dimensions are first entries of p, allows optimization of dummy u0
function predict(p, prob)
  u_init = vcat(u0,p[1:n-2])
  Array(prob(u_init, p[n-1:end]))
end

function loss(p, prob)
    sum(abs2, y_train .- predict(p, prob)[1:2,:])
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

p_orig = deepcopy(prob_node.p)
θ = vcat(randn(n-2),p_orig)
#@time for t in 1:45000
@time for t in 1:2000
	if t % 100 == 0 println("t = " , t) end
    grad = gradient(p -> loss(p, train_prob), θ)[1]
    sgld(grad, θ, t)
    tmp = deepcopy(θ)
    append!(losses, loss(θ, train_prob))
    append!(grad_norm, sum(abs2, grad))
    append!(parameters, [tmp])
    println(loss(θ, train_prob))
end

plot(losses, yscale = :log10)
plot(grad_norm, yscale =:log10)