using DiffEqFlux, DifferentialEquations, GalacticOptim, Flux, Plots, StatsPlots
using CSV, DataFrames, Statistics, Distributions

n = 3 				# must be >= 2
activation = tanh 	# activation function for first layer of NN
layer_size = 30		# nodes in each layer of NN
rtol = 1e-2			# relative tolerance for ODE solver
atol = 1e-3			# absolute tolerance for ODE solver
adm_learn = 0.0005	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 500		# max iterates for each incremental learning step
csv_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/input/lynx_hare_data.csv"
out_file = "/Users/steve/Desktop/output.jld2"
solver = TRBDF2()

train_end_time = 10.0

# wt_steps is smallest integer such that wt_base^wt_steps >=500.

wt_base = 1.1		# good default is 1.1
wt_steps = Int(ceil(log(500)/log(wt_base)))
wt_trunc = 1e-2		# truncation for weights

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

# function loss(p, prob)
#     sum(abs2, y_train .- predict(p, prob)[1:2,:])
# end

function loss(p, prob, w)
	pred_all = predict(p, prob)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss
end

sgld(∇L, θᵢ, t, a = 2.5e-3, b = 0.05, γ = 0.35) = begin
    ϵ = a*(b + t)^-γ
    η = ϵ.*randn(size(θᵢ))
    Δθᵢ = .5ϵ*∇L + η
    θᵢ .-= Δθᵢ
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

parameters = []
losses = Float64[]
grad_norm = Float64[]

p_orig = deepcopy(prob_node.p)type
θ = vcat(randn(n-2),p_orig)

#@time for t in 1:45000
for i in 1:length(beta_a)
	println(beta_a[i])
	w = weights(wt_base^beta_a[i]; trunc=wt_trunc)
	steps = length(w[1,:])
	last_time = tsteps[steps]
	train_prob = NeuralODE(dudt, (0., last_time), solver, saveat = tsteps[1:steps])
	for t in 1:1000
		if t % 100 == 0 println("t = " , t) end
		grad = gradient(p -> loss(p, train_prob, w), θ)[1]
		sgld(grad, θ, t)
		tmp = deepcopy(θ)
		curr_loss = loss(θ, train_prob, w)
		append!(losses, curr_loss)
		append!(grad_norm, sum(abs2, grad))
		append!(parameters, [tmp])
		println(curr_loss)
	end
end

# @time for t in 1:45000
# for t in 1:2000
# 	if t % 100 == 0 println("t = " , t) end
#     grad = gradient(p -> loss(p, train_prob), θ)[1]
#     sgld(grad, θ, t)
#     tmp = deepcopy(θ)
#     append!(losses, loss(θ, train_prob))
#     append!(grad_norm, sum(abs2, grad))
#     append!(parameters, [tmp])
#     println(loss(θ, train_prob))
# end

plot(losses, yscale = :log10)
plot(grad_norm, yscale =:log10)