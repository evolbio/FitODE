# Read in first part of lynx_hare_ode.jl

# read parameters
in_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/output/ode/n3-1-identity.jld2"
dt = load(in_file)

# train_end_time = 90.0
# train_end_index = findall(x->x==train_end_time, tsteps)[1]
# y_train = ode_data[1:2,1:train_end_index]

#train_prob = NeuralODE(dudt, (0., train_end_time), solver, saveat = tsteps[1:train_end_index],
#				reltol = rtol, abstol = atol)

sgld(∇L, θᵢ, t; a = 2.5e-3, b = 0.05, γ = 0.35) = begin
    ϵ = a*(b + t)^-γ
    η = ϵ.*randn(size(θᵢ))
    Δθᵢ = .5ϵ*∇L + η
    θᵢ .-= Δθᵢ
end

# use to test parameters for setting magnitude of ϵ
sgld_test(t; a = 2.5e-3, b = 0.05, γ = 0.35) = a*(b + t)^-γ

θ = dt["p3"] # or, θ = dt["p2"], because p3 may not work for some inputs

# get u0, first column of predicted values is u0 
u0 = dt["pred3"][:,1]
prob = ODEProblem(ode!, u0, tspan, θ, saveat = tsteps, reltol = S.rtol, abstol = S.atol)
ww = ones(2,length(tsteps))

if (length(θ) != length(prob.p)) error("Mismatch between input param length and model") end
if (length(u0) != n) error("Dimension mismatch between n variable and input") end

println("Loss in input = ", dt["loss3"]);
println("Calculated loss = ", loss(θ, prob, ww, u0)[1]);

parameters = []
losses = Float64[]
grad_norm = Float64[]

sgld_b=3e5;

#warmup
for t in 1:3000
	if t % 100 == 0 println("t = " , t) end
	grad = gradient(p -> loss(p, prob, ww, u0)[1], θ)[1]
	sgld(grad, θ, t, b=sgld_b)
	println(loss(θ, prob, ww, u0)[1])
end

for t in 3000:5000
	if t % 100 == 0 println("t = " , t) end
	grad = gradient(p -> loss(p, prob, ww, u0)[1], θ)[1]
	sgld(grad, θ, t, b=sgld_b)
	tmp = deepcopy(θ)
	curr_loss = loss(θ, prob, ww, u0)[1]
	append!(losses, curr_loss)
	append!(grad_norm, sum(abs2, grad))
	append!(parameters, [tmp])
	println(curr_loss)
end

# Plotting. First step is to check on convergence to min and sampling of posterior

plot(losses, yscale = :log10)
plot(grad_norm, yscale =:log10)

using StatsPlots
density(losses)

using StatsBase

# parameters is a vector with each entry for time as a vector of parameters
# this makes a matrix with rows for time and cols for parameter values
# see https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/14
pmatrix = reduce(hcat,parameters)';

# vector with each entry as a vector of autocorr vals 
auto=[autocor(pmatrix[:,i],1:10) for i=1:length(pmatrix[1,:])];
# matrix with each row for parameter and col for autocorr vals
amatrix = reduce(hcat,auto)';

# plot autocorrelations, this shows distn of nth autocorr val over all params
# should be concentrated near zero for lags > 1 or perhaps > small number
# if so, shows that posterior being sampled properly, goal is smallest
# sgld ϵ such that autocorr remains small, allowing best convergence to 
# near minimum cost and yet still good stochastic sampling of posterior
# if small number of parameters, use histogram(), else use density()
histogram(amatrix[:,10])
