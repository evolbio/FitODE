# Read in first part of lynx_hare_ode.jl
# Using pSGLD, see Rackauckas22.pdf, Bayesian Neural Ordinary Differential Equations
# and code in https://github.com/RajDandekar/MSML21_BayesianNODE.
# Theory in Li15.pdf, however, had to correct online code examples after reading
# original literature, because examples weighted noise term incorrectly.
# Code for SGLD in https://diffeqflux.sciml.ai/stable/examples/BayesianNODE_SGLD/
# also weights noise incorrectly, weight should be sqrt of ϵ.

# read parameters
in_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/output/ode/n3-1-identity.jld2"
dt = load(in_file)

# train_end_time = 90.0
# train_end_index = findall(x->x==train_end_time, tsteps)[1]
# y_train = ode_data[1:2,1:train_end_index]

#train_prob = NeuralODE(dudt, (0., train_end_time), solver, saveat = tsteps[1:train_end_index],
#				reltol = rtol, abstol = atol)

# Julia code at https://diffeqflux.sciml.ai/stable/examples/BayesianNODE_SGLD/
# may be incorrect, uses ϵ instead of sqrt(ϵ) for η, variance should be ϵ
function sgld(∇L, θᵢ, t; a = 2.5e-3, b = 0.05, γ = 0.35)
    ϵ = a*(b + t)^-γ
    η = sqrt(ϵ).*randn(size(θᵢ))
    Δθᵢ = .5ϵ*∇L + η
    θᵢ .-= Δθᵢ
end

# precondition pSGLD, weight by m, see Li15.pdf, with m=G in their notation
# corrected the bug in https://github.com/RajDandekar/MSML21_BayesianNODE, see above
function p_sgld(∇L, θᵢ, t, m; a = 2.5e-3, b = 0.05, γ = 0.35)
    ϵ = a*(b + t)^-γ
    η = sqrt.(ϵ.*m).*randn(size(θᵢ))
    Δθᵢ = .5ϵ*(∇L.*m) + η
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

sgld_a=1e-1;
sgld_b=1e4;

# setup

beta = 0.9;
λ =1e-8;

# do one update with m=1e-3, because very small initial gradients cause problem
p_sgld(grad, θ, 1, 1e-3.*ones(length(θ)); a=sgld_a, b=sgld_b)

grd = gradient(p -> loss(p, prob, ww, u0)[1], θ)[1];
precond = grd .* grd;

warmup = 20000
total = 40000
for t in 1:total
	if t % 100 == 0 println("t = " , t) end
	grad = gradient(p -> loss(p, prob, ww, u0)[1], θ)[1]
	# precondition gradient, normalizing magnitude in each dimension
	precond *= beta
	precond += (1-beta)*(grad .* grad)
	m = 1 ./ (λ .+ sqrt.(precond))
	p_sgld(grad, θ, t, m; a=sgld_a, b=sgld_b)
	# start collecting statistics after initial warmup period
	if t > warmup
		tmp = deepcopy(θ)
		curr_loss = loss(θ, prob, ww, u0)[1]
		append!(losses, curr_loss)
		append!(grad_norm, sum(abs2, grad))
		append!(parameters, [tmp])
		println(curr_loss)
	else
		println(loss(θ, prob, ww, u0)[1])
	end
end

# save values
jldsave("/Users/steve/Desktop/" * S.start_time * "bayes.jld2"; parameters, losses, grad_norm)

###############################################################################
# Plotting. First check ϵ values of sgld
# Then check on convergence to min and sampling of posterior for loss values
# Goal here is uncertainty for trajectories, so sufficient to look at 
# posterior distribution of loss values to check for sampling convergence

plot([sgld_test(i; a=sgld_a, b=sgld_b) for i=warmup:10:total], yscale=:log10)

plot(losses, yscale = :log10)
plot(grad_norm, yscale =:log10)

using StatsPlots, StatsBase

density(losses)

# compare density between time periods to see if converging
obs = length(pmatrix[:,1])
first_half = Int(floor(obs/2))

density(losses[1:first_half])
density!(losses[first_half+1:end])

plot(autocor(losses,1:100))

###############################################################################
# Look at individual parameters. Typically not necessary for this application
# because goal is to obtain uncertainty estimate for trajectories, not for
# parameters, so looking at loss as above is sufficient

# parameters is a vector with each entry for time as a vector of parameters
# this makes a matrix with rows for time and cols for parameter values
# see https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/14
pmatrix = reduce(hcat,parameters)';

# posterior distn for 8th parameter, choose index as needed
density(pmatrix[:,8])

function plot_posterior(pindex)
	density(pmatrix[1:first_half,pindex])
	density!(pmatrix[first_half+1:end,pindex])
end

# vector with each entry as a vector of autocorr vals 
auto=[autocor(pmatrix[:,i],1:50) for i=1:length(pmatrix[1,:])];
# matrix with each row for parameter and col for autocorr vals
amatrix = reduce(hcat,auto)';

# plot autocorrelations, this shows distn of nth autocorr val over all params
# should be concentrated near zero for lags > 1 or perhaps > small number
# if so, shows that posterior being sampled properly, goal is smallest
# sgld ϵ such that autocorr remains small, allowing best convergence to 
# near minimum cost and yet still good stochastic sampling of posterior
# if small number of parameters, use histogram(), else use density()

# distn for 10th lag over all parameters
histogram(amatrix[:,10])
# autocorr plot for ith parameter
plot(amatrix[3,:])
