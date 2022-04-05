module FitODE_bayes
using FitODE, StatsPlots, StatsBase, HypothesisTests, Printf, JLD2, Parameters
export psgld_sample, save_bayes, load_bayes, plot_loss_bayes, plot_sgld_epsilon,
			plot_autocorr, plot_moving_ave, p_matrix, p_ts, auto_matrix, plot_traj_bayes,
			plot_autocorr_hist, pSGLD
			
@with_kw struct pSGLD
	warmup =	2000
	sample =	5000
	a =			5e-3
	b =			1e4
	g =			0.35
	pre_beta =	0.9
	pre_λ =		1e-8
end

# pSGLD, see Rackauckas22.pdf, Bayesian Neural Ordinary Differential Equations
# and theory in Li et al. 2015 (arXiv:1512.07666v1)

# PUT TRAINING SETUP IN SETTINGS
# train_end_time = 90.0
# train_end_index = findall(x->x==train_end_time, tsteps)[1]
# y_train = ode_data[1:2,1:train_end_index]
#train_prob = NeuralODE(dudt, (0., train_end_time), solver, saveat = tsteps[1:train_end_index],
#				reltol = rtol, abstol = atol)

# Use p_sgld() below, however, useful to show original sgld() for background
# Code for SGLD in https://diffeqflux.sciml.ai/stable/examples/BayesianNODE_SGLD/
# However, that code incorrectly uses ϵ instead of sqrt(ϵ) for η, variance should be ϵ
function sgld(∇L, p, t; a = 2.5e-3, b = 0.05, g = 0.35)
    ϵ = a*(b + t)^-g
    η = sqrt(ϵ).*randn(size(p))
    Δp = .5ϵ*∇L + η
    p .-= Δp
end

# precondition SGLD (pSGLD), weight by m, see Li et al. 2015, with m=G in their notation
# Corrected bug in https://github.com/RajDandekar/MSML21_BayesianNODE, they failed
# to weight noise by sqrt.(ϵ.*m)
function p_sgld(∇L, p, t, m; a = 2.5e-3, b = 0.05, g = 0.35)
    ϵ = a*(b + t)^-g
    η = sqrt.(ϵ.*m).*randn(size(p))
    Δp = .5ϵ*(∇L.*m) + η
    p .-= Δp
end

# use to test parameters for setting magnitude of ϵ
sgld_test(t; a = 2.5e-3, b = 0.05, g = 0.35) = a*(b + t)^-g

function psgld_sample(p_in, S, L, B::pSGLD)
	
	p = deepcopy(p_in)		# copy to local variable, else p_in changed
	parameters = []
	losses = Float64[]
	ks = Float64[]
	ks_times = Int[]
	
	# initialize moving average for precondition values
	grd = gradient(p -> loss(p, S, L)[1], p)[1]
	precond = grd .* grd
	
	for t in 1:(B.warmup+B.sample)
		if t % 100 == 0 println("t = " , t) end
		grad = gradient(p -> loss(p, S, L)[1], p)[1]
		# precondition gradient, normalizing magnitude in each dimension
		precond *= B.pre_beta
		precond += (1-B.pre_beta)*(grad .* grad)
		m = 1 ./ (B.pre_λ .+ sqrt.(precond))
		p_sgld(grad, p, t, m; a=B.a, b=B.b, g=B.g)
		# start collecting statistics after initial warmup period
		if t > B.warmup
			tmp = deepcopy(p)
			curr_loss = loss(p, S, L)[1]
			append!(losses, curr_loss)
			append!(parameters, [tmp])
			println(curr_loss)
			if t % 100 == 0
				half = Int(floor((t-B.warmup) / 2))
				println("Sample timesteps = ", t - B.warmup)
				first_losses = losses[1:half]
				second_losses = losses[half+1:end]
				ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
				append!(ks, ks_diff.δ)
				append!(ks_times, t-B.warmup)
				# plot
				plt = plot(size=(600,400 * 2), layout=(2,1))
				density!(first_losses, subplot=1, plot_title="KS = "
						* @sprintf("%5.3e", ks_diff.δ) * ", samples per curve = "
						* @sprintf("%d", half), label="1st" )
				density!(second_losses, subplot=1, label="2nd")
				plot!(ks_times, ks, label="ks", subplot=2, legend=nothing)
				display(plt)
			end
		else
			println(loss(p, S, L)[1])
		end
	end
	return losses, parameters, ks, ks_times
end

# Saving and loading results
save_bayes(losses, parameters, ks, ks_times; file="/Users/steve/Desktop/bayes.jld2") =
					jldsave(file; losses, parameters, ks, ks_times)

function load_bayes(file)
	bt = load(file)
	(losses = bt["losses"], parameters = bt["parameters"],
			ks = bt["ks"], ks_times = bt["ks_times"])
end

# utility functions for extracting data

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

# params is a vector with each entry for time as a vector of parameters
# this makes a matrix with rows for time and cols for parameter values
# see https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/14
p_matrix(params) = reduce(hcat,params)' 
p_ts(params, index) = p_matrix(params)[:,index]

# matrix with each row for parameter and col for autocorr vals
function auto_matrix(params, arange=1:50)
	pm = p_matrix(params)
	auto = [autocor(pm[:,i],arange) for i=1:length(pm[1,:])]
	return reduce(hcat,auto)'
end

# plotting functions

plot_sgld_epsilon(end_time, a=1e-1, b=1e4, g=0.35) = 
	display(plot([i for i=1:10:end_time], [sgld_test(i; a=a, b=b, g=g) for i=1:10:end_time], 
		yscale=:log10))

plot_autocorr(data, auto_range=1:100) = display(plot(autocor(data, auto_range)))

plot_autocorr_hist(param, lag) = display(histogram(auto_matrix(param, 1:30)[:,lag]))

plot_moving_ave(data, n) = 
	display(plot(n/2 .+ Array(1:(length(data)-(n-1))),
		[sum(@view data[i:(i+n-1)])/n for i in 1:(length(data)-(n-1))]))
		
function plot_traj_bayes(param, L, S; samples=20)
	plt = plot(size=(600,800), layout=(2,1))
	for i in 1:samples 
		pred = L.predict(param[rand(1:length(param))], L.prob, L.u0)
		plot!(L.tsteps,pred[1,:], color=mma[2], label="", subplot=1)
		plot!(L.tsteps,pred[2,:], color=mma[2], label="", subplot=2)
	end
	plot!(L.tsteps, L.ode_data[1,:], color=mma[1], linewidth=3, label="hare", subplot=1)
	plot!(L.tsteps, L.ode_data[2,:], color=mma[1], linewidth=3, label="lynx", subplot=2)
	display(plt)
end

function plot_loss_bayes(losses; skip_frac=0.0, ks_intervals=10)
	plt = plot(size=(600,800), layout=(2,1))
	start_index = Int(ceil(skip_frac*length(losses)))
	start_index = (start_index==0) ? 1 : start_index
	losses = @view losses[start_index:end]
	if length(losses) < 5*ks_intervals
		println("\nWARNING: number of losses < 5*ks_intervals\n")
	end
	half = Int(floor(length(losses)/2))
	first_losses = @view losses[1:half]
	second_losses = @view losses[half+1:end]
	ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
	density!(first_losses, subplot=1, plot_title="KS = "
			* @sprintf("%5.3e", ks_diff.δ) * ", samples per curve = "
			* @sprintf("%d", half), label="1st" )
	density!(second_losses, subplot=1, label="2nd")
	
	ks_times = Int[]
	ks = []
	for i in 1:(ks_intervals-1)
		last_index = Int(floor(Float64(length(losses)*i)/ks_intervals))
		ks_losses = @view losses[1:last_index]
		half = Int(floor(length(ks_losses)/2))
	 	first_losses = @view ks_losses[1:half]
		second_losses = @view ks_losses[half+1:end]
		ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
		append!(ks, ks_diff.δ)
		append!(ks_times, length(ks_losses))
	end
	# and finally for full data set
	i = ks_intervals
	last_index = Int(floor(Float64(length(losses)*i)/ks_intervals))
	ks_losses = @view losses[1:last_index]
	half = Int(floor(length(ks_losses)/2))
	first_losses = @view ks_losses[1:half]
	second_losses = @view ks_losses[half+1:end]
	ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
	append!(ks, ks_diff.δ)
	append!(ks_times, length(ks_losses))
	plot!(ks_times, ks, subplot=2, legend=nothing)
	display(plt)
end

end # module
