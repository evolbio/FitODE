module FitODE
using CSV, DataFrames, Statistics, Interpolations, QuadGK, Printf, Flux,
		DifferentialEquations, JLD2, Parameters, Suppressor, Distributions
export calc_pred_loss

@with_kw struct loss_args
	u0
	prob				# problem for training data period only
	predict
	ode_data
	ode_data_orig
	tsteps				# steps for training data period only
	w
end

# When fit only to training data subset, then need full time period info
struct all_time
	prob_all			# problem for full data set	
	tsteps_all			# steps for full data set
end

function read_data(S)
	# Read data and store in matrix ode_data, row 1 for hare, row 2 for lynx
	df = CSV.read(S.csv_file, DataFrame);
	ode_data = permutedims(Array{Float64}(df[:,2:3]));

	# take log and then normalize by average, see DOI: 10.1111/2041-210X.13606
	ode_data = log.(ode_data)
	ode_data = ode_data .- mean(ode_data)
	
	datasize = length(ode_data[1,:]) # Number of time points
	tspan = (0.0f0, Float64(datasize-1)) # Time range
	tsteps = range(tspan[1], tspan[2], length = datasize) # Split to equal steps

	# fit spline to data and use fitted data to train, gauss smoothing is an option
	if S.use_splines
		ode_data_orig = ode_data	# store original data
		hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
		lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
		tsteps = range(tspan[1], tspan[2], length = 1+S.pts*(datasize-1)) # Split to equal steps
		if S.use_gauss_filter
			conv(y, spline, sd) = 
				quadgk(x -> spline(x) * pdf(Normal(0,sd),y-x), tspan[1], tspan[2])[1]
			ode_data = vcat([conv(y,hspline,S.filter_sd) for y=tsteps]',
								[conv(y,lspline,S.filter_sd) for y=tsteps]');
		else
			ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
		end
		#plot(tsteps,ode_data[1,:]); plot!(tsteps,ode_data[2,:])
	end
	
	u0 = ode_data[:,1] # Initial condition, first time point in data
	# if using not optimizing dummy vars, add random init condition for dummy dimensions
	if (S.opt_dummy_u0 == false) u0 = vcat(u0,randn(Float64,S.n-2)) end

	return ode_data, u0, tspan, tsteps, ode_data_orig
end

function setup_diffeq_func(S)
	# activation function to create nonlinearity, identity is no change
	activate = if (S.activate == 1) identity elseif (S.activate == 2) tanh
						elseif (S.activate == 3) sigmoid else swish end
	# If optimizing initial conditions for dummy dimensions, then for initial condition u0,
	# dummy dimensions are first entries of p
	predict_node_dummy(p, prob, u_init) =
	  		Array(prob(vcat(u_init,p[1:S.n-2]), p[S.n-1:end]))
	predict_node_nodummy(p, prob, u_init) = Array(prob(u_init, p))
	predict_ode_dummy(p, prob, u_init) =
			solve(prob, Rodas4P(), u0=vcat(u_init,p[1:S.n-2]), p=p[S.n-1:end])
	predict_ode_nodummy(p, prob, u_init) = solve(prob, Rodas4P(), p=p)

	# For NODE, many simple options to build alternative network architecture, see SciML docs
	if S.use_node
		dudt = Chain(Dense(S.n, S.layer_size, activate), Dense(S.layer_size, S.n))
		ode! = nothing
		predict = S.opt_dummy_u0 ?
			predict_node_dummy :
			predict_node_nodummy
	else
		dudt = nothing
		function ode!(du, u, p, t, n, nsqr)
			s = reshape(p[1:nsqr], n, n)
			du .= activate.(s*u .- p[nsqr+1:end])
		end
		predict = S.opt_dummy_u0 ?
			predict_ode_dummy :
			predict_ode_nodummy
	end
	return dudt, ode!, predict
end

function load_data(file)
	dt_string_keys = @suppress begin load(file) end
	dt_symbol_keys = Dict()
	for (k,v) in dt_string_keys
    	dt_symbol_keys[Symbol(k)] = v
	end
	(; dt_symbol_keys...)
end

function load_bayes(file)
	bt = @suppress begin load(file) end
	(B = bt["B"], losses = bt["losses"], parameters = bt["parameters"],
			ks = bt["ks"], ks_times = bt["ks_times"])
end

function calc_pred_loss(samples=1000)
	proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/FitODE/output/";
	train_time = "60";						# e.g., "all", "60", "75", etc
	train = "train_" * train_time * "/"; 	# directory for training period

	ode = ["ode-n" * string(i) * "-1.jld2" for i in 2:4];
	node = ["n" * x for x in ode];
	# interleave vectors a and b by [a b]'[:], but for strings, use permutedims
	dfiles = [proj_output * train * file for file in permutedims([ode node])[:]];
	bfiles = [proj_output * train * "bayes-" * file for file in permutedims([ode node])[:]];
	files = [(dfiles[i],bfiles[i]) for i in 1:length(dfiles)];

	start_idx = 121							# for timestep 60
	end_idx = 181							# for timestep 90

	for ftuple in files
		dt = load_data(ftuple[1])
		bt = load_bayes(ftuple[2])
		param = bt.parameters
		La = dt.L_all
		ode_data, u0, tspan, _, _ = read_data(dt.S)
		dudt, ode!, _ = setup_diffeq_func(dt.S)
		tsteps = La.tsteps
		p_init = 0.1*rand(dt.S.nsqr + dt.S.n)
		if dt.S.use_node
			p, re = Flux.destructure(dudt)
			node(u, p, t) = re(p)(u)
			prob = ODEProblem(node, Float32.(u0), Float32.(tspan), Float32.(p_init))
		else
			prob = ODEProblem((du, u, p, t) -> ode!(du, u, p, t, dt.S.n, dt.S.nsqr),
						u0, tspan, p_init)
		end
		hare_data = ode_data[1,start_idx:end_idx]
		lynx_data = ode_data[2,start_idx:end_idx]
		losses = zeros(samples)
		for i in 1:samples
			pred = solve(prob, Rodas4P(), p=param[rand(1:length(param))], saveat = La.tsteps, 
							reltol = dt.S.rtolR, abstol = dt.S.atolR)
			losses[i] = sum(abs2,[x[1] for x in pred.u][start_idx:end_idx] .- hare_data)
							+ sum(abs2,[x[2] for x in pred.u][start_idx:end_idx] .- lynx_data)
		end
		@printf("%s%s%s", dt.S.use_node ? "NODE" : " ODE", dt.S.n, ": ")
		@printf("median = %5.3e", median(losses))
		@printf("; mean = %5.3e", mean(losses))
		@printf("; sd = %5.3e\n", std(losses))
	end
	return
end

end # module
