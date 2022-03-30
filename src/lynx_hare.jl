module lynx_hare
using CSV, DataFrames, Statistics, Distributions, Interpolations, QuadGK,
		DifferentialEquations, Printf
export callback, loss, weights

# Combines ODE and NODE into single code base, with options to switch
# between ODE and NODE. Also provides switch to allow fitting of of initial
# conditions for extra dummy variable dimensions, ie, more variables in
# dynamics than in data, option opt_dummy_u0

# It may be that optimizing dummy vars sets the value when fitting for
# the initial timesteps and then gets fixed there, causing trap
# in local minimum. With random initial value (not optimized), sometimes
# better or worse but over repeated trials may be better than optimizing
# on first values.

# For lynx and hare data, start by log transform. Thus, linear (affine) ODE
# with "bias" term can be interpreted as LV ODE with extra squared term for
# each "species". Data for n=2 2D lynx and hare fit better by using 3 or more
# dimensions, ie, with extra 1 or 2 dummy variables. For NODE, no simple
# interpretation of terms. For NODE, can also use dummy variables, typically
# n=3 or 4 gives better fit than n=2. Perhaps 2D data sit in 3D manifold??

# Occasionally system will lock into local minimum that is clearly not
# a good fit. Rerun program, which seeds with different initial
# parameters and will typically avoid the same local minimum.

# Goal is to find one good fit. So possible need to do a few
# runs with different initial parameter seeds is not a difficulty.
# Program runs reasonably fast on desktop computer, depending on
# hyerparameters.

# If trouble converging, try varying adm_learn, the learning rate, the 
# solver tolerances, and the solver algorithm. If instabilities, lower
# tolerances and use "stiff" ODE solver

# See comments in lynx_hare_settings.jl

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

# define diff eq functions

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
			solve(prob, S.solver, u0=vcat(u_init,p[1:S.n-2]), p=p[S.n-1:end])
	predict_ode_nodummy(p, prob, u_init) = solve(prob, S.solver, p=p)

	# For NODE, many simple options to build alternative network architecture, see SciML docs
	if S.use_node
		dudt = FastChain(FastDense(S.n, S.layer_size, activate), FastDense(S.layer_size, S.n))
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

# This gets called for each iterate of DiffEqFlux.sciml_train(), args to left of ;
# are p plus the return values of loss(). Use this function for any intermediate
# plotting, display of status, or collection of statistics
function callback(p, l, pred, prob, u_init, w;
						doplot = true, show_lines = false, show_third = false)
  if (S.print_grad)
  	grad = gradient(p->loss(p,u_init,w,prob)[1], p)[1]
  	gnorm = sqrt(sum(abs2, grad))
  	println(@sprintf("%5.3e; %5.3e", l, gnorm))
  else
  	display(l)
  end
  # plot current prediction against data
  len = length(pred[1,:])
  ts = tsteps[1:len]
  ysize = if show_third 1200 else 800 end
  panels = if show_third 3 else 2 end
  plt = plot(size=(600,ysize), layout=(panels,1))
  plot_type! = if show_lines plot! else scatter! end
  plot_type!(ts, ode_data[1,1:len], label = "hare", subplot=1)
  plot_type!(plt, ts, pred[1,:], label = "pred", subplot=1)
  plot_type!(ts, ode_data[2,1:len], label = "lynx", subplot=2)
  plot_type!(plt, ts, pred[2,:], label = "pred", subplot=2)
  if show_third
  	plot_type!(plt, ts, pred[3,:], label = "3rdD", subplot=3)
  end
  if doplot
    display(plot(plt))
  end
  return false
end

function loss(p, u_init, w, prob)
	pred_all = predict(p, prob, u_init)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred_all, prob, u_init, w
end

# For iterative fitting of times series
function weights(a, tsteps; b=10, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

end # module
