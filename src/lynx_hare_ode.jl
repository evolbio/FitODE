using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics, Distributions, JLD2, Dates, Random, Printf

# LV ODE with extra squared term for each "species". Data for 2D hare
# and lynx. Better fit using n=3 or 4. For n=5 seems to be too high for
# easy fit. NODE with n=3 fits better.

# See lynx_hare_node.jl for comments on many aspects of this code. Much
# shared with that file.

now_name = Dates.format(now(),"yyyymmdd_HHMMSS")
proj_dir = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare"

###################### Start of settings section ######################

# Use single named tuple S to hold all settings, access by S.var

S = (

use_node = true,	# switch between NODE and ODE
layer_size = 20,	# size of layers for NODE

# number of variables to track in (N)ODE, first two are hare and lynx
n = 3, 					# must be >= 2, number dummy variables is n-2
opt_dummy_u0 = false,	# optimize dummy init values instead of using rand values

# larger tolerances are faster but errors make gradient descent more challenging
rtol = 1e-10,		# relative tolerance for ODE solver
atol = 1e-12,		# absolute tolerance for ODE solver
adm_learn = 0.0005,	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 500,		# max iterates for each incremental learning step
print_grad = true,	# show gradient on terminal, requires significant overhead

start_time = now_name,
csv_file = "$proj_dir/input/lynx_hare_data.csv",
out_file = "/Users/steve/Desktop/" * now_name * ".jld2",

git_vers = chomp(read(`git -C $proj_dir rev-parse --short HEAD`,String)),

generate_rand_seed = true,
rand_seed = 0x695db8870561193d,

wt_base = 1.1,		# good default is 1.1
wt_trunc = 1e-2,	# truncation for weights

# would be worthwhile to experiment with various solvers
# see https://diffeq.sciml.ai/stable/solvers/ode_solve/
solver = Rodas4P(), # TRBDF2() for large tol; Tsit5() faster? but check instability

# Activation function:
# tanh seems to give good fit, perhaps best fit, and maybe overfit
# however, gradient does not decline near best fit, may limit methods that
# require small gradient near fixed point; identity fit not as good but
# gradient declines properly near local optimum with BFGS()

activate = 1, # use one of 1 => identity, 2 => tanh, 3 => sigmoid, 4 => swish

use_splines = true,
# if using splines, increase data to pts per year by interpolation
pts = 2,
use_gauss_filter = true, # smooth the splines by gauss filtering
filter_sd = 1.2
)# end of named tuple of settings

###################### End of settings values ######################

wt_steps = Int(ceil(log(500)/log(S.wt_base)))
nsqr = S.n*S.n

###################### End of setting setup #######################

function set_rand_seed(gen_rand_seed=S.generate_rand_seed, new_seed_val=S.rand_seed)
	global rseed = gen_rand_seed ? rand(UInt) : new_seed_val
	Random.seed!(rseed)
	println("Random seed = ", rseed)
end

df = CSV.read(S.csv_file, DataFrame);
ode_data = permutedims(Array{Float64}(df[:,2:3]));

# take log and then normalize by average
ode_data = log.(ode_data)
ode_data = ode_data .- mean(ode_data)

datasize = length(ode_data[1,:]) # Number of time points
tspan = (0.0f0, Float64(datasize-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split to equal steps

# fit spline to data and use fitted data to train
if S.use_splines
	using Interpolations
	ode_data_orig = ode_data	# store original data
	hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
	lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
	tsteps = range(tspan[1], tspan[2], length = 1+S.pts*(datasize-1)) # Split to equal steps
	if S.use_gauss_filter
		using QuadGK
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
if (S.optimize_dummy_u0 == false) u0 = vcat(u0,randn(Float64,n-2))

# If optimizing initial conditions for dummy dimensions, then use parameters p 
# starting at initial condition u0, ie, u0 for dummy dimensions are first entries
# of p
if S.use_node
	if S.optimize_dummy_u0
		function predict(p, prob, u_init)
		  u_new = vcat(u_init,p[1:S.n-2])
		  Array(prob(u_new, p[S.n-1:end]))
		end
	else
		function predict(p, prob, u_init)
		  Array(prob(u_init, p))
		end
	end
else # ode instead of node, resove u_init and p in prob arg
	function predict(p, prob, u_init)
	  solve(prob, S.solver, p=p)
	end
end

activate = if (S.activate == 1) identity elseif (S.activate == 2) tanh
					elseif (S.activate == 3) sigmoid else swish end
if S.use_node
	dudt = FastChain(FastDense(S.n, S.layer_size, activate), FastDense(S.layer_size, S.n))
else
	function ode!(du, u, p, t)
		s = reshape(p[1:nsqr], S.n, S.n)
		du .= activate.(s*u .- p[nsqr+1:end])
	end
end

function problem(p, u_init, tspn, ts)
	if S.use_node
		prob = NeuralODE(dudt, tspn, S.solver, saveat = ts, reltol = S.rtol, abstol = S.atol)
	else
		if S.optimize_dummy_u0
			prob = ODEProblem(ode!, vcat(u_init,p[1:S.n-2]), tspn, p[S.n-1:end],
						saveat = ts, reltol = S.rtol, abstol = S.atol)
		else
			prob = ODEProblem(ode!, u_init, tspn, p, saveat = ts, reltol = S.rtol, abstol = S.atol)
	end
	return prob
end

callback = function (p, l, pred, prob, u_init, w, tspn, tstp; doplot = true, show_lines = false,
						show_third = false)
  if (S.print_grad)
  	grad = gradient(p->loss(p,u_init,w,tspn,tstp)[1], p)[1]
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

function loss(p, u_init, w, tspn, tstp)
	prob = problem(p, u_init, tspn, tstp)
	pred_all = predict(p, prob, u_init)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred_all, prob, u_init, w, tspn, tstp
end

function weights(a; b=10, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end

# Use Beta cdf weights for iterative fitting. Fits earlier parts of time
# series first with declining weights for later data points, then 
# keep fitted parameters and redo, slightly increasing weights for
# later time points.

beta_a = 1:1:wt_steps
set_rand_seed();
if !S.use_node p_init = 0.1*rand(nsqr + S.n) end;	# n^2 matrix plus vector n for individual growth
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(S.wt_base^beta_a[i]; trunc=S.wt_trunc)
	last_time = tsteps[length(w[1,:])]
	ts = tsteps[tsteps .<= last_time]
	# increase p length by adding u0 for dummy dimensions
	if (i == 1)
		if S.use_node
			prob = NeuralODE(dudt, (0.0,last_time), S.solver, saveat = ts,
						reltol = S.rtol, abstol = S.atol)
			p = prob.p
		else
			p = S.optimize_dummy_u0 ? vcat(randn(S.n-2),p_init) : p_init
	else
		p = result.u
	end
	result = DiffEqFlux.sciml_train(p -> loss(p,u_0,w,(0,last_time),ts), p,
					ADAM(S.adm_learn); cb = callback, maxiters=S.max_it)
end

# do additional optimization round with equal weights at all points
ww = ones(2,length(tsteps))
p1 = result.u
lossval = loss(p1,u_0,ww,tspan,tsteps);
loss1 = lossval[1]
pred1 = lossval[2]

result2 = DiffEqFlux.sciml_train(p -> loss(p,u_0,ww,tspan,tsteps), result.u, ADAM(S.adm_learn);
			cb = callback, maxiters=S.max_it)

p2 = result2.u
lossval = loss(p2,u_0,ww,tspan,tsteps);
loss2 = lossval[1]
pred2 = lossval[2]

grad = gradient(p->loss(p,u_0,ww,tspan,tsteps)[1], p2)

result3 = DiffEqFlux.sciml_train(p -> loss(p,u_0,ww,tspan,tsteps),p2,BFGS(),
			cb = callback, maxiters=S.max_it)

p3 = result3.u
lossval = loss(p3,u_0,ww,tspan,tsteps);
loss3 = lossval[1]
pred3 = lossval[2]

# final plot with third dimension and lines
third = if n >= 3 true else false end

prob = problem(p3, u0, tspan, tsteps)
callback(p3,loss3,pred3,prob,u_0,ww,tspan,tsteps); show_lines=true, show_third=third)

jldsave(S.out_file; S, rseed, p1, loss1, pred1, p2, loss2, pred2, p3, loss3, pred3)

# Also, could do fit back to ode_data_orig after fitting to splines or gaussian filter

# Could add code for pruning model (regularization) by adding costs to parameters
# and so reducing model size, perhaps searching for minimally sufficient model
			
# dt = load(out_file)
# dt["pred1"] # for prediction data for first set
