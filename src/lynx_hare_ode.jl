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

# number of variables to track in ODE, first two are hare and lynx
n = 3, # must be >= 2
# larger tolerances are faster but errors make gradient descent more challenging
rtol = 1e-10,		# relative tolerance for ODE solver
atol = 1e-12,		# absolute tolerance for ODE solver
adm_learn = 0.0005,	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 500,		# max iterates for each incremental learning step
print_grad = true,	# show gradient on terminal, requires significant overhead

csv_file = "$proj_dir/input/lynx_hare_data.csv",
out_file = "/Users/steve/Desktop/" * now_name * ".jld2",
rnd_file = "/Users/steve/Desktop/" * now_name * ".rnd",

git_vers = chomp(read(`git -C $proj_dir rev-parse --short HEAD`,String)),

generate_rand_seed = true,
rand_seed = 0x695db8870561193d,

wt_base = 1.1,		# good default is 1.1
wt_trunc = 1e-2,	# truncation for weights

# would be worthwhile to experiment with various solvers
# see https://diffeq.sciml.ai/stable/solvers/ode_solve/
solver = Rodas4P(), # TRBDF2() for large tol, Tsit5() faster? but check instability

# Activation function for ode!
# tanh seems to give good fit, perhaps best fit, however, gradient does not
# properly decline near best fit and so cannot use SGLD bayes methods, which
# require small gradient near fixed point; identity fit not as good but
# gradient declines properly near local optimum

activate = 1, # use one of 1 => identity, 2 => tanh, 3 => sigmoid, 4 => swish

use_splines = true,
# if using splines, increase data to pts per year by interpolation
pts = 2,
use_gauss_filter = true, # smooth the splines by gauss filtering
filter_sd = 1.2
)# end of named tuple of settings

###################### End of settings values ######################

wt_steps = Int(ceil(log(500)/log(S.wt_base)))
n = S.n
nsqr = n*n

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
# add additional initial values for dummy dimensions
# alternatively, can optimize initial conditions for dummy dimensions,
# search git log for old code
u0 = vcat(u0,randn(Float64,n-2))

# activation functions for ode!
swish(x) = x ./ (exp.(-x) .+ 1.0)
sigmoid(x) = 1.0 ./ (exp.(-x) .+ 1.0)
activate = if (S.activate == 1) identity elseif (S.activate == 2) tanh
				elseif (S.activate == 3) sigmoid else swish end

function ode!(du, u, p, t)
	s = reshape(p[1:nsqr], n, n)
	du .= activate.(s*u .- p[nsqr+1:end])
end

callback = function (p, l, pred, prob, w, u_init; doplot = true, show_lines = false,
						show_third = false)
  if (S.print_grad)
  	grad = gradient(p->loss(p,prob,w,u_init)[1], p)[1]
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

function loss(p, prob, w, u_init)
	pred_all = solve(prob, S.solver, p=p)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred_all, prob, w, u_init
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
p = 0.1*rand(nsqr + n);	# n^2 matrix plus vector n for individual growth
for i in 1:length(beta_a)
	global result
	println(beta_a[i])
	w = weights(S.wt_base^beta_a[i]; trunc=S.wt_trunc)
	last_time = tsteps[length(w[1,:])]
	prob = ODEProblem(ode!, u0, tspan, p, saveat = tsteps[tsteps .<= last_time],
					reltol = S.rtol, abstol = S.atol)
	p = if (i == 1) prob.p else result.u end
	result = DiffEqFlux.sciml_train(p -> loss(p,prob,w,u0), p,
					ADAM(S.adm_learn); cb = callback, maxiters=S.max_it)
end

# do additional optimization round with equal weights at all points
prob = ODEProblem(ode!, u0, tspan, p, saveat = tsteps, reltol = S.rtol, abstol = S.atol)
ww = ones(2,length(tsteps))
p1 = result.u
lossval = loss(p1,prob,ww,u0);
loss1 = lossval[1]
pred1 = lossval[2]

result2 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww,u0), result.u, ADAM(S.adm_learn);
			cb = callback, maxiters=S.max_it)

p2 = result2.u
lossval = loss(p2,prob,ww,u0);
loss2 = lossval[1]
pred2 = lossval[2]

grad = gradient(p->loss(p,prob,ww,u0)[1], p2)

result3 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww,u0),p2,BFGS(),
			cb = callback, maxiters=S.max_it)

p3 = result3.u
lossval = loss(p3,prob,ww,u0);
loss3 = lossval[1]
pred3 = lossval[2]

# final plot with third dimension and lines
third = if n >= 3 true else false end
callback(p3,loss3,pred3,prob,ww,u0; show_lines=true, show_third=third)

jldsave(S.out_file; S, rseed, p1, loss1, pred1, p2, loss2, pred2, p3, loss3, pred3)

# Also, could do fit back to ode_data_orig after fitting to splines
			
# dt = load(out_file)
# dt["pred1"] # for prediction data for first set
