using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, CSV, DataFrames,
		Statistics, Distributions, JLD2, Dates, Random, Printf

# LV ODE with extra squared term for each "species". Data for 2D hare
# and lynx. Better fit using n=3 or 4. For n=5 seems to be too high for
# easy fit. NODE with n=3 fits better.

# See lynx_hare_node.jl for comments on many aspects of this code. Much
# shared with that file.

# number of variables to track in ODE, first two are hare and lynx
n = 3 # must be >= 2
nsqr = n*n
wt_trunc = 1e-2		# truncation for weights
rtol = 1e-2			# relative tolerance for ODE solver
atol = 1e-3			# absolute tolerance for ODE solver
adm_learn = 0.0005	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 500		# max iterates for each incremental learning step
print_grad = true	# show gradient on terminal, requires significant overhead
csv_file = "/Users/steve/sim/zzOtherLang/julia/autodiff/lynx_hare/input/lynx_hare_data.csv"
out_file = "/Users/steve/Desktop/output.jld2"
seed_file = "/Users/steve/Desktop/seed.julia"

generate_rand_seed = true
rand_seed = 0x695db8870561193d
function set_rand_seed(gen_rand_seed=generate_rand_seed, new_seed_val=rand_seed)
	if gen_rand_seed
		seed_val = rand(UInt)
		Random.seed!(seed_val)
		open("/Users/steve/Desktop/rand.out", "w") do file
			write(file, string(seed_val))
		end
		println("New random seed is ", seed_val)
	else
		Random.seed!(rand_seed)
		println("Restored random seed to ", new_seed_val)
	end
end

wt_base = 1.1		# good default is 1.1
wt_steps = Int(ceil(log(500)/log(wt_base)))
wt_trunc = 1e-2		# truncation for weights

solver = TRBDF2()

use_splines = true
# if using splines, increase data to pts per year by interpolation
pts = 2
use_gauss_filter = true	# smooth the splines by gauss filtering
filter_sd = 1.2
 
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
	ode_data_orig = ode_data	# store original data
	hspline = CubicSplineInterpolation(tsteps,ode_data[1,:])
	lspline = CubicSplineInterpolation(tsteps,ode_data[2,:])
	tsteps = range(tspan[1], tspan[2], length = 1+pts*(datasize-1)) # Split to equal steps
	if use_gauss_filter
		using QuadGK
		conv(y, spline, sd) = 
			quadgk(x -> spline(x) * pdf(Normal(0,sd),y-x), tspan[1], tspan[2])[1]
		ode_data = vcat([conv(y,hspline,filter_sd) for y=tsteps]',
							[conv(y,lspline,filter_sd) for y=tsteps]');
	else
		ode_data = vcat(hspline.(tsteps)',lspline.(tsteps)');
	end
	#plot(tsteps,ode_data[1,:]); plot!(tsteps,ode_data[2,:])
end

u0 = ode_data[:,1] # Initial condition, first time point in data
# add additional initial values for dummy dimensions
u0 = vcat(u0,randn(Float32,n-2))

swish(x) = x ./ (exp.(-x) .+ 1.0)
sigmoid(x) = 1.0 ./ (exp.(-x) .+ 1.0)

# tanh seems to work best
function ode!(du, u, p, t)
	s = reshape(p[1:nsqr], n, n)
	du .= tanh.(s*u .- p[nsqr+1:end])
	#du .= sigmoid(s*u .- p[nsqr+1:end])
	#du .= swish(s*u .- p[nsqr+1:end])
	#du .= s*u .- p[nsqr+1:end]
end

callback = function (p, l, pred, prob, w, u_init; doplot = true, show_lines = false,
						show_third = false)
  if (print_grad)
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
	pred_all = solve(prob, solver, p=p)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(w[1,:]) println("Mismatch") end
	loss = sum(abs2, w[:,1:pred_length] .* (ode_data[:,1:pred_length] .- pred))
	return loss, pred_all, prob, w, u_init
end

function weights(a; b=10, trunc=1e-4) 
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
	w = weights(wt_base^beta_a[i]; trunc=wt_trunc)
	last_time = tsteps[length(w[1,:])]
	prob = ODEProblem(ode!, u0, tspan, p, saveat = tsteps[tsteps .<= last_time],
					reltol = rtol, abstol = atol)
	p = if (i == 1) prob.p else result.u end
	result = DiffEqFlux.sciml_train(p -> loss(p,prob,w,u0), p,
					ADAM(adm_learn); cb = callback, maxiters=max_it)
end

# do additional optimization round with equal weights at all points
prob = ODEProblem(ode!, u0, tspan, p, saveat = tsteps, reltol = rtol, abstol = atol)
ww = ones(2,length(tsteps))
p1 = result.u
lossval = loss(p1,prob,ww,u0);
loss1 = lossval[1]
pred1 = lossval[2]

result2 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww,u0), result.u, ADAM(adm_learn);
			cb = callback, maxiters=max_it)

p2 = result2.u
lossval = loss(p2,prob,ww,u0);
loss2 = lossval[1]
pred2 = lossval[2]

grad = gradient(p->loss(p,prob,ww,u0)[1], p2)

result3 = DiffEqFlux.sciml_train(p -> loss(p,prob,ww,u0),p2,BFGS(),
			cb = callback, maxiters=max_it)

p3 = result3.u
lossval = loss(p3,prob,ww,u0);
loss3 = lossval[1]
pred3 = lossval[2]

# final plot with third dimension and lines
third = if n >= 3 true else false end
callback(p3,loss3,pred3,prob,ww,u0; show_lines=true, show_third=third)

# out_file = Dates.format(now(),"yyyymmdd_HHMM") * ".jld2"
jldsave(out_file; p1, loss1, pred1, p2, loss2, pred2, p3, loss3, pred3)

# Also, could do fit back to ode_data_orig after fitting to splines
			
# dt = load(out_file)
# dt["pred1"] # for prediction data for first set
