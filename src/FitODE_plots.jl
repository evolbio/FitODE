module FitODE_plots
using Plots, FitODE
export plot_target_pred, plot_phase, plot_data_orig_smooth

num_rows(matr) = size(matr,1)
array_to_tuple(arr) = (arr...,)

function plot_target_pred(tsteps, target, pred; show_lines = false,
			num_dim = num_rows(target), target_labels = ("hare", "lynx"))
	target_dim = num_rows(target)
	num_labels = length(target_labels)
	@assert num_rows(pred) >= target_dim
	@assert num_rows(pred) >= num_dim
	labels = [i<=num_labels ? target_labels[i] : "none" for i in 1:num_dim]
	len = length(pred[1,:])
	ts = tsteps[1:len]
	plt = plot(size=(600,400 * num_dim), layout=(num_dim,1))
	plot_type! = if show_lines plot! else scatter! end
	for i in 1:num_dim
		if target_dim >= i
			plot_type!(ts, target[i,1:len], label = labels[i], subplot=i, color=1)
		end
		plot_type!(ts, pred[i,:], label = "pred", subplot=i, color=2)
	end
	display(plot(plt))	
end

# fix for cases in which number dimensions > 3
function plot_phase(target, pred; dummy_dim = true)
	@assert num_rows(target) <= 3 "More than 3 target dimensions, fix code"
	target_dim = num_rows(target)
	pred_dim = num_rows(pred)
	panels = (dummy_dim && target_dim == 2) ? 1 + pred_dim - target_dim : 1
	len = length(pred[1,:])
	plt = plot(size=(600,400 * panels), layout=(panels,1))
	plot!(array_to_tuple([target[i,:] for i in 1:num_rows(target)]),
			subplot=1,color=1,label="data", xlabel="hare", ylabel="lynx")
	plot!(array_to_tuple([pred[i,:] for i in 1:num_rows(target)]),
			subplot=1,color=2,label="pred")
	for j in 2:panels
		# here only if target_dim == 2
		tt = vcat(target, pred[j+1,:]')
		pp = vcat(pred[1:2,:], pred[j+1,:]')
		plot!(array_to_tuple([tt[i,:] for i in 1:3]),
				subplot=j,color=1,label="data", xlabel="hare", ylabel="lynx")
		plot!(array_to_tuple([pp[i,:] for i in 1:3]),
				subplot=j,color=2,label="pred", xlabel="hare", ylabel="lynx")
	end
	display(plt)
end

# requires reading in raw data from disk
function plot_data_orig_smooth(S; labels = ("hare", "lynx"))
	smooth, _, _, tsteps_smooth, orig = read_data(S)
	plt = plot(size=(600,400))
	orig_len = length(orig[1,:])
	tsteps_orig = range(tsteps_smooth[1], tsteps_smooth[end], length = length(orig[1,:]))
	num_dim = length(orig[:,1])
	labels = [i<=length(labels) ? labels[i] : "none" for i in 1:num_dim]
	plt = plot(size=(600,400*num_dim), layout=(num_dim,1))
	for i in 1:num_dim
		scatter!(tsteps_orig, orig[i,:], color=i, label=nothing, subplot=i)
		plot!(tsteps_smooth, smooth[i,:], color=i, label=labels[i], subplot=i)
	end
	display(plt)
end

# uses stored values as args to avoid rereading data from disk
function plot_data_orig_smooth(smooth, tsteps_smooth, orig; labels = ("hare", "lynx"))
	plt = plot(size=(600,400))
	orig_len = length(orig[1,:])
	tsteps_orig = range(tsteps_smooth[1], tsteps_smooth[end], length = length(orig[1,:]))
	num_dim = length(orig[:,1])
	labels = [i<=length(labels) ? labels[i] : "none" for i in 1:num_dim]
	plt = plot(size=(600,400*num_dim), layout=(num_dim,1))
	for i in 1:num_dim
		scatter!(tsteps_orig, orig[i,:], color=i, label=nothing, subplot=i)
		plot!(tsteps_smooth, smooth[i,:], color=i, label=labels[i], subplot=i)
	end
	display(plt)
end

end # module