using DelimitedFiles
using LinearAlgebra
using Statistics

# Directory where C++ Itensor is installed along with \"EffHubbardBFGS.cc\""
CODE_dir = "/data/homezvol2/rsawaya/ITensor/Hubbard"

function dorun(cmd::String)
	run(pipeline(`$(split(cmd))`, stdout=devnull))
end

function dorun(cmd::String, out::String)
	println("Running $cmd")
	run(pipeline(`$(split(cmd))`, stdout=out))
end

function to_mat(X::Array{Any,2}, N::Int)
	X_out = zeros(Float64, N, N)
	for row = 1:size(X,1)
		i,j,x = X[row,:]
		(i == 0) && continue
		X_out[i,j] = x
	end
	return X_out
end

function main()
	N = parse(Int, ARGS[1])

	max_iter = 50

	tij_var = readdlm("tijvar.txt", Any)
	Vij_var = readdlm("Vijvar.txt", Any)
	
	prev_tij = readdlm("tij_eff.txt", Any) |> x -> to_mat(x, N)
	prev_Vij = readdlm("Vij_eff.txt", Any) |> x -> to_mat(x, N)
	n_iter = 0
	dparam = 1.0
	trunc_err = 1e-6
	while (dparam > trunc_err) && (n_iter < max_iter)
		n_iter += 1

		# Optimize over Vij
		writedlm("tijvar.txt", zeros(Int64, 1, 3))
		dorun("$CODE_dir/EffHubbardBFGS_mutijVij inputfile", "Vij$(n_iter).out")	
		cp("BFGS_newtij_eff.txt", "tij_eff.txt", force=true)
		cp("BFGS_newVij_eff.txt", "Vij_eff.txt", force=true)
		
		# Optimize over tij
		writedlm("Vijvar.txt", zeros(Int64, 1, 3))
		writedlm("tijvar.txt", tij_var)
		dorun("$CODE_dir/EffHubbardBFGS_mutijVij inputfile", "tij$(n_iter).out")	
		cp("BFGS_newtij_eff.txt", "tij_eff.txt", force=true)
		cp("BFGS_newVij_eff.txt", "Vij_eff.txt", force=true)
		writedlm("Vijvar.txt", Vij_var)

		# Compute convergence criterion
		tij = readdlm("tij_eff.txt", Any) |> x -> to_mat(x, N)
		Vij = readdlm("Vij_eff.txt", Any) |> x -> to_mat(x, N)
		
		prev_tij_idx = abs.(prev_tij) .> 0
		dtij = abs.(tij - prev_tij) ./ abs.(prev_tij .+ 1e-8) |> 
			x -> mean(x[prev_tij_idx])
	
		prev_Vij_idx = abs.(prev_Vij) .> 0
		dVij = abs.(Vij - prev_Vij) ./ abs.(prev_Vij .+ 1e-8) |>
			x -> mean(x[prev_Vij_idx])

		dparam = 0.5 * (dtij + dVij)
		@show dparam
		prev_tij = tij
		prev_Vij = Vij
	end	
	writedlm("tijvar.txt", tij_var)
	writedlm("Vijvar.txt", Vij_var)
end
main()
