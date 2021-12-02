using ArgParse
using DelimitedFiles
using LinearAlgebra
using Statistics

s = ArgParseSettings()

@add_arg_table! s begin
    "N"
        help="System size(number of atoms)"
        required=true
	"fn_tij"
        help="File containing the single-particle terms for the full Hamiltonian in the form [...; i j tij; i (j+1) ti(j+1); ...]"
		required=true
	"fn_Vij"
        help="File containing the two-particle terms for the full Hamiltonian in the form [...; i j Vij; i (j+1) Vi(j+1); ...]"
		required=true
	"N_terms_tij"
		help="Number of variational terms in tij(including onsite term)"
		required=true
		arg_type=Int
	"N_terms_Vij"
		help="Number of variational terms in Vij(including onsite term)"
		required=true
		arg_type=Int
    "-o"
        help="Directory into which to output results"
        default="."
	"--sym"
		help="Flag, when called, symmetrized tij and Vij about the center of the 1D chain"
		action=:store_true
	"--cap_edge"
		help="Bounds the maximum edge region to cap_edge"
		arg_type=Int
		default=10000
end

args = parse_args(s)

N = args["N"]
@show N

fn_tij = args["fn_tij"]
fn_Vij = args["fn_Vij"]
@show fn_tij
@show fn_Vij

N_terms_tij = args["N_terms_tij"]
N_offdiag_tij = N_terms_tij - 1
@show N_terms_tij
@show N_offdiag_tij

N_terms_Vij = args["N_terms_Vij"]
N_offdiag_Vij = N_terms_Vij - 1
@show N_terms_Vij
@show N_offdiag_Vij

OUT_dir = args["o"]
@show OUT_dir

sym = args["sym"]
@show sym

cap_edge = args["cap_edge"]
@show cap_edge


function to_mat(x::Array{Any, 2}, N::Int)
	X = zeros(Float64, N, N)
	for row = 1:size(x, 1)
		i,j,a = x[row,:]
		(i == 0) && continue
		X[i,j] = a
	end
	X
end


function to_lst(x::Array{<:Any, 2})
	X = Array{Any}(undef, 0, 3)
	for i = 1:size(x, 1), j = 1:size(x, 2)
		tmp = Any[i, j, x[i,j]]
		X = [X; tmp']
	end
	X
end


function get_padding(X::Array{Float64, 2}, N_max::Int; lambda::Real=1e-4)
	N = size(X, 1)

	X_pad = zeros(Int64, N)
	for i = 0:(N - 1)
		band = diag(X, i) .|> abs
		for j = 0:(div(N - i, 2) - 1)
			tmp = std(band[(1 + j):(end - j)])
			if tmp < lambda
				X_pad[i+1] = min(clamp(j, 0, div(N_max - i, 2) - ((N_max - i) % 2 == 0)), cap_edge)
				break
			end
		end
	end
	return X_pad
end


function get_var_mats(N::Int, r_t::Int, r_v::Int)
	fn_path = split(fn, '/')
	fn_path[end] = replace(fn_path[end], "N=$N" => "N=100")
	full_fn = join(fn_path, '/')
	@show full_fn
		
	tij_full = readdlm("$full_fn/o1b/tij.txt", Any) |>
		x -> to_mat(x, 100)
	Vij_full = readdlm("$full_fn/o1b/Vij.txt", Any) |>
		x -> to_mat(x, 100)
	tij_full[diagind(tij_full)] += sum(Vij_full, dims=2)[:]
	
	t_pad = get_padding(tij_full, N)
	@show t_pad[1:r_t]	
	V_pad = get_padding(Vij_full, N)
	@show V_pad[1:r_v]

	t_var = zeros(Int64, N, N)
    prev_t = 0
    for i = 1:r_t
        diag_len = N - (i-1)
        N_pad = t_pad[i]
        pad_vec = [1:N_pad;] .+ 1
        vars = [pad_vec[end:-1:1]...,
                        ones(Int64, diag_len - (2 * N_pad))...,
                        pad_vec...]
        vars .+= prev_t

        idx  = diagind(t_var, i-1)
        idxf = diagind(t_var, -(i-1))
        t_var[idx] = t_var[idxf] = vars

        prev_t += t_pad[i] + 1
    end
    t_var_out = Array{Any}(undef, 0, 3)
    for i = 1:N, j = 1:N
        t = t_var[i,j]
        (t == 0) && continue
        t_var_out = [t_var_out; Any[i j t]]
    end

    V_var = zeros(Int64, N, N)
    prev_V = 0
    for i = 1:r_v
        diag_len = N - (i-1)
        N_pad = V_pad[i]
        pad_vec = [1:N_pad;] .+ 1
        vars = [pad_vec[end:-1:1]...,
                        ones(Int64, diag_len - (2 * N_pad))...,
                        pad_vec...]
        vars .+= prev_V

        idx  = diagind(V_var, i-1)
        idxf = diagind(V_var, -(i-1))
        V_var[idx] = V_var[idxf] = vars

        prev_V += V_pad[i] + 1
    end
    V_var_out = Array{Any}(undef, 0, 3)
    for i = 1:N, j = 1:N
        v = V_var[i,j]
        (v == 0) && continue
        V_var_out = [V_var_out; Any[i j v]]
    end

    return t_var_out, V_var_out
end


function write_input(DATA_dir::String, N::Int64, Nup::Int64, Ndn::Int64)
    # Parameters that usually stay the same, but can be changed
    #----------------------------------------------------------
    mfirst = 10
    mlast = 1600
    cutoff = 1E-10
    nsweeps = 10
    #----------------------------------------------------------

    f = open("$DATA_dir/inputfile", "w");
    write(f, "input\n\t { \n");
    write(f, "\t N = $N \n");
    write(f, "\t Nup = $Nup \n");
    write(f, "\t Ndn = $Ndn \n\n");

    write(f, "\t tij_eff_fn = tij_eff.txt\n")
    write(f, "\t Vij_eff_fn = Vij_eff.txt\n")
    write(f, "\t tijvar_fn = tijvar.txt\n")
    write(f, "\t Vijvar_fn = Vijvar.txt\n")
    write(f, "\t tij_full_fn = tij_full.txt\n")
    write(f, "\t Vij_full_fn = Vij_full.txt\n\n")

    write(f, "\t quiet = yes \n");
    write(f, "\t writem = 500 \n\n");

    write(f, "\t nsweeps = $nsweeps \n");
    write(f, "\t sweeps \n \t\t { \n");

    nfirst = -7
    nlast = -12
    xi_n = - (1 / (nsweeps - 1)) * log(nlast / nfirst)
    xi_m = - (1 / (nsweeps - 1)) * log(mlast / mfirst)
    exp_fn(x, xi, A) = A * exp(- (x - 1) * xi)
    for i=1:nsweeps
            ii = (i%2 == 1 ? i : i-1)
            noise = 10.0 ^ round(Int64, exp_fn(ii, xi_n, nfirst))
            (i%2 == 0) && (noise = 0.0)
            m = round(Int64, exp_fn(ii, xi_m, mfirst))
            mlow = min(m,10)
            println(f,"\t\t $m      $mlow   $cutoff 2       $noise")
    end

    write(f, "\t\t } \n\n");
    write(f, "\t }");

    close(f)
end


function main()
	tij = readdlm(fn_tij, Any) |>
		x -> to_mat(x, N)
	Vij = readdlm(fn_Vij, Any) |>
		x -> to_mat(x, N)

	if sym
		tij_flip = tij[end:-1:1, end:-1:1]
		tij += tij_flip
		tij .*= 0.5

		Vij_flip = Vij[end:-1:1, end:-1:1]
		Vij += Vij_flip
		Vij .*= 0.5
	end

	tij_full_mat = to_lst(tij) 
	Vij_full_mat = to_lst(Vij) 
    tij_var_mat, Vij_var_mat = get_var_mats(N, N_terms_tij, N_terms_Vij)
	
    # Vij
    Vij_eff = zeros(Float64, N, N)
    for i = 0:N_offdiag_Vij
        Vij_eff[diagind(Vij_eff, i)] = Vij_eff[diagind(Vij_eff, -i)] = diag(Vij, i)
    end
    Vij_eff_mat = to_lst(Vij_eff)

    # tij
    tij_eff = zeros(Float64, N, N)
    for i = 0:N_offdiag_tij
        tij_eff[diagind(tij_eff, i)] = tij_eff[diagind(tij_eff, -i)] = diag(tij, i)
    end
    tij_eff[diagind(tij_eff)] += sum(Vij, dims=2)[:] - sum(Vij_eff, dims=2)[:]
    tij_eff_mat = to_lst(tij_eff)

	EffH_dir = "$OUT_dir/EffH_mu+tij$(N_offdiag_tij)_U+V$(N_offdiag_Vij)_edge=N100_mu"
	(cap_edge < 10000) && (EffH_dir = "$OUT_dir/EffH_mu+tij$(N_offdiag_tij)_U+V$(N_offdiag_Vij)_edge=N100+$(cap_edge)_mu")
	if ~isdir(EffH_dir)
		println("Creating $EffH_dir")
		mkdir(EffH_dir)
	end

    write_input(EffH_dir, N, div(N,2), div(N,2))
    writedlm("$EffH_dir/tij_eff.txt", tij_eff_mat)
    writedlm("$EffH_dir/Vij_eff.txt", Vij_eff_mat)
    writedlm("$EffH_dir/tij_full.txt", tij_full_mat)
    writedlm("$EffH_dir/Vij_full.txt", Vij_full_mat)
    writedlm("$EffH_dir/tijvar.txt", tij_var_mat)
    writedlm("$EffH_dir/Vijvar.txt", Vij_var_mat)
    return
end
main()
