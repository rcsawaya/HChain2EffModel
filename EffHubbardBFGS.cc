#include "itensor/all.h"
#include "meas.h"
#include "BFGS.h"

#include <fstream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace itensor;
using std::string;
using std::min;
using std::max;

int N, N_vars, N_tijvars, N_Vijvars;
IQMPS psi;
Hubbard sites;
Matrix t_eff, tvar_mat, V_eff, Vvar_mat;
IQMPO H1_full, mpoV_full;
std::vector <IQMPO> Hset(2), Hset_epsilon(2);
Sweeps sweeps;
int nsweeps;
int writem;
bool write_wfs;

int func_calls = 0;
int dfunc_calls = 0;

void print_mat(const char* name, int N, Matrix mat)
	{
	FILE* tmp_f;
	tmp_f = fopen(name, "w");
	for (int i = 1; i <= N; i++)
	for (int j = 1; j <= N; j++)
		{
		fprintf(tmp_f, "%d\t%d\t%0.8f\n", i, j, mat(i-1,j-1));
		}
	fclose(tmp_f);
	}

void initialize_psi(int Nup, int Ndn)
    {
    if ((Nup > N) || (Ndn > N)) throw std::invalid_argument("Number of Ups or Dns exceeds number of sites");

    int dbl_occ = ((Nup + Ndn) > N) ? ((Nup + Ndn) - N) : 0;
    int Nup_eff = Nup - dbl_occ;
    int Ndn_eff = Ndn - dbl_occ;

    auto state = InitState(sites, "Emp");

    enum istate {Emp, Up, Dn, UpDn};
	std::vector <istate> init(N,Emp);
    for (int i = 0; i < Nup_eff; i++) {
        init[i] = Up;
    }
    for (int i = Nup_eff; i < (Nup_eff + Ndn_eff); i++) {
        init[i] = Dn;
    }
    for (int i = (Nup_eff + Ndn_eff); i < (Nup_eff + Ndn_eff + dbl_occ); i++) {
        init[i] = UpDn;
    }
    random_shuffle(init.begin(), init.end());

    for (int i = 0; i < N; i++) {
        if (init[i] == Up) state.set(i + 1, "Up");
        if (init[i] == Dn) state.set(i + 1, "Dn");
        if (init[i] == UpDn) state.set(i + 1, "UpDn");
    }

    psi = IQMPS(state);
    double Sztot = (Nup - Ndn);
    auto Sz = "Sztot = " + std::to_string(Sztot);

    Print(totalQN(psi));
    println(Sz);
    printf("\n\n");
	}

std::vector <double> tot_dens()
	{
	int N = psi.N();

	//
	// Measure spin densities
	//
	std::vector <double> upd(N), dnd(N);
	for(int j = 1; j <= N; ++j)
		{
		psi.position(j);
		upd[j-1] = (dag(prime(psi.A(j),Site))*sites.op("Nup",j)*psi.A(j)).real();
		dnd[j-1] = (dag(prime(psi.A(j),Site))*sites.op("Ndn",j)*psi.A(j)).real();
		}

	std::vector <double> dens(N,0);
	for(int j = 0; j < N; ++j) {
		dens[j] = upd[j] + dnd[j];
	}

	return dens;
	};

template<class Tensor>
class ChemObserver : public DMRGObserver<Tensor>
    {
    bool write_wfs_ = false;
    public:

    using Parent = DMRGObserver<Tensor>;

    ChemObserver(MPSt<Tensor> const& psi,
                Args const& args = Global::args())
        : Parent(psi,args)
        {
        write_wfs_ = args.getBool("WriteWFs",false);
        }

    virtual ~ChemObserver() { }

    void virtual
    measure(Args const& args = Global::args())
        {
        Parent::measure(args);

        auto& psi = Parent::psi();
        auto& sites = psi.sites();

        auto N = psi.N();
        auto b = args.getInt("AtBond");
        auto ha = args.getInt("HalfSweep");
        auto sw = args.getInt("Sweep");

        if(write_wfs_ && ha==2 && b==1)
            {
            auto fname = format("psi%02d",sw);
            println("    -> Writing wavefunction to file ",fname);
            writeToFile(fname,psi);
            }
        }
    }; //ChemObserver

Real DMRG()
	{
	// Get inital sweep configuration and define updates to sweep table
	int crnt_m = sweeps.maxm(nsweeps);
	Real crnt_cutoff = sweeps.cutoff(nsweeps);
	int n_swp = 1;
	int dm = 3;
	
	// Define DMRG arguments
	auto args = Args{"Quiet",true};
	if(writem > 0) args.add("WriteM",writem);
	auto obs = ChemObserver<IQTensor>(psi,{"WriteWFs=",write_wfs});
	
	// Initialize erros
	auto d_dens = 1.0;
	auto d_sym = 1.0;
	auto d_eng = 1.0;
	auto conv_thr = 1e-3;
	
	// Run DMRG
	Real energy, energy0=0.;
	auto dens0 = tot_dens();
	while ( ((d_dens > conv_thr) && (d_sym > conv_thr)) || (d_eng > conv_thr) ) {
		println(sweeps);
		std::ifstream stop("STOP_DMRG2");
		if(stop.good()) break;
		
		energy = dmrg(psi,Hset,sweeps,obs,args);

		// Update errors
		d_eng = fabs(energy0 - (energy / N));
		energy0 = energy / N;

		auto dens = tot_dens();

		d_dens = 0;
		d_sym = 0;
		int Nc = 0;
		for (int i = 0; i < N; i++) {
			if (dens[i] != 0) {
				d_dens += fabs(dens[i] - dens0[i]) / dens[i];
				d_sym += fabs(dens[i] - dens[(N-1) - i]) / dens[i];
				Nc += 1;
			}
		}
		d_dens /= Nc;
		d_sym /= Nc;

		printfln("Avg change in density = %.5f", d_dens);
		printfln("Avg symmetry in density = %.5f", d_sym);
		printfln("Change in energy per site = %.5f", d_eng);
		dens0 = dens;
			
		// Update sweep table
		crnt_m += dm;
		int prev_minm = sweeps.minm(1);
		if (prev_minm > 50) prev_minm = 49;

		sweeps = Sweeps(n_swp,prev_minm,crnt_m,crnt_cutoff);
		//sweeps.noise() = 1E-12;
		//sweeps.niter() = 4;
		sweeps.minm() = prev_minm + 1;
	}

	println("Writing wavefunction 'psi' to disk");
	writeToFile("psi",psi);

	return energy;
	}

// Define optimization function
float dmrg_func(float params[])
	{
	system("rm -r PH*");
	
	int telem, Velem;
	float tij, Vij;
	auto ampoH1 = AutoMPO(sites);
	auto ampoV = AutoMPO(sites);
	for (int i = 1; i <= N; i++)
	for (int j = 1; j <= N; j++)
		{
		telem = tvar_mat(i-1, j-1);
		Velem = Vvar_mat(i-1, j-1);
	
		// Assuming tij's are chemical potentials when i == j	
		if (telem > 0) tij = params[telem];
		else tij = t_eff(i-1, j-1);

		if (Velem > 0) Vij = params[Velem + N_tijvars];
		else Vij = V_eff(i-1, j-1);
		
		if (i == j) ampoH1 += tij,"Ntot",i;
		else
			{
			ampoH1 += tij,"Cdagup",i,"Cup",j;
			ampoH1 += tij,"Cdagdn",i,"Cdn",j;
			}
		ampoH1 += -Vij,"Ntot",i;  // Add in electron-lattice piece 
		
		if (i == j) ampoV += Vij,"Nupdn",i;
		else ampoV += (0.5 * Vij),"Ntot",i,"Ntot",j;
		}
	auto mpoH1_eff = IQMPO(ampoH1);
	auto mpoV_eff = IQMPO(ampoV);

	Hset.at(0) = mpoH1_eff;
	Hset.at(1) = mpoV_eff;

	nsweeps = 1;
	sweeps = Sweeps(nsweeps);
	sweeps.maxm() = 1500;
	sweeps.cutoff() = 1e-10;
	sweeps.noise() = 0.0;

	Real energy = DMRG();
	float full_overlap = (float) overlap(psi, H1_full, psi);
	full_overlap += overlap(psi, mpoV_full, psi);

	func_calls++;
	return full_overlap;
	}

// Define gradient of optimization function
void ddmrg_func(float overlap0, float params[], float grad[])
	{
	system("rm -r PH*");
	
	float epsilon = 1e-4; 
	IQMPS psi_epsilon = psi;
	
	// Define DMRG arguments
	auto args = Args{"Quiet",true};
	if(writem > 0) args.add("WriteM",writem);
	auto obs = ChemObserver<IQTensor>(psi_epsilon,{"WriteWFs=",write_wfs});

	// Modify variational parameters by depsilon
	int telem, Velem;
	float tij, Vij;
	for (int var = 1; var <= N_vars; var++)
		{
		auto ampoH1_epsilon = AutoMPO(sites);
		auto ampoV_epsilon = AutoMPO(sites);
		for (int i = 1; i <= N; i++)
		for (int j = 1; j <= N; j++)
			{
			telem = tvar_mat(i-1, j-1);
			Velem = Vvar_mat(i-1, j-1);
			
			// Assuming tij's are chemical potentials when i == j	
			if (telem == var) tij = params[var] + epsilon;
			else if (telem > 0) tij = params[telem];
			else tij = t_eff(i-1, j-1);

			if (Velem == (var - N_tijvars)) Vij = params[var] + epsilon;
			else if (Velem > 0) Vij = params[Velem + N_tijvars];
			else Vij = V_eff(i-1, j-1);
			
			if (i == j) ampoH1_epsilon += tij,"Ntot",i;
			else
				{
				ampoH1_epsilon += tij,"Cdagup",i,"Cup",j;
				ampoH1_epsilon += tij,"Cdagdn",i,"Cdn",j;
				}
			ampoH1_epsilon += -Vij,"Ntot",i;  // Add in electron-lattice piece 
		
			if (i == j) ampoV_epsilon += Vij,"Nupdn",i;
			else ampoV_epsilon += (0.5 * Vij),"Ntot",i,"Ntot",j;
			}
		auto H1_mpo = IQMPO(ampoH1_epsilon);
		auto V_mpo = IQMPO(ampoV_epsilon);

		Hset_epsilon.at(0) = H1_mpo;
		Hset_epsilon.at(1) = V_mpo;

		auto epsilon_sweeps = Sweeps(1);
		epsilon_sweeps.maxm() = 1500;
		epsilon_sweeps.cutoff() = 1e-10;
		epsilon_sweeps.noise() = 0.0;
		
		// Run DMRG
		Real energy_epsilon, overlap_epsilon;
		energy_epsilon = dmrg(psi_epsilon, Hset_epsilon, epsilon_sweeps, obs, args);
		
		overlap_epsilon = overlap(psi_epsilon, H1_full, psi_epsilon);
		overlap_epsilon += overlap(psi_epsilon, mpoV_full, psi_epsilon);

		// Calculate df/dvar
		grad[var] = (overlap_epsilon - overlap0) / epsilon;

		// Reset psi
		psi_epsilon = psi;
		}	
	dfunc_calls++;
	}

int main(int argc, char* argv[])
    {
    //Parse the input file
    if(argc != 2) { printfln("Usage: %s inputfile",argv[0]); return 0; }
    auto input = InputGroup(argv[1],"input");

    N = input.getInt("N");
    int Nup = (int) floor(N / 2) + 1;
    int Ndn = (int) floor(N / 2);
    Nup = input.getInt("Nup", Nup);
    Ndn = input.getInt("Ndn", Ndn);

    nsweeps = input.getInt("nsweeps");
    auto tij_eff_fn = input.getString("tij_eff_fn", "tij_eff.txt");
    auto Vij_eff_fn = input.getString("Vij_eff_fn", "Vij_eff.txt");
	auto tijvar_fn = input.getString("tijvar_fn", "tijvar.txt");
	auto Vijvar_fn = input.getString("Vijvar_fn", "Vijvar.txt");
	auto tij_full_fn = input.getString("tij_full_fn", "tij_full.txt");
	auto Vij_full_fn = input.getString("Vij_full_fn", "Vij_full.txt");
	auto epsilon = input.getReal("epsilon", 1e-4);
    auto quiet = input.getYesNo("quiet",false);
    writem = input.getInt("writem",0);
    write_wfs = input.getYesNo("write_wfs",false);

    auto table = InputGroup(input,"sweeps");
    sweeps = Sweeps(nsweeps,table);
    println(sweeps);

    
	//
    // Initialize the site degrees of freedom.
    //
    if (fileExists("sites"))
		{
		println("Reading sites from disk");
		sites = readFromFile<Hubbard>("sites");
    	}
    else
		{
		sites = Hubbard(N);
		writeToFile("sites",sites);
    	}

    
	//
    // Create the Hamiltonian using AutoMPO
    //
	auto ampoH1 = AutoMPO(sites);
	auto ampoH1_full = AutoMPO(sites);
	auto ampoV = AutoMPO(sites);
	auto ampoV_full = AutoMPO(sites);
	
	// Interactions
	int i,j;
	Real Vij;
	V_eff = Matrix(N,N);
	auto V_full = Matrix(N,N);
	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		{
		V_eff(i,j) = 0.0;
		V_full(i,j) = 0.0;
		}
	
	std::ifstream Vij_full_f(Vij_full_fn);
	while (Vij_full_f >> i >> j >> Vij)
		{
		if (i == 0) break;
		V_full(i-1, j-1) = Vij;
		
		if (i == j) ampoV_full += Vij,"Nupdn",i;
		else
			{
			ampoV_full += (0.5 * Vij),"Ntot",i,"Ntot",j;
			}
		}
	Vij_full_f.close();
    mpoV_full = IQMPO(ampoV_full);
	printf("%d terms in V\n", ampoV_full.terms().size());
	printf("Max bond dimension of V MPO: %d\n", maxM(mpoV_full));

	std::ifstream Vij_eff_f(Vij_eff_fn);
	while (Vij_eff_f >> i >> j >> Vij)
		{
		if (i == 0) break;
		V_eff(i-1, j-1) = Vij;

		if (i == j) ampoV += Vij,"Nupdn",i;
		else
			{
			ampoV += (0.5 * Vij),"Ntot",i,"Ntot",j;
			}
		}
	Vij_eff_f.close();
    auto mpoV_eff = IQMPO(ampoV);
	printf("%d terms in V\n", ampoV.terms().size());
	printf("Max bond dimension of V MPO: %d\n", maxM(mpoV_eff));

	// Kinetic
	Real tij;
	t_eff = Matrix(N,N);	
	auto t_full = Matrix(N,N);
	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		{
		t_eff(i,j) = 0.0;
		// t_eff(i,i) += V_full(i,j);
		
		t_full(i,j) = 0.0;
		}

	std::ifstream tij_full_f(tij_full_fn);
	while (tij_full_f >> i >> j >> tij)
		{
		if (i == 0) break;
		t_full(i-1, j-1) = tij;

		if (i == j) ampoH1_full += tij,"Ntot",i;
		else
			{
			ampoH1_full += tij,"Cdagup",i,"Cup",j; // Ignore Hermitian conj. since file will have i -> j terms
			ampoH1_full += tij,"Cdagdn",i,"Cdn",j;
			}
		}
	tij_full_f.close();
    H1_full = IQMPO(ampoH1_full);
	printf("%d terms in H1\n", ampoH1_full.terms().size());
	printf("Max bond dimension of H1 MPO: %d\n", maxM(H1_full));

	// Shift diagonal to get constant	
	float U_lattice[N];
	for (int i = 0; i < N; i++)
		{
		U_lattice[i] = 0.0;
		for (int j = 0; j < N; j++) U_lattice[i] += V_eff(i,j);
		}
			
	std::ifstream tij_eff_f(tij_eff_fn);
	while (tij_eff_f >> i >> j >> tij)
		{
		if (i == 0) break;
		t_eff(i-1, j-1) = tij;

		if (i == j)
			{
			t_eff(i-1, j-1) += U_lattice[i-1];
			tij += U_lattice[i-1];
			ampoH1 += tij,"Ntot",i;
			}
		else
			{
			ampoH1 += tij,"Cdagup",i,"Cup",j; // Ignore Hermitian conj. since file will have i -> j terms
			ampoH1 += tij,"Cdagdn",i,"Cdn",j;
			}
		}
	tij_eff_f.close();
    auto mpoH1_eff = IQMPO(ampoH1);
	printf("%d terms in V\n", ampoH1.terms().size());
	printf("Max bond dimension of V MPO: %d\n", maxM(mpoH1_eff));
	
	//std::vector <IQMPO> Hset(2);
	Hset.at(0) = mpoH1_eff;
	Hset.at(1) = mpoV_eff;
   
	Hset_epsilon.at(0) = mpoH1_eff;
	Hset_epsilon.at(1) = mpoV_eff;

	//
    // Set the initial wavefunction matrix product state
    // to be a Neel state.
    //
    if (fileExists("psi")) {
        println("Reading wavefunction from file 'psi'");
        psi = readFromFile<IQMPS>("psi",sites);
        Print(overlap(psi,psi));
    }
    else {
        initialize_psi(Nup, Ndn);
    }
    Print(totalQN(psi));


    //
    // Define lambda for DMRG calculation
    //
	// Get inital sweep configuration and define updates to sweep table
	int crnt_m = sweeps.maxm(nsweeps);
	Real crnt_cutoff = sweeps.cutoff(nsweeps);
	
	// Define DMRG arguments
	auto args = Args{"Quiet",true};
	if(writem > 0) args.add("WriteM",writem);
	auto obs = ChemObserver<IQTensor>(psi,{"WriteWFs=",write_wfs});
	
	// Run the initial DMRG 
	Real energy = DMRG();
	

	//
	// Begin Optimization
	//
	// Initialize variational parameters
    println("Initializing variational parameters");
    tvar_mat = Matrix(N,N);
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        {
        tvar_mat(i,j) = -9999;  // Fill value
        }
    
	N_tijvars = 0;
    int tijvar0;
    std::ifstream tijvar_in(tijvar_fn);
    while (tijvar_in >> i >> j >> tijvar0)
        {
        if (i == 0) break;
        tvar_mat(i-1, j-1) = tijvar0;
        N_tijvars = std::max(N_tijvars, tijvar0);
        }
    tijvar_in.close();
    printfln("Got %d tij variables", N_tijvars);

    Vvar_mat = Matrix(N,N);
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        {
        Vvar_mat(i,j) = -9999;  // Fill value
        }
	
	N_Vijvars = 0;
    int Vijvar0;
    std::ifstream Vijvar_in(Vijvar_fn);
    while (Vijvar_in >> i >> j >> Vijvar0)
        {
        if (i == 0) break;
        Vvar_mat(i-1, j-1) = Vijvar0;
        N_Vijvars = std::max(N_Vijvars, Vijvar0);
        }
    Vijvar_in.close();
    printfln("Got %d Vij variables", N_Vijvars);
	
	N_vars = N_tijvars + N_Vijvars;	
	
	// Run optimization
	//// Initialize starting point 
	printf("Initializing starting point\n");
	float tmp;
	int N_tmp;
	float p0[N_vars+1];
	for (int var = 1; var <= N_vars; var++)
		{
		if (var <= N_tijvars)
			{
			tmp = 0.0;
			N_tmp = 0;
			for (int i = 1; i <= N; i++)
			for (int j = 1; j <= N; j++)
				{
				if (tvar_mat(i-1, j-1) == var)
					{
					tmp += t_eff(i-1, j-1);
					N_tmp++;
					}
				}
			tmp /= N_tmp;
			}
		else
			{
			tmp = 0.0;
			N_tmp = 0;
			for (int i = 1; i <= N; i++)
			for (int j = 1; j <= N; j++)
				{
				if (Vvar_mat(i-1, j-1) == (var - N_tijvars))
					{
					tmp += V_eff(i-1, j-1);
					N_tmp++;
					}
				}
			tmp /= N_tmp;
			}

		p0[var] = tmp;
		}

	//// Initialize function values
	printf("Initializing function values at starting point\n");
	float fret = dmrg_func(p0);	

	//// Run algorthim
	printf("Running optimization\n");
	int ndim = N_vars;
	float gtol = 1e-6;
	int check;
	int iter = 0;

	float (*func)(float*);
	func = &dmrg_func;

	void (*dfunc)(float, float*, float*);
	dfunc = &ddmrg_func;
	BFGS(p0, ndim, gtol, &iter, &fret, func, dfunc);	

	printf("********* Final p **********\n");
	for (int i = 1; i <= N_vars; i++)
		{
		printf("%d\t%0.5f\n", i, p0[i]);
		}
	printf("********* Final Value **********\n");
	printf("%0.5f\n", fret);
	printf("Total number of iterations = %d\n", iter);
	printf("Total number of function calls = %d\n", func_calls);
	printf("Total number of dfunction calls = %d\n", dfunc_calls);
	
	FILE* fNitr;
	fNitr = fopen("BFGS_N_iterations.txt", "w");
	fprintf(fNitr, "%d\n", iter);
	fclose(fNitr);

	FILE* fNcalls;
	fNcalls = fopen("BFGS_N_calls.txt", "w");
	fprintf(fNcalls, "%d\n", func_calls);
	fclose(fNcalls);
	
	FILE* fNdcalls;
	fNdcalls = fopen("BFGS_N_dcalls.txt", "w");
	fprintf(fNdcalls, "%d\n", dfunc_calls);
	fclose(fNdcalls);
	
	float tmp_t, tmp_v;
	int idx, idx_v;
	auto tfromp = Matrix(N,N);
	auto Vfromp = Matrix(N,N);
	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		{
		idx = tvar_mat(i,j);
		idx_v = Vvar_mat(i,j) + N_tijvars;
		
		if (idx > 0) tmp_t = p0[idx];
		else tmp_t = t_eff(i,j);
		
		if (idx_v > 0) tmp_v = p0[idx_v];
		else tmp_v = V_eff(i,j);
		
		tfromp(i,j) = tmp_t;
		// tfromp(i,i) -= tmp_v;

		Vfromp(i,j) = tmp_v;
		}

	// Restore tii	
	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		{
		tfromp(i,i) -= Vfromp(i,j);
		}

	print_mat("BFGS_newtij_eff.txt", N, tfromp);
	print_mat("BFGS_newVij_eff.txt", N, Vfromp);
	
	return 0;
	} // main


































