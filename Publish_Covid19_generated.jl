################################################################################
import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()
Pkg.instantiate()

################################################################################
#Load stuff
_dataadjustments = "deathscases"
include("src/_preamble.jl");
include(string("src/_models/SEEIIR-SSM-NR-", _dataadjustments, ".jl"));

# Check Number of threads and set (existing) output directory for cloud
println("Threads: ", Threads.nthreads())
ENV["RESULTS_FILE_TO_UPLOAD"] = joinpath(@__DIR__, "results")

################################################################################
# Set Model
model_og = ModelWrapper(SEEIIRNR_NB4_deathscases(), param_seeiirnr_NB4_deathscases_daily_posteriormean, args_param_seeiirnr_deathscases_daily)
_vals = (:β, :γ, :ϵ, :p_states, :p_thirdstate, :r, :ψ, :ϕ_cases, :ϕ_deaths)
println("Model: ", Base.nameof(typeof(model_og.id)))

tagged_mcmc = Tagged(model_og, _vals)
tagged_latent = Tagged(model_og, :latent)

using JLD2
_subfolder ="\\saved\\"
f_generated   =   jldopen(string(pwd(), _subfolder, "Generated Data - PMCMC chain - Winning Model.jld2"))
data_true = read(f_generated, "data")
θ_true = read(f_generated, "θ")
latent_true = read(f_generated, "latent")

#Create data as integer
data_true_int = [(; deaths = Int( round(data_true[iter].deaths) ), cases = Int( round(data_true[iter].cases) )) for iter in eachindex(data_true)]
Ngenerated = 500
data = deepcopy(data_true_int[1:Ngenerated])

_pf = ParticleFilter(_rng, Objective(model_og, data, :latent), ParticleFilterDefault(; memory = _pfmemory, coverage = 3.0) )
propose!(_rng, _pf, model_og, data)
model_og.val.latent
objective = Objective(deepcopy(model_og), data, Tagged(model_og, _vals))
objective(objective.model.val)

################################################################################
#Set Algorithm
_pf = ParticleFilter(:latent;
        memory = _pfmemory,
        referencing = Ancestral(), coverage = Nparticlemultiplier, threshold = 0.75,
        init = OptimInitialization() #initial pf run after calling constructor
)
#MCMC
_mcmc = NUTS(_vals;
    proposal = ConfigProposal(; metric = MDense()),
    init = PriorInitialization(1000)
)

#!NOTE - In SMC2, use priorinitialization only for kernels in PMCMC, so jittekernels start with same parameter
_pmcmc2 = ParticleGibbs(_pf, _mcmc)

#!NOTE - In SMC2, use priorinitialization only for kernels in PMCMC, so jittekernels start with same parameter
#SMC2
_pf1 = ParticleFilter(:latent;
        memory = _pfmemory,
        referencing = Marginal(), coverage = Nparticlemultiplier, threshold = 0.75,
        init = NoInitialization() #Use trajectory from initial model to construct pf
)
_smc2 = SMC2(
    _pf1,
    _pmcmc2;
   jittermin = 1, jittermax = 5,
   jitterthreshold = 0.9,
   resamplingthreshold = .50,
   Ntuning = _NtuningSMC,
)

################################################################################
#Run Algorithm
#SMC
trace_smc, algorithm_smc = sample(_rng, deepcopy(model_og), data, _smc2;
    default = SampleDefault(;
    dataformat = Expanding(NinitSMC),
    iterations = Niterations, chains = NchainsSMC, burnin = 0,
    printoutput = false,
    )
);
plotChain(trace_smc, tagged_mcmc)
plotDiagnostics(trace_smc, algorithm_smc)
summary(trace_smc, algorithm_smc, TraceTransform(trace_smc, model_og))
Baytes.savetrace(trace_smc, model_og, algorithm_smc)

#PMCMC
trace_pmcmc, algorithm_pmcmc = sample(_rng, deepcopy(model_og), data, _pmcmc2;
    default = SampleDefault(;
        printoutput = true, safeoutput = false,
        iterations = 1500, chains = 4, burnin = 0,
    )
);
Baytes.savetrace(trace_pmcmc, model_og, algorithm_pmcmc)
plotChain(trace_pmcmc, tagged_mcmc)
summary(trace_pmcmc, algorithm_pmcmc, TraceTransform(trace_pmcmc, model_og))

################################################################################
#Save Output

#1 Set Name
result_name = join((
    Base.nameof(typeof(model_og.id)),
    "_",
    Base.nameof(typeof(algorithm_smc)),
    ".jld2"
))

#3 Save in subfolder
JLD2.jldsave(string("results/", result_name); #result_name;
    trace=trace_smc,
    model=model_og,
    algorithm=algorithm_smc,
)
