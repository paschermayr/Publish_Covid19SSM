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

objective = Objective(deepcopy(model_og), data, Tagged(model_og, _vals))
_pf = ParticleFilter(_rng, Objective(model_og, data, :latent), ParticleFilterDefault(; memory = _pfmemory, coverage = 3.0) )
propose!(_rng, _pf, model_og, data)

################################################################################
#Set Algorithm
include("src/_algorithm.jl");

################################################################################
#Run Algorithm
#SMC
trace_smc, algorithm_smc = sample(_rng, deepcopy(model_og), data[1:_Ndata], _smc2;
    default = SampleDefault(;
    dataformat = Expanding(NinitSMC),
    iterations = Niterations, chains = NchainsSMC, burnin = 0,
    printoutput = false,
    )
);
#Baytes.savetrace(trace_smc, model_og, algorithm_smc)
#PMCMC
#=
trace_pmcmc, algorithm_pmcmc = sample(_rng, deepcopy(model_og), data[1:_Ndata], _pmcmc2;
    default = SampleDefault(;
        printoutput = true, safeoutput = false,
        iterations = Niterations, chains = 4, burnin = 0,
    )
);
=#
#plotChain(trace_smc, tagged_mcmc)
#BaytesInference.plotDiagnostics(trace_smc.diagnostics, algorithm_smc)
#summary(trace_smc, algorithm_smc, TraceTransform(trace_smc, model_og))

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
