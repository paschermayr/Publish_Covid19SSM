################################################################################
#Set Algorithm

#PGibbs
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

if smc_tuned
    #Use Cov and Stepsize from tuned Covariance Kernel
    f_tuning = jldopen(string(pwd(), "/saved/", typeof(objective.model.id), "_posterior_tuning.jld2"))
    cov_pmcmc = read(f_tuning, "cov_pmcmc")
#    ϵ_pmcmc = read(f_tuning, "ϵ_pmcmc")

    _mcmc = NUTS(_vals;
        proposal = ConfigProposal(metric = MDense(), proposaladaption = UpdateFalse(), covariance = cov_pmcmc),
#        stepsize = ConfigStepsize(ϵ = ϵ_pmcmc, stepsizeadaption = UpdateFalse()),
        init = PriorInitialization(1000)
    )
end

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
