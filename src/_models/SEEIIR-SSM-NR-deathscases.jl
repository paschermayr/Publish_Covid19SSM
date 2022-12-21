using Bijectors, LinearAlgebra
################################################################################

param_seeiirnr_NB3_deathscases_daily = (;
# ODE Parameter
    β = Param(log.([0.15, 0.4, 1.25]),
        Bijectors.ordered(MvNormal(log.([0.15, 0.4, 1.25]), 1/2 .* Diagonal(ones(3))))
    ),
    γ = Param([.4, .5], [truncated(Gamma(1600,1/4000), .1, 1.5), truncated(Gamma(2500,1/5000), .1, 1.5)] ),
    ϵ = Param(1.0, truncated(Gamma(1000,1/1000), .1, 2.)),
    ρ = Param(0.5, Fixed()),
# Data Parameter
    ϕ_cases = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
    ϕ_deaths = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
# Latent state and transition parameter
    latent = Param(sample_states, Fixed()),
    p_states = Param([[1.], [1.]], [Fixed(), Fixed()]),
    p_thirdstate = Param([0.5, 0.5], Dirichlet(2,2)),
    r = Param([40., 30., 28.], [Gamma(40., 1.), Gamma(30., 1.), Gamma(28., 1.)]),
    ψ = Param([0.5,0.5,0.5], [Beta(5,5),Beta(5,5),Beta(5,5)]),
)
param_seeiirnr_NB4_deathscases_daily = (;
# ODE Parameter
    β = Param(log.([0.15, 0.4, 0.6, 1.20]),
        Bijectors.ordered(MvNormal(log.([0.15, 0.4, 0.6, 1.20]), 1/2 .* Diagonal(ones(4))))
    ),
    γ = Param([.4, .5], [truncated(Gamma(1600,1/4000), .1, 1.5), truncated(Gamma(2500,1/5000), .1, 1.5)] ),
    ϵ = Param(1.0, truncated(Gamma(1000,1/1000), .1, 2.)),
    ρ = Param(0.5, Fixed()),
# Data Parameter
    ϕ_cases = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
    ϕ_deaths = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
# Latent state and transition parameter
    latent = Param(sample_states, Fixed()),
    p_states = Param([[.7, .3], [.6, .4], [.5, .5]], [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)]),
    p_thirdstate = Param([.33, .33, .34], Dirichlet(3,3)),
    r = Param([40., 30., 20.0, 28.], [Gamma(40., 1.), Gamma(30., 1.), Gamma(20., 1.), Gamma(28., 1.)]),
    ψ = Param([0.5,0.5,0.5,0.5], [Beta(5,5),Beta(5,5),Beta(5,5),Beta(5,5)]),
)

param_seeiirnr_NB5_deathscases_daily = (;
# ODE Parameter
    β = Param(log.([0.15, 0.3, 0.4, 0.6, 1.25]),
        Bijectors.ordered(MvNormal(log.([0.15, 0.3, 0.4, 0.6, 1.25]), 1/2 .* Diagonal(ones(5))))
    ),
    γ = Param([.4, .5], [truncated(Gamma(1600,1/4000), .1, 1.5), truncated(Gamma(2500,1/5000), .1, 1.5)] ),
    ϵ = Param(1.0, truncated(Gamma(1000,1/1000), .1, 2.)),
    ρ = Param(0.5, Fixed()),
# Data Parameter
    ϕ_cases = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
    ϕ_deaths = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
# Latent state and transition parameter
    latent = Param(sample_states, Fixed()),
    p_states = Param([[.5, .3, .2], [.4, .4, .2], [.33, .33, .34], [.33, .33, .34]], [Dirichlet(3,3), Dirichlet(3,3), Dirichlet(3,3), Dirichlet(3,3)]),
    p_thirdstate = Param([.25, .25, .25, .25], Dirichlet(4,4)),
    r = Param([40., 30., 30., 20.0, 28.], [Gamma(40., 1.), Gamma(30., 1.), Gamma(30., 1.), Gamma(20., 1.), Gamma(28., 1.)]),
    ψ = Param([0.5, 0.5, 0.5, 0.5, 0.5], [Beta(5,5),Beta(5,5),Beta(5,5),Beta(5,5),Beta(5,5)]),
)

param_seeiirnr_NB4_deathscases_daily_posteriormean = (;
# ODE Parameter
    β = Param( [-1.86, -1.38, -0.74, 0.25], #[-2.5, -1.75, -1.5, -1.0], #[-1.86, -1.38, -0.74, 0.48]
        Bijectors.ordered(MvNormal([-1.86, -1.38, -0.74, 0.25], 1/2 .* Diagonal(ones(4))))
    ),
    γ = Param([.4, .5], [truncated(Gamma(160,1/400), .1, 1.5), truncated(Gamma(250,1/500), .1, 1.5)] ),
    ϵ = Param(1.0, truncated(Gamma(100,1/100), .1, 2.)),
    ρ = Param(0.5, Fixed()),
# Data Parameter
    ϕ_cases = Param(4.8, truncated(Gamma(2500, 1/500), 2.5, 20.)),
    ϕ_deaths = Param(5.2, truncated(Gamma(2500, 1/500), 2.5, 20.)),
# Latent state and transition parameter
    latent = Param(sample_states, Fixed()),
    p_states = Param([[0.878, 1-0.878], [0.588, 1-0.588], [0.249, 1-0.249]], [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)]),
#    p_states = Param([[0.95, 1-0.95], [0.8, 1-0.8], [0.8, 1-0.8]], [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)]),
    p_thirdstate = Param([0.35, 0.344, (1-0.344-0.35)], Dirichlet(3,3)),
    r = Param([34.205, 22.107, 13.773, 27.777], [Gamma(40., 1.), Gamma(30., 1.), Gamma(20., 1.), Gamma(28., 1.)]),
    ψ = Param([0.772, 0.644, 0.34, 0.493], [Beta(5,5),Beta(5,5),Beta(5,5),Beta(5,5)])
)

param_seeiirnr_P4_deathscases_daily = (;
# ODE Parameter
    β = Param(log.([0.15, 0.4, 0.6, 1.25]), #, 0.75
        Bijectors.ordered(MvNormal(log.([0.15, 0.4, 0.6, 1.25]), 1/2 .* Diagonal(ones(4))))
    ),
    γ = Param([.4, .5], [truncated(Gamma(1600,1/4000), .1, 1.5), truncated(Gamma(2500,1/5000), .1, 1.5)] ),
    ϵ = Param(1.0, truncated(Gamma(1000,1/1000), .1, 2.)),
    ρ = Param(0.5, Fixed()),
# Data Parameter
    ϕ_cases = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
    ϕ_deaths = Param(5.0, truncated(Gamma(2500, 1/500), 2.5, 20.)),
# Latent state and transition parameter
    latent = Param(sample_states, Fixed()),
    p_states = Param([[.7, .3], [.6, .4], [.5, .5]], [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)]),
    p_thirdstate = Param([.33, .33, .34], Dirichlet(3,3)),
    λ = Param([40., 30., 20., 28.], [Gamma(40., 1.), Gamma(30., 1.), Gamma(20., 1.), Gamma(28., 1.)]),
)

args_param_seeiirnr_deathscases_daily = (;
    state₀ = sample_state0_seeiir,
    vaccinations = data_vaccinations_immunity,
    gamma_change = gamma_change,
    N = UK_population,
    ifr = approx_ifr,
    reverse_death_probability = reverse_death_probability,
    index_change_ifr = index_change_ifr,
    underreporting = underreporting_vec
)


################################################################################
# Models

struct SEEIIRNR_P3_deathscases <: ModelName end
struct SEEIIRNR_P4_deathscases <: ModelName end

struct SEEIIRNR_NB3_deathscases <: ModelName end
struct SEEIIRNR_NB4_deathscases <: ModelName end
struct SEEIIRNR_NB5_deathscases <: ModelName end

struct SEEIIRNR_G4_deathscases <: ModelName end

seeiirnr_P4_deathscases = ModelWrapper(SEEIIRNR_P4_deathscases(), param_seeiirnr_P4_deathscases_daily, args_param_seeiirnr_deathscases_daily)

seeiirnr_NB3_deathscases = ModelWrapper(SEEIIRNR_NB3_deathscases(), param_seeiirnr_NB3_deathscases_daily, args_param_seeiirnr_deathscases_daily)
seeiirnr_NB4_deathscases = ModelWrapper(SEEIIRNR_NB4_deathscases(), param_seeiirnr_NB4_deathscases_daily, args_param_seeiirnr_deathscases_daily)
seeiirnr_NB5_deathscases = ModelWrapper(SEEIIRNR_NB5_deathscases(), param_seeiirnr_NB5_deathscases_daily, args_param_seeiirnr_deathscases_daily)

seeiirnr_NB4_deathscases_posteriormean = ModelWrapper(SEEIIRNR_NB4_deathscases(), param_seeiirnr_NB4_deathscases_daily_posteriormean, args_param_seeiirnr_deathscases_daily)

################################################################################
"Generate state and duration dynamics from parameter."
function get_dynamics(model::ModelWrapper{<:D}, θ) where {D<:Union{SEEIIRNR_P3_deathscases, SEEIIRNR_P4_deathscases}}
    @unpack p_states, p_thirdstate, λ = θ
    F = eltype(p_thirdstate)
    dynamicsᵈ = [Poisson(λ[iter]) for iter in eachindex(λ)]
    dynamicsˢ = [Categorical(extend_state_NR(F, p_states[iter], iter)) for iter in eachindex(p_states)]
    push!(dynamicsˢ, Categorical(extend_state(p_thirdstate, length(p_states) + 1) ))
    return dynamicsˢ, dynamicsᵈ
end
function get_dynamics(model::ModelWrapper{<:D}, θ) where {D<:Union{SEEIIRNR_NB3_deathscases, SEEIIRNR_NB4_deathscases, SEEIIRNR_NB5_deathscases}}
    @unpack p_states, p_thirdstate, r, ψ = θ
    F = eltype(p_thirdstate)
    dynamicsᵈ = [ NegativeBinomial(r[iter], ψ[iter]) for iter in eachindex(r) ]
    dynamicsˢ = [Categorical(extend_state_NR(F, p_states[iter], iter)) for iter in eachindex(p_states)]
    push!(dynamicsˢ, Categorical(extend_state(p_thirdstate, length(p_states) + 1) ))
    return dynamicsˢ, dynamicsᵈ
end

function get_dynamics(model::ModelWrapper{<:D}, θ) where {D<:Union{SEEIIRNR_G4_deathscases}}
    @unpack p_states, p_thirdstate, geo = θ
    F = eltype(p_thirdstate)
    dynamicsᵈ = [ Geometric(geo[iter]) for iter in eachindex(geo) ]
    dynamicsˢ = [Categorical(extend_state_NR(F, p_states[iter], iter)) for iter in eachindex(p_states)]
    push!(dynamicsˢ, Categorical(extend_state(p_thirdstate, length(p_states) + 1) ))
    return dynamicsˢ, dynamicsᵈ
end

################################################################################
# Generate Data
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{<:D}, Nsamples = 568) where {D<:Union{SEEIIRNR_P3_deathscases, SEEIIRNR_P4_deathscases, SEEIIRNR_NB3_deathscases, SEEIIRNR_NB4_deathscases, SEEIIRNR_NB5_deathscases, SEEIIRNR_G4_deathscases}}
    # Get Dynamics
    @unpack β, γ, ϵ, ϕ_cases, ϕ_deaths, ρ = model.val
    @unpack state₀, vaccinations, gamma_change, N, ifr, reverse_death_probability, index_change_ifr, underreporting = model.arg
    β = exp.(β)
    dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)
    # Assign initial states
    u0 = initial_state_to_population_seeiir(eltype(β), state₀, 300., N)
    # Assign buffer
    latentⁿᵉʷ = [(0,0) for _ in 1:Nsamples]
    states = [zeros(5) for _ in 1:Nsamples]
    cases = zeros(Float64, Nsamples)
    deaths = zeros(Float64, Nsamples)
    #
    iter = 1
    stateₜ = u0
# First sample states - always start with third state
    sₜ = Int(length(β)) #rand(rng, dynamicsˢ[rand(1:length(dynamicsˢ))])
    dₜ = rand(rng, dynamicsᵈ[sₜ])
    latentⁿᵉʷ[iter] = (sₜ, dₜ)
# Now Solve ODE for cases and states
    tspan = (float(iter-1), float(iter))
# Solve ODE
    βₜ = β[latentⁿᵉʷ[iter][1]]
    vaccinationsₜ = vaccinations[1]
    γₜ = gamma_interval(gamma_change, 1, γ)
    p = [βₜ, vaccinationsₜ, γₜ, ϵ, ρ]
    prob = ODEProblem(seeiir_ode!, stateₜ, tspan, p)
    sol = solve(prob, Euler(), dt = 0.001, saveat = 1.0)
# Compute Model implied cases and return model distribution
    states[iter] = sol.u[end]
    cases[iter] = cases_from_sol(sol) #sol.u[end][end] - sol.u[end-1][end]
    deaths[1] = 0.01
    death_lookback = length(reverse_death_probability)
    hyperparam = CovidDeathHyperParameter(ifr, index_change_ifr, reverse_death_probability, length(reverse_death_probability))
# Iterate over time
    for iter in 2:size(cases,1)
        # Sample states
        if dₜ > 0
            dₜ -=  1
            latentⁿᵉʷ[iter] = (sₜ, dₜ)
        else
            sₜ = rand(rng, dynamicsˢ[sₜ]) #stateₜ for t-1 overwritten
            dₜ = rand(rng, dynamicsᵈ[sₜ])
            latentⁿᵉʷ[iter] = (sₜ, dₜ)
        end
# Solve ODE
        tspan = (float(iter-1), float(iter))
# Solve ODE
        βₜ = β[latentⁿᵉʷ[iter][1]]
        vaccinationsₜ = vaccinations[iter]
        γₜ = gamma_interval(gamma_change, iter, γ)
        p[1] = βₜ
        p[2] = vaccinationsₜ
        p[3] = γₜ
        prob = ODEProblem(seeiir_ode!, states[iter-1], tspan, p)
        sol = solve(prob, Tsit5(), dt = 0.001, saveat = 1.0)
# Compute Model implied cases and return model distribution
        states[iter] = sol.u[end]
        cases[iter] = cases_from_sol(sol) #sol.u[end][end] - sol.u[end-1][end]
# Compute model implied deaths
        # iter-1 because deaths at t determined from cases up to t-1
        deaths[iter] = accumulate_deaths(cases, hyperparam, iter-1)
    end
        # Now assign noise around deaths:
        deaths_noisy =  [ rand(NegativeBinomial(ϕ_deaths, ϕ_deaths / (ϕ_deaths + deaths[iter]))) for iter in eachindex(deaths)]
        #!NOTE: Underreporting does NOT factor in here - is only used for logpdf, not for simulating data
        cases_noisy =   [ rand(NegativeBinomial(ϕ_cases, ϕ_cases / (ϕ_cases + cases[iter]))) for iter in eachindex(cases)]
# Return dates
    _data = [(deaths = deaths_noisy[iter], cases = cases_noisy[iter]) for iter in eachindex(deaths_noisy)]
    return _data, cases, [(latentⁿᵉʷ[idx], states[idx], cases[idx], deaths[idx]) for idx in 1:Nsamples]
end

function (objective::Objective{<:ModelWrapper{<:D}})(θ::NamedTuple) where {D<:Union{SEEIIRNR_P3_deathscases, SEEIIRNR_P4_deathscases, SEEIIRNR_NB3_deathscases, SEEIIRNR_NB4_deathscases, SEEIIRNR_NB5_deathscases, SEEIIRNR_G4_deathscases}}
    @unpack model, data, tagged = objective

    β = exp.(θ.β)
    if maximum(β) > 5.0 return -Inf end
    θ = merge(θ, (β = β, ), model.arg)
    ll_starting_time = 2
    # Prior
    lp = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
    # Likelihood
    #!NOTE: For Piecewise constant beta, need piecewise ODE solver - not marginally, else wrong gradients.
    sol_C, _ = get_incremental_ODE_SEEIIR(θ)
    @unpack latent, ϕ_cases, ϕ_deaths, underreporting, reverse_death_probability, ifr, index_change_ifr = θ
    hyperparam = CovidDeathHyperParameter(ifr, index_change_ifr, reverse_death_probability, length(reverse_death_probability))
    sol_D = deaths_from_cases(sol_C, hyperparam)
    # Compute ll and return log target function
    sol_deathscases = Sol_DeathsCases(sol_D, sol_C, ϕ_deaths, ϕ_cases)
    ll = cumulative_ll_nb_deathscases(sol_deathscases, data, underreporting, ll_starting_time)
    # Compute Latent state Parameter
    dynamicsˢ, dynamicsᵈ = get_dynamics(model, θ)
    ll_states = add_ll_states(dynamicsˢ, dynamicsᵈ, latent, ll_starting_time)
    return ll + lp + ll_states
end

############################################################################################
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{<:D}}) where {D<:Union{SEEIIRNR_P3_deathscases, SEEIIRNR_P4_deathscases, SEEIIRNR_NB3_deathscases, SEEIIRNR_NB4_deathscases, SEEIIRNR_NB5_deathscases, SEEIIRNR_G4_deathscases}}
    @unpack model, data = objective
    @unpack β, γ, ϵ, ϕ_cases, ϕ_deaths, ρ = model.val
    @unpack state₀, vaccinations, gamma_change, N, ifr, reverse_death_probability, index_change_ifr, underreporting = model.arg
    β = exp.(β)
    death_lookback = length(reverse_death_probability)
    hyperparam = CovidDeathHyperParameter(ifr, index_change_ifr, reverse_death_probability, death_lookback)
## Initial state
    u0 = initial_state_to_population_seeiir(typeof(ϵ), state₀, 300., N)
## Assign ODE closure
    param = (β, vaccinations, γ, ϵ, ρ, gamma_change)
    tspan = (0.0, 1.0)
    prob = ODEProblem(seeiir_ode!,
          u0,
          tspan,
          (β[begin], vaccinations[begin], γ[begin], ϵ, ρ))
## Assign PF distributions
    dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)
    #!NOTE: Initial particle always from final state
    p_initial = zeros(length(β))
    p_initial[end] = 1.0
    initialˢ = Categorical(p_initial)
    initialᵈ(sₜ) = dynamicsᵈ[sₜ]
    initial = SemiMarkovInitiation(initialˢ, initialᵈ)

    state(particles, iter) = dynamicsˢ[obtain_s(particles, iter-1)]
    duration(s, iter) = dynamicsᵈ[s]
    transition = SemiMarkovTransition(state, duration)

    to_NB = from_ODE_to_NB_deathscases(model.val, underreporting)
    observation(particles, iter) = to_NB(particles, iter)
    return SemiMarkovSEEIIR_deaths(initial, transition, observation, prob, u0, param, hyperparam) #, u0, param, N)
end

deaths_noisy_seeiirnr_P4, cases_seeiirnr_P4, lat_seeiirnr_P4 = ModelWrappers.simulate(_rng, seeiirnr_P4_deathscases, Int(sample_tmax))
fill!(seeiirnr_P4_deathscases, Tagged(seeiirnr_P4_deathscases, :latent), (; latent = lat_seeiirnr_P4,))
seeiirnr_P4_deathscases.val.latent

###
deaths_noisy_seeiirnr_NB4, cases_seeiirnr_NB4, lat_seeiirnr_NB4 = ModelWrappers.simulate(_rng, seeiirnr_NB4_deathscases, Int(sample_tmax))
fill!(seeiirnr_NB4_deathscases, Tagged(seeiirnr_NB4_deathscases, :latent), (; latent = lat_seeiirnr_NB4,))
seeiirnr_NB4_deathscases.val.latent

deaths_noisy_seeiirnr_NB3, cases_seeiirnr_NB3, lat_seeiirnr_NB3 = ModelWrappers.simulate(_rng, seeiirnr_NB3_deathscases, Int(sample_tmax))
fill!(seeiirnr_NB3_deathscases, Tagged(seeiirnr_NB3_deathscases, :latent), (; latent = lat_seeiirnr_NB3,))
seeiirnr_NB3_deathscases.val.latent

deaths_noisy_seeiirnr_NB5, cases_seeiirnr_NB5, lat_seeiirnr_NB5 = ModelWrappers.simulate(_rng, seeiirnr_NB5_deathscases, Int(sample_tmax))
fill!(seeiirnr_NB5_deathscases, Tagged(seeiirnr_NB5_deathscases, :latent), (; latent = lat_seeiirnr_NB5,))
seeiirnr_NB5_deathscases.val.latent

deaths_noisy_seeiirnr_NB4_posteriormean, cases_seeiirnr_NB4_posteriormean, lat_seeiirnr_NB4_posteriormean = ModelWrappers.simulate(_rng, seeiirnr_NB4_deathscases_posteriormean, Int(sample_tmax))
fill!(seeiirnr_NB4_deathscases_posteriormean, Tagged(seeiirnr_NB4_deathscases_posteriormean, :latent), (; latent = lat_seeiirnr_NB4_posteriormean,))
seeiirnr_NB4_deathscases_posteriormean.val.latent
