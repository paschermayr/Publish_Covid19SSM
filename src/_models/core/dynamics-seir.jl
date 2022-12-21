############################################################################################
# Define Particle Filter and generate valid states
mutable struct SemiMarkovSEIR_cases{A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P} <: ParticleKernel
    "Initial distribution, function of iter only."
    initial::A
    "Transition distribution, function of full particle trajectory and current iteration count."
    transition::B
    "Data distribution to weight particles. Function of full data, particle trajectory and current iteration count."
    evidence::C
    "ODE Problem including initial conditions - serves as buffer."
    problem::T
    "Initial state"
    u0::R
    "Parameter"
    param::P
    function SemiMarkovSEIR_cases(
        initial::A, transition::B, evidence::C, problem::T, u0::R, param::P#state₀::T, N::R
    ) where {A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P} #,P,R<:Real}
        return new{A,B,C,T,R,P}(initial, transition, evidence, problem, u0, param) #state₀, param, N) ,P,R
    end
end

mutable struct SemiMarkovSEIR_deaths{A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P, H} <: ParticleKernel
    "Initial distribution, function of iter only."
    initial::A
    "Transition distribution, function of full particle trajectory and current iteration count."
    transition::B
    "Data distribution to weight particles. Function of full data, particle trajectory and current iteration count."
    evidence::C
    "ODE Problem including initial conditions - serves as buffer."
    problem::T
    "Initial state"
    u0::R
    "Parameter"
    param::P
    "HyperParameter for COVID Death Rates"
    hyperparam::H
    function SemiMarkovSEIR_deaths(
        initial::A, transition::B, evidence::C, problem::T, u0::R, param::P, hyperparam::H #state₀::T, N::R
    ) where {A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P,H} #,P,R<:Real}
        return new{A,B,C,T,R,P,H}(initial, transition, evidence, problem, u0, param, hyperparam) #state₀, param, N) ,P,R
    end
end

############################################################################################
#!NOTE: Initial starts with initial (s,d) and kernel.state₀ -> we need to get first observation and cases from here.

function _initial(_rng::Random.AbstractRNG, kernel::K) where {K<:Union{SemiMarkovSEIR_cases, SemiMarkovSEIR_deaths}}
    # Obtain initial states
    s = rand(_rng, kernel.initial.state)
    d = rand(_rng, kernel.initial.duration(s))
    val_new = (s, d)

    state₀ = kernel.u0
    tspan = (float(0.0), float(1.0))
    # Solve ODE
    βₜ = kernel.param[1][val_new[1]] #!NOTE: Cannot use obtain_β as val_new is not yet in the particle trajectory
    p = (βₜ, kernel.param[2], kernel.param[3])
    kernel.problem = remake(kernel.problem;
        u0 = state₀,
        tspan = tspan,
        p = p
    )
    sol = solve(kernel.problem, Tsit5(), saveat = tspan[end])
    # Compute Model implied cases and return model distribution
    state_new = sol.u[end]
    cases = cases_from_sol(sol) #cases_new >= 0.0 ? cases_new : 0.0
    # Return
    return (val_new, state_new, cases)
end

function BaytesFilters.initial(_rng::Random.AbstractRNG, kernel::SemiMarkovSEIR_cases)
    return _initial(_rng, kernel)
end

function BaytesFilters.initial(_rng::Random.AbstractRNG, kernel::SemiMarkovSEIR_deaths)
    val_new, state_new, cases = _initial(_rng, kernel)
    deaths = 0.01
    return (val_new, state_new, cases, deaths)
end

############################################################################################
function _transition(
    _rng::Random.AbstractRNG, kernel::K, val::AbstractArray{P}, iter::Integer
) where {K<:Union{SemiMarkovSEIR_cases, SemiMarkovSEIR_deaths}, P}
## Transition Particles
    if val[iter - 1][1][2] > 0
        val_new = (val[iter - 1][1][1], val[iter - 1][1][2] - 1)
    else
        #!NOTE: Inconsistent that s depends on whole Vector, but d only on current s - but no other way as of now
        sₜ = rand(_rng, kernel.transition.state(val, iter))
        dₜ = rand(_rng, kernel.transition.duration(sₜ, iter))
        val_new = (sₜ, dₜ)
    end
## Solve ODE given current states as initial state
    # Assign parameter
    state₀ = val[iter - 1][2]
    tspan = (float(iter-1), float(iter))
    βₜ = kernel.param[1][val_new[1]] #!NOTE: Cannot use obtain_β as val_new is not yet in the particle trajectory
    p = (βₜ, kernel.param[2], kernel.param[3])
    # Solve ODE
    kernel.problem = remake(kernel.problem;
        u0 = state₀,
        tspan =tspan,
        p = p
    )
    sol = solve(kernel.problem,
        Tsit5(),
        saveat = float(iter)
    )
    # Compute Model implied cases and return model distribution
    state_new = sol.u[end]
    cases = cases_from_sol(sol)  #cases_new >= 0.0 ? cases_new : 0.0
    # Return
    return (val_new, state_new, cases)
end

function BaytesFilters.transition(
    _rng::Random.AbstractRNG, kernel::SemiMarkovSEIR_cases, val::AbstractArray{P}, iter::Integer
) where {P}
    return _transition(_rng, kernel, val, iter)
end

function BaytesFilters.transition(
    _rng::Random.AbstractRNG, kernel::SemiMarkovSEIR_deaths, val::AbstractArray{P}, iter::Integer
) where {P}
    val_new, state_new, cases = _transition(_rng, kernel, val, iter)
    # Deaths can be computed from cases and deaths at t-1
    deaths = accumulate_deaths_pf(val, kernel.hyperparam, iter-1)
    # Return
    return (val_new, state_new, cases, deaths)
end

############################################################################################
function BaytesFilters.ℓtransition(
    valₜ::Union{P,AbstractArray{P}},
    kernel::K,
    val::AbstractArray{P},
    iter::Integer,
) where {K<:Union{SemiMarkovSEIR_cases, SemiMarkovSEIR_deaths}, P}
    sₜ₋₁, dₜ₋₁ = val[iter - 1][1]
    sₜ, dₜ = valₜ[1]
    #!NOTE: If duration at t-1 is 0, and particles states are not the same from t-1 to t
    if dₜ₋₁ == 0 && sₜ₋₁ != sₜ
        ℓπ = logpdf(kernel.transition.state(val, iter), sₜ) #current state given past state
        ℓπ += logpdf(kernel.transition.duration(sₜ, iter), dₜ) #duration given current state
        return ℓπ
    elseif (dₜ₋₁ - 1) == (dₜ) && sₜ₋₁ == sₜ
        return 0.0 #log(1.0) = 0
    else
        return -Inf
    end
end

#!NOTE: This will have cases_smoothed as mean and ϕ as dispersion parameter
function from_ODE_to_NB_cases(θ, underreporting)
    @unpack ϕ = θ
    function to_NB(particles, iter)
        sol_X = particles[iter][3]
        underreporting_x = underreporting[iter]
        return NegativeBinomial(ϕ, ϕ / (ϕ + (underreporting_x * sol_X)))
    end
end
function from_ODE_to_NB_deaths(θ)
    @unpack ϕ = θ
    function to_NB(particles, iter)
        sol_X = particles[iter][4]
        return NegativeBinomial(ϕ, ϕ / (ϕ + sol_X))
    end
end

function from_ODE_to_Poisson_cases(θ, underreporting)
    function to_Poisson(particles, iter)
        sol_X = particles[iter][3]
        underreporting_x = underreporting[iter]
        return Poisson(underreporting_x * sol_X)
    end
end
function from_ODE_to_Poisson_deaths(θ)
    function to_Poisson(particles, iter)
        sol_X = particles[iter][4]
        return Poisson(sol_X)
    end
end


################################################################################
################################################################################
# Deathscases needs new distribution that can evaluate both cases and deaths
using Distributions
import Distributions: rand, logpdf, DiscreteUnivariateDistribution

struct DeathsCasesDistribution{A<:NegativeBinomial, B<:NegativeBinomial} <: Distributions.DiscreteUnivariateDistribution
    deaths        ::  B
    cases         ::  A
end

function logpdf(d::DeathsCasesDistribution, data::NamedTuple)
    ℓdeaths = logpdf(d.deaths, data.deaths)
    ℓcases  = logpdf(d.cases, data.cases)
    return ℓdeaths + ℓcases
end

function rand(rng::Random.AbstractRNG, d::DeathsCasesDistribution)
    case = rand(rng, d.cases)
    death = rand(rng, d.deaths)
    return (deaths = death, cases = case)
end

#!NOTE: This will have cases_smoothed as mean and ϕ as dispersion parameter
function from_ODE_to_NB_deathscases(θ, underreporting)
    @unpack ϕ_cases, ϕ_deaths = θ #, cases_reported
    function to_NB(particles, iter)
        # Model implied death
        sol_Cases = particles[iter][3]
        # Model implied case
        sol_Deaths = particles[iter][4]
        # Underreporting factor
        sol_underreporting = underreporting[iter]
        # Assign distributions
        d_death = NegativeBinomial(ϕ_deaths, ϕ_deaths / (ϕ_deaths + sol_Deaths) )
        d_case = NegativeBinomial(ϕ_cases, ϕ_cases / (ϕ_cases + (sol_underreporting * sol_Cases) ) )

        return DeathsCasesDistribution(d_death, d_case)
    end
end
