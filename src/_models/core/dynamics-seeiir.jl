############################################################################################
# Define Particle Filter and generate valid states
mutable struct SemiMarkovSEEIIR_cases{A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P} <: ParticleKernel
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
    function SemiMarkovSEEIIR_cases(
        initial::A, transition::B, evidence::C, problem::T, u0::R, param::P#state₀::T, N::R
    ) where {A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P} #,P,R<:Real}
        return new{A,B,C,T,R,P}(initial, transition, evidence, problem, u0, param) #state₀, param, N) ,P,R
    end
end

mutable struct SemiMarkovSEEIIR_deaths{A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P, H} <: ParticleKernel
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
    function SemiMarkovSEEIIR_deaths(
        initial::A, transition::B, evidence::C, problem::T, u0::R, param::P, hyperparam::H #state₀::T, N::R
    ) where {A<:SemiMarkovInitiation,B<:SemiMarkovTransition,C,T,R,P,H} #,P,R<:Real}
        return new{A,B,C,T,R,P,H}(initial, transition, evidence, problem, u0, param, hyperparam) #state₀, param, N) ,P,R
    end
end

############################################################################################
#!NOTE: Initial starts with initial (s,d) and kernel.state₀ -> we need to get first observation and cases from here.

function _initial(_rng::Random.AbstractRNG, kernel::K) where {K<:Union{SemiMarkovSEEIIR_cases, SemiMarkovSEEIIR_deaths}}
    # Obtain initial states
    s = rand(_rng, kernel.initial.state)
    d = rand(_rng, kernel.initial.duration(s))
    val_new = (s, d)

    state₀ = kernel.u0
    tspan = (float(0.0), float(1.0))
    # Solve ODE
    βₜ = kernel.param[1][val_new[1]] #!NOTE: Cannot use obtain_β as val_new is not yet in the particle trajectory
    vaccinationsₜ = kernel.param[2][1]
    γₜ = gamma_interval(kernel.param[6], 1, kernel.param[3])
    p = (βₜ, vaccinationsₜ, γₜ, kernel.param[4], kernel.param[5])
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

function BaytesFilters.initial(_rng::Random.AbstractRNG, kernel::SemiMarkovSEEIIR_cases)
    return _initial(_rng, kernel)
end

function BaytesFilters.initial(_rng::Random.AbstractRNG, kernel::SemiMarkovSEEIIR_deaths)
    val_new, state_new, cases = _initial(_rng, kernel)
    deaths = 0.01
    return (val_new, state_new, cases, deaths)
end

############################################################################################
function _transition(
    _rng::Random.AbstractRNG, kernel::K, val::AbstractArray{P}, iter::Integer
) where {K<:Union{SemiMarkovSEEIIR_cases, SemiMarkovSEEIIR_deaths}, P}
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
    vaccinationsₜ = kernel.param[2][iter]
    γₜ = gamma_interval(kernel.param[6], iter, kernel.param[3])
    p = (βₜ, vaccinationsₜ, γₜ, kernel.param[4], kernel.param[5])
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
    cases = cases_from_sol(sol) #cases_new >= 0.0 ? cases_new : 0.0
    # Return
    return (val_new, state_new, cases)
end

function BaytesFilters.transition(
    _rng::Random.AbstractRNG, kernel::SemiMarkovSEEIIR_cases, val::AbstractArray{P}, iter::Integer
) where {P}
    return _transition(_rng, kernel, val, iter)
end

function BaytesFilters.transition(
    _rng::Random.AbstractRNG, kernel::SemiMarkovSEEIIR_deaths, val::AbstractArray{P}, iter::Integer
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
) where {K<:Union{SemiMarkovSEEIIR_cases, SemiMarkovSEEIIR_deaths}, P}
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
