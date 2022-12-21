################################################################################
# HSMM helper
"Define function to extend Markov to Semi Markov Transition"
function extend_state(transition::T, state::R) where {T, R<:Real}
    transitionⁿᵉʷ = zeros( eltype(transition), length(transition) +1 )
    transitionⁿᵉʷ[1:end .!= state] .= transition
    return transitionⁿᵉʷ
end
function extend_state_NR(F, transition::T, state::R) where {T, R<:Real}
    transitionⁿᵉʷ = zeros(F, length(transition) +1 )
    transitionⁿᵉʷ[1:end .!= state] .= transition
    return push!(transitionⁿᵉʷ, 0.0)
end

################################################################################
#Check differences in values of hidden states (for checking pointer issues)
function latentdiff(latent₀::AbstractVector{Tuple{R, R}}, latent₁::AbstractVector{Tuple{R, R}}) where {R<:Real}
    s₀ = getfield.(latent₀, 1)
    d₀ = getfield.(latent₀, 2)
    s₁ = getfield.(latent₁, 1)
    d₁ = getfield.(latent₁, 2)
    return sum( abs.(s₀ - s₁)) + sum( abs.(d₀ - d₁))
end

function check_correctness(val::AbstractVector{T}) where {T}
## Get relevant fields
    s = getfield.(val, 1)
    d = getfield.(val, 2)
## Initate container that holds time when state changes
    StateIter = Int64[]
    DurationIter = Int64[]
## Compute all state changes
    statechanges = [s[iter]-s[iter-1] for iter in 2:length(s)]
    durationchanges = [d[iter]-d[iter-1] for iter in 2:length(d)]
## Get all iterations where s changes
    for iter in eachindex(statechanges)
        if statechanges[iter] != 0
            push!(StateIter, iter)
        end
    end
## Get all iterations where d changes
    for iter in eachindex(durationchanges)
        if durationchanges[iter] != -1
            push!(DurationIter, iter)
        end
    end
## Get all changes that are correct
    changes = [ StateIter[iter] == DurationIter[iter] for iter in eachindex(StateIter) ]
## Return total changes - correct changes (should b 0)
    return length(StateIter) - sum(changes)
end

#Check if HSMM has impossible transitions
function check_correctness(kernel::SemiMarkov, val::Vector{<:AbstractArray{T}}) where {T}
    return sum([check_correctness(val[iter]) for iter in eachindex(val)])
end
function check_correctness(kernel::SemiMarkov, val::AbstractMatrix{T}) where {T}
    return sum([check_correctness(val[iter, :]) for iter in Base.OneTo(size(val, 1))])
end

################################################################################
# HMM and HSMM helper for initial distribution
############################
"Compute Stationary distribution for given transition matrix"
function get_stationary!(Transition::AbstractMatrix{T}) where T<:Real
    # Taken From: https://github.com/QuantEcon/QuantEcon.jl/blob/f454d4dfbaf52f550ddd52eff52471e4b8fddb9d/src/markov/mc_tools.jl
    # Grassmann-Taksar-Heyman (GTH) algorithm (Grassmann, Taksar, and Heyman 1985)
    n = size(Transition, 1)
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = sum(Transition[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        Transition[k+1:n, k] /= scale

        for j in k+1:n, i in k+1:n
            Transition[i, j] += Transition[i, k] * Transition[k, j]
        end
    end

    # backsubstitution
    x[n] = 1
    @inbounds for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * Transition[i, k]
    end

    # normalisation
    x /= sum(x)

    return x
end
get_stationary(Transition::AbstractMatrix{T}) where {T<:Real} = get_stationary!(copy(Transition))
