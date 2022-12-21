using DifferentialEquations
using Random, Distributions, BenchmarkTools, UnPack, ArgCheck
using Dates, JLD2
using ModelWrappers, BaytesFilters, Baytes, BaytesInference

################################################################################
# Settings that are shared among different models

#Set Number of datapoints
_Ndata = 568

#MCMC/PMCMC setting
Niterations = 1000
Nchains = 4
Nburnin = 500

#SMC setting
NchainsSMC = Int(48)
NinitSMC = 180 # 6months
smc_tuned = true #Use tuned PMCMC kernel for SMC?
_NtuningSMC = 10
Nparticlemultiplier = 2.0

println("Number of data points: ", _Ndata)
println("Number of SMC chains: ", NchainsSMC)

################################################################################
#Data
using JLD2
deaths = jldopen(string(pwd(), "/src/data/UK_deaths_20220407.jld2"))
deaths_raw = ( read(deaths, "deaths_daily") )
deaths_dates_raw = ( read(deaths, "deaths_dates_daily") )

cases = jldopen(string(pwd(), "/src/data/UK_cases_20220407.jld2"))
cases_raw = ( read(cases, "cases_daily") )
cases_dates_raw = ( read(cases, "cases_dates_daily") )

#!NOTE: Cases at t show the reported cases from t-1, so fix this such that they are aligned with model implied cases.
#!NOTE: need to put t to t-1 for cases
cases_raw = vcat(cases_raw[2:end], cases_raw[end])

vaccinations = jldopen(string(pwd(), "/src/data/UK_vaccines_20220530.jld2"))
vaccinations_raw = ( read(vaccinations, "vaccinations_daily") )
vaccinations_dates_raw = ( read(vaccinations, "vaccinations_dates_daily") )

################################################################################
################################################################################
################################################################################
#Assign data for Sampling
if _dataadjustments == "deaths"
    data_temp = deaths_raw
    dates_temp = deaths_dates_raw
    _pfmemory = ParticleFilterMemory(29, 29, 1)
elseif  _dataadjustments == "cases"
    data_temp = cases_raw
    dates_temp = cases_dates_raw
    _pfmemory = ParticleFilterMemory(1,0,1)
elseif _dataadjustments == "deathscases"
    data_temp = deaths_raw
    dates_temp = deaths_dates_raw
    _pfmemory = ParticleFilterMemory(29, 29, 1)
    #Make data a NamedTuple with deaths and cases, where cases start where deaths start
    data_temp2 = cases_raw
else error(" _dataadjustments wrong")
end

################################################################################
#Start series when at least 10 deaths occured to avoid noise, see Flaxman et al. (2020), and end on 2021-09-30, to use ifrs values from papers
first_date = findfirst( data_temp .> 10)
last_date = findfirst( string.(dates_temp) .== "2021-09-30")
dates_adjusted = dates_temp[first_date:last_date]
data_adjusted = data_temp[first_date:last_date]

################################################################################
## Assign vaccinations based on dates
_firstdata = findfirst( string(vaccinations_dates_raw[begin]) .== string.(dates_temp))
vaccinations_dates_raw[begin]
dates_temp[_firstdata]

zerovaccines = zeros(Int64, _firstdata - 1)
vaccinations_temp = identity.(vcat(zerovaccines, vaccinations_raw))
vaccinations_adjusted = vaccinations_temp[first_date:last_date]

################################################################################
# not relevant anymore - Adjust data to per 1000 -> work with UK_population so not too much noise from rounding
base_population = 1.0e6
UK_population = 67886011    #6.722 * 1.0e7
data = data_adjusted
data_dates = dates_adjusted
data_vaccinations = vaccinations_adjusted

firstdoseimmunity = 45
#!NOTE: Length + 1 so can predict
data_vaccinations_immunity = vcat(zeros(Float64, firstdoseimmunity), data_vaccinations)[begin:length(data_vaccinations)+1]
date_first_vaccinations_immunity = findfirst( data_vaccinations_immunity .> 0 )

#Check when 1/1/2021 for gamma change
gamma_change = findfirst( string.(data_dates) .== "2021-01-01")

#For plotting, set cases equal to deaths
date_cases_for_deaths_start = findfirst( string.(cases_dates_raw) .== string(data_dates[begin]))
date_cases_for_deaths_end = findfirst( string.(cases_dates_raw) .== string(data_dates[end]))
data_cases_for_deaths = cases_raw[date_cases_for_deaths_start:date_cases_for_deaths_end]

date_deaths_for_cases_start = findfirst( string.(deaths_dates_raw) .== string(data_dates[begin]))
date_deaths_for_cases_end = findfirst( string.(deaths_dates_raw) .== string(data_dates[end]))
if date_deaths_for_cases_start isa Nothing
    date_deaths_for_cases_start = 1
    data_deaths_for_cases = deaths_raw[date_deaths_for_cases_start:date_deaths_for_cases_end]
    #push until equal length
    data_deaths_for_cases = vcat( zeros(length(data) - length(data_deaths_for_cases)), data_deaths_for_cases)
else
    data_deaths_for_cases = deaths_raw[date_deaths_for_cases_start:date_deaths_for_cases_end]
end

#!NOTE: Use case as Fixed Parameter, so can keep matrix structure for particle filter
#If both deaths and cases are used, make data to NamedTuple with deaths and cases, starting from data point of first death
if _dataadjustments == "deathscases"
    #deaths and cases
    data = [(deaths = data[iter], cases = data_cases_for_deaths[iter]) for iter in eachindex(data)]
end


################################################################################
# Maximum count for death probabilities
t_last = 28

# Dates for changing ifr probabilities
date_change_ifr = [
    Date(2020,07,18),
    Date(2020,10,01),
    Date(2021,01,30),
    Date(2021,06,01)
]
index_change_ifr = [findfirst( string.(data_dates) .== string(date_change_ifr[iter])) for iter in eachindex(date_change_ifr)]

# Initial ifr values
# Posterior values from Anastasia
_multiplier = 2.0
approx_ifr = _multiplier .* [0.01035, 0.0095, 0.007245, 0.004, 0.002] #[0.010350, 0.007245, 0.009500, 0.00004, 0.000018]
#this one is the one we have from sampling
approx_ifr_prior = [truncated(Normal(approx_ifr[iter], 1), 0.002, approx_ifr[iter]*_multiplier) for iter in eachindex(approx_ifr)]

################################################################################
sample_tmax = float( length(data) )
_rng = Random.Xoshiro(123) #use Xoshiro for thread safety

################################################################################
# Hyper Parameter FOR DATA GENERATION and Initial conditions
sample_tspan = (0.0, sample_tmax)
sample_N = UK_population #1.0e6
sample_s₀ = (67886011 - 1800 - 600)/67886011  #0.99
sample_e₀ = 1800/67886011 #0.0033
sample_i₀ = 600/67886011 #0.0033
sample_r₀ = 0/67886011 #1.0 - sample_s₀ - sample_e₀ - sample_i₀

sample_state0 = [sample_s₀, sample_e₀, sample_i₀, sample_r₀] # S,E,I,R
sum(sample_state0)
sample_state0 ./= sum(sample_state0)
@argcheck sum(sample_state0) ≈ 1

sample_state0_seeiir = [67883011.,900,900,600,600,0]
sample_state0_seeiir ./= sum(sample_state0_seeiir)
@argcheck sum(sample_state0_seeiir) ≈ 1

################################################################################
# Assign UnderReporting vector
break_1_60 = findfirst( string.(data_dates) .== "2020-08-15" )
break_2_30 = findfirst( string.(data_dates) .== "2020-12-01" )
break_3_100 = findfirst( string.(data_dates) .== "2021-04-01" )

break_1_60_vec = collect( range(0.1, 0.6, break_1_60) )
break_2_30_vec = collect( range(0.6, 0.3, break_2_30-break_1_60) )
break_3_100_vec = collect( range(0.3, 1.0, break_3_100-break_2_30) )

underreporting_vec = vcat(break_1_60_vec, break_2_30_vec, break_3_100_vec)
# +1 to have predictive available
underreporting_vec = vcat(underreporting_vec, repeat([1.0], length(data_dates) - length(underreporting_vec) + 1 ) )
#plot(underreporting_vec)

################################################################################
#Parameter that are shared for all SSMs

# Initial PF trajectory - will be overwritten after data is sampled
sample_u0 = u0 = zeros(Float64, 5) #initial_state_to_population(Float64, sample_state0, sample_N) # S,E,I,R,C
sample_states = [((rand(1:2), rand(1:10)), sample_u0, sample_u0[end], sample_u0[end]) for _ in 1:sample_tmax]
include("_models/core/core.jl")
