using Parameters
using ProgressMeter
using QuantEcon: MarkovChain, stationary_distributions
using QuadGK: quadgk
using Distributions
using Roots: find_zero
using Interpolations: linear_interpolation

function adda_cooper(N::Integer, ρ::Real, σ::Real; μ::Real=0.0)
    """
    Approximation of an autoregression process with a Markov chain proposed by Adda and Cooper (2003)
    """

    σ_ϵ = σ / sqrt(1.0 - ρ^2.0)
    ϵ = σ_ϵ .* quantile.(Normal(), [i / N for i = 0:N]) .+ μ
    z = zeros(N)
    for i = 1:N
        if i != (N + 1) / 2
            z[i] = N * σ_ϵ * (pdf(Normal(), (ϵ[i] - μ) / σ_ϵ) - pdf(Normal(), (ϵ[i+1] - μ) / σ_ϵ)) + μ
        end
    end
    Π = zeros(N, N)
    if ρ == 0.0
        Π .= 1.0 / N
    else
        for i = 1:N, j = 1:N
            f(u) = exp(-(u - μ)^2.0 / (2.0 * σ_ϵ^2.0)) * (cdf(Normal(), (ϵ[j+1] - μ * (1.0 - ρ) - ρ * u) / σ) - cdf(Normal(), (ϵ[j] - μ * (1.0 - ρ) - ρ * u) / σ))
            integral, err = quadgk(u -> f(u), ϵ[i], ϵ[i+1])
            Π[i, j] = (N / sqrt(2.0 * π * σ_ϵ^2.0)) * integral
        end
    end
    return z, Π
end

function parameters_function(;
    life_span::Int64=16,                # model life span
    period::Int64=3,                    # model period
    β::Float64=0.9730,                  # discount factor
    r_f::Float64=0.0344,                # risk-free rate 
    τ::Float64=0.0093,                  # transaction cost
    γ::Float64=2.00,                    # risk aversion coefficient
    ω::Float64=0.56,                    # consumption weight
    T::Float64=1.50,                    # time endowment
    ϕ::Float64=0.395,                   # wage garnishment rate
    ψ::Float64=0.011,                   # divorce probability
    e_m_ρ::Float64=0.9730,              # AR(1) of male persistent income
    e_m_σ::Float64=sqrt(0.016),         # s.d. of male persistent income 
    e_m_size::Int64=5,                  # number of male persistent income 
    a_min::Float64=-5.0,                # min of asset holding
    a_max::Float64=800.0,               # max of asset holding
    a_size_neg::Int64=501,              # number of grid of negative asset holding for VFI
    a_size_pos::Int64=101,              # number of grid of positive asset holding for VFI
    a_degree::Int64=3,                  # curvature of the positive asset gridpoints
)
    """
    contruct an immutable object containg all paramters
    """

    # lifecycle profile (Gourinchas and Parker, 2002)
    h_grid = [0.774482122, 0.819574547, 0.873895492, 0.9318168, 0.986069673, 1.036889326, 1.082870993, 1.121249981, 1.148476948, 1.161069822, 1.156650443, 1.134940682, 1.09844343, 1.05261516, 1.005569967, 0.9519]
    h_size = length(h_grid)

    # male persistent income
    e_m_grid, e_m_Γ = adda_cooper(e_m_size, e_m_ρ, e_m_σ)
    e_m_G = stationary_distributions(MarkovChain(e_m_Γ, e_m_grid))[1]

    # expenditure schock
    κ_grid = zeros(life_span)
    κ_size = length(κ_grid)

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = ((range(0.0, stop=a_size_pos - 1, length=a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims=1)
    a_size = length(a_grid)
    a_ind_zero = a_size_neg

    # iterators
    loop_VFI = collect(Iterators.product(1:κ_size, 1:e_m_size, 1:a_size))
    loop_thres_a = collect(Iterators.product(1:κ_size, 1:e_m_size))
    loop_thres_e = collect(Iterators.product(1:κ_size, 1:a_size_neg))

    # return values
    return (
        life_span=life_span,
        period=period,
        β=β,
        r_f=r_f,
        τ=τ,
        γ=γ,
        ω=ω,
        T=T,
        ϕ=ϕ,
        ψ=ψ,
        h_grid=h_grid,
        h_size=h_size,
        e_m_ρ=e_m_ρ,
        e_m_σ=e_m_σ,
        e_m_size=e_m_size,
        e_m_grid=e_m_grid,
        e_m_Γ=e_m_Γ,
        e_m_G=e_m_G,
        κ_grid=κ_grid,
        κ_size=κ_size,
        a_grid=a_grid,
        a_grid_neg=a_grid_neg,
        a_grid_pos=a_grid_pos,
        a_size=a_size,
        a_size_neg=a_size_neg,
        a_size_pos=a_size_pos,
        a_ind_zero=a_ind_zero,
        a_degree=a_degree,
        loop_VFI=loop_VFI,
        loop_thres_a=loop_thres_a,
        loop_thres_e=loop_thres_e,
    )
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    # single male
    V_r_s_m::Array{Float64,4}
    V_d_s_m::Array{Float64,2}
    policy_a_s_m::Array{Float64,4}
    policy_n_s_m::Array{Float64,4}
    policy_d_s_m::Array{Float64,4}
    thres_a::Array{Float64,3}
    thres_e::Array{Float64,3}
    R_s_m::Array{Float64,3}
    q_s_m::Array{Float64,3}
end

function utility_function(c::Float64, l::Float64, γ::Float64, ω::Float64)
    """
    compute utility of CRRA utility function with coefficient γ
    """
    if (c > 0.0) && (l > 0.0)
        return γ == 1.0 ? log(c^ω * l^(1.0 - ω)) : 1.0 / ((1.0 - γ) * (c^ω * l^(1.0 - ω))^(γ - 1.0))
    else
        return -Inf
    end
end

log_function(threshold_e::Float64) = threshold_e > 0.0 ? log(threshold_e) : -Inf

function bounds_function(V_d_j::Float64, V_r_j::Vector{Float64}, a_grid::Vector{Float64})
    """
    compute bounds for finding roots
    """
    
    if V_d_j > maximum(V_r_j)
        error("V_d > V_nd for all a")
    elseif V_d_j < minimum(V_r_j)
        error("V_d < V_nd for all a")
    else
        a_ind = findfirst(V_r_j .> V_d_j)
        @inbounds ub = a_grid[a_ind]
        @inbounds lb = a_grid[a_ind-1]
        return lb, ub
    end
end

function threshold_function!(j::Int64, thres_a::Array{Float64,2}, thres_e::Array{Float64,2}, V_d::Array{Float64,2}, V_r::Array{Float64,4}, policy_n_s_m::Array{Float64,4}, parameters::NamedTuple)
    """
    update default thresholds
    """
    # unpack parameters
    @unpack a_size_neg, a_grid, h_size, h_grid, e_m_size, e_m_grid, κ_size, κ_grid, loop_thres_a, loop_thres_e = parameters

    # defaulting thresholds in wealth
    Threads.@threads for (κ_i, e_m_i) in loop_thres_a

        @inbounds @views V_r_j = V_r[:, e_m_i, κ_i, j]
        @inbounds @views V_d_j = V_d[e_m_i, j]
        @inbounds @views V_r_j_wo_Inf = findall(V_r_j .!= -Inf)
        @inbounds @views a_grid_itp = a_grid[V_r_j_wo_Inf]
        @inbounds @views V_r_j_grid_itp = V_r_j[V_r_j_wo_Inf]

        if minimum(V_r_j_grid_itp) > V_d_j
            @inbounds thres_a[e_m_i, κ_i] = -Inf
        else
            V_r_j_itp = Akima(a_grid_itp, V_r_j_grid_itp)
            @inbounds V_j_diff_itp(a) = V_r_j_itp(a) - V_d_j
            @inbounds V_j_diff_lb, V_j_diff_ub = zero_bounds_function(V_d_j, V_r_j, a_grid)
            @inbounds thres_a[e_m_i, κ_i] = find_zero(a -> V_j_diff_itp(a), (V_j_diff_lb, V_j_diff_ub), Bisection())
        end
    end

    # defaulting thresholds in earnings
    Threads.@threads for (κ_i, a_i) in loop_thres_e
        
        @inbounds @views thres_a_wo_Inf = findall(thres_a[:, κ_i] .!= -Inf)
        @inbounds @views thres_a_grid_itp = -thres_a[thres_a_wo_Inf, κ_i]
        @inbounds @views policy_n_s_m_j = policy_n_s_m[a_i, :, κ_i, j]
        @inbounds @views policy_n_s_m_j_wo_Inf = policy_n_s_m_j[thres_a_wo_Inf]
        @inbounds @views earning_grid_itp = h_grid[j] .* e_m_grid[thres_a_wo_Inf] .* policy_n_s_m_j_wo_Inf .- κ_grid[κ_i]
        thres_n_itp = linear_interpolation(e_m_grid, policy_n_s_m_j, extrapolation_bc=Line())
        thres_e_itp = linear_interpolation(thres_a_grid_itp, earning_grid_itp, extrapolation_bc=Line())
        @inbounds thres_e[a_i, e_1_i, e_3_i, κ_i] = find_zero(e_m -> thres_e_itp(-a_grid[a_i]) - h_grid[j] * e_m * thres_n_itp(e_m) + κ_grid[κ_i], (e_m_grid[end]*1.1, e_m_grid[1]*0.9), Bisection())

    end
    return nothing
end

#==============================#
# Solve stationary equilibrium #
#==============================#
parameters = parameters_function();