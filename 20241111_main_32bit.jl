using Parameters
using ProgressMeter
using QuadGK
using Distributions
using LinearAlgebra
using BenchmarkTools
using Profile
import Base: *
using Polyester

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
            integral = quadgk(u -> f(u), ϵ[i], ϵ[i+1])[1]
            Π[i, j] = (N / sqrt(2.0 * π * σ_ϵ^2.0)) * integral
        end
    end
    for i = 1:N
        Π[i, :] .= Π[i, :] ./ sum(Π[i, :])
    end
    return Float32.(z), Float32.(Π)
end

struct i0 end
(*)(n, ::Type{i0}) = Int32(n)

function parameters_function(;
    life_span::Int32=16i0,                # model life span
    period::Int32=3i0,                    # model period
    β::Float32=0.9730f0,                  # discount factor
    r_f::Float32=0.0344f0,                # risk-free rate 
    τ::Float32=0.0093f0,                  # transaction cost
    γ::Float32=2.00f0,                    # risk aversion coefficient
    ω::Float32=0.56f0,                    # consumption weight
    T::Float32=1.50f0,                    # time endowment
    ϕ::Float32=0.395f0,                   # wage garnishment rate
    η::Float32=1.64f0,                    # equivalence scale
    ψ::Float32=0.011f0,                   # divorce probability
    κ_div::Float32=0.0818f0,              # divorce cost
    e_size::Int32=5i0,                    # number of persistent wage
    e_m_ρ::Float32=0.9730f0,              # AR(1) of male persistent wage
    e_m_σ::Float32=sqrt(0.016f0),         # s.d. of male persistent wage 
    e_f_ρ::Float32=0.9630f0,              # AR(1) of female persistent wage
    e_f_σ::Float32=sqrt(0.014f0),         # s.d. of female persistent wage 
    a_min::Float32=-4.0f0,                # min of asset holding
    a_max::Float32=800.0f0,               # max of asset holding
    a_size_neg::Int32=401i0,              # number of grid of negative asset holding for VFI
    a_size_pos::Int32=101i0,              # number of grid of positive asset holding for VFI
    a_degree::Int32=3i0,                  # curvature of the positive asset gridpoints
)
    """
    contruct an immutable object containg all paramters
    """

    # model-period-year parameters
    β = β^period
    r_f = (1.0f0 + r_f)^period - 1.0f0
    τ = (1.0f0 + τ)^period - 1.0f0
    e_m_σ = sqrt((e_m_ρ^4.0f0 + e_m_ρ^2.0f0 + 1.0f0) * e_m_σ^2.0f0)
    e_m_ρ = e_m_ρ^period
    e_f_σ = sqrt((e_f_ρ^4.0f0 + e_f_ρ^2.0f0 + 1.0f0) * e_f_σ^2.0f0)
    e_f_ρ = e_f_ρ^period
    R_f = 1.0f0 + r_f
    BR = 1.0f0 + r_f + τ

    # lifecycle profile (Gourinchas and Parker, 2002)
    h_grid = [
        0.774482122f0, 0.819574547f0, 0.873895492f0, 0.931816800f0, 0.986069673f0,
        1.036889326f0, 1.082870993f0, 1.121249981f0, 1.148476948f0, 1.161069822f0,
        1.156650443f0, 1.134940682f0, 1.098443430f0, 1.052615160f0, 1.005569967f0,
        0.951900000f0
    ]
    h_size = Int32(length(h_grid))

    # male persistent income
    e_m_grid, e_m_Γ = adda_cooper(e_size, e_m_ρ, e_m_σ)
    # e_m_G = stationary_distributions(MarkovChain(e_m_Γ, e_m_grid))[1]
    e_m_G = repeat([1.0f0 / e_size], e_size)
    e_m_grid = exp.(e_m_grid)


    # female persistent income
    e_f_grid, e_f_Γ = adda_cooper(e_size, e_f_ρ, e_f_σ)
    # e_f_G = stationary_distributions(MarkovChain(e_f_Γ, e_f_grid))[1]
    e_f_G = repeat([1.0f0 / e_size], e_size)
    e_f_grid = exp.(e_f_grid)

    # expenditure schock
    κ_grid = zeros(Float32, 3i0, life_span)
    κ_size = Int32(size(κ_grid)[1i0])
    κ_Γ = [1.0f0, 0.0f0, 0.0f0]

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0f0, length=a_size_neg))
    a_grid_pos = ((range(0.0f0, stop=a_size_pos - 1i0, length=a_size_pos) / (a_size_pos - 1i0)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1i0:(end-1i0)], a_grid_pos, dims=1i0)
    a_size = Int32(length(a_grid))
    a_ind_zero = a_size_neg

    # grids for couple
    a_grid_neg_c = a_grid_neg .* 2.0f0
    a_grid_pos_c = a_grid_pos .* 2.0f0
    a_grid_c = a_grid .* 2.0f0
    κ_grid_c = κ_grid

    # labor supply
    n_grid = collect(0.0f0:0.5f0:1.0f0)
    n_size = Int32(length(n_grid))


    # iterators
    loop_q_s = collect(Iterators.product(1i0:e_size, 1i0:(a_size_neg-1i0)))
    loop_EV_s = collect(Iterators.product(1i0:e_size, 1i0:a_size))
    loop_V_s = collect(Iterators.product(1i0:κ_size, 1i0:a_size))
    loop_q_c = collect(Iterators.product(1i0:e_size, 1i0:e_size, 1i0:(a_size_neg-1i0)))
    loop_EV_c = collect(Iterators.product(1i0:e_size, 1i0:e_size, 1i0:a_size))
    loop_V_c = collect(Iterators.product(1i0:κ_size, 1i0:κ_size, 1i0:a_size))

    # return values
    return (
        life_span=life_span,
        period=period,
        β=β,
        r_f=r_f,
        τ=τ,
        R_f=R_f,
        BR=BR,
        γ=γ,
        ω=ω,
        T=T,
        ϕ=ϕ,
        η=η,
        ψ=ψ,
        κ_div=κ_div,
        h_grid=h_grid,
        h_size=h_size,
        e_size=e_size,
        e_m_ρ=e_m_ρ,
        e_m_σ=e_m_σ,
        e_m_grid=e_m_grid,
        e_m_Γ=e_m_Γ,
        e_m_G=e_m_G,
        e_f_ρ=e_f_ρ,
        e_f_σ=e_f_σ,
        e_f_grid=e_f_grid,
        e_f_Γ=e_f_Γ,
        e_f_G=e_f_G,
        κ_grid=κ_grid,
        κ_size=κ_size,
        κ_Γ=κ_Γ,
        κ_grid_c=κ_grid_c,
        a_grid_neg=a_grid_neg,
        a_grid_pos=a_grid_pos,
        a_grid=a_grid,
        a_size=a_size,
        a_size_neg=a_size_neg,
        a_size_pos=a_size_pos,
        a_ind_zero=a_ind_zero,
        a_grid_neg_c=a_grid_neg_c,
        a_grid_pos_c=a_grid_pos_c,
        a_grid_c=a_grid_c,
        a_degree=a_degree,
        n_grid=n_grid,
        n_size=n_size,
        loop_q_s=loop_q_s,
        loop_EV_s=loop_EV_s,
        loop_V_s=loop_V_s,
        loop_q_c=loop_q_c,
        loop_EV_c=loop_EV_c,
        loop_V_c=loop_V_c,
    )
end

mutable struct Mutable_Variables
    """
    construct a type for mutable variables
    """
    # single male
    V_s_m::Array{Float32,4} # (a, κ, e, h)
    E_V_s_m::Array{Float32,3} # (a', e, h)
    V_s_m_r::Array{Float32,4} # (a, κ, e, h)
    V_s_m_d::Array{Float32,2} # (e, h)
    policy_s_m_r_a::Array{Float32,4}
    policy_s_m_r_n::Array{Float32,4}
    policy_s_m_d::Array{Float32,4}
    policy_s_m_d_n::Array{Float32,2}
    q_s_m::Array{Float32,3} # (a', e, h)
    # divorced male
    V_d_m::Array{Float32,4}
    V_d_m_r::Array{Float32,4}
    policy_d_m_r_a::Array{Float32,4}
    policy_d_m_r_n::Array{Float32,4}
    policy_d_m_d::Array{Float32,4}
    # single female
    V_s_f::Array{Float32,4}
    E_V_s_f::Array{Float32,3}
    V_s_f_r::Array{Float32,4}
    V_s_f_d::Array{Float32,2}
    policy_s_f_r_a::Array{Float32,4}
    policy_s_f_r_n::Array{Float32,4}
    policy_s_f_d::Array{Float32,4}
    policy_s_f_d_n::Array{Float32,2}
    q_s_f::Array{Float32,3}
    # divorced female
    V_d_f::Array{Float32,4}
    V_d_f_r::Array{Float32,4}
    policy_d_f_r_a::Array{Float32,4}
    policy_d_f_r_n::Array{Float32,4}
    policy_d_f_d::Array{Float32,4}
    # couple
    V_c::Array{Float32,6} # (a, κ_f, κ_m, e_f, e_m, h)
    E_V_c::Array{Float32,4} # (a', e_f, e_m, h)
    V_c_r::Array{Float32,6} # (a, κ_f, κ_m, e_f, e_m, h)
    V_c_d::Array{Float32,3} # (e_f, e_m, h)
    policy_c_r_a::Array{Float32,6}
    policy_c_r_n_m::Array{Float32,6}
    policy_c_r_n_f::Array{Float32,6}
    policy_c_d::Array{Float32,6}
    policy_c_d_n_m::Array{Float32,3}
    policy_c_d_n_f::Array{Float32,3}
    q_c::Array{Float32,4} # (a', e_f, e_m, h)
end

function utility_function(c::Float32, l::Float32, γ::Float32, ω::Float32)
    """
    compute utility of CRRA utility function with coefficient γ
    """
    if (c > 0.0f0) && (l > 0.0f0)
        return γ == 1.0f0 ? log(c^ω * l^(1.0f0 - ω)) : 1.0f0 / ((1.0f0 - γ) * (c^ω * l^(1.0f0 - ω))^(γ - 1.0f0))
    else
        return -Inf32
    end
end

function utility_function(c::Float32, l_m::Float32, l_f::Float32, γ::Float32, ω::Float32)
    """
    compute utility of CRRA utility function with coefficient γ
    """
    if (c > 0.0f0) && (l_m > 0.0f0) && (l_f > 0.0f0)
        return γ == 1.0f0 ? log(c^ω * l_m^(1.0f0 - ω)) + log(c^ω * l_f^(1.0f0 - ω)) : 1.0f0 / ((1.0f0 - γ) * (c^ω * l_m^(1.0f0 - ω))^(γ - 1.0f0)) + 1.0f0 / ((1.0f0 - γ) * (c^ω * l_f^(1.0f0 - ω))^(γ - 1.0f0))
    else
        return -Inf32
    end
end

function variables_function(parameters::NamedTuple)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack loop_V_s, loop_V_c, a_size, a_grid, a_grid_c, e_size, e_m_grid, e_f_grid, κ_size, κ_grid, κ_grid_c, h_size, h_grid, n_size, n_grid = parameters
    @unpack R_f, ϕ, γ, ω, T, κ_div, η = parameters

    # define value and policy functions for male
    V_s_m = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    E_V_s_m = zeros(Float32, a_size, e_size, h_size - 1i0)
    V_s_m_r = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    V_s_m_d = (-Inf32) .* ones(Float32, e_size, h_size)
    policy_s_m_r_a = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_s_m_r_n = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_s_m_d = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_s_m_d_n = zeros(Float32, e_size, h_size)
    V_d_m = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    V_d_m_r = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    policy_d_m_r_a = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_d_m_r_n = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_d_m_d = zeros(Float32, a_size, κ_size, e_size, h_size)

    # define value and policy functions for female
    V_s_f = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    E_V_s_f = zeros(Float32, a_size, e_size, h_size - 1i0)
    V_s_f_r = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    V_s_f_d = (-Inf32) .* ones(Float32, e_size, h_size)
    policy_s_f_r_a = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_s_f_r_n = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_s_f_d = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_s_f_d_n = zeros(Float32, e_size, h_size)
    V_d_f = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    V_d_f_r = (-Inf32) .* ones(Float32, a_size, κ_size, e_size, h_size)
    policy_d_f_r_a = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_d_f_r_n = zeros(Float32, a_size, κ_size, e_size, h_size)
    policy_d_f_d = zeros(Float32, a_size, κ_size, e_size, h_size)

    # define value and policy functions for couple
    V_c = (-Inf32) .* ones(Float32, a_size, κ_size, κ_size, e_size, e_size, h_size)
    E_V_c = zeros(Float32, a_size, e_size, e_size, h_size - 1i0)
    V_c_r = (-Inf32) .* ones(Float32, a_size, κ_size, κ_size, e_size, e_size, h_size)
    V_c_d = (-Inf32) .* ones(Float32, e_size, e_size, h_size)
    policy_c_r_a = zeros(Float32, a_size, κ_size, κ_size, e_size, e_size, h_size)
    policy_c_r_n_m = zeros(Float32, a_size, κ_size, κ_size, e_size, e_size, h_size)
    policy_c_r_n_f = zeros(Float32, a_size, κ_size, κ_size, e_size, e_size, h_size)
    policy_c_d = zeros(Float32, a_size, κ_size, κ_size, e_size, e_size, h_size)
    policy_c_d_n_m = zeros(Float32, e_size, e_size, h_size)
    policy_c_d_n_f = zeros(Float32, e_size, e_size, h_size)

    # extract the last-period life-cycle component
    h = h_grid[h_size]

    # male
    for e_i in 1i0:e_size
        e_m = e_m_grid[e_i]
        he_m = h * e_m

        # default (single)
        for n_i in 1i0:n_size
            n = n_grid[n_i]
            l = T - n
            c = he_m * n * (1.0f0 - ϕ)
            if c > 0.0f0
                u = utility_function.(c, l, γ, ω)
                if u > V_s_m_d[e_i, h_size]
                    V_s_m_d[e_i, h_size] = u
                    policy_s_m_d_n[e_i, h_size] = n
                end
            end
        end

        # repayment
        @batch for (κ_i, a_i) in loop_V_s
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_size]
            a_κ = a - κ

            for n_i in 1i0:n_size
                n = n_grid[n_i]
                l = T - n

                # single
                c = he_m * n + a_κ
                if c > 0.0f0
                    u = utility_function(c, l, γ, ω)
                    if u > V_s_m_r[a_i, κ_i, e_i, h_size]
                        V_s_m_r[a_i, κ_i, e_i, h_size] = u
                        policy_s_m_r_n[a_i, κ_i, e_i, h_size] = n
                    end
                end

                # divorced
                c = c - κ_div
                if c > 0.0f0
                    u = utility_function(c, l, γ, ω)
                    if u > V_d_m_r[a_i, κ_i, e_i, h_size]
                        V_d_m_r[a_i, κ_i, e_i, h_size] = u
                        policy_d_m_r_n[a_i, κ_i, e_i, h_size] = n
                    end
                end
            end

            # to default or not (single)
            if V_s_m_r[a_i, κ_i, e_i, h_size] <= V_s_m_d[e_i, h_size]
                V_s_m[a_i, κ_i, e_i, h_size] = V_s_m_d[e_i, h_size]
                policy_s_m_d[a_i, κ_i, e_i, h_size] = 1.0f0
            else
                V_s_m[a_i, κ_i, e_i, h_size] = V_s_m_r[a_i, κ_i, e_i, h_size]
            end

            # to default or not (divorced)
            if V_d_m_r[a_i, κ_i, e_i, h_size] <= V_s_m_d[e_i, h_size]
                V_d_m[a_i, κ_i, e_i, h_size] = V_s_m_d[e_i, h_size]
                policy_d_m_d[a_i, κ_i, e_i, h_size] = 1.0f0
            else
                V_d_m[a_i, κ_i, e_i, h_size] = V_d_m_r[a_i, κ_i, e_i, h_size]
            end
        end
    end

    # female
    for e_i in 1i0:e_size
        e_f = e_f_grid[e_i]
        he_f = h * e_f

        # default (single)
        for n_i in 1i0:n_size
            n = n_grid[n_i]
            l = T - n
            c = he_f * n * (1.0f0 - ϕ)
            if c > 0.0f0
                u = utility_function.(c, l, γ, ω)
                if u > V_s_f_d[e_i, h_size]
                    V_s_f_d[e_i, h_size] = u
                    policy_s_f_d_n[e_i, h_size] = n
                end
            end
        end

        # repayment (single)
        @batch for (κ_i, a_i) in loop_V_s
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_size]
            a_κ = a - κ

            for n_i in 1i0:n_size
                n = n_grid[n_i]
                l = T - n

                # single
                c = he_f * n + a_κ
                if c > 0.0f0
                    u = utility_function(c, l, γ, ω)
                    if u > V_s_f_r[a_i, κ_i, e_i, h_size]
                        V_s_f_r[a_i, κ_i, e_i, h_size] = u
                        policy_s_f_r_n[a_i, κ_i, e_i, h_size] = n
                    end
                end

                # divorced
                c = c - κ_div
                if c > 0.0f0
                    u = utility_function(c, l, γ, ω)
                    if u > V_d_f_r[a_i, κ_i, e_i, h_size]
                        V_d_f_r[a_i, κ_i, e_i, h_size] = u
                        policy_d_f_r_n[a_i, κ_i, e_i, h_size] = n
                    end
                end
            end

            # to default or not (single)
            if V_s_f_r[a_i, κ_i, e_i, h_size] <= V_s_f_d[e_i, h_size]
                V_s_f[a_i, κ_i, e_i, h_size] = V_s_f_d[e_i, h_size]
                policy_s_f_d[a_i, κ_i, e_i, h_size] = 1.0f0
            else
                V_s_f[a_i, κ_i, e_i, h_size] = V_s_f_r[a_i, κ_i, e_i, h_size]
            end

            # to default or not (divorced)
            if V_d_f_r[a_i, κ_i, e_i, h_size] <= V_s_f_d[e_i, h_size]
                V_d_f[a_i, κ_i, e_i, h_size] = V_s_f_d[e_i, h_size]
                policy_d_f_d[a_i, κ_i, e_i, h_size] = 1.0f0
            else
                V_d_f[a_i, κ_i, e_i, h_size] = V_d_f_r[a_i, κ_i, e_i, h_size]
            end
        end
    end

    # couple
    for e_m_i in 1i0:e_size, e_f_i in 1i0:e_size
        e_m = e_m_grid[e_m_i]
        he_m = h * e_m
        e_f = e_f_grid[e_f_i]
        he_f = h * e_f

        # default
        for n_m_i in 1i0:n_size, n_f_i in 1i0:n_size
            n_m = n_grid[n_m_i]
            n_f = n_grid[n_f_i]
            l_m = T - n_m
            l_f = T - n_f
            c = (he_m * n_m + he_f * n_f) * (1.0f0 - ϕ)
            if c > 0.0f0
                u = utility_function(c / η, l_m, γ, ω) + utility_function(c / η, l_f, γ, ω)
                if u > V_c_d[e_f_i, e_m_i, h_size]
                    V_c_d[e_f_i, e_m_i, h_size] = u
                    policy_c_d_n_m[e_f_i, e_m_i, h_size] = n_m
                    policy_c_d_n_f[e_f_i, e_m_i, h_size] = n_f
                end
            end
        end

        # repayment
        @batch for (κ_m_i, κ_f_i, a_i) in loop_V_c
            a = a_grid_c[a_i]
            κ_m = κ_grid_c[κ_m_i, h_size]
            κ_f = κ_grid_c[κ_f_i, h_size]
            a_κ = a - κ_m - κ_f

            for n_m_i in 1i0:n_size, n_f_i in 1i0:n_size
                n_m = n_grid[n_m_i]
                n_f = n_grid[n_f_i]
                l_m = T - n_m
                l_f = T - n_f
                c = he_m * n_m + he_f * n_f + a_κ
                if c > 0.0f0
                    u = utility_function(c / η, l_m, γ, ω) + utility_function(c / η, l_f, γ, ω)
                    if u > V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size]
                        V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] = u
                        policy_c_r_n_m[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] = n_m
                        policy_c_r_n_f[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] = n_f
                    end
                end
            end

            # to default or not
            if V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] <= V_c_d[e_f_i, e_m_i, h_size]
                V_c[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] = V_c_d[e_f_i, e_m_i, h_size]
                policy_c_d[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] = 1.0f0
            else
                V_c[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size] = V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_size]
            end
        end
    end

    # define pricing functions
    q_s_m = ones(a_size, e_size, h_size - 1i0) ./ R_f
    q_s_f = ones(a_size, e_size, h_size - 1i0) ./ R_f
    q_c = ones(a_size, e_size, e_size, h_size - 1i0) ./ R_f

    # return outputs
    variables = Mutable_Variables(
        V_s_m, E_V_s_m, V_s_m_r, V_s_m_d, policy_s_m_r_a, policy_s_m_r_n, policy_s_m_d, policy_s_m_d_n, q_s_m,
        V_d_m, V_d_m_r, policy_d_m_r_a, policy_d_m_r_n, policy_d_m_d,
        V_s_f, E_V_s_f, V_s_f_r, V_s_f_d, policy_s_f_r_a, policy_s_f_r_n, policy_s_f_d, policy_s_f_d_n, q_s_f,
        V_d_f, V_d_f_r, policy_d_f_r_a, policy_d_f_r_n, policy_d_f_d,
        V_c, E_V_c, V_c_r, V_c_d, policy_c_r_a, policy_c_r_n_m, policy_c_r_n_f, policy_c_d, policy_c_d_n_m, policy_c_d_n_f, q_c
    )
    return variables
end

function pricing_and_rbl_function!(
    h_i::Int32,
    variables::Mutable_Variables,
    parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack loop_q_s, loop_q_c, e_size, e_m_grid, e_m_Γ, e_f_grid, e_f_Γ, a_grid_neg, h_grid, κ_size, κ_grid = parameters
    @unpack κ_Γ, BR, ϕ, ψ = parameters

    # extract the life-cycle wage
    h_p = h_grid[h_i+1i0]

    # single
    # loop over current states
    @batch for (e_i, a_p_i) in loop_q_s
        a_p = a_grid_neg[a_p_i]
        variables.q_s_m[a_p_i, e_i, h_i] = 0.0f0
        variables.q_s_f[a_p_i, e_i, h_i] = 0.0f0

        # loop over future states
        for e_p_i in 1i0:e_size, κ_p_i in 1i0:κ_size
            e_m_p = e_m_grid[e_p_i]
            e_f_p = e_f_grid[e_p_i]
            κ_p = κ_grid[κ_p_i]

            # male
            variables.q_s_m[a_p_i, e_i, h_i] +=
                variables.policy_s_m_d[a_p_i, κ_p_i, e_p_i, h_i+1i0] * κ_Γ[κ_p_i] * e_m_Γ[e_i, e_p_i] * ((h_p * e_m_p * variables.policy_s_m_d_n[e_p_i, h_i+1i0]) * ϕ / (κ_p - a_p)) +
                (1.0f0 - variables.policy_s_m_d[a_p_i, κ_p_i, e_p_i, h_i+1i0]) * κ_Γ[κ_p_i] * e_m_Γ[e_i, e_p_i]

            # female
            variables.q_s_f[a_p_i, e_i, h_i] +=
                variables.policy_s_f_d[a_p_i, κ_p_i, e_p_i, h_i+1i0] * κ_Γ[κ_p_i] * e_f_Γ[e_i, e_p_i] * ((h_p * e_f_p * variables.policy_s_f_d_n[e_p_i, h_i+1i0]) * ϕ / (κ_p - a_p)) +
                (1.0f0 - variables.policy_s_f_d[a_p_i, κ_p_i, e_p_i, h_i+1i0]) * κ_Γ[κ_p_i] * e_f_Γ[e_i, e_p_i]
        end

        # make sure the risk-based price is bounded between zero and one
        variables.q_s_m[a_p_i, e_i, h_i] = clamp(variables.q_s_m[a_p_i, e_i, h_i], 0.0f0, 1.0f0) / BR
        variables.q_s_f[a_p_i, e_i, h_i] = clamp(variables.q_s_f[a_p_i, e_i, h_i], 0.0f0, 1.0f0) / BR
    end

    # couple
    # loop over current states
    @batch for (e_m_i, e_f_i, a_p_i) in loop_q_c
        a_p = a_grid_neg[a_p_i]
        variables.q_c[a_p_i, e_f_i, e_m_i, h_i] = 0.0f0

        # loop over future states
        for e_m_p_i in 1i0:e_size, e_f_p_i in 1i0:e_size, κ_m_p_i in 1i0:κ_size, κ_f_p_i in 1i0:κ_size
            e_m_p = e_m_grid[e_m_p_i]
            e_f_p = e_f_grid[e_f_p_i]
            κ_m_p = κ_grid[κ_m_p_i]
            κ_f_p = κ_grid[κ_f_p_i]

            # remain couple
            variables.q_c[a_p_i, e_f_i, e_m_i, h_i] +=
                (1.0f0 - ψ) *
                (variables.policy_c_d[a_p_i, κ_f_p_i, κ_m_p_i, e_f_p_i, e_m_p_i, h_i+1i0] * κ_Γ[κ_m_p_i] * e_m_Γ[e_m_i, e_m_p_i] * κ_Γ[κ_f_p_i] * e_f_Γ[e_f_i, e_f_p_i] *
                 ((h_p * e_m_p * variables.policy_c_d_n_m[e_f_p_i, e_m_p_i, h_i+1i0] + h_p * e_f_p * variables.policy_c_d_n_f[e_f_p_i, e_m_p_i, h_i+1i0]) * ϕ / (κ_m_p + κ_f_p - a_p)) +
                 (1.0f0 - variables.policy_c_d[a_p_i, κ_f_p_i, κ_m_p_i, e_f_p_i, e_m_p_i, h_i+1i0]) * κ_Γ[κ_m_p_i] * e_m_Γ[e_m_i, e_m_p_i] * κ_Γ[κ_f_p_i] * e_f_Γ[e_f_i, e_f_p_i])

            # divorced
            variables.q_c[a_p_i, e_f_i, e_m_i, h_i] +=
                ψ *
                (variables.policy_s_m_d[a_p_i, κ_m_p_i, e_m_p_i, h_i+1i0] * κ_Γ[κ_m_p_i] * e_m_Γ[e_m_i, e_m_p_i] *
                 ((h_p * e_m_p * variables.policy_s_m_d_n[e_m_p_i, h_i+1i0] * ϕ / (κ_m_p - a_p / 2.0f0))) +
                 (1.0f0 - variables.policy_s_m_d[a_p_i, κ_m_p_i, e_m_p_i, h_i+1i0]) * κ_Γ[κ_m_p_i] * e_m_Γ[e_m_i, e_m_p_i] / 2.0f0 +
                 variables.policy_s_f_d[a_p_i, κ_f_p_i, e_f_p_i, h_i+1i0] * κ_Γ[κ_f_p_i] * e_f_Γ[e_f_i, e_f_p_i] *
                 ((h_p * e_f_p * variables.policy_s_f_d_n[e_f_p_i, h_i+1i0] * ϕ / (κ_f_p - a_p / 2.0f0))) +
                 (1.0f0 - variables.policy_s_f_d[a_p_i, κ_f_p_i, e_f_p_i, h_i+1i0]) * κ_Γ[κ_f_p_i] * e_f_Γ[e_f_i, e_f_p_i] / 2.0f0)
        end

        # make sure the risk-based price is bounded between zero and one
        variables.q_c[a_p_i, e_f_i, e_m_i, h_i] = clamp(variables.q_c[a_p_i, e_f_i, e_m_i, h_i], 0.0f0, 1.0f0) / BR
    end

    # return results
    return nothing
end

function E_V_function!(
    h_i::Int32,
    variables::Mutable_Variables,
    parameters::NamedTuple
)
    """
    construct expected value functions
    """

    # unpack parameters
    @unpack loop_EV_s, loop_EV_c, e_size, e_m_Γ, e_f_Γ, κ_size, κ_Γ, β, ψ = parameters

    # update expected value for single
    @batch for (e_i, a_p_i) in loop_EV_s
        for e_p_i in 1i0:e_size, κ_p_i in 1i0:κ_size
            variables.E_V_s_m[a_p_i, e_i, h_i] += β * κ_Γ[κ_p_i] * e_m_Γ[e_i, e_p_i] * variables.V_s_m[a_p_i, κ_p_i, e_p_i, h_i+1i0]
            variables.E_V_s_f[a_p_i, e_i, h_i] += β * κ_Γ[κ_p_i] * e_f_Γ[e_i, e_p_i] * variables.V_s_f[a_p_i, κ_p_i, e_p_i, h_i+1i0]
        end
    end

    # update expected value for couple
    @batch for (e_m_i, e_f_i, a_p_i) in loop_EV_c
        for e_m_p_i in 1i0:e_size, e_f_p_i in 1i0:e_size, κ_m_p_i in 1i0:κ_size, κ_f_p_i in 1i0:κ_size
            variables.E_V_c[a_p_i, e_f_i, e_m_i, h_i] +=
                β *
                ((1.0f0 - ψ) * κ_Γ[κ_m_p_i] * e_m_Γ[e_m_i, e_m_p_i] * κ_Γ[κ_f_p_i] * e_f_Γ[e_f_i, e_f_p_i] * variables.V_c[a_p_i, κ_f_p_i, κ_m_p_i, e_f_p_i, e_m_p_i, h_i+1i0] +
                 ψ * κ_Γ[κ_m_p_i] * e_m_Γ[e_m_i, e_m_p_i] * variables.V_d_m[a_p_i, κ_m_p_i, e_m_p_i, h_i+1i0] +
                 ψ * κ_Γ[κ_f_p_i] * e_f_Γ[e_f_i, e_f_p_i] * variables.V_d_f[a_p_i, κ_f_p_i, e_f_p_i, h_i+1i0])
        end
    end

    # return results
    return nothing
end

function value_and_policy_function!(
    h_i::Int32,
    variables::Mutable_Variables,
    parameters::NamedTuple
)
    """
    one-step update of value and policy functions at age h_i
    """

    # unpack parameters
    @unpack loop_V_s, loop_V_c, h_grid, a_size, a_grid, a_grid_c, a_ind_zero, e_size, e_m_grid, e_f_grid, κ_grid, κ_grid_c, n_size, n_grid = parameters
    @unpack T, γ, ω, ϕ, κ_div, ψ, η = parameters
    @unpack κ_size = parameters


    # extract the life-cycle component at age h_i
    h = h_grid[h_i]

    # male
    for e_i in 1i0:e_size
        e_m = e_m_grid[e_i]
        he_m = h * e_m

        # construct useful vectors
        @views qa_m = variables.q_s_m[:, e_i, h_i] .* a_grid
        @views EV_m = variables.E_V_s_m[:, e_i, h_i]

        # default (single)
        for n_i in 1i0:n_size
            n = n_grid[n_i]
            l = T - n
            c = he_m * n * (1.0f0 - ϕ)
            if c > 0.0f0
                u = utility_function(c, l, γ, ω) + EV_m[a_ind_zero]
                if u > variables.V_s_m_d[e_i, h_i]
                    variables.V_s_m_d[e_i, h_i] = u
                    variables.policy_s_m_d_n[e_i, h_i] = n
                end
            end
        end

        # repayment (single)
        @batch for (κ_i, a_i) in loop_V_s
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_i]
            a_κ = a - κ

            for n_i in 1i0:n_size
                n = n_grid[n_i]
                l = T - n
                income = he_m * n + a_κ

                for a_p_i in 1i0:a_size

                    # single
                    c = income - qa_m[a_p_i]
                    if c > 0.0f0
                        u = utility_function(c, l, γ, ω) + EV_m[a_p_i]
                        if u > variables.V_s_m_r[a_i, κ_i, e_i, h_i]
                            variables.V_s_m_r[a_i, κ_i, e_i, h_i] = u
                            variables.policy_s_m_r_a[a_i, κ_i, e_i, h_i] = a_grid[a_p_i]
                            variables.policy_s_m_r_n[a_i, κ_i, e_i, h_i] = n
                        end
                    end

                    # divorced
                    c = c - κ_div
                    if c > 0.0f0
                        u = utility_function(c, l, γ, ω) + EV_m[a_p_i]
                        if u > variables.V_d_m_r[a_i, κ_i, e_i, h_i]
                            variables.V_d_m_r[a_i, κ_i, e_i, h_i] = u
                            variables.policy_d_m_r_a[a_i, κ_i, e_i, h_i] = a_grid[a_p_i]
                            variables.policy_d_m_r_n[a_i, κ_i, e_i, h_i] = n
                        end
                    end
                end
            end

            # to default or not (single)
            if variables.V_s_m_r[a_i, κ_i, e_i, h_i] <= variables.V_s_m_d[e_i, h_i]
                variables.V_s_m[a_i, κ_i, e_i, h_i] = variables.V_s_m_d[e_i, h_i]
                variables.policy_s_m_d[a_i, κ_i, e_i, h_i] = 1.0f0
            else
                variables.V_s_m[a_i, κ_i, e_i, h_i] = variables.V_s_m_r[a_i, κ_i, e_i, h_i]
            end

            # to default or not (divorced)
            if variables.V_d_m_r[a_i, κ_i, e_i, h_i] <= variables.V_s_m_d[e_i, h_i]
                variables.V_d_m[a_i, κ_i, e_i, h_i] = variables.V_s_m_d[e_i, h_i]
                variables.policy_d_m_d[a_i, κ_i, e_i, h_i] = 1.0f0
            else
                variables.V_d_m[a_i, κ_i, e_i, h_i] = variables.V_d_m_r[a_i, κ_i, e_i, h_i]
            end
        end
    end

    # female
    for e_i in 1i0:e_size
        e_f = e_f_grid[e_i]
        he_f = h * e_f

        # construct useful vectors
        @views qa_f = variables.q_s_f[:, e_i, h_i] .* a_grid
        @views EV_f = variables.E_V_s_f[:, e_i, h_i]

        # default (single)
        for n_i in 1i0:n_size
            n = n_grid[n_i]
            l = T - n
            c = he_f * n * (1.0f0 - ϕ)
            if c > 0.0f0
                u = utility_function(c, l, γ, ω) + EV_f[a_ind_zero]
                if u > variables.V_s_f_d[e_i, h_i]
                    variables.V_s_f_d[e_i, h_i] = u
                    variables.policy_s_f_d_n[e_i, h_i] = n
                end
            end
        end

        # repayment (single)
        @batch for (κ_i, a_i) in loop_V_s
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_i]
            a_κ = a - κ

            for n_i in 1i0:n_size
                n = n_grid[n_i]
                l = T - n
                income = he_f * n + a_κ

                for a_p_i in 1i0:a_size

                    # single
                    c = income - qa_f[a_p_i]
                    if c > 0.0f0
                        u = utility_function(c, l, γ, ω) + EV_f[a_p_i]
                        if u > variables.V_s_f_r[a_i, κ_i, e_i, h_i]
                            variables.V_s_f_r[a_i, κ_i, e_i, h_i] = u
                            variables.policy_s_f_r_a[a_i, κ_i, e_i, h_i] = a_grid[a_p_i]
                            variables.policy_s_f_r_n[a_i, κ_i, e_i, h_i] = n
                        end
                    end

                    # divorced
                    c = c - κ_div
                    if c > 0.0f0
                        u = utility_function(c, l, γ, ω) + EV_f[a_p_i]
                        if u > variables.V_d_f_r[a_i, κ_i, e_i, h_i]
                            variables.V_d_f_r[a_i, κ_i, e_i, h_i] = u
                            variables.policy_d_f_r_a[a_i, κ_i, e_i, h_i] = a_grid[a_p_i]
                            variables.policy_d_f_r_n[a_i, κ_i, e_i, h_i] = n
                        end
                    end
                end
            end

            # to default or not (single)
            if variables.V_s_f_r[a_i, κ_i, e_i, h_i] <= variables.V_s_f_d[e_i, h_i]
                variables.V_s_f[a_i, κ_i, e_i, h_i] = variables.V_s_f_d[e_i, h_i]
                variables.policy_s_f_d[a_i, κ_i, e_i, h_i] = 1.0f0
            else
                variables.V_s_f[a_i, κ_i, e_i, h_i] = variables.V_s_f_r[a_i, κ_i, e_i, h_i]
            end

            # to default or not (divorced)
            if variables.V_d_f_r[a_i, κ_i, e_i, h_i] <= variables.V_s_f_d[e_i, h_i]
                variables.V_d_f[a_i, κ_i, e_i, h_i] = variables.V_s_f_d[e_i, h_i]
                variables.policy_d_f_d[a_i, κ_i, e_i, h_i] = 1.0f0
            else
                variables.V_d_f[a_i, κ_i, e_i, h_i] = variables.V_d_f_r[a_i, κ_i, e_i, h_i]
            end
        end
    end

    # couple
    # loop over all states
    for e_m_i in 1i0:e_size, e_f_i in 1i0:e_size
        e_m = e_m_grid[e_m_i]
        he_m = h * e_m
        e_f = e_f_grid[e_f_i]
        he_f = h * e_f

        # construct useful vectors
        @views qa_c = variables.q_c[:, e_f_i, e_m_i, h_i] .* a_grid_c
        @views EV_c = variables.E_V_c[:, e_f_i, e_m_i, h_i]

        # default
        for n_m_i in 1i0:n_size, n_f_i in 1i0:n_size
            n_m = n_grid[n_m_i]
            n_f = n_grid[n_f_i]
            l_m = T - n_m
            l_f = T - n_f
            c = (he_m * n_m + he_f * n_f) * (1.0f0 - ϕ)
            if c > 0.0f0
                u = utility_function(c / η, l_m, l_f, γ, ω) + EV_c[a_ind_zero]
                if u > variables.V_c_d[e_f_i, e_m_i, h_i]
                    variables.V_c_d[e_f_i, e_m_i, h_i] = u
                    variables.policy_c_d_n_m[e_f_i, e_m_i, h_i] = n_m
                    variables.policy_c_d_n_f[e_f_i, e_m_i, h_i] = n_f
                end
            end
        end

        # repayment
        # for κ_m_i in 1i0:κ_size, κ_f_i in 1i0:κ_size, a_i in 1i0:a_size
        @batch for (κ_m_i, κ_f_i, a_i) in loop_V_c
            a = a_grid_c[a_i]
            κ_m = κ_grid_c[κ_m_i, h_i]
            κ_f = κ_grid_c[κ_f_i, h_i]
            a_κ = a - κ_m - κ_f

            for n_m_i in 1i0:n_size, n_f_i in 1i0:n_size
                n_m = n_grid[n_m_i]
                n_f = n_grid[n_f_i]
                l_m = T - n_m
                l_f = T - n_f
                income = he_m * n_m + he_f * n_f + a_κ
                for a_p_i in 1i0:a_size
                    c = income - qa_c[a_p_i]
                    if c > 0.0f0
                        u = utility_function(c / η, l_m, l_f, γ, ω) + EV_c[a_p_i]
                        if u > variables.V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i]
                            variables.V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = u
                            variables.policy_c_r_a[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = a_grid_c[a_p_i]
                            variables.policy_c_r_n_m[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = n_m
                            variables.policy_c_r_n_f[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = n_f
                        end
                    end
                end
            end

            # to default or not
            if variables.V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] <= variables.V_c_d[e_f_i, e_m_i, h_i]
                variables.V_c[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = variables.V_c_d[e_f_i, e_m_i, h_i]
                variables.policy_c_d[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = 1.0f0
            else
                variables.V_c[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i] = variables.V_c_r[a_i, κ_f_i, κ_m_i, e_f_i, e_m_i, h_i]
            end
        end
    end

    # return results
    return nothing
end

function solve_function!(
    variables::Mutable_Variables,
    parameters::NamedTuple
)
    """
    solve the model backward
    """

    # unpack parameters
    @unpack life_span = parameters

    # loop over life span
    @showprogress dt = 1i0 desc = "Computing..." for h_i = (life_span-1i0):(-1i0):1i0
        pricing_and_rbl_function!(h_i, variables, parameters)
        E_V_function!(h_i, variables, parameters)
        value_and_policy_function!(h_i, variables, parameters)
    end

    # return results
    return nothing
end

#==============================#
# Solve stationary equilibrium #
#==============================#
parameters = parameters_function();
variables = variables_function(parameters);
@btime solve_function!($variables, $parameters);

# testing functions
# pricing_and_rbl_function!(parameters.h_size - 1i0, variables, parameters)
# E_V_function!(parameters.h_size - 1i0, variables, parameters)
# @btime value_and_policy_function!(parameters.h_size - 1i0, $variables, $parameters)

# pricing_and_rbl_function!(parameters.h_size - 2, variables, parameters)
# E_V_function!(parameters.h_size - 2, variables, parameters)
# value_and_policy_function!(parameters.h_size - 2, variables, parameters)

# pricing_and_rbl_function!(parameters.h_size - 3, variables, parameters)
# E_V_function!(parameters.h_size - 3, variables, parameters)
# value_and_policy_function!(parameters.h_size - 3, variables, parameters)

# pricing_and_rbl_function!(parameters.h_size - 4, variables, parameters)
# E_V_function!(parameters.h_size - 4, variables, parameters)
# value_and_policy_function!(parameters.h_size - 4, variables, parameters)

# pricing_and_rbl_function!(parameters.h_size - 5, variables, parameters)
# E_V_function!(parameters.h_size - 5, variables, parameters)
# value_and_policy_function!(parameters.h_size - 5, variables, parameters)

# pricing_and_rbl_function!(parameters.h_size - 6, variables, parameters)
# E_V_function!(parameters.h_size - 6, variables, parameters)
# value_and_policy_function!(parameters.h_size - 6, variables, parameters)

# cheching figures
using Plots
h_i = parameters.h_size - 9
plot_q_s = plot(bg=:black, legend=:none, box=:on, ylims=(0.0, 1.0))
for e_i in 1:parameters.e_size
    plot!(plot_q_s, parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, e_i, h_i], color=e_i)
    plot!(plot_q_s, parameters.a_grid_neg, variables.q_s_f[1:parameters.a_ind_zero, e_i, h_i], color=e_i, linestyle=:dash)
end
plot_q_s

h_i = parameters.h_size - 12
plot_q_c = plot(bg=:black, legend=:none, box=:on, ylims=(0.0, 1.0))
for e_i in 1:parameters.e_size
    plot!(plot_q_c, parameters.a_grid_neg_c, variables.q_c[1:parameters.a_ind_zero, e_i, 1, h_i], color=e_i)
    plot!(plot_q_c, parameters.a_grid_neg_c, variables.q_c[1:parameters.a_ind_zero, e_i, 5, h_i], color=e_i, linestyle=:dash)
end
plot_q_c

h_i = 4
e_i = 1
plot_q_c_s = plot(legend=:none, box=:on, ylims=(0.0, 1.0))
plot!(plot_q_c_s, parameters.a_grid_neg, variables.q_s_f[1:parameters.a_ind_zero, e_i, h_i], color=e_i)
plot!(plot_q_c_s, parameters.a_grid_neg, variables.q_c[1:parameters.a_ind_zero, e_i, :, h_i] * parameters.e_f_G, color=e_i, linestyle=:dash)
# plot!(plot_q_c_s, parameters.a_grid_neg, variables.q_c[1:parameters.a_ind_zero, e_i, 1, h_i], color=e_i, linestyle=:dash)
plot_q_c_s

# plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end-3] .* parameters.a_grid_neg)
# plot(parameters.a_grid, variables.q_s_m[:, :, end] .* parameters.a_grid)

# plot(parameters.a_grid_neg, variables.q_s_f[1:parameters.a_ind_zero, 5, 9:11], legend=:none, bg=:black)

# plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end-3])
