# Prototype model, version based on ...-test where gradient-using algos were merged in

using Parameters
using LinearAlgebra

using ProgressMeter
using NLopt
# using Optim
# using Plots
# using StatsFuns

println("Julia is running with $(Threads.nthreads()) threads...")

include("GQ_algorithm.jl")

function parameters_function(;
    β::Real = 0.95,                              # Discount factor
    λ::Real = 0.5,                              # Utility weight for couples
    r::Real = 0.04,
    ρ::Real = 0.975,                            # survival probability
    a_size_neg::Integer = 21,                   # number of  assets
    # a_size_neg::Integer = 6,
    a_size_pos::Integer = 20,
    # a_size_pos::Integer = 5,
    a_min::Real = -1.00,                        # minimum of assets
    a_max::Real = 3.00,                         # maximum of assets
    # a_degree::Real = 1.0,                      # governs how grid is spaced
    # d_size::Integer = 2                         # Number of default/repayment options
    T::Real = 1.2,
    α::Real = 0.5,                                 # Weight of leisure in utility function # From Alon et al. (2020)
    # α::Real = 0.2,
    ϕ::Real = 0.355,                                 # Wage garnishment
    # ϕ::Real = 0.99,
    θ::Real = 0.55,                              # Returns to labor # From Alon et al. (2020)
    # θ::Real = 1.0,
    ν::Real = 0.4                               # Share of singles in economy
    )
    """
    contruct an immutable object containing all parameters
    """

    # persistent productivity shock
    z_grid = [0.37996271996003, 0.63109515876990, 0.86127208980130, 1.17540057527388, 1.95226945619490]
    # z_grid = [0.35, 0.6, 0.85, 1.2, 1.9]
    z_size = length(z_grid)
    Γ_z = [0.86379548987220  0.13510724642768  0.00109682214183  0.00000033823300  0.00000010332529;
    0.13510724429090  0.67772725268309  0.18381378713248  0.00335137854872  0.00000033734481;
    0.00109682516680  0.18381378410751  0.63017878145138  0.18381378410751  0.00109682516680;
    0.00000033734481  0.00335137854872 0.18381378713248  0.67772725268309  0.13510724429090;
    0.00000010332529  0.00000033823300  0.00109682214183  0.13510724642768  0.86379548987220]
    # Γ_z = [0.8  0.1  0.06  0.03  0.01;
    #        0.1  0.65  0.15  0.07  0.03;
    #        0.06  0.15  0.59  0.15  0.05;
    #        0.03  0.07 0.15  0.63  0.12;
    #        0.01  0.03  0.05  0.12  0.79]
    # G_z = [G_e_L G_e_M 1-G_e_L-G_e_M]

    # Transitory productivity shock
    η_grid = [0.61505879, 0.978521538, 1.556768907]
    # η_grid = [0.25, 0.5, 1.0, 1.5, 1.75]
    # η_grid = [0.15, 0.25, 0.5, 1.0, 1.5, 1.75]
    η_size = length(η_grid)
    Γ_η = [0.1, 0.8, 0.1]
    # Γ_η = [0.05, 0.1, 0.7, 0.1, 0.05]
    # Γ_η = [0.02, 0.05, 0.1, 0.7, 0.1, 0.03]

    # Expense shocks
    # κ_grid = [0.0 0.264 0.8218]
    # κ_grid = [0.0 0.1 0.4]
    κ_grid = [0.0 0.1 0.2]
    # κ_grid = [0.0 0.05 0.1]
    # κ_grid = [0.0]
    κ_size = length(κ_grid)
    # Γ_κ = [(1.0-0.07104-0.0046), 0.07104, 0.0046]
    Γ_κ = [(1.0-0.15-0.05), 0.15, 0.05]
    # Γ_κ = [1.0]

    # Asset grid
    # a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    # a_grid_pos = collect(range(0.0, a_max, length=a_size_pos))
    # a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    a_grid_neg = reverse((exp.(collect(range(log(1.0), stop=log(-a_min+1), length = a_size_neg))).-1)*(-1))
    a_grid_pos = exp.(collect(range(log(1.0), stop=log(a_max+1), length = a_size_pos))).-1
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    a_size_neg = length(a_grid_neg)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # Labor
    # n_grid = collect(range(0.0, 1.0, length=15))
    # n_grid = collect(range(0.0, 1.0, length=5))
    # n_size = length(n_grid)

    BM_indices = BM_function(a_size)

    # return the outcome
    return (β=β, λ=λ, r=r, ρ=ρ, T=T, α=α, ϕ=ϕ, θ=θ, ν=ν,
    a_grid=a_grid, a_size=a_size, a_ind_zero=a_ind_zero,
    z_grid=z_grid, z_size=z_size, Γ_z=Γ_z, η_grid=η_grid, η_size=η_size, Γ_η=Γ_η, κ_grid=κ_grid, κ_size = κ_size, Γ_κ = Γ_κ, BM_indices = BM_indices)
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    # U::Array{Float64,2}
    # V_S_R::Array{Float64,3}
    # V_S_D::Array{Float64,2}
    V_S::Array{Float64,4}
    # V_C_R::Array{Float64,5}
    # V_C_D::Array{Float64,4}
    V_C::Array{Float64,7}
    # c_S::Array{Float64,3}
    # c_C_1::Array{Float64,5}
    # c_C_2::Array{Float64,5}
    n_S_i::Array{Float64,4}
    n_C_1_i::Array{Float64,7}
    n_C_2_i::Array{Float64,7}
    # l_S::Array{Float64,3}
    # l_C_1::Array{Float64,5}
    # l_C_2::Array{Float64,5}
    a_S_i::Array{Int64,4}
    a_C_i::Array{Int64,7}
    d_S_i::Array{Int64,4}
    d_C_i::Array{Int64,7}
    P_S::Array{Float64,2}
    q_S::Array{Float64,2}
    P_C::Array{Float64,3}
    q_C::Array{Float64,3}
    μ_S::Array{Float64,4}
    μ_C::Array{Float64,7}
end

function variables_function(
    parameters::NamedTuple;
    load_initial_value::Bool = false
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack  a_size, z_size, η_size, κ_size, ρ, r, ν = parameters

    # Singles' value functions
    # V_S_R = zeros(a_size, z_size, η_size)
    # V_S_D = zeros(z_size, η_size)
    V_S = zeros(a_size, z_size, η_size, κ_size)

    # V_C_R = zeros(a_size, z_size, z_size, η_size, η_size)
    # V_C_D = zeros(z_size, z_size, η_size, η_size)
    V_C = zeros(a_size, z_size, z_size, η_size, η_size, κ_size, κ_size)

    # Policy Functions
    # Consumption
    # c_S = zeros(a_size, z_size, η_size)
    # c_C_1 = zeros(a_size, z_size, z_size, η_size, η_size)
    # c_C_2 = zeros(a_size, z_size, z_size, η_size, η_size)

    # Labor
    n_S_i = ones(Float64,a_size, z_size, η_size, κ_size).*0.5
    n_C_1_i = ones(Float64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size).*0.5
    n_C_2_i = ones(Float64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size).*0.5

    # Leisure
    # l_S = zeros(a_size, z_size, η_size)
    # l_C_1 = zeros(a_size, z_size, z_size, η_size, η_size)
    # l_C_2 = zeros(a_size, z_size, z_size, η_size, η_size)

    # Asset
    a_S_i = zeros(Int64,a_size, z_size, η_size, κ_size)
    a_C_i = zeros(Int64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size)

    # Default
    d_S_i = zeros(Int64,a_size, z_size, η_size, κ_size)
    d_C_i = zeros(Int64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size)

    # Loan pricing
    # Singles
    P_S = ones(a_size, z_size)
    q_S = ones(a_size, z_size) .* ρ/(1.0 + r)

    # Couples
    P_C = ones(a_size, z_size, z_size)
    q_C = ones(a_size, z_size, z_size) .* ρ/(1.0 + r)

    # cross-sectional distribution
    μ_S = ones(a_size, z_size, η_size, κ_size) ./ (a_size*z_size*η_size*κ_size)
    μ_C = ones(a_size, z_size, z_size, η_size, η_size, κ_size, κ_size) ./ (a_size*z_size*z_size*η_size*η_size*κ_size*κ_size)

    # return the outcome
    variables = MutableVariables(V_S, V_C, n_S_i, n_C_1_i, n_C_2_i, a_S_i, a_C_i, d_S_i, d_C_i, P_S, q_S, P_C, q_C, μ_S, μ_C)
    return variables
end

function utility_function(
    c::Real,
    l::Real
    )
    """
    compute utility of CRRA utility function
    """
    @unpack α = parameters
    if (c > 0.0) && (l > 0.0)
        return log(c) + α*log(l)
    else
        return -Inf
    end
end

function Util_R_S_optim_function(
    n::Vector, grad::Vector, z, η, a, q_S, a_p, κ, V_expect, θ, α, T, β, ρ
    )

    @inbounds l = T - n[1]
    @inbounds c = z*η*(n[1]^(θ))+a-q_S*a_p-κ

    if (c > 0.0) && (l > 0.0)
        grad[1] = (θ*z*η*(n[1]^(θ-1.0))/(z*η*(n[1]^θ)+a-q_S*a_p-κ))-(α/(T-n[1]))
        return log(c) + α*log(l) + β*ρ*V_expect
    else
        # grad[1] = 10^10
        return -Inf
    end
end

function Util_D_S_optim_function(
    n::Vector, grad::Vector, z, η, V_expect, θ, α, T, β, ρ, ϕ
    )

    @inbounds l = T - n[1]
    # c = z*η*(n^(θ))-ϕ
    @inbounds c = (z*η*(n[1]^(θ)))*(1.0-ϕ)

    if (c > 0.0) && (l > 0.0)
        grad[1] = (θ*z*η*(n[1]^(θ-1.0))*(1-ϕ)/(z*η*(n[1]^θ)*(1-ϕ)))-(α/(T-n[1]))
        return log(c) + α*log(l) + β*ρ*V_expect
    else
        return -Inf
    end
end

function Util_R_C_optim_function(
    n::Vector, grad::Vector, z_1, z_2, η_1, η_2, a, q_C, a_p, κ_1, κ_2, V_expect,θ,α,T,λ,β,ρ
    )

    @inbounds l_1 = T - n[1]
    @inbounds l_2 = T - n[2]
    @inbounds c = z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ)+a-q_C*a_p-κ_1-κ_2

    if (c > 0.0) && (l_1 > 0.0) && (l_2 > 0.0)
        grad[1] = θ*z_1*η_1*(n[1]^(θ-1.0))/(z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ)+a-q_C*a_p-κ_1-κ_2)-(λ*α/(T-n[1]))
        grad[2] = θ*z_2*η_2*(n[2]^(θ-1.0))/(z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ)+a-q_C*a_p-κ_1-κ_2)-((1-λ)*α/(T-n[2]))
        return λ*log(c/2) + (1-λ)*log(c/2) + λ*α*log(l_1) + (1-λ)*α*log(l_2) + β*ρ*V_expect
    else
        grad[1] = 10^10
        grad[2] = 10^10
        return -Inf
    end
end

function Util_D_C_optim_function(
    n::Vector, grad::Vector, z_1, z_2, η_1, η_2, V_expect, θ, α, T, λ, β, ρ, ϕ
    )

    @inbounds l_1 = T - n[1]
    @inbounds l_2 = T - n[2]
    # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
    @inbounds c = (z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ))*(1.0-ϕ)

    if (c > 0.0) && (l_1 > 0.0) && (l_2 > 0.0)
        grad[1] = θ*z_1*η_1*(n[1]^(θ-1.0))/(z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ))-(λ*α/(T-n[1]))
        grad[2] = θ*z_2*η_2*(n[2]^(θ-1.0))/(z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ))-((1.0-λ)*α/(T-n[2]))
        return λ*log(c/2) + (1-λ)*log(c/2) + λ*α*log(l_1) + (1-λ)*α*log(l_2) + β*ρ*V_expect
    else
        return -Inf
    end
end

function value_function_singles!(
    V_S_p::Array{Float64,4},
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, BM_indices, α = parameters

    variables.V_S .= -Inf

    # For Repayment
    # variables.V_S_R .= -Inf
    # Threads.@threads for a_i in 1:a_size
    #     for  κ_i in 1:κ_size, η_i in 1:η_size, z_i in 1:z_size

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
        for κ_i in 1:κ_size
        # lb_new = 1

        # non-defaulting value
        BM_bounds = zeros(Int,a_size,3)
        BM_bounds[:,1] = 1:a_size
        BM_bounds[1,2] = 1
        BM_bounds[1,3] = a_size
        BM_bounds[a_size,3] = a_size

        for BM_i in 1:a_size

            @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
            @inbounds lb = BM_bounds[lb_i,2]
            @inbounds ub = BM_bounds[ub_i,3]

            if ub != 0

                @inbounds z_i = i[1]
                @inbounds η_i = i[2]
                # @inbounds κ_i = i[3]

                @inbounds a = a_grid[a_i]
                @inbounds z = z_grid[z_i]
                @inbounds η = η_grid[η_i]
                @inbounds κ = κ_grid[κ_i]

                # lb = 1
                # if a_i > 1
                #     lb = lb_new
                # end
                #
                # v_temp_pre = -Inf

                # Test
                # v_highest = -Inf
                # v_flag = 0

                for a_p_i in lb:ub #, n_i in 1:n_size
                    @inbounds q_S = variables.q_S[a_p_i,z_i]

                    @inbounds a_p = a_grid[a_p_i]
                    # @inbounds n = n_grid[n_i]

                    # l = T - n
                    # @inbounds c = z*η*(n^(θ))+a-variables.q_S[a_p_i,z_i]*a_p-κ

                    V_expect = 0.0

                    for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                        @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p[a_p_i,z_p_i,η_p_i,κ_p_i]
                    end

                    n_min = max(((q_S*a_p+κ-a)/(z*η)),0.0)^(1/θ)

                    if n_min > 1
                        optf = -Inf
                    else

                        f(n::Vector,grad::Vector) = Util_R_S_optim_function(n,grad,z,η,a,q_S,a_p,κ,V_expect,θ,α,T,β,ρ)

                        opt = Opt(:LD_MMA,1)
                        opt.max_objective = f
                        opt.lower_bounds = max(n_min,0.001)
                        opt.upper_bounds = 1.0
                        opt.xtol_rel = 1e-4

                        if variables.n_S_i[a_i,z_i,η_i,κ_i] < n_min
                            n_initial = n_min
                        else
                            n_initial = variables.n_S_i[a_i,z_i,η_i,κ_i]
                        end

                        (optf,optx,ret) = optimize(opt,[n_initial])
                    end

                    # if isnan(optf) || isnan(optx[1])
                    #     println(optf)
                    #     println(optx)
                    #     println(ret)
                    #     error("NaN in opt S,R")
                    # end

                    # Test
                    # if optf > v_highest
                    #     v_highest = optf
                    #     if v_flag == 1
                    #         error("Non-concavity detected")
                    #     end
                    # elseif optf < v_highest
                    #     v_flag = 1
                    # end

                    # results = optimize(n -> -1*Util_optim_function(n,z,η,θ,a,q_S,a_p,κ,V_expect,T),0.0, 1.0)

                    # v_temp = utility_function(c,l) + β*ρ*V_expect

                    # v_temp = optf

                    # if v_temp > v_temp_pre
                    #     v_temp_pre = v_temp
                    #     lb_new = a_p_i
                    # end

                    # if (a_p_i == a_size) && (n_i == n_size) && (v_temp_pre == -Inf)
                    #     lb_new = 1
                    # end

                    if optf > variables.V_S[a_i,z_i,η_i,κ_i]
                        @inbounds variables.V_S[a_i,z_i,η_i,κ_i] = optf
                        @inbounds variables.d_S_i[a_i,z_i,η_i,κ_i] = 1
                        @inbounds variables.a_S_i[a_i,z_i,η_i,κ_i] = a_p_i
                        @inbounds variables.n_S_i[a_i,z_i,η_i,κ_i] = optx[1]
                        BM_bounds[a_i,2] = a_p_i
                        BM_bounds[a_i,3] = a_p_i
                    # elseif optf < variables.V_S[a_i,z_i,η_i,κ_i]
                    #     break
                    end

                end

                if variables.V_S[a_i,z_i,η_i,κ_i] == -Inf
                    BM_bounds[a_i,2] = 1
                    BM_bounds[a_i,3] = 0
                end

            else
                BM_bounds[a_i,2] = 1
                BM_bounds[a_i,3] = 0
            end
        end
    end
    end

    # For Default
    # variables.V_S_D .= -Inf
    # Threads.@threads for z_i in 1:z_size
    #     for η_i in 1:η_size

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))

        @inbounds z_i = i[1]
        @inbounds η_i = i[2]

            @inbounds z = z_grid[z_i]
            @inbounds η = η_grid[η_i]

            # for n_i in 1:n_size
                # @inbounds n = n_grid[n_i]

                # l = T - n
                # c = z*η*(n^(θ))-ϕ
                # c = (z*η*(n^(θ)))*(1.0-ϕ)

                V_expect = 0.0

                for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                    @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p[a_ind_zero,z_p_i,η_p_i,κ_p_i]
                end

                f(n::Vector,grad::Vector) = Util_D_S_optim_function(n,grad,z,η,V_expect,θ,α,T,β,ρ,ϕ)

                opt = Opt(:LD_MMA,1)
                opt.max_objective = f
                opt.lower_bounds = 0.001
                opt.upper_bounds = 1.0
                opt.xtol_rel = 1e-4

                (optf,optx,ret) = optimize(opt,[0.5])

                # if isnan(optf) || isnan(optx[1])
                #     error("NaN in opt S,D")
                # end

                # v_temp = utility_function(c,l) + β*ρ*V_expect

                for κ_i in 1:κ_size, a_i in 1:a_size
                    if optf > variables.V_S[a_i,z_i,η_i,κ_i]
                        @inbounds variables.V_S[a_i,z_i,η_i,κ_i] = optf
                        @inbounds variables.d_S_i[a_i,z_i,η_i,κ_i] = 2
                        @inbounds variables.a_S_i[a_i,z_i,η_i,κ_i] = a_ind_zero
                        @inbounds variables.n_S_i[a_i,z_i,η_i,κ_i] = optx[1]
                    end
                end
            # end
#        end
    end

    # if slow_updating != 1.0
    #     variables.W = slow_updating*variables.W + (1.0-slow_updating)*W_p
    # end
end

function howard_singles!(
    V_S_p_howard,
    variables::MutableVariables,
    parameters::NamedTuple;
    howard_iter::Real = 15
    )

    @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, BM_indices = parameters

    iter = 1

    while iter <= howard_iter
        copyto!(V_S_p_howard, variables.V_S)
        Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
            for κ_i in 1:κ_size, a_i in 1:a_size
            # @inbounds a_i = i[1]
            @inbounds z_i = i[1]
            @inbounds η_i = i[2]
            # @inbounds κ_i = i[4]

            @inbounds a = a_grid[a_i]
            @inbounds z = z_grid[z_i]
            @inbounds η = η_grid[η_i]
            @inbounds κ = κ_grid[κ_i]

            @inbounds a_p_i = variables.a_S_i[a_i,z_i,η_i,κ_i]
            @inbounds a_p = a_grid[a_p_i]
            @inbounds n = variables.n_S_i[a_i,z_i,η_i,κ_i]

            if variables.d_S_i[a_i,z_i,η_i,κ_i] == 1
                l = T - n
                @inbounds c = z*η*(n^(θ))+a-variables.q_S[a_p_i,z_i]*a_p-κ

                V_expect = 0.0

                for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                    @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p_howard[a_p_i,z_p_i,η_p_i,κ_p_i]
                end

                variables.V_S[a_i,z_i,η_i,κ_i] = utility_function(c,l) + β*ρ*V_expect
            else
                l = T - n
                # c = z*η*(n^(θ))-ϕ
                c = (z*η*(n^(θ)))*(1.0-ϕ)

                V_expect = 0.0

                for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                    @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p_howard[a_ind_zero,z_p_i,η_p_i,κ_p_i]
                end

                variables.V_S[a_i,z_i,η_i,κ_i] = utility_function(c,l) + β*ρ*V_expect
            end
        end
        end
        iter += 1
    end
end

# Value function for couples
function value_function_couples!(
    V_C_p::Array{Float64,7},
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, λ, BM_indices, α = parameters

    variables.V_C .= -Inf

    # For Repayment
    # variables.V_S_R .= -Inf
    # Threads.@threads for a_i in 1:a_size
    #     for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size, z_1_i in 1:z_size

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
        for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, z_2_i in 1:z_size
        # lb_new = 1

        # non-defaulting value
        BM_bounds = zeros(Int,a_size,3)
        BM_bounds[:,1] = 1:a_size
        BM_bounds[1,2] = 1
        BM_bounds[1,3] = a_size
        BM_bounds[a_size,3] = a_size

        for BM_i in 1:a_size

            @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
            @inbounds lb = BM_bounds[lb_i,2]
            @inbounds ub = BM_bounds[ub_i,3]

            if ub != 0

                @inbounds z_1_i = i[1]
                # @inbounds z_2_i = i[2]
                @inbounds η_1_i = i[2]
                # @inbounds η_2_i = i[4]
                # @inbounds κ_1_i = i[5]
                # @inbounds κ_2_i = i[6]

                @inbounds a = a_grid[a_i]
                @inbounds z_1 = z_grid[z_1_i]
                @inbounds z_2 = z_grid[z_2_i]
                @inbounds η_1 = η_grid[η_1_i]
                @inbounds η_2 = η_grid[η_2_i]
                @inbounds κ_1 = κ_grid[κ_1_i]
                @inbounds κ_2 = κ_grid[κ_2_i]

                # lb = 1
                # if a_i > 1
                #     lb = lb_new
                # end
                #
                # v_temp_pre = -Inf

                for a_p_i in lb:ub #, n_1_i in 1:n_size, n_2_i in 1:n_size

                    @inbounds q_C = variables.q_C[a_p_i,z_1_i,z_2_i]

                    @inbounds a_p = a_grid[a_p_i]
                    # @inbounds n_1 = n_grid[n_1_i]
                    # @inbounds n_2 = n_grid[n_2_i]

                    # @inbounds l_1 = T - n_1
                    # @inbounds l_2 = T - n_2
                    # @inbounds c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)+a-variables.q_C[a_p_i,z_1_i,z_2_i]*a_p-κ_1-κ_2

                    V_expect = 0.0

                    for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
                        @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
                    end

                    n_1_min = max(((q_C*a_p+κ_1+κ_2-a-z_2*η_2*(0.99^θ))/(z_1*η_1)),0.0)^(1/θ)
                    n_2_min = max(((q_C*a_p+κ_1+κ_2-a-z_1*η_1*(0.99^θ))/(z_2*η_2)),0.0)^(1/θ)

                    if (n_1_min > 1) || (n_2_min > 1)
                        optf = -Inf
                    else

                        f(n::Vector,grad::Vector) = Util_R_C_optim_function(n,grad,z_1,z_2,η_1,η_2,a,q_C,a_p,κ_1,κ_2,V_expect,θ,α,T,λ,β,ρ)

                        opt = Opt(:LD_MMA,2)
                        opt.max_objective = f
                        opt.lower_bounds = [max(n_1_min,0.001), max(n_2_min,0.001)]
                        opt.upper_bounds = 1.0
                        opt.xtol_rel = 1e-4

                        n_1_initial = max(n_1_min,0.99)
                        n_2_initial = max(n_2_min,0.99)

                        (optf,optx,ret) = optimize(opt,[n_1_initial,n_2_initial])

                        # (optf,optx,ret) = optimize(opt,[variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i],variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]])
                    end

                    # if isnan(optf) || any(isnan.(optx))
                    #     error("NaN in opt C,R")
                    # end

                    # v_temp = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect

                    # if v_temp > v_temp_pre
                    #     v_temp_pre = v_temp
                    #     lb_new = a_p_i
                    # end

                    # if (a_p_i == a_size) && (n_1_i == n_size) && (n_2_i == n_size) && (v_temp_pre == -Inf)
                    #     lb_new = 1
                    # end

                    if optf > variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
                        @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optf
                        @inbounds variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = 1
                        @inbounds variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = a_p_i
                        @inbounds variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[1]
                        @inbounds variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[2]
                        BM_bounds[a_i,2] = a_p_i
                        BM_bounds[a_i,3] = a_p_i
                    # elseif optf < variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
                    #     break
                    end
                end

                if variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == -Inf
                    BM_bounds[a_i,2] = 1
                    BM_bounds[a_i,3] = 0

                end
            else
                BM_bounds[a_i,2] = 1
                BM_bounds[a_i,3] = 0
            end
        end
    end
    end

    # For Default
    # variables.V_S_D .= -Inf
    # Threads.@threads for z_1_i in 1:z_size
    #     for η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
        for η_2_i in 1:η_size, z_2_i in 1:z_size

            @inbounds z_1_i = i[1]
            # @inbounds z_2_i = i[2]
            @inbounds η_1_i = i[2]
            # @inbounds η_2_i = i[4]

            @inbounds z_1 = z_grid[z_1_i]
            @inbounds z_2 = z_grid[z_2_i]
            @inbounds η_1 = η_grid[η_1_i]
            @inbounds η_2 = η_grid[η_2_i]

            # for n_1_i in 1:n_size, n_2_i in 1:n_size
                # @inbounds n_1 = n_grid[n_1_i]
                # @inbounds n_2 = n_grid[n_2_i]

                # l_1 = T - n_1
                # l_2 = T - n_2
                # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
                # c = (z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ))*(1.0-ϕ)

                V_expect = 0.0

                for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
                    @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p[a_ind_zero,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
                end

                f(n::Vector,grad::Vector) = Util_D_C_optim_function(n,grad,z_1,z_2,η_1,η_2,V_expect,θ,α,T,λ,β,ρ,ϕ)

                opt = Opt(:LD_MMA,2)
                opt.max_objective = f
                opt.lower_bounds = 0.001
                opt.upper_bounds = 1.0
                opt.xtol_rel = 1e-4

                (optf,optx,ret) = optimize(opt,[0.5,0.5])

                if isinf(optf)
                    error("Inf in C-D")
                end

                # if isnan(optf) || any(isnan.(optx))
                #     error("NaN in opt C,D")
                # end

                # v_temp = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect

                for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, a_i in 1:a_size
                    if optf > variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
                        @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optf
                        @inbounds variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = 2
                        @inbounds variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = a_ind_zero
                        @inbounds variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[1]
                        @inbounds variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[2]
                    end
                end
            # end
#        end
    end
    end

    # if slow_updating != 1.0
    #     variables.W = slow_updating*variables.W + (1.0-slow_updating)*W_p
    # end
end

function howard_couples!(
    V_C_p_howard,
    variables::MutableVariables,
    parameters::NamedTuple;
    howard_iter::Real = 15
    )

    @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, λ, BM_indices = parameters

    iter = 1

    while iter <= howard_iter
        copyto!(V_C_p_howard, variables.V_C)

        Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
            for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, z_2_i in 1:z_size, a_i in 1:a_size
            # @inbounds a_i = i[1]
            @inbounds z_1_i = i[1]
            # @inbounds z_2_i = i[3]
            @inbounds η_1_i = i[2]
            # @inbounds η_2_i = i[5]
            # @inbounds κ_1_i = i[6]
            # @inbounds κ_2_i = i[7]

            @inbounds a = a_grid[a_i]
            @inbounds z_1 = z_grid[z_1_i]
            @inbounds z_2 = z_grid[z_2_i]
            @inbounds η_1 = η_grid[η_1_i]
            @inbounds η_2 = η_grid[η_2_i]
            @inbounds κ_1 = κ_grid[κ_1_i]
            @inbounds κ_2 = κ_grid[κ_2_i]

            @inbounds a_p_i = variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            @inbounds a_p = a_grid[a_p_i]
            @inbounds n_1 = variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            @inbounds n_2 = variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]

            if variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 1
                @inbounds l_1 = T - n_1
                @inbounds l_2 = T - n_2
                @inbounds c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)+a-variables.q_C[a_p_i,z_1_i,z_2_i]*a_p-κ_1-κ_2

                V_expect = 0.0

                for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
                    @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p_howard[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
                end

                variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
            else
                l_1 = T - n_1
                l_2 = T - n_2
                # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
                c = (z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ))*(1.0-ϕ)

                V_expect = 0.0

                for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
                    @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p_howard[a_ind_zero,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
                end

                variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
            end
        end
    end
        iter += 1
    end
end

function pricing_function!(
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )

    @unpack  a_size, z_size, η_size, κ_size, ρ, r, Γ_z, Γ_η, Γ_κ, a_ind_zero = parameters

    # Singles
    # variables.P_S .= 0.0
    Threads.@threads for a_p_i in 1:(a_ind_zero-1)
        for z_i in 1:z_size
            temp = 0.0

            for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                @inbounds temp += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*(variables.d_S_i[a_p_i,z_p_i,η_p_i,κ_p_i] == 1)
            end
            @inbounds variables.P_S[a_p_i,z_i] = temp

            @inbounds variables.q_S[a_p_i,z_i] = ρ*temp/(1+r)
        end
    end

    # Couples
    Threads.@threads for a_p_i in 1:(a_ind_zero-1)
        for z_2_i in 1:z_size, z_1_i in 1:z_size
            temp = 0.0

            for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
                @inbounds temp += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*(variables.d_C_i[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i] == 1)
            end
            @inbounds variables.P_C[a_p_i,z_1_i,z_2_i] = temp

            @inbounds variables.q_C[a_p_i,z_1_i,z_2_i] = ρ*temp/(1+r)
        end
    end

    # if slow_updating != 1.0
    #     # store previous pricing function
    #     q_b_p = similar(variables.q_b)
    #     copyto!(q_b_p, variables.q_b)
    # end
    #
    # if slow_updating != 1.0
    #     variables.q_b = slow_updating*variables.q_b + (1.0-slow_updating)*q_b_p
    # end
end

function stationary_distribution!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol::Real = tol_h,
    iter_max::Integer = iter_max,
    )

    # Unpack grids
    @unpack  a_size, z_size, η_size, κ_size, Γ_z, Γ_η, Γ_κ, a_ind_zero, ρ = parameters

    # initialize matrices
    μ_S_p = similar(variables.μ_S)
    μ_C_p = similar(variables.μ_C)

    # set up the criterion and the iteration number
    crit_μ = Inf
    iter_μ = 0
    prog_μ = ProgressThresh(tol, "Computing stationary distribution: ")

    while crit_μ > tol && iter_μ < iter_max

        # update distribution
        copyto!(μ_S_p, variables.μ_S)
        copyto!(μ_C_p, variables.μ_C)

        # nullify distribution
        variables.μ_S .= 0.0
        variables.μ_C .= 0.0

        Threads.@threads for a_p_i in 1:a_size
            # Singles
            for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size

                μ_temp_survivor = 0.0
                μ_temp_newborn = 0.0

                for κ_i in 1:κ_size, η_i in 1:η_size, z_i in 1:z_size, a_i in 1:a_size

                    @inbounds μ_temp_survivor += ρ*Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*(variables.a_S_i[a_i,z_i,η_i,κ_i] == a_p_i)*μ_S_p[a_i,z_i,η_i,κ_i]
                    if a_p_i == a_ind_zero
                        @inbounds μ_temp_newborn += (1-ρ)*Γ_η[η_p_i]*(z_p_i == 3)*(κ_p_i == 1)*μ_S_p[a_i,z_i,η_i,κ_i]
                    end
                end

                # assign the result
                @inbounds variables.μ_S[a_p_i,z_p_i,η_p_i,κ_p_i] += μ_temp_survivor + μ_temp_newborn
            end

            # Couples
            for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size

                μ_temp_survivor = 0.0
                μ_temp_newborn = 0.0

                for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size, z_1_i in 1:z_size, a_i in 1:a_size

                    @inbounds μ_temp_survivor += ρ*Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*(variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == a_p_i)*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
                    if a_p_i == a_ind_zero
                        @inbounds μ_temp_newborn += (1-ρ)*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*(z_1_p_i == 3)*(z_2_p_i == 3)*(κ_1_p_i == 1)*(κ_2_p_i == 1)*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
                    end
                end

                # assign the result
                @inbounds variables.μ_C[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i] += μ_temp_survivor + μ_temp_newborn
            end
        end

        # report the progress
        crit_μ = max(norm(variables.μ_S-μ_S_p, Inf),norm(variables.μ_C-μ_C_p, Inf))
        ProgressMeter.update!(prog_μ, crit_μ)

        # update the iteration number
        iter_μ += 1
    end
end

function solve_function!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol::Real = tol_h,
    iter_max::Integer = iter_max,
    one_loop::Bool = true
    )
    """
    solve model
    """

    if one_loop == true

        # initialize the iteration number and criterion
        iter = 0
        crit = Inf
        prog = ProgressThresh(tol, "Solving model (one-loop): ")

        # initialize the next-period functions
        V_S_p = similar(variables.V_S)
        V_C_p = similar(variables.V_C)
        q_S_p = similar(variables.q_S)
        q_C_p = similar(variables.q_C)
        # μ_p = similar(variables.μ)

        V_S_p_howard = similar(variables.V_S)
        V_C_p_howard = similar(variables.V_C)

        while crit > tol && iter < iter_max

            # copy previous unconditional value and loan pricing functions
            copyto!(V_S_p, variables.V_S)
            copyto!(V_C_p, variables.V_C)
            copyto!(q_S_p, variables.q_S)
            copyto!(q_C_p, variables.q_C)
            # copyto!(μ_p, variables.μ)

            # update value functions
            value_function_singles!(V_S_p, variables, parameters; slow_updating = 1.0)

            howard_singles!(V_S_p_howard,variables, parameters)

            value_function_couples!(V_C_p, variables, parameters; slow_updating = 1.0)

            howard_couples!(V_C_p_howard,variables, parameters)

            # compute payday loan price
            pricing_function!(variables, parameters; slow_updating = 1.0)

            # Compute stationary distribution
            # stationary_distribution!(μ_p, variables, parameters)

            # check convergence
            crit = max(norm(variables.V_S .- V_S_p, Inf), norm(variables.V_C .- V_C_p, Inf), norm(variables.q_S .- q_S_p, Inf), norm(variables.q_C .- q_C_p, Inf))

            # V_S_crit = norm(variables.V_S .- V_S_p, Inf)
            # V_S_crit_ind = argmax(abs.(variables.V_S .- V_S_p))
            # V_S_ind = variables.V_S[V_S_crit_ind]
            # V_S_p_ind = V_S_p[V_S_crit_ind]
            #
            # V_C_crit = norm(variables.V_C .- V_C_p, Inf)
            # V_C_crit_ind = argmax(abs.(variables.V_C .- V_C_p))
            # V_C_ind = variables.V_C[V_C_crit_ind]
            # V_C_p_ind = V_C_p[V_C_crit_ind]
            #
            # q_S_crit = norm(variables.q_S .- q_S_p, Inf)
            # q_S_crit_ind = argmax(abs.(variables.q_S .- q_S_p))
            #
            # q_C_crit = norm(variables.q_C .- q_C_p, Inf)
            # q_C_crit_ind = argmax(abs.(variables.q_C .- q_C_p))
            #
            # crit = max(V_S_crit, V_C_crit, q_S_crit, q_C_crit)

            # report progress
            ProgressMeter.update!(prog, crit)

            # println("V_S_crit = $V_S_crit at $V_S_crit_ind, V_S = $V_S_ind, V_S_p = $V_S_p_ind")
            # println("V_C_crit = $V_C_crit at $V_C_crit_ind, V_C = $V_C_ind, V_C_p = $V_C_p_ind")
            # println("q_S_crit = $q_S_crit at $q_S_crit_ind")
            # println("q_C_crit = $q_C_crit at $q_C_crit_ind")

            # update the iteration number
            iter += 1
            # println("iter = ", iter)
        end

    else

        # initialize the iteration number and criterion
        iter_q = 0
        crit_q = Inf
        prog_q = ProgressThresh(tol, "Solving loan pricing and type updating functions: ")

        # initialize the next-period function
        q_S_p = similar(variables.q_S)
        q_C_p = similar(variables.q_C)
        # μ_p = similar(variables.μ)

        while crit_q > tol && iter_q < iter_max

            # copy previous loan pricing function and type updating function
            copyto!(q_S_p, variables.q_S)
            copyto!(q_C_p, variables.q_C)
            # copyto!(μ_p, variables.μ)

            # initialize the iteration number and criterion
            iter_W = 0
            crit_W = Inf
            prog_W = ProgressThresh(tol, "Solving unconditional value function: ")

            # initialize the next-period function
            V_S_p = similar(variables.V_S)
            V_C_p = similar(variables.V_C)

            while crit_W > tol && iter_W < iter_max

                # copy previous loan pricing function
                copyto!(V_S_p, variables.V_S)
                copyto!(V_C_p, variables.V_C)

                # update value functions
                value_function_singles!(V_S_p, variables, parameters; slow_updating = 1.0)

                value_function_couples!(V_C_p, variables, parameters; slow_updating = 1.0)

                # check convergence
                crit_W = max(norm(variables.V_S .- V_S_p, Inf), norm(variables.V_C .- V_C_p, Inf))

                # report preogress
                ProgressMeter.update!(prog_W, crit_W)

                # update the iteration number
                iter_W += 1

            end

            # compute payday price
            pricing_function!(variables, parameters; slow_updating = 1.0)

            # Compute stationary distribution
            # stationary_distribution!(μ_p, variables, parameters)

            # check convergence
            crit_q = max(norm(variables.q_S .- q_S_p, Inf), norm(variables.q_C .- q_C_p, Inf))

            # report preogress
            ProgressMeter.update!(prog_q, crit_q)

            # update the iteration number
            iter_q += 1
            println("")
        end
    end
end

parameters = parameters_function()
variables = variables_function(parameters; load_initial_value = true)
solve_function!(variables, parameters; tol = 1E-4, iter_max = 1000, one_loop = true)
stationary_distribution!(variables, parameters; tol = 1E-9, iter_max = 1000)

# check whether the sum of choice probability, given any individual state, equals one
# all(sum(variables.σ, dims=1) .≈ 1.0)

# save and load workspace
V_S = variables.V_S
V_C = variables.V_C
n_S_i = variables.n_S_i
n_C_1_i = variables.n_C_1_i
n_C_2_i = variables.n_C_2_i
# l_S::Array{Float64,3}
# l_C_1::Array{Float64,5}
# l_C_2::Array{Float64,5}
a_S_i = variables.a_S_i
a_C_i = variables.a_C_i
d_S_i = variables.d_S_i
d_C_i = variables.d_C_i
P_S = variables.P_S
q_S = variables.q_S
P_C = variables.P_C
q_C = variables.q_C
μ_S = variables.μ_S
μ_C = variables.μ_C

using JLD2
@save "workspace.jld2" V_S V_C n_S_i n_C_1_i n_C_2_i a_S_i a_C_i d_S_i d_C_i P_S q_S P_C q_C μ_S μ_C

# cd("C:/Users/JanSun/Dropbox/Bankruptcy-Family/Results/7")
# @load "workspace.jld2"
