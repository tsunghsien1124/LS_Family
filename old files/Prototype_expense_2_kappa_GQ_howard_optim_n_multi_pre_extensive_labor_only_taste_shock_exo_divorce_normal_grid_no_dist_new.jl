# Prototype model, version where the multi-threading was adjusted for cluster speed

using Parameters
using LinearAlgebra

using ProgressMeter
# using NLopt
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
    # a_size_neg::Integer = 31,                   # number of  assets
    a_size_neg::Integer = 61,
    # a_size_neg::Integer = 6,
    # a_size_pos::Integer = 40,
    # a_size_pos::Integer = 230,
    a_size_pos::Integer = 100,
    # a_size_pos::Integer = 10,
    a_min::Real = -2.00,                        # minimum of assets
    a_max::Real = 10.00,                         # maximum of assets
    # a_degree::Real = 1.0,                      # governs how grid is spaced
    d_size::Integer = 2,                         # Number of default/repayment options
    # T::Real = 1.2,
    T::Real = 1.4,
    # α::Real = 0.5,                                 # Weight of leisure in utility function # From Alon et al. (2020)
    # α::Real = 1.27,
    α::Real = 0.2,
    γ_c::Real = 2.0,                             # CRRA parameter on consumption
    γ_l::Real = 3.0,                             # CRRA parameter on leisure
    # ϕ::Real = 0.355,                                 # Wage garnishment
    ϕ::Real = 0.9,
    # θ::Real = 0.55,                              # Returns to labor # From Alon et al. (2020)
    θ::Real = 1.0,
    ν::Real = 0.4,                               # Share of singles in economy
    # ζ::Real = 0.05,                             # Extreme value parameter
    ζ::Real = 0.1,
    # ψ::Real = 0.1,                              # Exogenous divorce probability
    ψ::Real = 0.5,
    κ::Real = 0.2,                              # Divorce cost shock
    # κ::Real = 0.0
    )
    """
    contruct an immutable object containing all parameters
    """

    # persistent productivity shock
    z_grid = [0.574824402185664, 1.0, 1.7396617057273214]
    z_size = length(z_grid)
    Γ_z = [ 0.818    0.178  0.004;
            0.178    0.644  0.178;
            0.004  0.178  0.818]

    # Transitory productivity shock
    η_grid = [0.777792, 1.0, 1.28569]
    η_size = length(η_grid)
    Γ_η = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]

    # Expense shock
    # κ_grid = [0.0 0.2 0.5]
    # Γ_κ = [0.8, 0.15, 0.05]
    # κ_grid = [0.0 0.2 4.0]
    κ_grid = [0.0 0.2 2.0]
    Γ_κ = [0.81, 0.16, 0.03]
    κ_size = length(κ_grid)

    # Asset grid
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = collect(range(0.0, a_max, length=a_size_pos))
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    # a_grid_neg = reverse((exp.(collect(range(log(1.0), stop=log(-a_min+1), length = a_size_neg))).-1)*(-1))
    # a_grid_pos = exp.(collect(range(log(1.0), stop=log(a_max+1), length = a_size_pos))).-1
    # a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    a_size_neg = length(a_grid_neg)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # Labor
    # n_grid = collect(range(0.0, 1.0, length=5))
    # n_grid = collect(range(0.0, 1.0, length=3))
    n_grid = collect(range(1.0, 1.0, length=1))
    n_size = length(n_grid)

    BM_indices = BM_function(a_size)

    # numerically relevant lower bound
    L_ζ = ζ*log(eps(Float64))

    # return the outcome
    return (β=β, λ=λ, r=r, ρ=ρ, T=T, α=α, γ_c=γ_c, γ_l=γ_l, ϕ=ϕ, θ=θ, ν=ν, κ_grid = κ_grid, Γ_κ = Γ_κ, κ_size = κ_size,
    a_grid=a_grid, a_size=a_size, a_ind_zero=a_ind_zero,
    z_grid=z_grid, z_size=z_size, Γ_z=Γ_z, η_grid=η_grid, η_size=η_size, Γ_η=Γ_η, BM_indices = BM_indices, n_grid=n_grid, n_size = n_size, d_size=d_size, ζ=ζ, L_ζ=L_ζ, ψ=ψ, κ = κ)
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    v_S_R::Array{Float64,6}
    v_S_D::Array{Float64,4}
    W_S_D::Array{Float64,3}
    W_S_R::Array{Float64,4}
    W_S::Array{Float64,4}
    # F_S::Array{Float64,6}
    σ_S_R::Array{Float64,6}
    σ_S_D::Array{Float64,4}
    σ_S_d::Array{Float64,5}
    σ_S::Array{Float64,6}
    P_S::Array{Float64,2}
    q_S::Array{Float64,2}
    # v_div_R::Array{Float64,5}
    # v_div_D::Array{Float64,4}
    # W_div_D::Array{Float64,3}
    # W_div_R::Array{Float64,3}
    # W_div::Array{Float64,3}
    # σ_div_R::Array{Float64,5}
    # σ_div_D::Array{Float64,4}
    # σ_div_d::Array{Float64,4}
    # σ_div::Array{Float64,5}
    # v_C_R::Array{Float64,8}
    # v_C_D::Array{Float64,7}
    # W_C_D::Array{Float64,5}
    # W_C_R::Array{Float64,5}
    # W_C::Array{Float64,5}
    # # F_C::Array{Float64,6}
    # σ_C_R::Array{Float64,8}
    # σ_C_D::Array{Float64,7}
    # σ_C_d::Array{Float64,6}
    # σ_C::Array{Float64,8}
    # P_C::Array{Float64,3}
    # q_C::Array{Float64,3}
    μ_S::Array{Float64,4}
    # μ_C::Array{Float64,5}
end

function variables_function(
    parameters::NamedTuple;
    load_initial_value::Bool = false
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack  a_size, n_size, z_size, η_size, κ_size, d_size, ρ, r, ν = parameters

    # Singles' value functions
    v_S_R = zeros(a_size, n_size, a_size, z_size, η_size, κ_size)
    v_S_D = zeros(n_size, a_size, z_size, η_size)
    W_S_D = zeros(a_size, z_size, η_size)
    W_S_R = zeros(a_size, z_size, η_size, κ_size)
    W_S = zeros(a_size, z_size, η_size, κ_size)

    # Singles' feasible set
    # F_S = zeros(a_size, n_size, a_size, z_size, η_size, κ_size)

    # Singles' choice probabilities
    σ_S_R = ones(a_size, n_size, a_size, z_size, η_size, κ_size) ./ (a_size*n_size)
    σ_S_D = ones(n_size, a_size, z_size, η_size) ./ n_size
    σ_S_d = ones(d_size, a_size, z_size, η_size, κ_size) ./ d_size
    σ_S = ones(a_size+1, n_size, a_size, z_size, η_size, κ_size) ./ ((a_size+1)*n_size)

    # Divorced value functions
    # v_div_R = zeros(a_size, n_size, a_size, z_size, η_size)
    # v_div_D = zeros(n_size, a_size, z_size, η_size)
    # W_div_D = zeros(a_size, z_size, η_size)
    # W_div_R = zeros(a_size, z_size, η_size)
    # W_div = zeros(a_size, z_size, η_size)

    # Divorced choice probabilities
    # σ_div_R = ones(a_size, n_size, a_size, z_size, η_size) ./ (a_size*n_size)
    # σ_div_D = ones(n_size, a_size, z_size, η_size) ./ n_size
    # σ_div_d = ones(d_size, a_size, z_size, η_size) ./ d_size
    # σ_div = ones(a_size+1, n_size, a_size, z_size, η_size) ./ ((a_size+1)*n_size)

    # Couples' value functions
    # v_C_R = zeros(a_size, n_size, n_size, a_size, z_size, z_size, η_size, η_size)
    # v_C_D = zeros(n_size, n_size, a_size, z_size, z_size, η_size, η_size)
    # W_C_D = zeros(a_size, z_size, z_size, η_size, η_size)
    # W_C_R = zeros(a_size, z_size, z_size, η_size, η_size)
    # W_C = zeros(a_size, z_size, z_size, η_size, η_size)

    # Couples' choice probabilities
    # σ_C_R = ones(a_size, n_size, n_size, a_size, z_size, z_size, η_size, η_size) ./ (a_size*n_size*n_size)
    # σ_C_D = ones(n_size, n_size, a_size, z_size, z_size, η_size, η_size) ./ (n_size*n_size)
    # σ_C_d = ones(d_size, a_size, z_size, z_size, η_size, η_size) ./ d_size
    # σ_C = ones(a_size+1, n_size, n_size, a_size, z_size, z_size, η_size, η_size) ./ ((a_size+1)*n_size*n_size)

    # Loan pricing
    # Singles
    P_S = ones(a_size, z_size)
    q_S = ones(a_size, z_size) .* ρ/(1.0 + r)

    # Couples
    # P_C = ones(a_size, z_size, z_size)
    # q_C = ones(a_size, z_size, z_size) .* ρ/(1.0 + r)

    # cross-sectional distribution
    μ_S = ones(a_size, z_size, η_size, κ_size) ./ (a_size*z_size*η_size*κ_size)
    # μ_C = ones(a_size, z_size, z_size, η_size, η_size) ./ (a_size*z_size*z_size*η_size*η_size)

    # return the outcome
    # variables = MutableVariables(V_S, V_C, n_S_i, n_C_1_i, n_C_2_i, a_S_i, a_C_i, d_S_i, d_C_i, P_S, q_S, P_C, q_C, μ_S, μ_C)
    # variables = MutableVariables(v_S_R, v_S_D, W_S_D, W_S_R, W_S, σ_S_R, σ_S_D, σ_S_d, σ_S, P_S, q_S, v_div_R, v_div_D, W_div_D, W_div_R, W_div, σ_div_R, σ_div_D, σ_div_d, σ_div, v_C_R, v_C_D, W_C_D, W_C_R, W_C, σ_C_R, σ_C_D, σ_C_d, σ_C, P_C, q_C)
    variables = MutableVariables(v_S_R, v_S_D, W_S_D, W_S_R, W_S, σ_S_R, σ_S_D, σ_S_d, σ_S, P_S, q_S, μ_S)
    return variables
end

function utility_function(
    c::Real,
    l::Real
    )
    """
    compute utility of CRRA utility function
    """
    @unpack α, γ_c, γ_l = parameters
    if (c > 0.0) && (l > 0.0)
        # return log(c) + α*log(l)
        # return (c^(1.0-γ_c)-1.0)/(1.0-γ_c) + α*((l^(1.0-γ_l)-1.0)/(1.0-γ_l))
        return (c^(1.0-γ_c)-1.0)/(1.0-γ_c)
    else
        return -Inf
    end
end

function value_function_singles!(
    W_S_p::Array{Float64,4},
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, BM_indices, α, n_grid, n_size, ζ, L_ζ = parameters

    # W_expect_mat = reshape(W_S_p,a_size,:)*transpose(kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z))
    W_expect_mat = reshape(W_S_p,a_size,:)*transpose(kron(reshape(Γ_κ,1,:),kron(reshape(Γ_η,1,:),Γ_z)))

    variables.v_S_R .= -Inf
    variables.v_S_D .= -Inf

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))

        @inbounds z_i = i[1]
        @inbounds η_i = i[2]

        @inbounds z = z_grid[z_i]
        @inbounds η = η_grid[η_i]

        # Formal default
        for n_i in 1:n_size

            @inbounds n = n_grid[n_i]

            l = T - n
            c = (z*η*(n^(θ)))*(1.0-ϕ)

            if c <= 0.0
                @inbounds variables.v_S_D[n_i, :, z_i, η_i] .= -Inf
            else
                @inbounds variables.v_S_D[n_i, :, z_i, η_i] .= utility_function(c,l) + β*ρ*W_expect_mat[a_ind_zero,z_i]
            end
        end

        for κ_i in 1:κ_size

            @inbounds κ = κ_grid[κ_i]

            # Repayment
            for n_i in 1:n_size

                @inbounds n = n_grid[n_i]

                l = T - n

                BM_bounds = zeros(Int,a_size,3)
                @inbounds BM_bounds[:,1] = 1:a_size
                @inbounds BM_bounds[1,2] = 1
                @inbounds BM_bounds[1,3] = a_size
                @inbounds BM_bounds[a_size,3] = a_size

                for BM_i in 1:a_size

                    @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
                    # @inbounds lb = BM_bounds[lb_i,2]
                    # @inbounds ub = BM_bounds[ub_i,3]

                    lb = 1
                    ub = a_size

                    @inbounds a = a_grid[a_i]

                    if ub != 0

                        for a_p_i in lb:ub
                            @inbounds q_S = variables.q_S[a_p_i,z_i]

                            @inbounds a_p = a_grid[a_p_i]

                            @inbounds c = z*η*(n^(θ))+a-variables.q_S[a_p_i,z_i]*a_p-κ

                            if c <= 0.0
                                @inbounds variables.v_S_R[a_p_i,n_i, a_i, z_i, η_i, κ_i] = -Inf
                            else
                                @inbounds variables.v_S_R[a_p_i,n_i, a_i, z_i, η_i, κ_i] = utility_function(c,l) + β*ρ*W_expect_mat[a_p_i,z_i]
                            end
                        end

                        if all(variables.v_S_R[:,n_i,a_i,z_i,η_i,κ_i] .== -Inf)
                            @inbounds BM_bounds[a_i,2] = 1
                            @inbounds BM_bounds[a_i,3] = 0
                        else
                            @inbounds @views U_star = maximum(variables.v_S_R[:,n_i,a_i,z_i,η_i,κ_i])

                            # Get new bounds (in terms of asset grid index!)
                            @inbounds @views lb_new = findfirst(variables.v_S_R[:,n_i,a_i,z_i,η_i,κ_i] .- U_star .>= L_ζ)
                            @inbounds @views ub_new = findlast(variables.v_S_R[:,n_i,a_i,z_i,η_i,κ_i] .- U_star .>= L_ζ)

                            @inbounds BM_bounds[a_i,2] = lb_new
                            @inbounds BM_bounds[a_i,3] = ub_new
                        end

                    else
                        @inbounds BM_bounds[a_i,2] = 1
                        @inbounds BM_bounds[a_i,3] = 0
                    end
                end
            end
        end

        for a_i in 1:a_size

            @inbounds @views v_S_D_max = maximum(variables.v_S_D[:,a_i,z_i,η_i])
            if v_S_D_max == -Inf
                @inbounds variables.W_S_D[a_i,z_i,η_i] = v_S_D_max
                @inbounds variables.σ_S_D[:,a_i,z_i,η_i] .= 0.0
            else
                @inbounds @views v_S_D_diff = variables.v_S_D[:,a_i,z_i,η_i] .- v_S_D_max
                v_S_D_exp = exp.(v_S_D_diff ./ ζ)
                v_S_D_sum = sum(v_S_D_exp)
                @inbounds variables.W_S_D[a_i,z_i,η_i] = v_S_D_max + ζ*log(v_S_D_sum)
                @inbounds variables.σ_S_D[:,a_i,z_i,η_i] = v_S_D_exp ./ v_S_D_sum
            end

            for κ_i in 1:κ_size

                @inbounds @views v_S_R_max = maximum(variables.v_S_R[:,:,a_i,z_i,η_i,κ_i])
                if v_S_R_max == -Inf
                    @inbounds variables.W_S_R[a_i,z_i,η_i,κ_i] = v_S_R_max
                    # variables.σ_S_R[:,:,a_i,z_i,η_i,κ_i] .= 1.0/(a_size*n_size)
                    @inbounds variables.σ_S_R[:,:,a_i,z_i,η_i,κ_i] .= 0.0
                else
                    @inbounds @views v_S_R_diff = variables.v_S_R[:,:,a_i,z_i,η_i,κ_i] .- v_S_R_max
                    v_S_R_exp = exp.(v_S_R_diff ./ ζ)
                    v_S_R_sum = sum(v_S_R_exp)
                    @inbounds variables.W_S_R[a_i,z_i,η_i,κ_i] = v_S_R_max + ζ*log(v_S_R_sum)
                    @inbounds variables.σ_S_R[:,:,a_i,z_i,η_i,κ_i] = v_S_R_exp ./ v_S_R_sum
                end

                # compute defaut/repayment probabilities
                @inbounds d_vec = vcat(variables.W_S_D[a_i,z_i,η_i],variables.W_S_R[a_i,z_i,η_i,κ_i])
                d_max = maximum(d_vec)
                d_diff = d_vec .- d_max
                d_exp = exp.(d_diff ./ ζ)
                d_sum = sum(d_exp)
                @inbounds variables.W_S[a_i,z_i,η_i,κ_i] = d_max + ζ*log(d_sum)
                @inbounds variables.σ_S_d[:,a_i,z_i,η_i,κ_i] = d_exp ./ d_sum

                @inbounds variables.σ_S[1,:,a_i,z_i,η_i,κ_i] = variables.σ_S_d[1,a_i,z_i,η_i,κ_i] .* variables.σ_S_D[:,a_i,z_i,η_i]
                @inbounds variables.σ_S[2:end,:,a_i,z_i,η_i,κ_i] = variables.σ_S_d[2,a_i,z_i,η_i,κ_i] .* variables.σ_S_R[:,:,a_i,z_i,η_i,κ_i]
            end
        end

    end

    # variables.F_S = variables.σ_S .> 0.0

    # if slow_updating != 1.0
    #     variables.W = slow_updating*variables.W + (1.0-slow_updating)*W_p
    # end
end

# function howard_singles!(
#     variables::MutableVariables,
#     parameters::NamedTuple;
#     # howard_iter::Real = 15
#     howard_iter::Real = 0
#     )
#
#     @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, BM_indices = parameters
#
#     iter = 1
#
#     while iter <= howard_iter
#         # copyto!(V_S_p_howard, variables.V_S)
#
#         # V_expect_mat = reshape(variables.V_S,a_size,:)*transpose(kron(parameters.Γ_z,kron(reshape(parameters.Γ_η,1,:),reshape(parameters.Γ_κ,1,:))))
#
#         V_expect_mat = reshape(variables.V_S,a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z)))
#
#         Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#             for κ_i in 1:κ_size, a_i in 1:a_size
#             # @inbounds a_i = i[1]
#             @inbounds z_i = i[1]
#             @inbounds η_i = i[2]
#             # @inbounds κ_i = i[4]
#
#             @inbounds a = a_grid[a_i]
#             @inbounds z = z_grid[z_i]
#             @inbounds η = η_grid[η_i]
#             @inbounds κ = κ_grid[κ_i]
#
#             @inbounds a_p_i = variables.a_S_i[a_i,z_i,η_i,κ_i]
#             @inbounds a_p = a_grid[a_p_i]
#             @inbounds n = variables.n_S_i[a_i,z_i,η_i,κ_i]
#
#             if variables.d_S_i[a_i,z_i,η_i,κ_i] == 1
#                 l = T - n
#                 @inbounds c = z*η*(n^(θ))+a-variables.q_S[a_p_i,z_i]*a_p-κ
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p_howard[a_p_i,z_p_i,η_p_i,κ_p_i]
#                 # end
#
#                 @inbounds V_expect = V_expect_mat[a_p_i,z_i]
#
#                 @inbounds variables.V_S[a_i,z_i,η_i,κ_i] = utility_function(c,l) + β*ρ*V_expect
#             else
#                 l = T - n
#                 # c = z*η*(n^(θ))-ϕ
#                 c = (z*η*(n^(θ)))*(1.0-ϕ)
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p_howard[a_ind_zero,z_p_i,η_p_i,κ_p_i]
#                 # end
#
#                 @inbounds V_expect = V_expect_mat[a_ind_zero,z_i]
#
#                 @inbounds variables.V_S[a_i,z_i,η_i,κ_i] = utility_function(c,l) + β*ρ*V_expect
#             end
#         end
#         end
#         iter += 1
#     end
# end

# function value_function_divorced!(
#     W_S_p::Array{Float64,3},
#     variables::MutableVariables,
#     parameters::NamedTuple;
#     slow_updating::Real = 1.0
#     )
#     """
#     compute feasible set and (un)conditional value functions
#     """
#
#     @unpack  a_size, z_size, η_size, a_grid, z_grid, η_grid, T, θ, β, ρ, Γ_z, Γ_η, ϕ, a_ind_zero, BM_indices, α, n_grid, n_size, ζ, L_ζ, κ = parameters
#
#     W_expect_mat = reshape(W_S_p,a_size,:)*transpose(kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z))
#
#     variables.v_div_R .= -Inf
#     variables.v_div_D .= -Inf
#
#     Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#
#         @inbounds z_i = i[1]
#         @inbounds η_i = i[2]
#
#         @inbounds z = z_grid[z_i]
#         @inbounds η = η_grid[η_i]
#
#         # Formal default
#         for a_i in 1:a_size, n_i in 1:n_size
#
#             @inbounds a = a_grid[a_i]
#             @inbounds n = n_grid[n_i]
#
#             if (a<0)
#                 l = T - n
#                 c = (z*η*(n^(θ)))*(1.0-ϕ)
#
#                 if c <= 0.0
#                     @inbounds variables.v_div_D[n_i, a_i, z_i, η_i] = -Inf
#                 else
#                     @inbounds variables.v_div_D[n_i, a_i, z_i, η_i] = utility_function(c,l) + β*ρ*W_expect_mat[a_ind_zero,z_i]
#                 end
#             end
#         end
#
#         # Repayment
#         for n_i in 1:n_size
#
#             @inbounds n = n_grid[n_i]
#
#             l = T - n
#
#             BM_bounds = zeros(Int,a_size,3)
#             @inbounds BM_bounds[:,1] = 1:a_size
#             @inbounds BM_bounds[1,2] = 1
#             @inbounds BM_bounds[1,3] = a_size
#             @inbounds BM_bounds[a_size,3] = a_size
#
#             for BM_i in 1:a_size
#
#                 @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
#                 @inbounds lb = BM_bounds[lb_i,2]
#                 @inbounds ub = BM_bounds[ub_i,3]
#
#                 @inbounds a = a_grid[a_i]
#
#                 if ub != 0
#
#                     for a_p_i in lb:ub
#                         @inbounds q_S = variables.q_S[a_p_i,z_i]
#
#                         @inbounds a_p = a_grid[a_p_i]
#
#                         @inbounds c = z*η*(n^(θ))+a-variables.q_S[a_p_i,z_i]*a_p - κ
#
#                         if c <= 0.0
#                             @inbounds variables.v_div_R[a_p_i,n_i, a_i, z_i, η_i] = -Inf
#                         else
#                             @inbounds variables.v_div_R[a_p_i,n_i, a_i, z_i, η_i] = utility_function(c,l) + β*ρ*W_expect_mat[a_p_i,z_i]
#                         end
#                     end
#
#                     if all(variables.v_div_R[:,n_i,a_i,z_i,η_i] .== -Inf)
#                         @inbounds BM_bounds[a_i,2] = 1
#                         @inbounds BM_bounds[a_i,3] = 0
#                     else
#                         @inbounds @views U_star = maximum(variables.v_div_R[:,n_i,a_i,z_i,η_i])
#
#                         # Get new bounds (in terms of asset grid index!)
#                         @inbounds @views lb_new = findfirst(variables.v_div_R[:,n_i,a_i,z_i,η_i] .- U_star .>= L_ζ)
#                         @inbounds @views ub_new = findlast(variables.v_div_R[:,n_i,a_i,z_i,η_i] .- U_star .>= L_ζ)
#
#                         @inbounds BM_bounds[a_i,2] = lb_new
#                         @inbounds BM_bounds[a_i,3] = ub_new
#                     end
#
#                 else
#                     @inbounds BM_bounds[a_i,2] = 1
#                     @inbounds BM_bounds[a_i,3] = 0
#                 end
#             end
#         end
#
#         for a_i in 1:a_size
#
#             @inbounds @views v_div_D_max = maximum(variables.v_div_D[:,a_i,z_i,η_i])
#             if v_div_D_max == -Inf
#                 @inbounds variables.W_div_D[a_i,z_i,η_i] = v_div_D_max
#                 @inbounds variables.σ_div_D[:,a_i,z_i,η_i] .= 0.0
#             else
#                 @inbounds @views v_div_D_diff = variables.v_div_D[:,a_i,z_i,η_i] .- v_div_D_max
#                 v_div_D_exp = exp.(v_div_D_diff ./ ζ)
#                 v_div_D_sum = sum(v_div_D_exp)
#                 @inbounds variables.W_div_D[a_i,z_i,η_i] = v_div_D_max + ζ*log(v_div_D_sum)
#                 @inbounds variables.σ_div_D[:,a_i,z_i,η_i] = v_div_D_exp ./ v_div_D_sum
#             end
#
#             @inbounds @views v_div_R_max = maximum(variables.v_div_R[:,:,a_i,z_i,η_i])
#             if v_div_R_max == -Inf
#                 @inbounds variables.W_div_R[a_i,z_i,η_i] = v_div_R_max
#                 # variables.σ_S_R[:,:,a_i,z_i,η_i,κ_i] .= 1.0/(a_size*n_size)
#                 @inbounds variables.σ_div_R[:,:,a_i,z_i,η_i] .= 0.0
#             else
#                 @inbounds @views v_div_R_diff = variables.v_div_R[:,:,a_i,z_i,η_i] .- v_div_R_max
#                 v_div_R_exp = exp.(v_div_R_diff ./ ζ)
#                 v_div_R_sum = sum(v_div_R_exp)
#                 @inbounds variables.W_div_R[a_i,z_i,η_i] = v_div_R_max + ζ*log(v_div_R_sum)
#                 @inbounds variables.σ_div_R[:,:,a_i,z_i,η_i] = v_div_R_exp ./ v_div_R_sum
#             end
#
#             # compute defaut/repayment probabilities
#             @inbounds d_vec = vcat(variables.W_div_D[a_i,z_i,η_i],variables.W_div_R[a_i,z_i,η_i])
#             d_max = maximum(d_vec)
#             d_diff = d_vec .- d_max
#             d_exp = exp.(d_diff ./ ζ)
#             d_sum = sum(d_exp)
#             @inbounds variables.W_div[a_i,z_i,η_i] = d_max + ζ*log(d_sum)
#             @inbounds variables.σ_div_d[:,a_i,z_i,η_i] = d_exp ./ d_sum
#
#             @inbounds variables.σ_div[1,:,a_i,z_i,η_i] = variables.σ_div_d[1,a_i,z_i,η_i] .* variables.σ_div_D[:,a_i,z_i,η_i]
#             @inbounds variables.σ_div[2:end,:,a_i,z_i,η_i] = variables.σ_div_d[2,a_i,z_i,η_i] .* variables.σ_div_R[:,:,a_i,z_i,η_i]
#         end
#     end
#
#     # variables.F_S = variables.σ_S .> 0.0
#
#     # if slow_updating != 1.0
#     #     variables.W = slow_updating*variables.W + (1.0-slow_updating)*W_p
#     # end
# end

# Value function for couples
# function value_function_couples!(
#     W_C_p::Array{Float64,5},
#     variables::MutableVariables,
#     parameters::NamedTuple;
#     slow_updating::Real = 1.0
#     )
#     """
#     compute feasible set and (un)conditional value functions
#     """
#
#     @unpack  a_size, z_size, η_size, a_grid, z_grid, η_grid, T, θ, β, ρ, Γ_z, Γ_η, ϕ, a_ind_zero, λ, BM_indices, α, n_grid, n_size, ζ, L_ζ, ψ = parameters
#
#     W_expect_C_mat = reshape(W_C_p,a_size,:)*transpose(kron(reshape(parameters.Γ_η,1,:),kron(reshape(parameters.Γ_η,1,:),kron(parameters.Γ_z,parameters.Γ_z))))
#
#     W_expect_div_mat = reshape(variables.W_div,a_size,:)*transpose(kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z))
#
#     variables.v_C_R .= -Inf
#     variables.v_C_D .= -Inf
#
#     Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#         for η_2_i in 1:η_size, z_2_i in 1:z_size
#
#             @inbounds z_1_i = i[1]
#             @inbounds η_1_i = i[2]
#
#             @inbounds z_1 = z_grid[z_1_i]
#             @inbounds z_2 = z_grid[z_2_i]
#             @inbounds η_1 = η_grid[η_1_i]
#             @inbounds η_2 = η_grid[η_2_i]
#
#             @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#         # Formal default
#         for a_i in 1:a_size, n_2_i in 1:n_size, n_1_i in 1:n_size
#
#             @inbounds a = a_grid[a_i]
#             @inbounds n_1 = n_grid[n_1_i]
#             @inbounds n_2 = n_grid[n_2_i]
#
#             if (a<0)
#                 l_1 = T - n_1
#                 l_2 = T - n_2
#                 # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
#                 c = (z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ))*(1.0-ϕ)
#
#                 if c <= 0.0
#                     @inbounds variables.v_C_D[n_1_i, n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] = -Inf
#                 else
#                     @inbounds variables.v_C_D[n_1_i, n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*(1-ψ)*W_expect_C_mat[a_ind_zero,z_i] + β*ρ*ψ*(λ*W_expect_div_mat[a_ind_zero, z_1_i] + (1-λ)*W_expect_div_mat[a_ind_zero, z_2_i])
#                 end
#             end
#         end
#
#         # Repayment
#         for n_2_i in 1:n_size, n_1_i in 1:n_size
#
#             @inbounds n_1 = n_grid[n_1_i]
#             @inbounds n_2 = n_grid[n_2_i]
#
#             l_1 = T - n_1
#             l_2 = T - n_2
#
#             BM_bounds = zeros(Int,a_size,3)
#             @inbounds BM_bounds[:,1] = 1:a_size
#             @inbounds BM_bounds[1,2] = 1
#             @inbounds BM_bounds[1,3] = a_size
#             @inbounds BM_bounds[a_size,3] = a_size
#
#             for BM_i in 1:a_size
#
#                 @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
#                 @inbounds lb = BM_bounds[lb_i,2]
#                 @inbounds ub = BM_bounds[ub_i,3]
#
#                 @inbounds a = a_grid[a_i]
#
#                 if ub != 0
#
#                     for a_p_i in lb:ub
#                         @inbounds q_C = variables.q_C[a_p_i,z_1_i,z_2_i]
#
#                         @inbounds a_p = a_grid[a_p_i]
#
#                         @inbounds c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)+a-variables.q_C[a_p_i,z_1_i,z_2_i]*a_p
#
#                         if c <= 0.0
#                             @inbounds variables.v_C_R[a_p_i, n_1_i, n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] = -Inf
#                         else
#                             # if a_p_i < a_ind_zero
#                             #     a_p_i_half = a_p_i + 1
#                             # elseif a_p_i == a_ind_zero
#                             #     a_p_i_half = a_p_i
#                             # else
#                             #     a_p_i_half = a_p_i - 1
#                             # end
#
#                             if a_p/2 in a_grid
#                                 a_p_i_half_1 = findfirst(a_grid .== a_p/2)
#                                 a_p_i_half_2 = a_p_i_half_1
#                             else
#                                 a_p_i_half_1 = findlast(a_grid .< a_p/2)
#                                 a_p_i_half_2 = findfirst(a_grid .> a_p/2)
#                             end
#
#                             @inbounds variables.v_C_R[a_p_i, n_1_i, n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*(1-ψ)*W_expect_C_mat[a_p_i,z_i] + β*ρ*ψ*(λ*W_expect_div_mat[a_p_i_half_1, z_1_i] + (1-λ)*W_expect_div_mat[a_p_i_half_2, z_2_i])
#                         end
#                     end
#
#                     if all(variables.v_C_R[:,n_1_i,n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] .== -Inf)
#                         @inbounds BM_bounds[a_i,2] = 1
#                         @inbounds BM_bounds[a_i,3] = 0
#                     else
#                         @inbounds @views U_star = maximum(variables.v_C_R[:,n_1_i,n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i])
#
#                         # Get new bounds (in terms of asset grid index!)
#                         @inbounds @views lb_new = findfirst(variables.v_C_R[:,n_1_i,n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] .- U_star .>= L_ζ)
#                         @inbounds @views ub_new = findlast(variables.v_C_R[:,n_1_i,n_2_i, a_i, z_1_i, z_2_i, η_1_i, η_2_i] .- U_star .>= L_ζ)
#
#                         @inbounds BM_bounds[a_i,2] = lb_new
#                         @inbounds BM_bounds[a_i,3] = ub_new
#                     end
#
#                 else
#                     @inbounds BM_bounds[a_i,2] = 1
#                     @inbounds BM_bounds[a_i,3] = 0
#                 end
#             end
#         end
#
#         for a_i in 1:a_size
#
#             @inbounds @views v_C_D_max = maximum(variables.v_C_D[:,:, a_i, z_1_i, z_2_i, η_1_i, η_2_i])
#             if v_C_D_max == -Inf
#                 @inbounds variables.W_C_D[a_i,z_1_i,z_2_i,η_1_i,η_2_i] = v_C_D_max
#                 @inbounds variables.σ_C_D[:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] .= 0.0
#             else
#                 @inbounds @views v_C_D_diff = variables.v_C_D[:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] .- v_C_D_max
#                 v_C_D_exp = exp.(v_C_D_diff ./ ζ)
#                 v_C_D_sum = sum(v_C_D_exp)
#                 @inbounds variables.W_C_D[a_i,z_1_i,z_2_i,η_1_i,η_2_i] = v_C_D_max + ζ*log(v_C_D_sum)
#                 @inbounds variables.σ_C_D[:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] = v_C_D_exp ./ v_C_D_sum
#             end
#
#             @inbounds @views v_C_R_max = maximum(variables.v_C_R[:,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i])
#             if v_C_R_max == -Inf
#                 @inbounds variables.W_C_R[a_i,z_1_i,z_2_i,η_1_i,η_2_i] = v_C_R_max
#                 # variables.σ_S_R[:,:,a_i,z_i,η_i,κ_i] .= 1.0/(a_size*n_size)
#                 @inbounds variables.σ_C_R[:,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] .= 0.0
#             else
#                 @inbounds @views v_C_R_diff = variables.v_C_R[:,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] .- v_C_R_max
#                 v_C_R_exp = exp.(v_C_R_diff ./ ζ)
#                 v_C_R_sum = sum(v_C_R_exp)
#                 @inbounds variables.W_C_R[a_i,z_1_i,z_2_i,η_1_i,η_2_i] = v_C_R_max + ζ*log(v_C_R_sum)
#                 @inbounds variables.σ_C_R[:,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] = v_C_R_exp ./ v_C_R_sum
#             end
#
#             # compute default/repayment probabilities
#             @inbounds d_vec = vcat(variables.W_C_D[a_i,z_1_i,z_2_i,η_1_i,η_2_i],variables.W_C_R[a_i,z_1_i,z_2_i,η_1_i,η_2_i])
#             d_max = maximum(d_vec)
#             d_diff = d_vec .- d_max
#             d_exp = exp.(d_diff ./ ζ)
#             d_sum = sum(d_exp)
#             @inbounds variables.W_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i] = d_max + ζ*log(d_sum)
#             @inbounds variables.σ_C_d[:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] = d_exp ./ d_sum
#
#             @inbounds variables.σ_C[1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] = variables.σ_C_d[1,a_i,z_1_i,z_2_i,η_1_i,η_2_i] .* variables.σ_C_D[:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i]
#             @inbounds variables.σ_C[2:end,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i] = variables.σ_C_d[2,a_i,z_1_i,z_2_i,η_1_i,η_2_i] .* variables.σ_C_R[:,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i]
#         end
#     end
#     end
# end
#
# function howard_couples!(
#     variables::MutableVariables,
#     parameters::NamedTuple;
#     howard_iter::Real = 15
#     # howard_iter::Real = 0
#     )
#
#     @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, λ, BM_indices = parameters
#
#     iter = 1
#
#     while iter <= howard_iter
#         # copyto!(V_C_p_howard, variables.V_C)
#
#         V_expect_mat = reshape(variables.V_C,a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_η,1,:),kron(reshape(parameters.Γ_η,1,:),kron(parameters.Γ_z,parameters.Γ_z))))))
#
#         Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#             for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, z_2_i in 1:z_size, a_i in 1:a_size
#             # @inbounds a_i = i[1]
#             @inbounds z_1_i = i[1]
#             # @inbounds z_2_i = i[3]
#             @inbounds η_1_i = i[2]
#             # @inbounds η_2_i = i[5]
#             # @inbounds κ_1_i = i[6]
#             # @inbounds κ_2_i = i[7]
#
#             @inbounds a = a_grid[a_i]
#             @inbounds z_1 = z_grid[z_1_i]
#             @inbounds z_2 = z_grid[z_2_i]
#             @inbounds η_1 = η_grid[η_1_i]
#             @inbounds η_2 = η_grid[η_2_i]
#             @inbounds κ_1 = κ_grid[κ_1_i]
#             @inbounds κ_2 = κ_grid[κ_2_i]
#
#             @inbounds a_p_i = variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#             @inbounds a_p = a_grid[a_p_i]
#             @inbounds n_1 = variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#             @inbounds n_2 = variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#
#             if variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 1
#                 @inbounds l_1 = T - n_1
#                 @inbounds l_2 = T - n_2
#                 @inbounds c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)+a-variables.q_C[a_p_i,z_1_i,z_2_i]*a_p-κ_1-κ_2
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p_howard[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
#                 # end
#
#                 @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#                 @inbounds V_expect = V_expect_mat[a_p_i,z_i]
#
#                 @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#             else
#                 l_1 = T - n_1
#                 l_2 = T - n_2
#                 # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
#                 c = (z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ))*(1.0-ϕ)
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p_howard[a_ind_zero,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
#                 # end
#
#                 @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#                 @inbounds V_expect = V_expect_mat[a_ind_zero,z_i]
#
#                 @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#             end
#         end
#     end
#         iter += 1
#     end
# end

function pricing_function!(
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )

    @unpack  a_size, z_size, η_size, κ_size, a_grid, ρ, r, Γ_z, Γ_η, Γ_κ, a_ind_zero, ψ = parameters

    # Singles
    # variables.P_S .= 0.0
    Threads.@threads for a_p_i in 1:(a_ind_zero-1)
        for z_i in 1:z_size

            P_expect = 0.0

            for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                @inbounds @views P_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*(1.0-sum(variables.σ_S[1,:,a_p_i,z_p_i,η_p_i,κ_p_i]))
            end

            @inbounds variables.P_S[a_p_i,z_i] = P_expect

            @inbounds variables.q_S[a_p_i,z_i] = ρ*P_expect/(1+r)
        end
    end

    # Couples
    # Threads.@threads for a_p_i in 1:(a_ind_zero-1)
    #
    #     a_p = a_grid[a_p_i]
    #
    #     if a_p/2 in a_grid
    #         a_p_i_half_1 = findfirst(a_grid .== a_p/2)
    #         a_p_i_half_2 = a_p_i_half_1
    #     else
    #         a_p_i_half_1 = findlast(a_grid .< a_p/2)
    #         a_p_i_half_2 = findfirst(a_grid .> a_p/2)
    #     end
    #
    #     a_p_half_1 = a_grid[a_p_i_half_1]
    #     a_p_half_2 = a_grid[a_p_i_half_2]
    #
    #     for z_2_i in 1:z_size, z_1_i in 1:z_size
    #         P_expect_remain = 0.0
    #         P_expect_div_1 = 0.0
    #         P_expect_div_2 = 0.0
    #
    #         for η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
    #             @inbounds @views P_expect_remain += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*(1.0-sum(variables.σ_C[1,:,:,a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i]))
    #         end
    #
    #         for η_1_p_i in 1:η_size, z_1_p_i in 1:z_size
    #             @inbounds @views P_expect_div_1 += Γ_z[z_1_i,z_1_p_i]*Γ_η[η_1_p_i]*(1.0-sum(variables.σ_div[1,:,a_p_i_half_1,z_1_p_i,η_1_p_i]))
    #         end
    #
    #         for η_2_p_i in 1:η_size, z_2_p_i in 1:z_size
    #             @inbounds @views P_expect_div_2 += Γ_z[z_2_i,z_2_p_i]*Γ_η[η_2_p_i]*(1.0-sum(variables.σ_div[1,:,a_p_i_half_2,z_2_p_i,η_2_p_i]))
    #         end
    #
    #         @inbounds variables.P_C[a_p_i,z_1_i,z_2_i] = ρ*(1.0-ψ)*P_expect_remain + ρ*ψ*((a_p_half_1/a_p)*P_expect_div_1 + (a_p_half_2/a_p)*P_expect_div_2)
    #
    #         @inbounds variables.q_C[a_p_i,z_1_i,z_2_i] = variables.P_C[a_p_i,z_1_i,z_2_i]/(1+r)
    #     end
    # end

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
    # μ_C_p = similar(variables.μ_C)

    # set up the criterion and the iteration number
    crit_μ = Inf
    iter_μ = 0
    prog_μ = ProgressThresh(tol, "Computing stationary distribution: ")

    while crit_μ > tol && iter_μ < iter_max

        # update distribution
        copyto!(μ_S_p, variables.μ_S)
        # copyto!(μ_C_p, variables.μ_C)

        # nullify distribution
        variables.μ_S .= 0.0
        # variables.μ_C .= 0.0

        # Threads.@threads for η_p_i in 1:η_size, z_p_i in 1:z_size
        Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
            @inbounds z_p_i = i[1]
            @inbounds η_p_i = i[2]

            for κ_p_i in 1:κ_size

                # Singles
                for a_p_i in 1:a_size

                    μ_temp_survivor = 0.0
                    μ_temp_newborn = 0.0

                    for κ_i in 1:κ_size, η_i in 1:η_size, z_i in 1:z_size, a_i in 1:a_size

                        @inbounds @views temp = sum(variables.σ_S[1+a_p_i,:,a_i,z_i,η_i,κ_i])

                        if temp > 0.0
                            # @inbounds @views μ_temp_survivor += ρ*Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*sum(variables.σ_S[1+a_p_i,:,a_i,z_i,η_i,κ_i])*μ_S_p[a_i,z_i,η_i,κ_i]
                            @inbounds μ_temp_survivor += ρ*Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*temp*μ_S_p[a_i,z_i,η_i,κ_i]
                        end
                        if a_p_i == a_ind_zero
                            @inbounds @views temp_2 = sum(variables.σ_S[1,:,a_i,z_i,η_i,κ_i])
                            if temp_2 > 0.0
                                # @inbounds @views μ_temp_survivor += ρ*Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*sum(variables.σ_S[1,:,a_i,z_i,η_i,κ_i])*μ_S_p[a_i,z_i,η_i,κ_i]
                                @inbounds μ_temp_survivor += ρ*Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*temp_2*μ_S_p[a_i,z_i,η_i,κ_i]
                            end
                            @inbounds μ_temp_newborn += (1-ρ)*Γ_η[η_p_i]*(z_p_i == 2)*(κ_p_i == 1)*μ_S_p[a_i,z_i,η_i,κ_i]
                        end
                    end

                    # assign the result
                    @inbounds variables.μ_S[a_p_i,z_p_i,η_p_i,κ_p_i] = μ_temp_survivor + μ_temp_newborn
                end
            end

            # Couples
            # for η_2_p_i in 1:η_size, z_2_p_i in 1:z_size, a_p_i in 1:a_size
            #
            #     μ_temp_survivor = 0.0
            #     μ_temp_newborn = 0.0
            #
            #     for η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size, z_1_i in 1:z_size, a_i in 1:a_size
            #
            #         @inbounds @views temp = sum(variables.σ_C[1+a_p_i,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i])
            #
            #         if temp > 0.0
            #             # @inbounds @views μ_temp_survivor += ρ*Γ_z[z_1_i,z_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*sum(variables.σ_C[1+a_p_i,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            #             @inbounds μ_temp_survivor += ρ*Γ_z[z_1_i,z_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_p_i]*Γ_η[η_2_p_i]*temp*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i]
            #         end
            #         if a_p_i == a_ind_zero
            #             @inbounds @views temp_2 = sum(variables.σ_C[1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i])
            #             if temp_2 > 0.0
            #                 # @inbounds @views μ_temp_survivor += ρ*Γ_z[z_1_i,z_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*sum(variables.σ_C[1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            #                 @inbounds μ_temp_survivor += ρ*Γ_z[z_1_i,z_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_p_i]*Γ_η[η_2_p_i]*temp_2*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i]
            #             end
            #             @inbounds μ_temp_newborn += (1-ρ)*Γ_η[η_p_i]*Γ_η[η_2_p_i]*(z_p_i == 2)*(z_2_p_i == 2)*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i]
            #         end
            #     end
            #
            #     # assign the result
            #     @inbounds variables.μ_C[a_p_i,z_p_i,z_2_p_i,η_p_i,η_2_p_i] = μ_temp_survivor + μ_temp_newborn
            # end
        end

        # report the progress
        # crit_μ = max(norm(variables.μ_S-μ_S_p, Inf),norm(variables.μ_C-μ_C_p, Inf))
        crit_μ = norm(variables.μ_S-μ_S_p, Inf)
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
        W_S_p = similar(variables.W_S)
        # W_C_p = similar(variables.W_C)
        q_S_p = similar(variables.q_S)
        # q_C_p = similar(variables.q_C)

        # V_S_p_howard = similar(variables.V_S)
        # V_C_p_howard = similar(variables.V_C)

        while crit > tol && iter < iter_max

            # copy previous unconditional value and loan pricing functions
            copyto!(W_S_p, variables.W_S)
            # copyto!(W_C_p, variables.W_C)
            copyto!(q_S_p, variables.q_S)
            # copyto!(q_C_p, variables.q_C)

            # copyto!(μ_p, variables.μ)

            # update value functions
            value_function_singles!(W_S_p, variables, parameters; slow_updating = 1.0)

            # howard_singles!(variables, parameters)

            # value_function_divorced!(W_S_p, variables, parameters; slow_updating = 1.0)
            #
            # value_function_couples!(W_C_p, variables, parameters; slow_updating = 1.0)

            # howard_couples!(variables, parameters)

            # compute payday loan price
            pricing_function!(variables, parameters; slow_updating = 1.0)

            # check convergence
            # crit = max(norm(variables.W_S .- W_S_p, Inf), norm(variables.W_C .- W_C_p, Inf), norm(variables.q_S .- q_S_p, Inf), norm(variables.q_C .- q_C_p, Inf))
            crit = max(norm(variables.W_S .- W_S_p, Inf), norm(variables.q_S .- q_S_p, Inf))

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
stationary_distribution!(variables, parameters; tol = 1E-6, iter_max = 1000)

# check whether the sum of choice probability, given any individual state, equals one
# all(sum(variables.σ, dims=1) .≈ 1.0)

# save and load workspace
v_S_R = variables.v_S_R
v_S_D = variables.v_S_D
W_S = variables.W_S
σ_S = variables.σ_S
# v_div_R = variables.v_div_R
# v_div_D = variables.v_div_D
# W_div = variables.W_div
# σ_div = variables.σ_div
# v_C_R = variables.v_C_R
# v_C_D = variables.v_C_D
# W_C = variables.W_C
# σ_C = variables.σ_C
P_S = variables.P_S
q_S = variables.q_S
# P_C = variables.P_C
# q_C = variables.q_C
μ_S = variables.μ_S
# μ_C = variables.μ_C

using JLD2
# @save "workspace.jld2" parameters v_S_R v_S_D W_S σ_S v_div_R v_div_D W_div σ_div v_C_R v_C_D W_C σ_C P_S q_S P_C q_C
@save "workspace.jld2" parameters v_S_R v_S_D W_S σ_S P_S q_S μ_S

# cd("C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock")
# @load "workspace.jld2"
