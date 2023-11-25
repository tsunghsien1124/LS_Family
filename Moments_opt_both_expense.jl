# Moments

#===============#
# Model Moments #
#===============#
#================#
# Default #
#================#
# Unconditional
# Singles
# d_S_i_dist = zeros(parameters.a_size_dist,parameters.z_size,parameters.η_size)
#
# for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size, a_i in 1:parameters.a_size_dist
#     default_thresh_i = findfirst(d_S_i[:,z_i,η_i] .≈ 1.0)
#     default_thresh = parameters.a_grid[default_thresh_i]
#
#     if parameters.a_grid_dist[a_i] < default_thresh
#         d_S_i_dist[a_i,z_i,η_i] = 2
#     else
#         d_S_i_dist[a_i,z_i,η_i] = 1
#     end
# end
#
# fraction_S_default = sum((d_S_i_dist.==2).*μ_S)*100

fraction_S_default = sum((d_S_i.==2).*μ_S)*100
# Couples
# fraction_C_default = sum((d_C_i.==2).*μ_C)*100

# Across z
# Singles
# fraction_S_default_z = zeros(parameters.z_size)
# for i in 1:parameters.z_size
#     fraction_S_default_z[i] = sum((d_S_i_dist[:,i,:].==2).*μ_S[:,i,:])/sum(μ_S[:,i,:])*100
# end

fraction_S_default_z = zeros(parameters.z_size)
for i in 1:parameters.z_size
    fraction_S_default_z[i] = sum((d_S_i[:,i,:,:].==2).*μ_S[:,i,:,:])/sum(μ_S[:,i,:,:])*100
end
# Couples
# fraction_C_default_z = zeros(parameters.z_size,parameters.z_size)
# for i in 1:parameters.z_size, j in 1:parameters.z_size
#     fraction_C_default_z[i,j] = sum((d_C_i[:,i,j,:,:,:,:].==2).*μ_C[:,i,j,:,:,:,:])/sum(μ_C[:,i,j,:,:,:,:])*100
# end
#
# fraction_C_default_z_reduced = zeros(parameters.z_size)
# for i in 1:parameters.z_size
#     fraction_C_default_z_reduced[i] = sum((d_C_i[:,i,:,:,:,:,:].==2).*μ_C[:,i,:,:,:,:,:])/sum(μ_C[:,i,:,:,:,:,:])*100
# end

# # Across z
# fraction_full_default_e = zeros(parameters.e_size)
# for i in 1:parameters.e_size
#     fraction_full_default_e[i] = sum(σ[1,:,i,:,:].*μ[:,i,:,:])/sum(μ[:,i,:,:])*100
# end
#
# # Across η
# fraction_full_default_β = zeros(parameters.β_size)
# for i in 1:parameters.β_size
#     fraction_full_default_β[i] = sum(σ[1,i,:,:,:].*μ[i,:,:,:])/sum(μ[i,:,:,:])*100
# end

#=============================#
# Fraction of loan users #
#=============================#
# Unconditional
# Singles
a_i = [(parameters.a_grid_dist .< 0.0)][1]
fraction_S_loan_users = sum(μ_S[a_i,:,:,:])*100
# Couples
# fraction_C_loan_users = sum(μ_C[a_i,:,:,:,:,:,:])*100

# Across z
# Singles
fraction_S_loan_users_z = zeros(parameters.z_size)
for i in 1:parameters.z_size
    fraction_S_loan_users_z[i] = sum(μ_S[a_i,i,:,:])/sum(μ_S[:,i,:,:])*100
end

# Couples
# fraction_C_loan_users_z = zeros(parameters.z_size,parameters.z_size)
# for i in 1:parameters.z_size, j in 1:parameters.z_size
#     fraction_C_loan_users_z[i,j] = sum(μ_C[a_i,i,j,:,:,:,:])/sum(μ_C[:,i,j,:,:,:,:])*100
# end
#
# fraction_C_loan_users_z_reduced = zeros(parameters.z_size)
# for i in 1:parameters.z_size
#     fraction_C_loan_users_z_reduced[i] = sum(μ_C[a_i,i,:,:,:,:,:])/sum(μ_C[:,i,:,:,:,:,:])*100
# end

# Across η
# Singles
# a_S_dist = zeros(parameters.a_size_dist,parameters.z_size,parameters.η_size)
#
# for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size
#       default_thresh_i = findfirst(variables.a_S[:,z_i,η_i] .!= 0.0)
#       default_thresh = parameters.a_grid[default_thresh_i]
#
#       interp_a_S = LinearInterpolation(parameters.a_grid, variables.a_S[:,z_i,η_i])
#
#       for a_i in 1:parameters.a_size_dist
#
#             if parameters.a_grid_dist[a_i] < default_thresh
#                 a_S_dist[a_i,z_i,η_i] = 0.0
#             else
#                 a_S_dist[a_i,z_i,η_i] = interp_a_S(parameters.a_grid_dist[a_i])
#             end
#       end
# end

# fraction_S_loan_users_η = zeros(parameters.η_size)
# for i in 1:parameters.η_size
#     fraction_S_loan_users_η[i] = sum((a_S_dist[:,:,i] .< 0.0).*μ_S[:,:,i])/sum(μ_S[:,:,i])*100
# end

fraction_S_loan_users_η = zeros(parameters.η_size)
for i in 1:parameters.η_size
    fraction_S_loan_users_η[i] = sum((a_S[:,:,i,:] .< 0.0).*μ_S[:,:,i,:])/sum(μ_S[:,:,i,:])*100
end

# Couples
# fraction_C_loan_users_η = zeros(parameters.η_size, parameters.η_size)
# for i in 1:parameters.η_size, j in 1:parameters.η_size
#     fraction_C_loan_users_η[i,j] = sum(μ_C[a_i,:,:,i,j,:,:])/sum(μ_C[:,:,:,i,j,:,:])*100
# end

#=========================#
#     Debt-to-income      #
#=========================#
# Singles
# debt_to_wage_income_S = zeros(parameters.a_ind_zero_dist-1,parameters.z_size,parameters.η_size)
# μ_in_debt_S = zeros(parameters.a_ind_zero_dist-1,parameters.z_size,parameters.η_size)
#
# for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size
#       interp_n_S = LinearInterpolation(parameters.a_grid, variables.n_S[:,z_i,η_i])
#       for a_i in 1:(parameters.a_ind_zero_dist-1)
#             debt_to_wage_income_S[a_i,z_i,η_i] = abs(parameters.a_grid_dist[a_i])/(parameters.z_grid[z_i]*parameters.η_grid[η_i]*(interp_n_S(parameters.a_grid_dist[a_i])^parameters.θ))
#             μ_in_debt_S[a_i,z_i,η_i] = μ_S[a_i,z_i,η_i]
#       end
# end
#
# debt_to_wage_income_S_cond = sum(debt_to_wage_income_S.*μ_in_debt_S)/sum(μ_in_debt_S)*100

debt_to_wage_income_S = zeros(parameters.a_ind_zero_dist-1,parameters.z_size,parameters.η_size,parameters.κ_size)
μ_in_debt_S = zeros(parameters.a_ind_zero_dist-1,parameters.z_size,parameters.η_size,parameters.κ_size)

for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size
      for a_i in 1:(parameters.a_ind_zero_dist-1)
            debt_to_wage_income_S[a_i,z_i,η_i,κ_i] = abs(parameters.a_grid_dist[a_i])/(parameters.z_grid[z_i]*parameters.η_grid[η_i]*(variables.n_S[a_i,z_i,η_i,κ_i]^parameters.θ))
            μ_in_debt_S[a_i,z_i,η_i,κ_i] = μ_S[a_i,z_i,η_i,κ_i]
      end
end

debt_to_wage_income_S_cond = sum(debt_to_wage_income_S.*μ_in_debt_S)/sum(μ_in_debt_S)*100

#=============================#
# Average labor supply       #
#=============================#
# Unconditional
# Singles
# n_S_dist = zeros(parameters.a_size_dist,parameters.z_size,parameters.η_size)
#
# for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size
#       interp_n_S = LinearInterpolation(parameters.a_grid, variables.n_S[:,z_i,η_i])
#
#       for a_i in 1:parameters.a_size_dist
#             n_S_dist[a_i,z_i,η_i] = interp_n_S(parameters.a_grid_dist[a_i])
#       end
# end
#
# ave_S_n = sum(n_S_dist.*μ_S)

ave_S_n = sum(n_S.*μ_S)
# Couples
# ave_C_n = sum(n_C_1_i.*μ_C+n_C_2_i.*μ_C)

#=============================#
# Average interest rate       #
#=============================#
# Singles
num_S = 0.0
den_S = 0.0
num_total_S = 0.0
den_total_S = 0.0

# for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size
#
#       interp_q_S = LinearInterpolation(parameters.a_grid, q_S[:,z_i])
#       interp_a_S = LinearInterpolation(parameters.a_grid, a_S[:,z_i,η_i])
#
#       # default_thresh_i = findfirst(a_S[:,z_i,η_i] .!= 0.0)
#       # default_thresh = parameters.a_grid[default_thresh_i]
#
#       default_thresh_i = findfirst(d_S_i[:,z_i,η_i] .≈ 1.0)
#       default_thresh = parameters.a_grid[default_thresh_i]
#
#       for a_i in 1:parameters.a_size_dist
#
#             if parameters.a_grid_dist[a_i] < default_thresh
#                 a_p = 0.0
#             else
#                 a_p = interp_a_S(parameters.a_grid_dist[a_i])
#             end
#
#             num_total_S += interp_q_S(a_p)*μ_S[a_i,z_i,η_i]
#             den_total_S += μ_S[a_i,z_i,η_i]
#
#             if a_p < 0.0
#                   num_S += interp_q_S(a_p)*μ_S[a_i,z_i,η_i]
#                   den_S += μ_S[a_i,z_i,η_i]
#             end
#       end
# end
#
# ave_S_price = num_S/den_S
# ave_S_rate = ((1.0/ave_S_price) - 1.0)*100
#
# ave_S_price_total = num_total_S/den_total_S
# ave_S_rate_total = ((1.0/ave_S_price_total) - 1.0)*100

for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size

      interp_q_S = LinearInterpolation(parameters.a_grid, q_S[:,z_i])

      for a_i in 1:parameters.a_size_dist

            a_p = a_S[a_i,z_i,η_i,κ_i]

            num_total_S += interp_q_S(a_p)*μ_S[a_i,z_i,η_i,κ_i]
            den_total_S += μ_S[a_i,z_i,η_i,κ_i]

            if a_p < 0.0
                  num_S += interp_q_S(a_p)*μ_S[a_i,z_i,η_i,κ_i]
                  den_S += μ_S[a_i,z_i,η_i,κ_i]
            end
      end
end

ave_S_price = num_S/den_S
ave_S_rate = ((1.0/ave_S_price) - 1.0)*100

ave_S_price_total = num_total_S/den_total_S
ave_S_rate_total = ((1.0/ave_S_price_total) - 1.0)*100

# Alternative
# num_S_alt = 0.0
# den_S_alt = 0.0
#
# for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size
#
#       interp_q_S = LinearInterpolation(parameters.a_grid, q_S[:,z_i])
#       interp_a_S = LinearInterpolation(parameters.a_grid, a_S[:,z_i,η_i])
#
#       # default_thresh_i = findfirst(a_S[:,z_i,η_i] .!= 0.0)
#       # default_thresh = parameters.a_grid[default_thresh_i]
#
#       default_thresh_i = findfirst(d_S_i[:,z_i,η_i] .≈ 1.0)
#       default_thresh = parameters.a_grid[default_thresh_i]
#
#       for a_i in 1:parameters.a_size_dist
#
#             if parameters.a_grid_dist[a_i] < default_thresh
#                 a_p = 0.0
#             else
#                 a_p = interp_a_S(parameters.a_grid_dist[a_i])
#             end
#
#             if a_p < 0.0
#                   num_S_alt += ((1.0/interp_q_S(a_p)) - 1.0)*μ_S[a_i,z_i,η_i]
#                   den_S_alt += μ_S[a_i,z_i,η_i]
#             end
#       end
# end
#
# ave_S_rate_alt = num_S_alt/den_S_alt*100

num_S_alt = 0.0
den_S_alt = 0.0

for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size

      interp_q_S = LinearInterpolation(parameters.a_grid, q_S[:,z_i])

      for a_i in 1:parameters.a_size_dist

            a_p = a_S[a_i,z_i,η_i,κ_i]

            if a_p < 0.0
                  num_S_alt += ((1.0/interp_q_S(a_p)) - 1.0)*μ_S[a_i,z_i,η_i,κ_i]
                  den_S_alt += μ_S[a_i,z_i,η_i,κ_i]
            end
      end
end

ave_S_rate_alt = num_S_alt/den_S_alt*100

# Couples
# global num_C = 0.0
# global den_C = 0.0
# global num_total_C = 0.0
# global den_total_C = 0.0
# for κ_2_i in 1:parameters.κ_size
#     for κ_1_i in 1:parameters.κ_size, η_2_i in 1:parameters.η_size, η_1_i in 1:parameters.η_size, z_2_i in 1:parameters.z_size, z_1_i in 1:parameters.z_size, a_i in 1:parameters.a_size, a_p_i in 1:parameters.a_size
#
#         global num_total_C += q_C[a_p_i,z_1_i,z_2_i]*(a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == a_p_i)*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#         global den_total_C += (a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == a_p_i)*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#
#         if a_p_i < parameters.a_ind_zero
#             global num_C += q_C[a_p_i,z_1_i,z_2_i]*(a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == a_p_i)*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#             global den_C += (a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == a_p_i)*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#         end
#     end
# end
# ave_C_price = num_C/den_C
# ave_C_rate = ((1.0/ave_C_price) - 1.0)*100
#
# ave_C_price_total = num_total_C/den_total_C
# ave_C_rate_total = ((1.0/ave_C_price_total) - 1.0)*100

#=============================#
#       Ex-ante Welfare       #
#=============================#
# Singles
welfare_S = 0.0

for η_i in 1:parameters.η_size
    welfare_S += parameters.Γ_η[η_i]*V_S[parameters.a_ind_zero,2,η_i,1]
end

# Couples
# welfare_C = 0.0
#
# for η_1_i in 1:parameters.η_size, η_2_i in 1:parameters.η_size
#     global welfare_C += parameters.Γ_η[η_1_i]*parameters.Γ_η[η_2_i]*V_C[parameters.a_ind_zero,3,3,η_1_i,η_2_i,1,1]
# end

# welfare_S_fix = V_S[parameters.a_ind_zero,2,2]

# welfare_C_fix = V_C[parameters.a_ind_zero,3,3,2,2,1,1]

# Collect moments
# moments = [
#       fraction_S_default
#       fraction_C_default
#       fraction_S_loan_users
#       fraction_C_loan_users
#       fraction_S_loan_users_z[1]
#       fraction_S_loan_users_z[2]
#       fraction_S_loan_users_z[3]
#       fraction_S_loan_users_z[4]
#       fraction_S_loan_users_z[5]
#       fraction_C_loan_users_z[1,1]
#       fraction_C_loan_users_z[2,2]
#       fraction_C_loan_users_z[3,3]
#       fraction_C_loan_users_z[4,4]
#       fraction_C_loan_users_z[5,5]
#       fraction_S_loan_users_η[1]
#       fraction_S_loan_users_η[2]
#       fraction_S_loan_users_η[3]
#       fraction_C_loan_users_η[1,1]
#       fraction_C_loan_users_η[2,2]
#       fraction_C_loan_users_η[3,3]
#       ave_S_n
#       ave_C_n
#       ave_S_rate
#       ave_S_rate_total
#       ave_C_rate
#       ave_C_rate_total
#       welfare_S
#       welfare_C
#       welfare_S_fix
#       welfare_C_fix
#       ]

moments = [
      fraction_S_default
      # fraction_C_default
      fraction_S_loan_users
      # fraction_C_loan_users
      fraction_S_loan_users_z[1]
      fraction_S_loan_users_z[2]
      fraction_S_loan_users_z[3]
      # fraction_C_loan_users_z[1,1]
      # fraction_C_loan_users_z[2,2]
      # fraction_C_loan_users_z[3,3]
      fraction_S_loan_users_η[1]
      fraction_S_loan_users_η[2]
      fraction_S_loan_users_η[3]
      # fraction_C_loan_users_η[1,1]
      # fraction_C_loan_users_η[2,2]
      # fraction_C_loan_users_η[3,3]
      debt_to_wage_income_S_cond
      ave_S_n
      # ave_C_n
      ave_S_rate
      ave_S_rate_total
      # ave_C_rate
      # ave_C_rate_total
      welfare_S
      # welfare_C
      # welfare_S_fix
      # welfare_C_fix
      ]

# @save "moments.jld2" moments
