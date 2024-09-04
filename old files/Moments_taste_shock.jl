# Moments

#===============#
# Model Moments #
#===============#
#================#
# Default #
#================#
# Unconditional
# Singles
fraction_S_default = sum(dropdims(sum(σ_S[1,:,:,:,:,:],dims=1),dims=1).*μ_S)*100
# Couples
fraction_C_default = sum(dropdims(sum(σ_C[1,:,:,:,:,:,:,:,:,:],dims=[1 2]),dims=(1,2)).*μ_C)*100

# Across z
# Singles
fraction_S_default_z = zeros(parameters.z_size)
for i in 1:parameters.z_size
    fraction_S_default_z[i] = sum(dropdims(sum(σ_S[1,:,:,i,:,:],dims=1),dims=1).*μ_S[:,i,:,:])/sum(μ_S[:,i,:,:])*100
end
# Couples
fraction_C_default_z = zeros(parameters.z_size,parameters.z_size)
for i in 1:parameters.z_size, j in 1:parameters.z_size
    fraction_C_default_z[i,j] = sum(dropdims(sum(σ_C[1,:,:,:,i,j,:,:,:,:],dims=[1 2]),dims=(1,2)).*μ_C[:,i,j,:,:,:,:])/sum(μ_C[:,i,j,:,:,:,:])*100
end

fraction_C_default_z_reduced = zeros(parameters.z_size)
for i in 1:parameters.z_size
    fraction_C_default_z_reduced[i] = sum(dropdims(sum(σ_C[1,:,:,:,i,:,:,:,:,:],dims=[1 2]),dims=(1,2)).*μ_C[:,i,:,:,:,:,:])/sum(μ_C[:,i,:,:,:,:,:])*100
end

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
a_i = [(parameters.a_grid .< 0.0)][1]
fraction_S_loan_users = sum(μ_S[a_i,:,:,:])*100
# Couples
fraction_C_loan_users = sum(μ_C[a_i,:,:,:,:,:,:])*100

a_p_i = [(parameters.a_grid .< 0.0)][1]
σ_S_temp = σ_S[2:end,:,:,:,:,:]
fraction_S_loan_users = sum(dropdims(sum(σ_S_temp[a_p_i,:,:,:,:,:],dims=[1 2]),dims=(1,2)).*μ_S)*100

# Across z
# Singles
fraction_S_loan_users_z = zeros(parameters.z_size)
for i in 1:parameters.z_size
    fraction_S_loan_users_z[i] = sum(μ_S[a_i,i,:,:])/sum(μ_S[:,i,:,:])*100
end
# Couples
fraction_C_loan_users_z = zeros(parameters.z_size,parameters.z_size)
for i in 1:parameters.z_size, j in 1:parameters.z_size
    fraction_C_loan_users_z[i,j] = sum(μ_C[a_i,i,j,:,:,:,:])/sum(μ_C[:,i,j,:,:,:,:])*100
end

fraction_C_loan_users_z_reduced = zeros(parameters.z_size)
for i in 1:parameters.z_size
    fraction_C_loan_users_z_reduced[i] = sum(μ_C[a_i,i,:,:,:,:,:])/sum(μ_C[:,i,:,:,:,:,:])*100
end

# Across η
# Singles
fraction_S_loan_users_η = zeros(parameters.η_size)
for i in 1:parameters.η_size
    fraction_S_loan_users_η[i] = sum(μ_S[a_i,:,i,:])/sum(μ_S[:,:,i,:])*100
end
# Couples
fraction_C_loan_users_η = zeros(parameters.η_size, parameters.η_size)
for i in 1:parameters.η_size, j in 1:parameters.η_size
    fraction_C_loan_users_η[i,j] = sum(μ_C[a_i,:,:,i,j,:,:])/sum(μ_C[:,:,:,i,j,:,:])*100
end

#=============================#
# Average labor supply       #
#=============================#
# Unconditional
# Singles
ave_S_n = sum(n_S_i.*μ_S)
# Couples
ave_C_n = sum(n_C_1_i.*μ_C+n_C_2_i.*μ_C)


#=============================#
# Average interest rate       #
#=============================#
# Singles
global num_S = 0.0
global den_S = 0.0
global num_total_S = 0.0
global den_total_S = 0.0
for κ_i in 1:parameters.κ_size
    for η_i in 1:parameters.η_size, z_i in 1:parameters.z_size, a_i in 1:parameters.a_size, a_p_i in 1:parameters.a_size

        global num_total_S += q_S[a_p_i,z_i]*sum(σ_S[a_p_i+1,:,a_i,z_i,η_i,κ_i])*μ_S[a_i,z_i,η_i,κ_i]
        global den_total_S += sum(σ_S[a_p_i+1,:,a_i,z_i,η_i,κ_i])*μ_S[a_i,z_i,η_i,κ_i]

        if a_p_i < parameters.a_ind_zero
            global num_S += q_S[a_p_i,z_i]*sum(σ_S[a_p_i+1,:,a_i,z_i,η_i,κ_i])*μ_S[a_i,z_i,η_i,κ_i]
            global den_S += sum(σ_S[a_p_i+1,:,a_i,z_i,η_i,κ_i])*μ_S[a_i,z_i,η_i,κ_i]
        end
    end
end
ave_S_price = num_S/den_S
ave_S_rate = ((1.0/ave_S_price) - 1.0)*100

ave_S_price_total = num_total_S/den_total_S
ave_S_rate_total = ((1.0/ave_S_price_total) - 1.0)*100

# Couples
global num_C = 0.0
global den_C = 0.0
global num_total_C = 0.0
global den_total_C = 0.0
for κ_2_i in 1:parameters.κ_size
    for κ_1_i in 1:parameters.κ_size, η_2_i in 1:parameters.η_size, η_1_i in 1:parameters.η_size, z_2_i in 1:parameters.z_size, z_1_i in 1:parameters.z_size, a_i in 1:parameters.a_size, a_p_i in 1:parameters.a_size

        global num_total_C += q_C[a_p_i,z_1_i,z_2_i]*sum(σ_C[a_p_i+1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
        global den_total_C += sum(σ_C[a_p_i+1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]

        if a_p_i < parameters.a_ind_zero
            global num_C += q_C[a_p_i,z_1_i,z_2_i]*sum(σ_C[a_p_i+1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            global den_C += sum(σ_C[a_p_i+1,:,:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])*μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
        end
    end
end
ave_C_price = num_C/den_C
ave_C_rate = ((1.0/ave_C_price) - 1.0)*100

ave_C_price_total = num_total_C/den_total_C
ave_C_rate_total = ((1.0/ave_C_price_total) - 1.0)*100

#=============================#
#       Ex-ante Welfare       #
#=============================#
# Singles
welfare_S = 0.0

for η_i in 1:parameters.η_size
    global welfare_S += parameters.Γ_η[η_i]*V_S[parameters.a_ind_zero,3,η_i,1]
end

# Couples
welfare_C = 0.0

for η_1_i in 1:parameters.η_size, η_2_i in 1:parameters.η_size
    global welfare_C += parameters.Γ_η[η_1_i]*parameters.Γ_η[η_2_i]*V_C[parameters.a_ind_zero,3,3,η_1_i,η_2_i,1,1]
end

welfare_S_fix = V_S[parameters.a_ind_zero,3,2,1,1]

welfare_C_fix = V_C[parameters.a_ind_zero,3,3,2,2,1,1]

# Collect moments
moments = [
      fraction_S_default
      fraction_C_default
      fraction_S_loan_users
      fraction_C_loan_users
      fraction_S_loan_users_z[1]
      fraction_S_loan_users_z[2]
      fraction_S_loan_users_z[3]
      fraction_S_loan_users_z[4]
      fraction_S_loan_users_z[5]
      fraction_C_loan_users_z[1,1]
      fraction_C_loan_users_z[2,2]
      fraction_C_loan_users_z[3,3]
      fraction_C_loan_users_z[4,4]
      fraction_C_loan_users_z[5,5]
      fraction_S_loan_users_η[1]
      fraction_S_loan_users_η[2]
      fraction_S_loan_users_η[3]
      fraction_C_loan_users_η[1,1]
      fraction_C_loan_users_η[2,2]
      fraction_C_loan_users_η[3,3]
      ave_S_n
      ave_C_n
      ave_S_rate
      ave_S_rate_total
      ave_C_rate
      ave_C_rate_total
      welfare_S
      welfare_C
      welfare_S_fix
      welfare_C_fix
      ]

# @save "moments.jld2" moments
