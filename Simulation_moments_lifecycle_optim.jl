#===============#
# Check moments #
#===============#
#======================#
# 1) Default rate #
#======================#
# Singles
fraction_S_default_sim = zeros(parameters.lifespan,2)
for age in 1:parameters.lifespan, gender in 1:2
    fraction_S_default_sim[age,gender] = sum(panel_S_d[:,age,gender,:] .== 2)/(num_S*monte_sim)*100
end

fraction_S_default_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    fraction_S_default_sim_ave_age[age] = sum(fraction_S_default_sim[age,:])/2
end

fraction_S_default_sim_ave_gender = zeros(2)
for gender in 1:2
    fraction_S_default_sim_ave_gender[gender] = sum(fraction_S_default_sim[:,gender])/parameters.lifespan
end

fraction_S_default_sim_ave = sum(fraction_S_default_sim_ave_gender)/2

# defaulters_ave_assets_num = 0.0
# defaulters_ave_assets_den = 0.0
# for age in 1:parameters.lifespan, gender in 1:2, hh_i in 1:num_S, monte_i in 1:monte_sim
#     if panel_S_d[hh_i,age,gender,monte_i] .== 2
#         defaulters_ave_assets_num += parameters.a_grid[panel_S_a[hh_i,age,gender,monte_i]]
#         defaulters_ave_assets_den += 1.0
#     end
# end
#
# defaulters_ave_assets = defaulters_ave_assets_num/defaulters_ave_assets_den
#
# defaulters_max_assets = 0.0
# for age in 1:parameters.lifespan, gender in 1:2, hh_i in 1:num_S, monte_i in 1:monte_sim
#     if panel_S_d[hh_i,age,gender,monte_i] .== 2
#         if (parameters.a_grid[panel_S_a[hh_i,age,gender,monte_i]] > defaulters_max_assets)
#             defaulters_max_assets = parameters.a_grid[panel_S_a[hh_i,age,gender,monte_i]]
#         end
#     end
# end

# using Plots
# # Plot across time
# plot(1:period_all, fraction_full_default_sim_all_periods)
#
# # In specific period
# fraction_full_default_sim = sum(panel_default[:,period] .== 1)/num_hh*100

# Divorced
fraction_div_default_sim = zeros(parameters.lifespan,2)
for age in 1:parameters.lifespan, gender in 1:2
    fraction_div_default_sim_num = 0.0
    fraction_div_default_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 1
            fraction_div_default_sim_num += (panel_C_d[hh_i,age,gender,monte_i] == 2)
            fraction_div_default_sim_den += 1.0
        end
    end
    fraction_div_default_sim[age,gender] = fraction_div_default_sim_num/fraction_div_default_sim_den*100
end

fraction_div_default_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    fraction_div_default_sim_ave_age[age] = sum(fraction_div_default_sim[age,:])/2
end

fraction_div_default_sim_ave_gender = zeros(2)
for gender in 1:2
    fraction_div_default_sim_ave_gender[gender] = sum(fraction_div_default_sim[2:end,gender])/(parameters.lifespan-1)
end

fraction_div_default_sim_ave = sum(fraction_div_default_sim_ave_gender)/2

# # Recently Divorced
# fraction_div_rec_default_sim_all_periods = zeros(period_all)
# for i in 1:period_all
#     fraction_div_rec_default_sim_all_periods[i] = (sum(panel_div_1_d[panel_div_1_ind[:,i] .== 1,i] .== 2) + sum(panel_div_2_d[panel_div_2_ind[:,i] .== 1,i] .== 2))/(sum(panel_div_1_ind[:,i] .== 1) + sum(panel_div_2_ind[:,i] .== 1))*100
# end
# fraction_div_rec_default_sim_all_periods_ave = sum(fraction_div_rec_default_sim_all_periods)/period_all
#
# # Non-recent divorced
# fraction_div_non_rec_default_sim_all_periods = zeros(period_all)
# for i in 1:period_all
#     fraction_div_non_rec_default_sim_all_periods[i] = (sum(panel_div_1_d[panel_div_1_ind[:,i] .== 2,i] .== 2) + sum(panel_div_2_d[panel_div_2_ind[:,i] .== 2,i] .== 2))/(sum(panel_div_1_ind[:,i] .== 2) + sum(panel_div_2_ind[:,i] .== 2))*100
# end
# fraction_div_non_rec_default_sim_all_periods_ave = sum(fraction_div_non_rec_default_sim_all_periods)/period_all

# Couples
fraction_C_default_sim = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    fraction_C_default_sim_num = 0.0
    fraction_C_default_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 2
            fraction_C_default_sim_num += (panel_C_d[hh_i,age,1,monte_i] == 2)
            fraction_C_default_sim_den += 1.0
        end
    end
    fraction_C_default_sim[age] = fraction_C_default_sim_num/fraction_C_default_sim_den*100
end

fraction_C_default_sim_ave = sum(fraction_C_default_sim)/parameters.lifespan

#================================#
# 2) Fraction of bank loan users #
#================================#
# Singles
fraction_S_loan_users_sim = zeros(parameters.lifespan-1,2)
for age in 1:(parameters.lifespan-1), gender in 1:2
    fraction_S_loan_users_sim[age,gender] = sum(panel_S_a[:,age+1,gender,:] .< parameters.a_ind_zero)/(num_S*monte_sim)*100
end

fraction_S_loan_users_sim_ave_age = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    fraction_S_loan_users_sim_ave_age[age] = sum(fraction_S_loan_users_sim[age,:])/2
end

# fraction_S_loan_users_sim_ave_age./100

fraction_S_loan_users_sim_ave_gender = zeros(2)
for gender in 1:2
    fraction_S_loan_users_sim_ave_gender[gender] = sum(fraction_S_loan_users_sim[:,gender])/(parameters.lifespan-1)
end

fraction_S_loan_users_sim_ave = sum(fraction_S_loan_users_sim_ave_gender)/2

# Plot across time
# plot(1:period_all, fraction_S_loan_users_sim_all_periods)
#
# # In specific period
# fraction_S_loan_users_sim = sum(panel_S_a[:,period] .< parameters.a_ind_zero)/num_S*100

# Singles, across z
# fraction_S_loan_users_sim_z_num = zeros(3)
# fraction_S_loan_users_sim_z_den = zeros(3)
#
# for i in 1:period_all-1
#     for hh_i in 1:num_S
#         if (shock_S_ρ[hh_i,i+1] == 1)
#             z_i = shock_S_z[hh_i,i]
#             if (panel_S_a[hh_i,i+1] .< parameters.a_ind_zero)
#                 fraction_S_loan_users_sim_z_num[z_i] += 1.0
#             end
#             fraction_S_loan_users_sim_z_den[z_i] += 1.0
#         end
#     end
# end
# fraction_S_loan_users_sim_z_ave = fraction_S_loan_users_sim_z_num./fraction_S_loan_users_sim_z_den.*100

# Among Divorced
fraction_div_loan_users_sim = zeros(parameters.lifespan-1,2)
for age in 1:(parameters.lifespan-1), gender in 1:2
    fraction_div_loan_users_sim_num = 0.0
    fraction_div_loan_users_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 1
            fraction_div_loan_users_sim_num += (panel_C_a[hh_i,age+1,gender,monte_i] < parameters.a_ind_zero)
            fraction_div_loan_users_sim_den += 1.0
        end
    end
    fraction_div_loan_users_sim[age,gender] = fraction_div_loan_users_sim_num/fraction_div_loan_users_sim_den*100
end

fraction_div_loan_users_sim_ave_age = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    fraction_div_loan_users_sim_ave_age[age] = sum(fraction_div_loan_users_sim[age,:])/2
end

# fraction_div_loan_users_sim_ave_age./100

fraction_div_loan_users_sim_ave_gender = zeros(2)
for gender in 1:2
    fraction_div_loan_users_sim_ave_gender[gender] = sum(fraction_div_loan_users_sim[2:end,gender])/(parameters.lifespan-2)
end

fraction_div_loan_users_sim_ave = sum(fraction_div_loan_users_sim_ave_gender)/2

# # Divorced, across z
# fraction_div_loan_users_sim_z_num = zeros(3)
# fraction_div_loan_users_sim_z_den = zeros(3)
#
# for i in 1:period_all-1
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .!= 0) && (panel_div_1_ind[hh_i,i+1] .== 2)
#             z_i = shock_div_1_z[hh_i,i]
#             if (panel_div_1_a[hh_i,i+1] .< parameters.a_ind_zero)
#                 fraction_div_loan_users_sim_z_num[z_i] += 1.0
#             end
#             fraction_div_loan_users_sim_z_den[z_i] += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .!= 0) && (panel_div_2_ind[hh_i,i+1] .== 2)
#             z_i = shock_div_2_z[hh_i,i]
#             if (panel_div_2_a[hh_i,i+1] .< parameters.a_ind_zero)
#                 fraction_div_loan_users_sim_z_num[z_i] += 1.0
#             end
#             fraction_div_loan_users_sim_z_den[z_i] += 1.0
#         end
#     end
# end
# fraction_div_loan_users_sim_z_ave = fraction_div_loan_users_sim_z_num./fraction_div_loan_users_sim_z_den.*100
#
# # Among Recently Divorced
# fraction_div_rec_loan_users_sim_all_periods = zeros(period_all)
# for i in 1:period_all
#     fraction_div_rec_loan_users_sim_all_periods[i] = (sum(panel_div_1_a[panel_div_1_ind[:,i] .== 1,i] .< parameters.a_ind_zero) + sum(panel_div_2_a[panel_div_2_ind[:,i] .== 1,i] .< parameters.a_ind_zero))/(sum(panel_div_1_ind[:,i] .== 1) + sum(panel_div_2_ind[:,i] .== 1))*100
# end
# fraction_div_rec_loan_users_sim_all_periods_ave = sum(fraction_div_rec_loan_users_sim_all_periods)/period_all
#
# # Among Non-recent divorced
# fraction_div_non_rec_loan_users_sim_all_periods = zeros(period_all)
# for i in 1:period_all
#     fraction_div_non_rec_loan_users_sim_all_periods[i] = (sum(panel_div_1_a[panel_div_1_ind[:,i] .== 2,i] .< parameters.a_ind_zero) + sum(panel_div_2_a[panel_div_2_ind[:,i] .== 2,i] .< parameters.a_ind_zero))/(sum(panel_div_1_ind[:,i] .== 2) + sum(panel_div_2_ind[:,i] .== 2))*100
# end
# fraction_div_non_rec_loan_users_sim_all_periods_ave = sum(fraction_div_non_rec_loan_users_sim_all_periods)/period_all

# Couples
fraction_C_loan_users_sim = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    fraction_C_loan_users_sim_num = 0.0
    fraction_C_loan_users_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (shock_C_div[hh_i,age,monte_i] == 2) && (shock_C_div[hh_i,age+1,monte_i] == 2)
            fraction_C_loan_users_sim_num += (panel_C_a[hh_i,age+1,1,monte_i] < parameters.a_ind_zero)
            fraction_C_loan_users_sim_den += 1.0
        end
    end
    fraction_C_loan_users_sim[age] = fraction_C_loan_users_sim_num/fraction_C_loan_users_sim_den*100
end

# fraction_C_loan_users_sim./100

fraction_C_loan_users_sim_ave = sum(fraction_C_loan_users_sim)/(parameters.lifespan-1)

# # Couples, across z
# fraction_C_loan_users_sim_z_num = zeros(3,3)
# fraction_C_loan_users_sim_z_den = zeros(3,3)
#
# for i in 1:period_all-1
#     for hh_i in 1:num_C
#         if (shock_C_ρ[hh_i,i+1] == 1) && (shock_C_div[hh_i,i+1] == 2)
#             z_1_i = shock_C_z_1[hh_i,i]
#             z_2_i = shock_C_z_2[hh_i,i]
#             if (panel_C_a[hh_i,i+1] .< parameters.a_ind_zero)
#                 fraction_C_loan_users_sim_z_num[z_1_i,z_2_i] += 1.0
#             end
#             fraction_C_loan_users_sim_z_den[z_1_i,z_2_i] += 1.0
#         end
#     end
# end
# fraction_C_loan_users_sim_z_ave = fraction_C_loan_users_sim_z_num./fraction_C_loan_users_sim_z_den.*100

#====================#
# Income  #
#====================#
# Singles
income_S_sim = zeros(parameters.lifespan,2)

for age in 1:(parameters.lifespan), gender in 1:2
    temp = 0.0
    for hh_i in 1:num_S, monte_i in 1:monte_sim
            temp += parameters.z_grid[shock_S_z[hh_i,age,gender,monte_i],gender]*parameters.h_grid[age]*parameters.n_grid[panel_S_n[hh_i,age,gender,monte_i]]
    end

    income_S_sim[age,gender] = temp/(num_S*monte_sim)
end

income_S_sim_ave_age = zeros(parameters.lifespan)

for age in 1:parameters.lifespan
    income_S_sim_ave_age[age] = sum(income_S_sim[age,:])/2
end

# Divorced
income_div_sim = zeros(parameters.lifespan,2)

for age in 1:(parameters.lifespan), gender in 1:2
    temp = 0.0
    den = 0.0
    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (shock_C_div[hh_i,age,monte_i] == 1) # Divorced
            temp += parameters.z_grid[shock_C_z[hh_i,age,gender,monte_i],gender]*parameters.h_grid[age]*parameters.n_grid[panel_C_n[hh_i,age,gender,monte_i]]

            den += 1.0
        end
    end

    income_div_sim[age,gender] = temp / den
end

income_div_sim_ave_age = zeros(parameters.lifespan)

for age in 1:parameters.lifespan
    income_div_sim_ave_age[age] = sum(income_div_sim[age,:])/2
end

# Couples
income_C_sim = zeros(parameters.lifespan)

for age in 1:parameters.lifespan
    temp = 0.0
    den = 0.0
    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (shock_C_div[hh_i,age,monte_i] == 2) # Married
            temp += parameters.z_grid[shock_C_z[hh_i,age,1,monte_i],1]*parameters.h_grid[age]*parameters.n_grid[panel_C_n[hh_i,age,1,monte_i]] + parameters.z_grid[shock_C_z[hh_i,age,2,monte_i],2]*parameters.h_grid[age]*parameters.n_grid[panel_C_n[hh_i,age,2,monte_i]]

            den += 1.0
        end
    end
    income_C_sim[age] = temp / (2*den)
end


#====================#
# 3) Debt-to-income  #
#====================#

# Ratio first, then average
# Singles
debt_to_inc_S_sim_ratio_first = zeros(parameters.lifespan-1,2)

for age in 1:(parameters.lifespan-1), gender in 1:2
    temp_num = 0.0
    temp_den = 0.0
    for hh_i in 1:num_S, monte_i in 1:monte_sim
        income_temp = parameters.z_grid[shock_S_z[hh_i,age,gender,monte_i],gender]*parameters.h_grid[age]*parameters.n_grid[panel_S_n[hh_i,age,gender,monte_i]]

        if (panel_S_a[hh_i,age+1,gender,monte_i] .< parameters.a_ind_zero) && (income_temp != 0.0)
            temp_num += (abs(parameters.a_grid[panel_S_a[hh_i,age+1,gender,monte_i]])/income_temp)
            temp_den += 1.0
        end
    end
    debt_to_inc_S_sim_ratio_first[age,gender] = temp_num/temp_den
end

# Average first, then ratio
# Singles
debt_S_sim = zeros(parameters.lifespan-1,2)
income_S_sim = zeros(parameters.lifespan-1,2)

for age in 1:(parameters.lifespan-1), gender in 1:2
    for hh_i in 1:num_S, monte_i in 1:monte_sim
        if (panel_S_a[hh_i,age+1,gender,monte_i] .< parameters.a_ind_zero)
            debt_S_sim[age,gender] += abs(parameters.a_grid[panel_S_a[hh_i,age+1,gender,monte_i]])

            income_S_sim[age,gender] += parameters.z_grid[shock_S_z[hh_i,age,gender,monte_i],gender]*parameters.h_grid[age]*parameters.n_grid[panel_S_n[hh_i,age,gender,monte_i]]
        end
    end
end

debt_to_inc_S_sim = (debt_S_sim ./ income_S_sim)*100

debt_to_inc_S_sim_ave_age = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    temp_num = 0.0
    temp_den = 0.0
    for gender in 1:2
        if (isnan(debt_to_inc_S_sim[age,gender]) == false) && (isinf(debt_to_inc_S_sim[age,gender]) == false)
            temp_num += debt_to_inc_S_sim[age,gender]
            temp_den += 1.0
        end
    end
    debt_to_inc_S_sim_ave_age[age] = temp_num/temp_den
end

debt_to_inc_S_sim_ave_age/100

debt_to_inc_S_sim_ave_gender = zeros(2)
for gender in 1:2
    temp_num = 0.0
    temp_den = 0.0
    for age in 1:(parameters.lifespan-1)
        if (isnan(debt_to_inc_S_sim[age,gender]) == false) && (isinf(debt_to_inc_S_sim[age,gender]) == false)
            temp_num += debt_to_inc_S_sim[age,gender]
            temp_den += 1.0
        end
    end
    debt_to_inc_S_sim_ave_gender[gender] = temp_num/temp_den
end

temp_num = 0.0
temp_den = 0.0
for gender in 1:2
    if (isnan(debt_to_inc_S_sim_ave_gender[gender]) == false) && (isinf(debt_to_inc_S_sim_ave_gender[gender]) == false)
        temp_num += debt_to_inc_S_sim_ave_gender[gender]
        temp_den += 1.0
    end
end

debt_to_inc_S_sim_ave = temp_num/temp_den

# Among Divorced
debt_div_sim = zeros(parameters.lifespan-1,2)
income_div_sim = zeros(parameters.lifespan-1,2)

for age in 1:(parameters.lifespan-1), gender in 1:2
    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (panel_C_a[hh_i,age+1,gender,monte_i] .< parameters.a_ind_zero) && (shock_C_div[hh_i,age,monte_i] == 1)
            debt_div_sim[age,gender] += abs(parameters.a_grid[panel_C_a[hh_i,age+1,gender,monte_i]])

            income_div_sim[age,gender] += parameters.z_grid[shock_C_z[hh_i,age,gender,monte_i],gender]*parameters.h_grid[age]*parameters.n_grid[panel_C_n[hh_i,age,gender,monte_i]]
        end
    end
end

debt_to_inc_div_sim = (debt_div_sim ./ income_div_sim)*100

debt_to_inc_div_sim_ave_age = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    temp_num = 0.0
    temp_den = 0.0
    for gender in 1:2
        if (isnan(debt_to_inc_div_sim[age,gender]) == false) && (isinf(debt_to_inc_div_sim[age,gender]) == false)
            temp_num += debt_to_inc_div_sim[age,gender]
            temp_den += 1.0
        end
    end
    debt_to_inc_div_sim_ave_age[age] = temp_num/temp_den
end

debt_to_inc_div_sim_ave_age/100

debt_to_inc_div_sim_ave_gender = zeros(2)
for gender in 1:2
    temp_num = 0.0
    temp_den = 0.0
    for age in 1:(parameters.lifespan-1)
        if (isnan(debt_to_inc_div_sim[age,gender]) == false) && (isinf(debt_to_inc_div_sim[age,gender]) == false)
            temp_num += debt_to_inc_div_sim[age,gender]
            temp_den += 1.0
        end
    end
    debt_to_inc_div_sim_ave_gender[gender] = temp_num/temp_den
end

temp_num = 0.0
temp_den = 0.0
for gender in 1:2
    if (isnan(debt_to_inc_div_sim_ave_gender[gender]) == false) && (isinf(debt_to_inc_div_sim_ave_gender[gender]) == false)
        temp_num += debt_to_inc_div_sim_ave_gender[gender]
        temp_den += 1.0
    end
end

debt_to_inc_div_sim_ave = temp_num/temp_den

# # Among recent Divorced
# debt_div_rec_sim_all_periods = 0.0
# income_div_rec_sim_all_periods = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_1_ind[hh_i,i] .== 1)
#             debt_div_rec_sim_all_periods += abs(parameters.a_grid[panel_div_1_a[hh_i,i]])
#             income_div_rec_sim_all_periods += parameters.z_grid[shock_div_1_z[hh_i,i]]*parameters.η_grid[shock_div_1_η[hh_i,i]]*(parameters.n_grid[panel_div_1_n[hh_i,i]]^(parameters.θ))
#         end
#
#         if (panel_div_2_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_2_ind[hh_i,i] .== 1)
#             debt_div_rec_sim_all_periods += abs(parameters.a_grid[panel_div_2_a[hh_i,i]])
#             income_div_rec_sim_all_periods += parameters.z_grid[shock_div_2_z[hh_i,i]]*parameters.η_grid[shock_div_2_η[hh_i,i]]*(parameters.n_grid[panel_div_2_n[hh_i,i]]^(parameters.θ))
#         end
#     end
# end
#
# debt_to_inc_div_rec_sim_all_periods = (debt_div_rec_sim_all_periods/income_div_rec_sim_all_periods)*100
#
# # Among non-recent Divorced
# debt_div_non_rec_sim_all_periods = 0.0
# income_div_non_rec_sim_all_periods = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_1_ind[hh_i,i] .== 2)
#             debt_div_non_rec_sim_all_periods += abs(parameters.a_grid[panel_div_1_a[hh_i,i]])
#             income_div_non_rec_sim_all_periods += parameters.z_grid[shock_div_1_z[hh_i,i]]*parameters.η_grid[shock_div_1_η[hh_i,i]]*(parameters.n_grid[panel_div_1_n[hh_i,i]]^(parameters.θ))
#         end
#
#         if (panel_div_2_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_2_ind[hh_i,i] .== 2)
#             debt_div_non_rec_sim_all_periods += abs(parameters.a_grid[panel_div_2_a[hh_i,i]])
#             income_div_non_rec_sim_all_periods += parameters.z_grid[shock_div_2_z[hh_i,i]]*parameters.η_grid[shock_div_2_η[hh_i,i]]*(parameters.n_grid[panel_div_2_n[hh_i,i]]^(parameters.θ))
#         end
#     end
# end
#
# debt_to_inc_div_non_rec_sim_all_periods = (debt_div_non_rec_sim_all_periods/income_div_non_rec_sim_all_periods)*100

# Couples
debt_C_sim = zeros(parameters.lifespan-1)
income_C_sim = zeros(parameters.lifespan-1)

for age in 1:(parameters.lifespan-1)
    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (panel_C_a[hh_i,age+1,1,monte_i] .< parameters.a_ind_zero) && (shock_C_div[hh_i,age+1,monte_i] == 2)
            debt_C_sim[age] += abs(parameters.a_grid[panel_C_a[hh_i,age+1,1,monte_i]])
            income_C_sim[age] += parameters.z_grid[shock_C_z[hh_i,age,1,monte_i],1]*parameters.h_grid[age]*parameters.n_grid[panel_C_n[hh_i,age,1,monte_i]] + parameters.z_grid[shock_C_z[hh_i,age,2,monte_i],2]*parameters.h_grid[age]*parameters.n_grid[panel_C_n[hh_i,age,2,monte_i]]
        end
    end
end

debt_to_inc_C_sim = (debt_C_sim ./ income_C_sim)*100

temp_num = 0.0
temp_den = 0.0

for age in 1:(parameters.lifespan-1)
    if (isnan(debt_to_inc_C_sim[age]) == false) && (isinf(debt_to_inc_C_sim[age]) == false)
        temp_num += debt_to_inc_C_sim[age]
        temp_den += 1.0
    end
end

debt_to_inc_C_sim_ave = temp_num/temp_den
# debt_to_inc_C_sim_ave = sum(debt_to_inc_C_sim)/(parameters.lifespan-1)

#============================#
# 4) Average loan price #
#============================#

# Singles
ave_bank_rate_S_sim = zeros(parameters.lifespan-1,2)

for age in 1:(parameters.lifespan-1), gender in 1:2
    ave_bank_price_S_sim_num = 0.0
    ave_bank_price_S_sim_den = 0.0

    for hh_i in 1:num_S, monte_i in 1:monte_sim
        if (panel_S_a[hh_i,age+1,gender,monte_i] .< parameters.a_ind_zero)
            ave_bank_price_S_sim_num += q_S[panel_S_a[hh_i,age+1,gender,monte_i],shock_S_z[hh_i,age,gender,monte_i],age,gender]
            ave_bank_price_S_sim_den += 1.0
        end
    end
    ave_bank_price_S_sim = ave_bank_price_S_sim_num/ave_bank_price_S_sim_den
    ave_bank_rate_S_sim[age,gender] = (1.0/ave_bank_price_S_sim) - 1.0
end

ave_bank_rate_S_sim_ave_age = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    temp_num = 0.0
    temp_den = 0.0
    for gender in 1:2
        if (isnan(ave_bank_rate_S_sim[age,gender]) == false)
            temp_num += ave_bank_rate_S_sim[age,gender]
            temp_den += 1.0
        end
    end
    ave_bank_rate_S_sim_ave_age[age] = temp_num/temp_den
end

# ave_bank_rate_S_sim_ave_age*100

ave_bank_rate_S_sim_ave_gender = zeros(2)
for gender in 1:2
    temp_num = 0.0
    temp_den = 0.0
    for age in 1:parameters.lifespan-1
        if (isnan(ave_bank_rate_S_sim[age,gender]) == false)
            temp_num += ave_bank_rate_S_sim[age,gender]
            temp_den += 1.0
        end
    end
    ave_bank_rate_S_sim_ave_gender[gender] = temp_num/temp_den
end

temp_num = 0.0
temp_den = 0.0
for gender in 1:2
    if (isnan(ave_bank_rate_S_sim_ave_gender[gender]) == false)
        temp_num += ave_bank_rate_S_sim_ave_gender[gender]
        temp_den += 1.0
    end
end
ave_bank_rate_S_sim_ave = temp_num/temp_den

# Among Divorced
ave_bank_rate_div_sim = zeros(parameters.lifespan-1,2)

for age in 1:(parameters.lifespan-1), gender in 1:2
    ave_bank_price_div_sim_num = 0.0
    ave_bank_price_div_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (panel_C_a[hh_i,age+1,gender,monte_i] .< parameters.a_ind_zero) && (shock_C_div[hh_i,age,monte_i] == 1)
            ave_bank_price_div_sim_num += q_S[panel_C_a[hh_i,age+1,gender,monte_i],shock_C_z[hh_i,age,gender,monte_i],age,gender]
            ave_bank_price_div_sim_den += 1.0
        end
    end
    ave_bank_price_div_sim = ave_bank_price_div_sim_num/ave_bank_price_div_sim_den
    ave_bank_rate_div_sim[age,gender] = (1.0/ave_bank_price_div_sim) - 1.0
end

ave_bank_rate_div_sim_ave_age = zeros(parameters.lifespan-1)
for age in 1:(parameters.lifespan-1)
    temp_num = 0.0
    temp_den = 0.0
    for gender in 1:2
        if (isnan(ave_bank_rate_div_sim[age,gender]) == false)
            temp_num += ave_bank_rate_div_sim[age,gender]
            temp_den += 1.0
        end
    end
    ave_bank_rate_div_sim_ave_age[age] = temp_num/temp_den
    # ave_bank_rate_div_sim_ave[gender] = sum(ave_bank_rate_div_sim[2:end,gender])/(parameters.lifespan-1)
end

# ave_bank_rate_div_sim_ave_age*100

ave_bank_rate_div_sim_ave_gender = zeros(2)
for gender in 1:2
    temp_num = 0.0
    temp_den = 0.0
    for age in 1:parameters.lifespan-1
        if (isnan(ave_bank_rate_div_sim[age,gender]) == false)
            temp_num += ave_bank_rate_div_sim[age,gender]
            temp_den += 1.0
        end
    end
    ave_bank_rate_div_sim_ave_gender[gender] = temp_num/temp_den
    # ave_bank_rate_div_sim_ave[gender] = sum(ave_bank_rate_div_sim[2:end,gender])/(parameters.lifespan-1)
end

temp_num = 0.0
temp_den = 0.0
for gender in 1:2
    if (isnan(ave_bank_rate_div_sim_ave_gender[gender]) == false)
        temp_num += ave_bank_rate_div_sim_ave_gender[gender]
        temp_den += 1.0
    end
end
ave_bank_rate_div_sim_ave = temp_num/temp_den

# # Among Recently Divorced
# ave_bank_rate_div_rec_sim_all_periods = zeros(period_all-1)
# for i in 2:period_all
#     ave_bank_price_div_rec_sim_num = 0.0
#     ave_bank_price_div_rec_sim_den = 0.0
#
#     for hh_i in 1:num_div
#
#         if (panel_div_1_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_1_ind[hh_i,i-1] .== 1) && (panel_div_1_ind[hh_i,i] .== 2)
#             ave_bank_price_div_rec_sim_num += q_S[panel_div_1_a[hh_i,i],shock_div_1_z[hh_i,i-1]]
#             ave_bank_price_div_rec_sim_den += 1.0
#         end
#
#         if (panel_div_2_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_2_ind[hh_i,i-1] .== 1) && (panel_div_2_ind[hh_i,i] .== 2)
#             ave_bank_price_div_rec_sim_num += q_S[panel_div_2_a[hh_i,i],shock_div_2_z[hh_i,i-1]]
#             ave_bank_price_div_rec_sim_den += 1.0
#         end
#     end
#
#     ave_bank_price_div_rec_sim = ave_bank_price_div_rec_sim_num/ave_bank_price_div_rec_sim_den
#     ave_bank_rate_div_rec_sim = (1.0/ave_bank_price_div_rec_sim) - 1.0
#
#     ave_bank_rate_div_rec_sim_all_periods[i-1] = ave_bank_rate_div_rec_sim
# end
#
# ave_bank_rate_div_rec_sim_all_periods_ave = sum(ave_bank_rate_div_rec_sim_all_periods)/(period_all-1)
#
# # Among non-recent Divorced
# ave_bank_rate_div_non_rec_sim_all_periods = zeros(period_all-1)
# for i in 2:period_all
#     ave_bank_price_div_non_rec_sim_num = 0.0
#     ave_bank_price_div_non_rec_sim_den = 0.0
#
#     for hh_i in 1:num_div
#
#         if (panel_div_1_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_1_ind[hh_i,i-1] .== 2) && (panel_div_1_ind[hh_i,i] .== 2)
#             ave_bank_price_div_non_rec_sim_num += q_S[panel_div_1_a[hh_i,i],shock_div_1_z[hh_i,i-1]]
#             ave_bank_price_div_non_rec_sim_den += 1.0
#         end
#
#         if (panel_div_2_a[hh_i,i] .< parameters.a_ind_zero) && (panel_div_2_ind[hh_i,i-1] .== 2) && (panel_div_2_ind[hh_i,i] .== 2)
#             ave_bank_price_div_non_rec_sim_num += q_S[panel_div_2_a[hh_i,i],shock_div_2_z[hh_i,i-1]]
#             ave_bank_price_div_non_rec_sim_den += 1.0
#         end
#     end
#
#     ave_bank_price_div_non_rec_sim = ave_bank_price_div_non_rec_sim_num/ave_bank_price_div_non_rec_sim_den
#     ave_bank_rate_div_non_rec_sim = (1.0/ave_bank_price_div_non_rec_sim) - 1.0
#
#     ave_bank_rate_div_non_rec_sim_all_periods[i-1] = ave_bank_rate_div_non_rec_sim
# end
#
# ave_bank_rate_div_non_rec_sim_all_periods_ave = sum(ave_bank_rate_div_non_rec_sim_all_periods)/(period_all-1)

# Couples
ave_bank_rate_C_sim = zeros(parameters.lifespan-1)

for age in 1:(parameters.lifespan-1)
    ave_bank_price_C_sim_num = 0.0
    ave_bank_price_C_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if (panel_C_a[hh_i,age+1,1,monte_i] .< parameters.a_ind_zero) && (shock_C_div[hh_i,age,monte_i] == 2) && (shock_C_div[hh_i,age+1,monte_i] == 2)
            ave_bank_price_C_sim_num += q_C[panel_C_a[hh_i,age+1,1,monte_i],shock_C_z[hh_i,age,1,monte_i],shock_C_z[hh_i,age,2,monte_i],age]
            ave_bank_price_C_sim_den += 1.0
        end
    end
    ave_bank_price_C_sim = ave_bank_price_C_sim_num/ave_bank_price_C_sim_den
    ave_bank_rate_C_sim[age] = (1.0/ave_bank_price_C_sim) - 1.0
end

# ave_bank_rate_C_sim*100

temp_num = 0.0
temp_den = 0.0
for age in 1:(parameters.lifespan-1)
    if (isnan(ave_bank_rate_C_sim[age]) == false)
        temp_num += ave_bank_rate_C_sim[age]
        temp_den += 1.0
    end
end
ave_bank_rate_C_sim_ave = temp_num/temp_den
# ave_bank_rate_C_sim_ave = sum(ave_bank_rate_C_sim)/(parameters.lifespan-1)


# For specific period
# ave_bank_price_sim_num = 0.0
# ave_bank_price_sim_den = 0.0
#
# for hh_i in 1:num_hh
#     a_p_i, b_p_i = parameters.asset_ind[panel_asset[hh_i,period],:]
#
#     if a_p_i < parameters.a_ind_zero
#         #=
#         if panel_default[hh_i,period-1] == 1
#             action_a_i = 1
#         elseif panel_default[hh_i,period-1] == 2
#             action_a_i = 1 + a_p_i
#         else
#             action_a_i = 1 + parameters.a_size + a_p_i
#         end
#         =#
#
#         action_a_i = 1  + a_p_i
#
#         a_i, b_i = parameters.asset_ind[panel_asset[hh_i,period-1],:]
#
#         ave_bank_price_sim_num += q_a[action_a_i-1,shock_e[hh_i,period-1],a_i,panel_score[hh_i,period-1]]
#         ave_bank_price_sim_den += 1.0
#     end
# end
#
# ave_bank_price_sim = ave_bank_price_sim_num/ave_bank_price_sim_den
# ave_bank_rate_sim = (1.0/ave_bank_price_sim) - 1.0

#=========================#
# 5) Average Consumption  #
#=========================#

# Singles
consumption_S_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    consumption_S_sim[age,gender] = sum(panel_S_c[:,age,gender,:])/(num_S*monte_sim)
end

consumption_S_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    consumption_S_sim_ave_age[age] = sum(consumption_S_sim[age,:])/2
end

consumption_S_sim_ave_gender = zeros(2)
for gender in 1:2
    consumption_S_sim_ave_gender[gender] = sum(consumption_S_sim[:,gender])/parameters.lifespan
end

consumption_S_sim_ave = sum(consumption_S_sim_ave_gender)/2

# Among divorced
consumption_div_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    consumption_div_sim_num = 0.0
    consumption_div_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 1
            consumption_div_sim_num += panel_C_c[hh_i,age,gender,monte_i]
            consumption_div_sim_den += 1.0
        end
    end

    consumption_div_sim[age,gender] = consumption_div_sim_num/consumption_div_sim_den
end

consumption_div_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    consumption_div_sim_ave_age[age] = sum(consumption_div_sim[age,:])/2
end

consumption_div_sim_ave_gender = zeros(2)
for gender in 1:2
    consumption_div_sim_ave_gender[gender] = sum(consumption_div_sim[2:end,gender])/(parameters.lifespan-1)
end

consumption_div_sim_ave = sum(consumption_div_sim_ave_gender)/2

# # Among recent divorced
# consumption_div_rec_sim_all_periods_num = 0.0
# consumption_div_rec_sim_all_periods_den = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .== 1)
#             consumption_div_rec_sim_all_periods_num += panel_div_1_c[hh_i,i]
#             consumption_div_rec_sim_all_periods_den += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .== 1)
#             consumption_div_rec_sim_all_periods_num += panel_div_2_c[hh_i,i]
#             consumption_div_rec_sim_all_periods_den += 1.0
#         end
#     end
# end
#
# consumption_div_rec_sim_all_periods_ave = consumption_div_rec_sim_all_periods_num / consumption_div_rec_sim_all_periods_den
#
# # Among non-recent divorced
# consumption_div_non_rec_sim_all_periods_num = 0.0
# consumption_div_non_rec_sim_all_periods_den = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .== 2)
#             consumption_div_non_rec_sim_all_periods_num += panel_div_1_c[hh_i,i]
#             consumption_div_non_rec_sim_all_periods_den += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .== 2)
#             consumption_div_non_rec_sim_all_periods_num += panel_div_2_c[hh_i,i]
#             consumption_div_non_rec_sim_all_periods_den += 1.0
#         end
#     end
# end
#
# consumption_div_non_rec_sim_all_periods_ave = consumption_div_non_rec_sim_all_periods_num / consumption_div_non_rec_sim_all_periods_den

# Couples
consumption_C_sim = zeros(parameters.lifespan)

for age in 1:parameters.lifespan
    consumption_C_sim_num = 0.0
    consumption_C_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 2
            consumption_C_sim_num += panel_C_c[hh_i,age,1,monte_i]
            consumption_C_sim_den += 1.0
        end
    end

    consumption_C_sim[age] = consumption_C_sim_num/consumption_C_sim_den
end

consumption_C_sim_ave = sum(consumption_C_sim)/parameters.lifespan

#===================#
# 6) Average Labor  #
#===================#
# Singles
labor_S_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    labor_S_sim[age,gender] = sum(parameters.n_grid[panel_S_n[:,age,gender,:]])/(num_S*monte_sim)
end

labor_S_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    labor_S_sim_ave_age[age] = sum(labor_S_sim[age,:])/2
end

labor_S_sim_ave_gender = zeros(2)
for gender in 1:2
    labor_S_sim_ave_gender[gender] = sum(labor_S_sim[:,gender])/parameters.lifespan
end

labor_S_sim_ave = sum(labor_S_sim_ave_gender)/2

# Average hours of single females at age 50
hours_single_women_50 = labor_S_sim[11,1]*40*52

# Average hours of single males at age 50
hours_single_men_50 = labor_S_sim[11,2]*40*52

hours_single_50 = (hours_single_women_50 + hours_single_men_50)/2

# Among Divorced
labor_div_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    labor_div_sim_num = 0.0
    labor_div_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 1
            labor_div_sim_num += parameters.n_grid[panel_C_n[hh_i,age,gender,monte_i]]
            labor_div_sim_den += 1.0
        end
    end
    labor_div_sim[age,gender] = labor_div_sim_num/labor_div_sim_den
end

labor_div_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    labor_div_sim_ave_age[age] = sum(labor_div_sim[age,:])/2
end

labor_div_sim_ave_gender = zeros(2)
for gender in 1:2
    labor_div_sim_ave_gender[gender] = sum(labor_div_sim[2:end,gender])/(parameters.lifespan-1)
end

labor_div_sim_ave = sum(labor_div_sim_ave_gender)/2

# # Among recent Divorced
# labor_div_rec_sim_all_periods_num = 0.0
# labor_div_rec_sim_all_periods_den = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .== 1)
#             labor_div_rec_sim_all_periods_num += parameters.n_grid[panel_div_1_n[hh_i,i]]
#             labor_div_rec_sim_all_periods_den += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .== 1)
#             labor_div_rec_sim_all_periods_num += parameters.n_grid[panel_div_2_n[hh_i,i]]
#             labor_div_rec_sim_all_periods_den += 1.0
#         end
#     end
# end
#
# labor_div_rec_sim_all_periods_ave = labor_div_rec_sim_all_periods_num / labor_div_rec_sim_all_periods_den
#
# # Among non-recent Divorced
# labor_div_non_rec_sim_all_periods_num = 0.0
# labor_div_non_rec_sim_all_periods_den = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .== 2)
#             labor_div_non_rec_sim_all_periods_num += parameters.n_grid[panel_div_1_n[hh_i,i]]
#             labor_div_non_rec_sim_all_periods_den += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .== 2)
#             labor_div_non_rec_sim_all_periods_num += parameters.n_grid[panel_div_2_n[hh_i,i]]
#             labor_div_non_rec_sim_all_periods_den += 1.0
#         end
#     end
# end
#
# labor_div_non_rec_sim_all_periods_ave = labor_div_non_rec_sim_all_periods_num / labor_div_non_rec_sim_all_periods_den

# Couples
labor_C_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    labor_C_sim_num = 0.0
    labor_C_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 2
            labor_C_sim_num += parameters.n_grid[panel_C_n[hh_i,age,gender,monte_i]]
            labor_C_sim_den += 1.0
        end
    end
    labor_C_sim[age,gender] = labor_C_sim_num/labor_C_sim_den
end

labor_C_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    labor_C_sim_ave_age[age] = sum(labor_C_sim[age,:])/2
end

labor_C_sim_ave_gender = zeros(2)
for gender in 1:2
    labor_C_sim_ave_gender[gender] = sum(labor_C_sim[:,gender])/parameters.lifespan
end

labor_C_sim_ave = sum(labor_C_sim_ave_gender)/2

#===================#
# 6) Average Assets #
#===================#
# Singles
assets_S_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    assets_S_sim[age,gender] = sum(parameters.a_grid[panel_S_a[:,age,gender,:]])/(num_S*monte_sim)
end

assets_S_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    assets_S_sim_ave_age[age] = sum(assets_S_sim[age,:])/2
end

assets_S_sim_ave_gender = zeros(2)
for gender in 1:2
    assets_S_sim_ave_gender[gender] = sum(assets_S_sim[:,gender])/parameters.lifespan
end

assets_S_sim_ave = sum(assets_S_sim_ave_gender)/2

# # Among Divorced
assets_div_sim = zeros(parameters.lifespan,2)

for age in 1:parameters.lifespan, gender in 1:2
    assets_div_sim_num = 0.0
    assets_div_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 1
            assets_div_sim_num += parameters.a_grid[panel_C_a[hh_i,age,gender,monte_i]]
            assets_div_sim_den += 1.0
        end
    end
    assets_div_sim[age,gender] = assets_div_sim_num/assets_div_sim_den
end

assets_div_sim_ave_age = zeros(parameters.lifespan)
for age in 1:parameters.lifespan
    assets_div_sim_ave_age[age] = sum(assets_div_sim[age,:])/2
end

assets_div_sim_ave_gender = zeros(2)
for gender in 1:2
    assets_div_sim_ave_gender[gender] = sum(assets_div_sim[2:end,gender])/(parameters.lifespan-1)
end

assets_div_sim_ave = sum(assets_div_sim_ave_gender)/2
#
# # Among recent Divorced
# assets_div_rec_sim_all_periods_num = 0.0
# assets_div_rec_sim_all_periods_den = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .== 1)
#             assets_div_rec_sim_all_periods_num += parameters.a_grid[panel_div_1_a[hh_i,i]]
#             assets_div_rec_sim_all_periods_den += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .== 1)
#             assets_div_rec_sim_all_periods_num += parameters.a_grid[panel_div_2_a[hh_i,i]]
#             assets_div_rec_sim_all_periods_den += 1.0
#         end
#     end
# end
#
# assets_div_rec_sim_all_periods_ave = assets_div_rec_sim_all_periods_num / assets_div_rec_sim_all_periods_den
#
# # Among non-recent Divorced
# assets_div_non_rec_sim_all_periods_num = 0.0
# assets_div_non_rec_sim_all_periods_den = 0.0
#
# for i in 1:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,i] .== 2)
#             assets_div_non_rec_sim_all_periods_num += parameters.a_grid[panel_div_1_a[hh_i,i]]
#             assets_div_non_rec_sim_all_periods_den += 1.0
#         end
#
#         if (panel_div_2_ind[hh_i,i] .== 2)
#             assets_div_non_rec_sim_all_periods_num += parameters.a_grid[panel_div_2_a[hh_i,i]]
#             assets_div_non_rec_sim_all_periods_den += 1.0
#         end
#     end
# end
#
# assets_div_non_rec_sim_all_periods_ave = assets_div_non_rec_sim_all_periods_num / assets_div_non_rec_sim_all_periods_den

# Couples
assets_C_sim = zeros(parameters.lifespan)

for age in 1:parameters.lifespan
    assets_C_sim_num = 0.0
    assets_C_sim_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        if shock_C_div[hh_i,age,monte_i] == 2
            assets_C_sim_num += parameters.a_grid[panel_C_a[hh_i,age,1,monte_i]]/2
            assets_C_sim_den += 1.0
        end
    end
    assets_C_sim[age] = assets_C_sim_num/assets_C_sim_den
end

assets_C_sim_ave = sum(assets_C_sim)/parameters.lifespan
#=======================#
# Consumption Insurance #
#=======================#

# Using change in productivity
# Singles
gender = 2

c_growth_S = zeros(num_S, parameters.lifespan-1, monte_sim)

for monte_i in 1:monte_sim, hh_i in 1:num_S, age in 2:parameters.lifespan
    c_growth_S[hh_i,age-1,monte_i] = log(panel_S_c[hh_i,age,gender,monte_i]) - log(panel_S_c[hh_i,age-1,gender,monte_i])
end

productivity_growth_S = zeros(num_S, parameters.lifespan-1, monte_sim)

for monte_i in 1:monte_sim, hh_i in 1:num_S, age in 2:parameters.lifespan
    productivity_growth_S[hh_i,age-1,monte_i] = log(parameters.z_grid[shock_S_z[hh_i,age,gender,monte_i]]*parameters.h_grid[age]) - log(parameters.z_grid[shock_S_z[hh_i,age-1,gender,monte_i]]*parameters.h_grid[age-1])
end

y_S_prod = c_growth_S[:]

productivity_growth_S_X = productivity_growth_S[:]

age_S_prod = zeros(num_S, parameters.lifespan-1, monte_sim)

for age in 2:parameters.lifespan
    age_S_prod[:,age-1,:] .= age
end

age_S_X_prod = age_S_prod[:]

row_num = size(age_S_X_prod)[1]

X_S_prod = zeros(row_num,4)

X_S_prod[:,1] .= 1.0
X_S_prod[:,2] = productivity_growth_S_X
X_S_prod[:,3] = age_S_X_prod
X_S_prod[:,4] = age_S_X_prod.^2

β_S_prod = (X_S_prod'*X_S_prod)\(X_S_prod'*y_S_prod)

# # Divorced
# c_growth_div_1 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if panel_div_1_ind[hh_i,period_i] == 2
#             c_growth_div_1[hh_i,period_i] = log(panel_div_1_c[hh_i,period_i]) - log(panel_div_1_c[hh_i,period_i-1])
#         end
#     end
# end
#
# c_growth_div_2 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if panel_div_2_ind[hh_i,period_i] == 2
#             c_growth_div_2[hh_i,period_i] = log(panel_div_2_c[hh_i,period_i]) - log(panel_div_2_c[hh_i,period_i-1])
#         end
#     end
# end
#
# productivity_growth_div_1 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,period_i] == 2)
#             productivity_growth_div_1[hh_i,period_i] = log(parameters.z_grid[shock_div_1_z[hh_i,period_i]]*parameters.η_grid[shock_div_1_η[hh_i,period_i]]) - log(parameters.z_grid[shock_div_1_z[hh_i,period_i-1]]*parameters.η_grid[shock_div_1_η[hh_i,period_i-1]])
#         end
#     end
# end
#
# productivity_growth_div_2 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if (panel_div_2_ind[hh_i,period_i] == 2)
#             productivity_growth_div_2[hh_i,period_i] = log(parameters.z_grid[shock_div_2_z[hh_i,period_i]]*parameters.η_grid[shock_div_2_η[hh_i,period_i]]) - log(parameters.z_grid[shock_div_2_z[hh_i,period_i-1]]*parameters.η_grid[shock_div_2_η[hh_i,period_i-1]])
#         end
#     end
# end
#
# c_growth_div_1_new = c_growth_div_1[:,2:end]
# c_growth_div_2_new = c_growth_div_2[:,2:end]
#
# y_div_prod = vcat(c_growth_div_1_new[(panel_div_1_ind[:,2:end] .== 2)],c_growth_div_2_new[(panel_div_2_ind[:,2:end] .== 2)])
#
# productivity_growth_div_1_new = productivity_growth_div_1[:,2:end]
# productivity_growth_div_2_new = productivity_growth_div_2[:,2:end]
#
# productivity_growth_div_X = vcat(productivity_growth_div_1_new[(panel_div_1_ind[:,2:end] .== 2)], productivity_growth_div_2_new[(panel_div_2_ind[:,2:end] .== 2)])
#
# age_div_1 = panel_div_1_age[:,2:end]
# age_div_2 = panel_div_2_age[:,2:end]
#
# age_div_X = vcat(age_div_1[(panel_div_1_ind[:,2:end] .== 2)],age_div_2[(panel_div_2_ind[:,2:end] .== 2)])
#
# # row_num = sum((shock_S_ρ[:,2:end] .== 1) .& (panel_S_n[:,2:end] .!= 1))
#
# X_div_prod = zeros(size(y_div_prod)[1],4)
#
# X_div_prod[:,1] .= 1.0
# X_div_prod[:,2] = productivity_growth_div_X
# X_div_prod[:,3] = age_div_X
# X_div_prod[:,4] = age_div_X.^2
#
# β_div_prod = (X_div_prod'*X_div_prod)\(X_div_prod'*y_div_prod)

# Couples
c_growth_C = zeros(num_C, parameters.lifespan-1, monte_sim)

for monte_i in 1:monte_sim, hh_i in 1:num_C, age in 2:parameters.lifespan
    if (shock_C_div[hh_i,age,monte_i] == 2)
        c_growth_C[hh_i,age-1,monte_i] = log(panel_C_c[hh_i,age,gender,monte_i]) - log(panel_C_c[hh_i,age-1,gender,monte_i])
    end
end

productivity_growth_C = zeros(num_C, parameters.lifespan-1, monte_sim)

for monte_i in 1:monte_sim, hh_i in 1:num_C, age in 2:parameters.lifespan
    if (shock_C_div[hh_i,age,monte_i] == 2)
        productivity_growth_C[hh_i,age-1,monte_i] = log(parameters.z_grid[shock_C_z[hh_i,age,gender,monte_i]]*parameters.h_grid[age]) - log(parameters.z_grid[shock_C_z[hh_i,age-1,gender,monte_i]]*parameters.h_grid[age-1])
    end
end

y_C_prod = c_growth_C[shock_C_div[:,2:end,:] .== 2]

productivity_growth_C_X = productivity_growth_C[shock_C_div[:,2:end,:] .== 2]

age_C_prod = zeros(num_C, parameters.lifespan-1, monte_sim)

for age in 2:parameters.lifespan
    age_C_prod[:,age-1,:] .= age
end

age_C_X_prod = age_C_prod[shock_C_div[:,2:end,:] .== 2]

row_num = size(age_C_X_prod)[1]

X_C_prod = zeros(row_num,4)

X_C_prod[:,1] .= 1.0
X_C_prod[:,2] = productivity_growth_C_X
X_C_prod[:,3] = age_C_X_prod
X_C_prod[:,4] = age_C_X_prod.^2

β_C_prod = (X_C_prod'*X_C_prod)\(X_C_prod'*y_C_prod)

# # Maximum asset level
# # Single
# maximum(parameters.a_grid[panel_S_a])
#
# # Couple and Divorced
# maximum(parameters.a_grid[panel_C_a])
#
# # Minimum asset level
# # Single
# minimum(parameters.a_grid[panel_S_a])
#
# # Couple and Divorced
# minimum(parameters.a_grid[panel_C_a])

#=========#
# Moments #
#=========#
moments = [fraction_S_default_sim_ave
            fraction_C_default_sim_ave
            fraction_div_default_sim_ave
            fraction_S_loan_users_sim_ave
            fraction_C_loan_users_sim_ave
            fraction_div_loan_users_sim_ave
            debt_to_inc_S_sim_ave
            debt_to_inc_C_sim_ave
            debt_to_inc_div_sim_ave
            ave_bank_rate_S_sim_ave
            ave_bank_rate_C_sim_ave
            ave_bank_rate_div_sim_ave
            consumption_S_sim_ave
            consumption_C_sim_ave
            consumption_div_sim_ave
            labor_S_sim_ave
            labor_C_sim_ave
            labor_div_sim_ave
            assets_S_sim_ave
            assets_C_sim_ave
            assets_div_sim_ave
            β_S_prod[2]
            β_C_prod[2]
            hours_single_50
]

#=======================#
# Default Behavior #
#=======================#
lifespan = 16
# Fraction of defaulters after bad income shock

# Singles
fraction_S_default_expense = zeros(lifespan,2)

for age in 1:lifespan, gender in 1:2
    fraction_S_default_expense_num = 0.0
    fraction_S_default_expense_den = 0.0

    for hh_i in 1:num_S, monte_i in 1:monte_sim
        # if shock_S_κ[hh_i,age,gender,monte_i] > 1 # Any bad expense shock
        # if shock_S_κ[hh_i,age,gender,monte_i] == 2 # Small expense shock
        if shock_S_κ[hh_i,age,gender,monte_i] == 3 # Large expense shock
            if panel_S_d[hh_i,age,gender,monte_i] == 2
                fraction_S_default_expense_num += 1.0
            end
            fraction_S_default_expense_den += 1.0
        end
    end

    fraction_S_default_expense[age,gender] = fraction_S_default_expense_num/fraction_S_default_expense_den*100
end

fraction_S_default_expense_ave_gender = zeros(2)
for gender in 1:2
    fraction_S_default_expense_ave_gender[gender] = sum(fraction_S_default_expense[2:end,gender])/(lifespan-1)
end

fraction_S_default_expense_ave = sum(fraction_S_default_expense_ave_gender)/2

# Couples
fraction_C_default_expense = zeros(lifespan)

for age in 1:lifespan
    fraction_C_default_expense_num = 0.0
    fraction_C_default_expense_den = 0.0

    for hh_i in 1:num_C, monte_i in 1:monte_sim
        # if any(shock_C_κ[hh_i,age,:,monte_i] .> 1) && (shock_C_div[hh_i,age,monte_i] == 2) # Either one gets any bad shock
        # if any(shock_C_κ[hh_i,age,:,monte_i] .== 2) && any(shock_C_κ[hh_i,age,:,monte_i] .== 1) && (shock_C_div[hh_i,age,monte_i] == 2) # One gets small, other no shock
        # if all(shock_C_κ[hh_i,age,:,monte_i] .== 2) && (shock_C_div[hh_i,age,monte_i] == 2) # Both get small
        # if any(shock_C_κ[hh_i,age,:,monte_i] .== 3) && any(shock_C_κ[hh_i,age,:,monte_i] .== 1) && (shock_C_div[hh_i,age,monte_i] == 2) # One gets large, other no shock
        # if any(shock_C_κ[hh_i,age,:,monte_i] .== 3) && any(shock_C_κ[hh_i,age,:,monte_i] .== 2) && (shock_C_div[hh_i,age,monte_i] == 2) # One gets large, other small shock
        if all(shock_C_κ[hh_i,age,:,monte_i] .== 3) && (shock_C_div[hh_i,age,monte_i] == 2) # Both get large
            if panel_C_d[hh_i,age,1,monte_i] == 2
                fraction_C_default_expense_num += 1.0
            end
            fraction_C_default_expense_den += 1.0
        end
    end

    fraction_C_default_expense[age] = fraction_C_default_expense_num/fraction_C_default_expense_den*100
end

temp_num = 0.0
temp_den = 0.0

for age in 2:lifespan
    if (isnan(fraction_C_default_expense[age]) == false)
        temp_num += fraction_C_default_expense[age]
        temp_den += 1.0
    end
end

fraction_C_default_expense_ave = temp_num/temp_den

# fraction_C_default_expense_ave = sum(fraction_C_default_expense[2:end])/(lifespan-1)

#=======================#
# Consumption Insurance #
#=======================#
# Using Change in Earnings
# Singles
# c_growth_S = zeros(num_S, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_S
#         if shock_S_ρ[hh_i,period_i] == 1
#             c_growth_S[hh_i,period_i] = log(panel_S_c[hh_i,period_i]) - log(panel_S_c[hh_i,period_i-1])
#         end
#     end
# end
#
# earnings_growth_S = zeros(num_S, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_S
#         if (shock_S_ρ[hh_i,period_i] == 1) && (panel_S_n[hh_i,period_i] != 1) && (panel_S_n[hh_i,period_i-1] != 1)
#             earnings_growth_S[hh_i,period_i] = log(parameters.z_grid[shock_S_z[hh_i,period_i]]*parameters.η_grid[shock_S_η[hh_i,period_i]]*(parameters.n_grid[panel_S_n[hh_i,period_i]]^parameters.θ)) - log(parameters.z_grid[shock_S_z[hh_i,period_i-1]]*parameters.η_grid[shock_S_η[hh_i,period_i-1]]*(parameters.n_grid[panel_S_n[hh_i,period_i-1]]^parameters.θ))
#         end
#     end
# end
#
# c_growth_S_new = c_growth_S[:,2:end]
#
# y_S = c_growth_S_new[(shock_S_ρ[:,2:end] .== 1) .& (panel_S_n[:,2:end] .!= 1)]
#
# earnings_growth_S_new = earnings_growth_S[:,2:end]
#
# earnings_growth_S_X = earnings_growth_S_new[(shock_S_ρ[:,2:end] .== 1) .& (panel_S_n[:,2:end] .!= 1)]
#
# age_S = panel_S_age[:,2:end]
#
# age_S_X = age_S[(shock_S_ρ[:,2:end] .== 1) .& (panel_S_n[:,2:end] .!= 1)]
#
# row_num = sum((shock_S_ρ[:,2:end] .== 1) .& (panel_S_n[:,2:end] .!= 1))
#
# X_S = zeros(row_num,4)
#
# X_S[:,1] .= 1.0
# X_S[:,2] = earnings_growth_S_X
# X_S[:,3] = age_S_X
# X_S[:,4] = age_S_X.^2
#
# β_S = (X_S'*X_S)\(X_S'*y_S)
#
# # Divorced
# c_growth_div_1 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if panel_div_1_ind[hh_i,period_i] == 2
#             c_growth_div_1[hh_i,period_i] = log(panel_div_1_c[hh_i,period_i]) - log(panel_div_1_c[hh_i,period_i-1])
#         end
#     end
# end
#
# c_growth_div_2 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if panel_div_2_ind[hh_i,period_i] == 2
#             c_growth_div_2[hh_i,period_i] = log(panel_div_2_c[hh_i,period_i]) - log(panel_div_2_c[hh_i,period_i-1])
#         end
#     end
# end
#
# earnings_growth_div_1 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if (panel_div_1_ind[hh_i,period_i] == 2) && (panel_div_1_n[hh_i,period_i] != 1) && (panel_div_1_n[hh_i,period_i-1] != 1)
#             earnings_growth_div_1[hh_i,period_i] = log(parameters.z_grid[shock_div_1_z[hh_i,period_i]]*parameters.η_grid[shock_div_1_η[hh_i,period_i]]*(parameters.n_grid[panel_div_1_n[hh_i,period_i]]^parameters.θ)) - log(parameters.z_grid[shock_div_1_z[hh_i,period_i-1]]*parameters.η_grid[shock_div_1_η[hh_i,period_i-1]]*(parameters.n_grid[panel_div_1_n[hh_i,period_i-1]]^parameters.θ))
#         end
#     end
# end
#
# earnings_growth_div_2 = zeros(num_div, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_div
#         if (panel_div_2_ind[hh_i,period_i] == 2) && (panel_div_2_n[hh_i,period_i] != 1) && (panel_div_2_n[hh_i,period_i-1] != 1)
#             earnings_growth_div_2[hh_i,period_i] = log(parameters.z_grid[shock_div_2_z[hh_i,period_i]]*parameters.η_grid[shock_div_2_η[hh_i,period_i]]*(parameters.n_grid[panel_div_2_n[hh_i,period_i]]^parameters.θ)) - log(parameters.z_grid[shock_div_2_z[hh_i,period_i-1]]*parameters.η_grid[shock_div_2_η[hh_i,period_i-1]]*(parameters.n_grid[panel_div_2_n[hh_i,period_i-1]]^parameters.θ))
#         end
#     end
# end
#
# c_growth_div_1_new = c_growth_div_1[:,2:end]
# c_growth_div_2_new = c_growth_div_2[:,2:end]
#
# y_div = vcat(c_growth_div_1_new[(panel_div_1_ind[:,2:end] .== 2)  .& (panel_div_1_n[:,2:end] .!= 1)],c_growth_div_2_new[(panel_div_2_ind[:,2:end] .== 2)  .& (panel_div_2_n[:,2:end] .!= 1)])
#
# earnings_growth_div_1_new = earnings_growth_div_1[:,2:end]
# earnings_growth_div_2_new = earnings_growth_div_2[:,2:end]
#
# earnings_growth_div_X = vcat(earnings_growth_div_1_new[(panel_div_1_ind[:,2:end] .== 2)  .& (panel_div_1_n[:,2:end] .!= 1)], earnings_growth_div_2_new[(panel_div_2_ind[:,2:end] .== 2)  .& (panel_div_2_n[:,2:end] .!= 1)])
#
# age_div_1 = panel_div_1_age[:,2:end]
# age_div_2 = panel_div_2_age[:,2:end]
#
# age_div_X = vcat(age_div_1[(panel_div_1_ind[:,2:end] .== 2)  .& (panel_div_1_n[:,2:end] .!= 1)],age_div_2[(panel_div_2_ind[:,2:end] .== 2)  .& (panel_div_2_n[:,2:end] .!= 1)])
#
# # row_num = sum((shock_S_ρ[:,2:end] .== 1) .& (panel_S_n[:,2:end] .!= 1))
#
# X_div = zeros(size(y_div)[1],4)
#
# X_div[:,1] .= 1.0
# X_div[:,2] = earnings_growth_div_X
# X_div[:,3] = age_div_X
# X_div[:,4] = age_div_X.^2
#
# β_div = (X_div'*X_div)\(X_div'*y_div)
#
# # Couples
# c_growth_C = zeros(num_C, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_C
#         if (shock_C_ρ[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i] == 2)
#             c_growth_C[hh_i,period_i] = log(panel_C_c[hh_i,period_i]) - log(panel_C_c[hh_i,period_i-1])
#         end
#     end
# end
#
# earnings_growth_C = zeros(num_C, period_all)
#
# for period_i in 2:period_all
#     for hh_i in 1:num_C
#         if (shock_C_ρ[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i] == 2) && ((panel_C_1_n[hh_i,period_i] != 1) ||(panel_C_2_n[hh_i,period_i] != 1)) && ((panel_C_1_n[hh_i,period_i-1] != 1) || (panel_C_2_n[hh_i,period_i-1] != 1))
#             earnings_growth_C[hh_i,period_i] = log(parameters.z_grid[shock_C_z_1[hh_i,period_i]]*parameters.η_grid[shock_C_η_1[hh_i,period_i]]*(parameters.n_grid[panel_C_1_n[hh_i,period_i]]^parameters.θ) + parameters.z_grid[shock_C_z_2[hh_i,period_i]]*parameters.η_grid[shock_C_η_2[hh_i,period_i]]*(parameters.n_grid[panel_C_2_n[hh_i,period_i]]^parameters.θ)) - log(parameters.z_grid[shock_C_z_1[hh_i,period_i-1]]*parameters.η_grid[shock_C_η_1[hh_i,period_i-1]]*(parameters.n_grid[panel_C_1_n[hh_i,period_i-1]]^parameters.θ) + parameters.z_grid[shock_C_z_2[hh_i,period_i-1]]*parameters.η_grid[shock_C_η_2[hh_i,period_i-1]]*(parameters.n_grid[panel_C_2_n[hh_i,period_i-1]]^parameters.θ))
#         end
#     end
# end
#
# c_growth_C_new = c_growth_C[:,2:end]
#
# y_C = c_growth_C_new[(shock_C_ρ[:,2:end] .== 1) .& (shock_C_div[:,2:end] .== 2) .& ((panel_C_1_n[:,2:end] .!= 1) .| (panel_C_2_n[:,2:end] .!= 1))]
#
# earnings_growth_C_new = earnings_growth_C[:,2:end]
#
# earnings_growth_C_X = earnings_growth_C_new[(shock_C_ρ[:,2:end] .== 1) .& (shock_C_div[:,2:end] .== 2) .& ((panel_C_1_n[:,2:end] .!= 1) .| (panel_C_2_n[:,2:end] .!= 1))]
#
# age_C = panel_C_age[:,2:end]
#
# age_C_X = age_C[(shock_C_ρ[:,2:end] .== 1) .& (shock_C_div[:,2:end] .== 2) .& ((panel_C_1_n[:,2:end] .!= 1) .| (panel_C_2_n[:,2:end] .!= 1))]
#
# row_num_C = sum((shock_C_ρ[:,2:end] .== 1) .& (shock_C_div[:,2:end] .== 2) .& ((panel_C_1_n[:,2:end] .!= 1) .| (panel_C_2_n[:,2:end] .!= 1)))
#
# X_C = zeros(row_num_C,4)
#
# X_C[:,1] .= 1.0
# X_C[:,2] = earnings_growth_C_X
# X_C[:,3] = age_C_X
# X_C[:,4] = age_C_X.^2
#
# β_C = (X_C'*X_C)\(X_C'*y_C)

#=============================#
#       Ex-ante Welfare       #
#=============================#
# Singles
welfare_S = 0.0

for η_i in 1:parameters.η_size
    welfare_S += parameters.Γ_η[η_i]*W_S[parameters.a_ind_zero,2,η_i]
end

# Couples
welfare_C = 0.0

for η_1_i in 1:parameters.η_size, η_2_i in 1:parameters.η_size
    welfare_C += parameters.Γ_η[η_1_i]*parameters.Γ_η[η_2_i]*W_C[parameters.a_ind_zero,2,2,η_1_i,η_2_i]
end

#===================#
# Summarize moments #
#===================#

moments = [fraction_S_default_sim_all_periods_ave
            fraction_div_default_sim_all_periods_ave
            fraction_div_rec_default_sim_all_periods_ave
            fraction_div_non_rec_default_sim_all_periods_ave
            fraction_C_default_sim_all_periods_ave
            fraction_S_loan_users_sim_all_periods_ave
            fraction_S_loan_users_sim_z_ave[1]
            fraction_S_loan_users_sim_z_ave[2]
            fraction_S_loan_users_sim_z_ave[3]
            fraction_div_loan_users_sim_all_periods_ave
            fraction_div_loan_users_sim_z_ave[1]
            fraction_div_loan_users_sim_z_ave[2]
            fraction_div_loan_users_sim_z_ave[3]
            fraction_div_rec_loan_users_sim_all_periods_ave
            fraction_div_non_rec_loan_users_sim_all_periods_ave
            fraction_C_loan_users_sim_all_periods_ave
            fraction_C_loan_users_sim_z_ave[1,1]
            fraction_C_loan_users_sim_z_ave[2,2]
            fraction_C_loan_users_sim_z_ave[3,3]
            debt_to_inc_S_sim_all_periods
            debt_to_inc_div_sim_all_periods
            debt_to_inc_div_rec_sim_all_periods
            debt_to_inc_div_non_rec_sim_all_periods
            debt_to_inc_C_sim_all_periods
            ave_bank_rate_S_sim_all_periods_ave
            ave_bank_rate_div_sim_all_periods_ave
            ave_bank_rate_div_rec_sim_all_periods_ave
            ave_bank_rate_div_non_rec_sim_all_periods_ave
            ave_bank_rate_C_sim_all_periods_ave
            consumption_S_sim_all_periods_ave
            consumption_div_sim_all_periods_ave
            consumption_div_rec_sim_all_periods_ave
            consumption_div_non_rec_sim_all_periods_ave
            consumption_C_sim_all_periods_ave
            labor_S_sim_all_periods_ave
            labor_div_sim_all_periods_ave
            labor_div_rec_sim_all_periods_ave
            labor_div_non_rec_sim_all_periods_ave
            labor_C_sim_all_periods_ave
            assets_S_sim_all_periods_ave
            assets_div_sim_all_periods_ave
            assets_div_rec_sim_all_periods_ave
            assets_div_non_rec_sim_all_periods_ave
            assets_C_sim_all_periods_ave
            β_S[2]
            β_div[2]
            β_C[2]
            β_S_prod[2]
            β_div_prod[2]
            β_C_prod[2]
            welfare_S
            welfare_C
]

#=
#========================#
# Simulated Distribution #
#========================#
μ_sim = zeros(parameters.β_size, parameters.z_size, parameters.e_size, parameters.asset_size, parameters.s_size)

@showprogress 1 "Computing..." for s_i in 1:parameters.s_size, asset_i in 1:parameters.asset_size, e_i in 1:parameters.e_size, z_i in 1:parameters.z_size, β_i in 1:parameters.β_size
    # HH's type
    ind_hh_type = [(shock_β .== β_i) .& (shock_z .== z_i) .& (shock_e .== e_i) .& (panel_asset .== asset_i) .& (panel_score .== s_i)][1]

    # simulated distribution
    μ_sim[β_i,z_i,e_i,asset_i,s_i] = sum(ind_hh_type)/(num_hh*period_all)
end
=#

#====================#
# Compute dist stats #
#====================#
using Statistics

# Compute 20th percentile of asset holdings
# Singles
asset_perc_S = quantile(parameters.a_grid[panel_S_a][:],0.1)

# Couples
asset_perc_C = quantile(parameters.a_grid[panel_C_a][:],0.1)

#===============#
# Event studies #
#===============#
event_window = 5

# Effect of divorce event

# On Consumption
divorce_event_consumption_num = zeros(event_window)
divorce_event_consumption_den = zeros(event_window)

for period_i in 3:period_all
    for hh_i in 1:num_div
        # Event time 0
        if panel_div_1_ind[hh_i,period_i] == 1
            divorce_event_consumption_num[3] += panel_div_1_c[hh_i,period_i]
            divorce_event_consumption_den[3] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1
            divorce_event_consumption_num[3] += panel_div_2_c[hh_i,period_i]
            divorce_event_consumption_den[3] += 1.0
        end

        # Event time 1
        if (panel_div_1_ind[hh_i,period_i-1] == 1) && (panel_div_1_ind[hh_i,period_i] == 2)
            divorce_event_consumption_num[4] += panel_div_1_c[hh_i,period_i]
            divorce_event_consumption_den[4] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-1] == 1) && (panel_div_2_ind[hh_i,period_i] == 2)
            divorce_event_consumption_num[4] += panel_div_2_c[hh_i,period_i]
            divorce_event_consumption_den[4] += 1.0
        end

        # Event time 3
        if (panel_div_1_ind[hh_i,period_i-2] == 1) && all(panel_div_1_ind[hh_i,period_i-1:period_i] .== 2)
            divorce_event_consumption_num[5] += panel_div_1_c[hh_i,period_i]
            divorce_event_consumption_den[5] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-2] == 1) && all(panel_div_2_ind[hh_i,period_i-1:period_i] .== 2)
            divorce_event_consumption_num[5] += panel_div_2_c[hh_i,period_i]
            divorce_event_consumption_den[5] += 1.0
        end
    end

    for hh_i in 1:num_C

        # Event time -1
        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_consumption_num[2] += panel_C_c[hh_i,period_i-1]/2.0
            divorce_event_consumption_den[2] += 1.0
        end

        # Event time -2
        if (shock_C_div[hh_i,period_i] == 1) && all(shock_C_div[hh_i,period_i-2:period_i-1] .== 2) && all(shock_C_ρ[hh_i,period_i-2:period_i] .== 1)
            divorce_event_consumption_num[1] += panel_C_c[hh_i,period_i-2]/2.0
            divorce_event_consumption_den[1] += 1.0
        end
    end
end

divorce_event_consumption = divorce_event_consumption_num ./ divorce_event_consumption_den

# On mean assets
divorce_event_assets_num = zeros(event_window)
divorce_event_assets_den = zeros(event_window)

for period_i in 3:period_all-1
    for hh_i in 1:num_div

        # Event time -1
        if panel_div_1_ind[hh_i,period_i] == 1
            divorce_event_assets_num[2] += parameters.a_grid[panel_div_1_a[hh_i,period_i]]
            divorce_event_assets_den[2] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1
            divorce_event_assets_num[2] += parameters.a_grid[panel_div_2_a[hh_i,period_i]]
            divorce_event_assets_den[2] += 1.0
        end

        # Event time 0
        if panel_div_1_ind[hh_i,period_i] == 1 && (panel_div_1_ind[hh_i,period_i+1] == 2)
            divorce_event_assets_num[3] += parameters.a_grid[panel_div_1_a[hh_i,period_i+1]]
            divorce_event_assets_den[3] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1 && (panel_div_2_ind[hh_i,period_i+1] == 2)
            divorce_event_assets_num[3] += parameters.a_grid[panel_div_2_a[hh_i,period_i+1]]
            divorce_event_assets_den[3] += 1.0
        end

        # Event time 1
        if (panel_div_1_ind[hh_i,period_i-1] == 1) && (panel_div_1_ind[hh_i,period_i] == 2) && (panel_div_1_ind[hh_i,period_i+1] == 2)
            divorce_event_assets_num[4] += parameters.a_grid[panel_div_1_a[hh_i,period_i+1]]
            divorce_event_assets_den[4] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-1] == 1) && (panel_div_2_ind[hh_i,period_i] == 2) && (panel_div_2_ind[hh_i,period_i+1] == 2)
            divorce_event_assets_num[4] += parameters.a_grid[panel_div_2_a[hh_i,period_i+1]]
            divorce_event_assets_den[4] += 1.0
        end

        # Event time 3
        if (panel_div_1_ind[hh_i,period_i-2] == 1) && all(panel_div_1_ind[hh_i,period_i-1:period_i+1] .== 2)
            divorce_event_assets_num[5] += parameters.a_grid[panel_div_1_a[hh_i,period_i+1]]
            divorce_event_assets_den[5] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-2] == 1) && all(panel_div_2_ind[hh_i,period_i-1:period_i+1] .== 2)
            divorce_event_assets_num[5] += parameters.a_grid[panel_div_2_a[hh_i,period_i+1]]
            divorce_event_assets_den[5] += 1.0
        end
    end

    for hh_i in 1:num_C

        # Event time -1
        # if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
        #     divorce_event_assets_num[2] += parameters.a_grid[panel_C_a[hh_i,period_i-1]]/2.0
        #     divorce_event_assets_den[2] += 1.0
        # end

        # Event time -2
        # if (shock_C_div[hh_i,period_i] == 1) && all(shock_C_div[hh_i,period_i-2:period_i-1] .== 2) && all(shock_C_ρ[hh_i,period_i-2:period_i] .== 1)
        #     divorce_event_assets_num[1] += parameters.a_grid[panel_C_a[hh_i,period_i-2]]/2.0
        #     divorce_event_assets_den[1] += 1.0
        # end

        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_assets_num[1] += parameters.a_grid[panel_C_a[hh_i,period_i-1]]/2.0
            divorce_event_assets_den[1] += 1.0
        end
    end
end

divorce_event_assets = divorce_event_assets_num ./ divorce_event_assets_den

# On mean assets (cond. on not defaulting)
divorce_event_assets_no_default_num = zeros(event_window)
divorce_event_assets_no_default_den = zeros(event_window)

for period_i in 3:period_all-1
    for hh_i in 1:num_div
        # Event time -1
        if panel_div_1_ind[hh_i,period_i] == 1
            divorce_event_assets_no_default_num[2] += parameters.a_grid[panel_div_1_a[hh_i,period_i]]
            divorce_event_assets_no_default_den[2] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1
            divorce_event_assets_no_default_num[2] += parameters.a_grid[panel_div_2_a[hh_i,period_i]]
            divorce_event_assets_no_default_den[2] += 1.0
        end

        # Event time 0
        if (panel_div_1_ind[hh_i,period_i] == 1) && (panel_div_1_ind[hh_i,period_i+1] == 2) && (panel_div_1_d[hh_i,period_i] == 1)
            divorce_event_assets_no_default_num[3] += parameters.a_grid[panel_div_1_a[hh_i,period_i+1]]
            divorce_event_assets_no_default_den[3] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1 && (panel_div_2_ind[hh_i,period_i+1] == 2) && (panel_div_2_d[hh_i,period_i] == 1)
            divorce_event_assets_no_default_num[3] += parameters.a_grid[panel_div_2_a[hh_i,period_i+1]]
            divorce_event_assets_no_default_den[3] += 1.0
        end

        # Event time 1
        if (panel_div_1_ind[hh_i,period_i-1] == 1) && (panel_div_1_ind[hh_i,period_i] == 2) && (panel_div_1_ind[hh_i,period_i+1] == 2) && all(panel_div_1_d[hh_i,period_i-1:period_i] .== 1)
            divorce_event_assets_no_default_num[4] += parameters.a_grid[panel_div_1_a[hh_i,period_i+1]]
            divorce_event_assets_no_default_den[4] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-1] == 1) && (panel_div_2_ind[hh_i,period_i] == 2) && (panel_div_2_ind[hh_i,period_i+1] == 2) && all(panel_div_2_d[hh_i,period_i-1:period_i] .== 1)
            divorce_event_assets_no_default_num[4] += parameters.a_grid[panel_div_2_a[hh_i,period_i+1]]
            divorce_event_assets_no_default_den[4] += 1.0
        end

        # Event time 3

        if (panel_div_1_ind[hh_i,period_i-2] == 1) && all(panel_div_1_ind[hh_i,period_i-1:period_i+1] .== 2) && all(panel_div_1_d[hh_i,period_i-2:period_i] .== 1)
            divorce_event_assets_no_default_num[5] += parameters.a_grid[panel_div_1_a[hh_i,period_i+1]]
            divorce_event_assets_no_default_den[5] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-2] == 1) && all(panel_div_2_ind[hh_i,period_i-1:period_i+1] .== 2) && all(panel_div_2_d[hh_i,period_i-2:period_i] .== 1)
            divorce_event_assets_no_default_num[5] += parameters.a_grid[panel_div_2_a[hh_i,period_i+1]]
            divorce_event_assets_no_default_den[5] += 1.0
        end
    end

    for hh_i in 1:num_C

        # Event time -1
        # if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
        #     divorce_event_assets_no_default_num[2] += parameters.a_grid[panel_C_a[hh_i,period_i-1]]/2.0
        #     divorce_event_assets_no_default_den[2] += 1.0
        # end

        # Event time -2
        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] .== 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_assets_no_default_num[1] += parameters.a_grid[panel_C_a[hh_i,period_i-1]]/2.0
            divorce_event_assets_no_default_den[1] += 1.0
        end
    end
end

divorce_event_assets_no_default = divorce_event_assets_no_default_num ./ divorce_event_assets_no_default_den

# On fraction of borrowers
divorce_event_fraction_borrow_num = zeros(event_window)
divorce_event_fraction_borrow_den = zeros(event_window)

for period_i in 3:period_all-1
    for hh_i in 1:num_div

        # Event time -1
        if panel_div_1_ind[hh_i,period_i] == 1
            divorce_event_fraction_borrow_num[2] += panel_div_1_a[hh_i,period_i] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[2] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1
            divorce_event_fraction_borrow_num[2] += panel_div_2_a[hh_i,period_i] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[2] += 1.0
        end

        # Event time 0
        if panel_div_1_ind[hh_i,period_i] == 1 && (panel_div_1_ind[hh_i,period_i+1] == 2)
            divorce_event_fraction_borrow_num[3] += panel_div_1_a[hh_i,period_i+1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[3] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1 && (panel_div_2_ind[hh_i,period_i+1] == 2)
            divorce_event_fraction_borrow_num[3] += panel_div_2_a[hh_i,period_i+1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[3] += 1.0
        end

        # Event time 1
        if (panel_div_1_ind[hh_i,period_i-1] == 1) && (panel_div_1_ind[hh_i,period_i] == 2) && (panel_div_1_ind[hh_i,period_i+1] == 2)
            divorce_event_fraction_borrow_num[4] += panel_div_1_a[hh_i,period_i+1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[4] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-1] == 1) && (panel_div_2_ind[hh_i,period_i] == 2) && (panel_div_2_ind[hh_i,period_i+1] == 2)
            divorce_event_fraction_borrow_num[4] += panel_div_2_a[hh_i,period_i+1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[4] += 1.0
        end

        # Event time 3
        if (panel_div_1_ind[hh_i,period_i-2] == 1) && all(panel_div_1_ind[hh_i,period_i-1:period_i+1] .== 2)
            divorce_event_fraction_borrow_num[5] += panel_div_1_a[hh_i,period_i+1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[5] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-2] == 1) && all(panel_div_2_ind[hh_i,period_i-1:period_i+1] .== 2)
            divorce_event_fraction_borrow_num[5] += panel_div_2_a[hh_i,period_i+1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[5] += 1.0
        end
    end

    for hh_i in 1:num_C

        # Event time -1
        # if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
        #     divorce_event_assets_num[2] += parameters.a_grid[panel_C_a[hh_i,period_i-1]]/2.0
        #     divorce_event_assets_den[2] += 1.0
        # end

        # Event time -2
        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_fraction_borrow_num[1] += panel_C_a[hh_i,period_i-1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[1] += 1.0
        end

        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_fraction_borrow_num[1] += panel_C_a[hh_i,period_i-1] < parameters.a_ind_zero
            divorce_event_fraction_borrow_den[1] += 1.0
        end
    end
end

divorce_event_fraction_borrow = divorce_event_fraction_borrow_num ./ divorce_event_fraction_borrow_den

# On fraction of defaulters
event_window = 5

divorce_event_default_num = zeros(event_window)
divorce_event_default_den = zeros(event_window)

for hh_i in 1:num_C, age in 4:(parameters.lifespan-3), monte_i in 1:monte_sim
    if (shock_C_div[hh_i,age,monte_i] == 1) && (shock_C_div[hh_i,age-1,monte_i] == 2)
        divorce_event_default_num[3] += (panel_C_d[hh_i,age,1,monte_i]==2)
        divorce_event_default_num[3] += (panel_C_d[hh_i,age,2,monte_i]==2)
        divorce_event_default_den[3] += 2.0

        divorce_event_default_num[4] += (panel_C_d[hh_i,age+1,1,monte_i]==2)
        divorce_event_default_num[4] += (panel_C_d[hh_i,age+1,2,monte_i]==2)
        divorce_event_default_den[4] += 2.0

        divorce_event_default_num[5] += (panel_C_d[hh_i,age+2,1,monte_i]==2)
        divorce_event_default_num[5] += (panel_C_d[hh_i,age+2,2,monte_i]==2)
        divorce_event_default_den[5] += 2.0

        divorce_event_default_num[1] += (panel_C_d[hh_i,age-2,1,monte_i]==2)
        divorce_event_default_den[1] += 1.0

        divorce_event_default_num[2] += (panel_C_d[hh_i,age-1,1,monte_i]==2)
        divorce_event_default_den[2] += 1.0
    end
end

divorce_event_default = divorce_event_default_num ./ divorce_event_default_den

for period_i in 3:period_all
    for hh_i in 1:num_div
        # Event time 0
        if panel_div_1_ind[hh_i,period_i] == 1
            divorce_event_default_num[3] += (panel_div_1_d[hh_i,period_i] == 2)
            divorce_event_default_den[3] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1
            divorce_event_default_num[3] += (panel_div_2_d[hh_i,period_i] == 2)
            divorce_event_default_den[3] += 1.0
        end

        # Event time 1
        if (panel_div_1_ind[hh_i,period_i-1] == 1) && (panel_div_1_ind[hh_i,period_i] == 2)
            divorce_event_default_num[4] += (panel_div_1_d[hh_i,period_i] == 2)
            divorce_event_default_den[4] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-1] == 1) && (panel_div_2_ind[hh_i,period_i] == 2)
            divorce_event_default_num[4] += (panel_div_2_d[hh_i,period_i] == 2)
            divorce_event_default_den[4] += 1.0
        end

        # Event time 3
        if (panel_div_1_ind[hh_i,period_i-2] == 1) && all(panel_div_1_ind[hh_i,period_i-1:period_i] .== 2)
            divorce_event_default_num[5] += (panel_div_1_d[hh_i,period_i] == 2)
            divorce_event_default_den[5] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-2] == 1) && all(panel_div_2_ind[hh_i,period_i-1:period_i] .== 2)
            divorce_event_default_num[5] += (panel_div_2_d[hh_i,period_i] == 2)
            divorce_event_default_den[5] += 1.0
        end
    end

    for hh_i in 1:num_C

        # Event time -1
        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_default_num[2] += (panel_C_d[hh_i,period_i-1] == 2)
            divorce_event_default_den[2] += 1.0
        end

        # Event time -2
        if (shock_C_div[hh_i,period_i] == 1) && all(shock_C_div[hh_i,period_i-2:period_i-1] .== 2) && all(shock_C_ρ[hh_i,period_i-2:period_i] .== 1)
            divorce_event_default_num[1] += (panel_C_d[hh_i,period_i-2] == 2)
            divorce_event_default_den[1] += 1.0
        end
    end
end

divorce_event_default = divorce_event_default_num ./ divorce_event_default_den

# On labor
divorce_event_labor_num = zeros(event_window)
divorce_event_labor_den = zeros(event_window)

for period_i in 3:period_all
    for hh_i in 1:num_div
        # Event time 0
        if panel_div_1_ind[hh_i,period_i] == 1
            divorce_event_labor_num[3] += parameters.n_grid[panel_div_1_n[hh_i,period_i]]
            divorce_event_labor_den[3] += 1.0
        end

        if panel_div_2_ind[hh_i,period_i] == 1
            divorce_event_labor_num[3] += parameters.n_grid[panel_div_2_n[hh_i,period_i]]
            divorce_event_labor_den[3] += 1.0
        end

        # Event time 1
        if (panel_div_1_ind[hh_i,period_i-1] == 1) && (panel_div_1_ind[hh_i,period_i] == 2)
            divorce_event_labor_num[4] += parameters.n_grid[panel_div_1_n[hh_i,period_i]]
            divorce_event_labor_den[4] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-1] == 1) && (panel_div_2_ind[hh_i,period_i] == 2)
            divorce_event_labor_num[4] += parameters.n_grid[panel_div_2_n[hh_i,period_i]]
            divorce_event_labor_den[4] += 1.0
        end

        # Event time 3
        if (panel_div_1_ind[hh_i,period_i-2] == 1) && all(panel_div_1_ind[hh_i,period_i-1:period_i] .== 2)
            divorce_event_labor_num[5] += parameters.n_grid[panel_div_1_n[hh_i,period_i]]
            divorce_event_labor_den[5] += 1.0
        end

        if (panel_div_2_ind[hh_i,period_i-2] == 1) && all(panel_div_2_ind[hh_i,period_i-1:period_i] .== 2)
            divorce_event_labor_num[5] += parameters.n_grid[panel_div_2_n[hh_i,period_i]]
            divorce_event_labor_den[5] += 1.0
        end
    end

    for hh_i in 1:num_C

        # Event time -1
        if (shock_C_div[hh_i,period_i] == 1) && (shock_C_div[hh_i,period_i-1] == 2) && all(shock_C_ρ[hh_i,period_i-1:period_i] .== 1)
            divorce_event_labor_num[2] += parameters.n_grid[panel_C_1_n[hh_i,period_i-1]] + parameters.n_grid[panel_C_2_n[hh_i,period_i-1]]
            divorce_event_labor_den[2] += 2.0
        end

        # Event time -2
        if (shock_C_div[hh_i,period_i] == 1) && all(shock_C_div[hh_i,period_i-2:period_i-1] .== 2) && all(shock_C_ρ[hh_i,period_i-2:period_i] .== 1)
            divorce_event_labor_num[1] += parameters.n_grid[panel_C_1_n[hh_i,period_i-2]] + parameters.n_grid[panel_C_2_n[hh_i,period_i-2]]
            divorce_event_labor_den[1] += 2.0
        end
    end
end

divorce_event_labor = divorce_event_labor_num ./ divorce_event_labor_den

# Old code

# Effect of switch from z_3 to z_1
z_before = 3
z_after = 2

# event_S = zeros(num_S,period_all-2*event_window)
# for hh_i in 1:num_S, period_i in (event_window+1):(period_all-event_window)
#     event_S[hh_i,period_i-event_window] = (shock_S_z[hh_i,period_i] == z_after) & (shock_S_z[hh_i,period_i-1] == z_before)
# end

################################################
# On default probability (singles vs. couples) #
################################################

d_grid = [0.0 1.0]
############
# Singles  #
############

# Among all HH
event_S = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_S[hh_i,period_i] = (shock_S_z[hh_i,period_i] == z_after) & (shock_S_z[hh_i,period_i-1] == z_before)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_S =  Int(sum(event_S[:,3:(period_all-event_window)]))

event_S_default = zeros(num_events_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_S_default[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_S_default_ave = sum(event_S_default,dims=1)./counter

event_S_default_ave_norm = event_S_default_ave ./ event_S_default_ave[2]

# Among liquidity constrained HH
event_S_lc = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_S_lc[hh_i,period_i] = (shock_S_z[hh_i,period_i] == z_after) & (shock_S_z[hh_i,period_i-1] == z_before) & (parameters.a_grid[panel_S_a[hh_i,period_i-1]] < asset_perc_S)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_S_lc =  Int(sum(event_S_lc[:,3:(period_all-event_window)]))

event_S_default_lc = zeros(num_events_S_lc,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_S_lc[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_S_default_lc[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_S_default_ave_lc = sum(event_S_default_lc,dims=1)./counter

event_S_default_ave_norm_lc = event_S_default_ave_lc ./ event_S_default_ave_lc[2]

###########
# Couples #
###########

# Among all HH
event_C = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_C[hh_i,period_i] = (shock_C_z_1[hh_i,period_i] == z_after) & (shock_C_z_1[hh_i,period_i-1] == z_before)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_C =  Int(sum(event_C[:,3:(period_all-event_window)]))

event_C_default = zeros(num_events_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_C_default[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_C_default_ave = sum(event_C_default,dims=1)./counter

event_C_default_ave_norm = event_C_default_ave ./ event_C_default_ave[2]

# Among liquidity constrained HH
event_C_lc = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_C_lc[hh_i,period_i] = (shock_C_z_1[hh_i,period_i] == z_after) & (shock_C_z_1[hh_i,period_i-1] == z_before) & (parameters.a_grid[panel_C_a[hh_i,period_i-1]] < asset_perc_C)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_C_lc =  Int(sum(event_C_lc[:,3:(period_all-event_window)]))

event_C_default_lc = zeros(num_events_C_lc,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_C_lc[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_C_default_lc[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_C_default_ave_lc = sum(event_C_default_lc,dims=1)./counter

event_C_default_ave_norm_lc = event_C_default_ave_lc ./ event_C_default_ave_lc[2]

##########################################
# On labour supply (singles vs. couples) #
##########################################

###########
# Singles #
###########
event_S_labor = zeros(num_events_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_S_labor[counter,:] = panel_S_n[hh_i,(period_i-2):(period_i+event_window)]
    end
end

event_S_labor_ave = sum(event_S_labor,dims=1)./counter

event_S_labor_ave_norm = event_S_labor_ave ./ event_S_labor_ave[2]

###########
# Couples #
###########
event_C_labor_1 = zeros(num_events_C,event_window+3)
event_C_labor_2 = zeros(num_events_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_C_labor_1[counter,:] = panel_C_1_n[hh_i,(period_i-2):(period_i+event_window)]
        event_C_labor_2[counter,:] = panel_C_2_n[hh_i,(period_i-2):(period_i+event_window)]
    end
end

event_C_labor_1_ave = sum(event_C_labor_1,dims=1)./counter
event_C_labor_2_ave = sum(event_C_labor_2,dims=1)./counter

event_C_labor_1_ave_norm = event_C_labor_1_ave ./ event_C_labor_1_ave[2]
event_C_labor_2_ave_norm = event_C_labor_2_ave ./ event_C_labor_2_ave[2]

# On asset choices
# Singles
# event_S_assets = zeros(num_events_S,event_window+3)
#
# counter = 0
#
# for hh_i in 1:num_S, period_i in 2:period_all
#     if (event_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window-1) && (panel_S_d[hh_i,period_i] == 1)
#         counter += 1
#         event_S_assets[counter,:] = parameters.a_grid[panel_S_a[hh_i,(period_i-1):(period_i+event_window+1)]]
#     end
# end
#
# event_S_assets_ave = sum(event_S_assets,dims=1)./counter
#
# event_S_assets_ave_norm = event_S_assets_ave ./ event_S_assets_ave[2]

# Couples

################################################################################
# Effect of transitory shock on default rate
η_shock = 1
# Singles
d_grid = [0.0 1.0]
# Singles
event_η_S = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_η_S[hh_i,period_i] = (shock_S_η[hh_i,period_i] == η_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_η_S =  Int(sum(event_η_S[:,3:(period_all-event_window)]))

event_η_S_default = zeros(num_events_η_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_η_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_η_S_default[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_η_S_default_ave = sum(event_η_S_default,dims=1)./counter

event_η_S_default_ave_norm = event_η_S_default_ave ./ event_η_S_default_ave[2]

# Couples
event_η_C = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_η_C[hh_i,period_i] = (shock_C_η_1[hh_i,period_i] == η_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_η_C =  Int(sum(event_η_C[:,3:(period_all-event_window)]))

event_η_C_default = zeros(num_events_η_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_η_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_η_C_default[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_η_C_default_ave = sum(event_η_C_default,dims=1)./counter

event_η_C_default_ave_norm = event_η_C_default_ave ./ event_η_C_default_ave[2]

################################################################################
# Effect of expense shock on default rate
κ_shock = 2
d_grid = [0.0 1.0]
# Singles
event_κ_S = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_κ_S[hh_i,period_i] = (shock_S_κ[hh_i,period_i] == κ_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_κ_S =  Int(sum(event_κ_S[:,3:(period_all-event_window)]))

event_κ_S_default = zeros(num_events_κ_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_κ_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_S_default[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_S_default_ave = sum(event_κ_S_default,dims=1)./counter

event_κ_S_default_ave_norm = event_κ_S_default_ave ./ event_κ_S_default_ave[2]

event_κ_S_default_ave_norm_sum = zeros(3)
event_κ_S_default_ave_norm_sum[1] = event_κ_S_default_ave[2] ./ event_κ_S_default_ave[2]
event_κ_S_default_ave_norm_sum[2] = event_κ_S_default_ave[3] ./ event_κ_S_default_ave[2]
event_κ_S_default_ave_norm_sum[3] = (event_κ_S_default_ave[3]+event_κ_S_default_ave[4]) ./ event_κ_S_default_ave[2]

# Couples
event_κ_1_C = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_κ_1_C[hh_i,period_i] = (shock_C_κ_1[hh_i,period_i] == κ_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_κ_1_C =  Int(sum(event_κ_1_C[:,3:(period_all-event_window)]))

event_κ_1_C_default = zeros(num_events_κ_1_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_κ_1_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_1_C_default[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_1_C_default_ave = sum(event_κ_1_C_default,dims=1)./counter

event_κ_1_C_default_ave_norm = event_κ_1_C_default_ave ./ event_κ_1_C_default_ave[2]

event_κ_1_C_default_ave_norm_sum = zeros(3)
event_κ_1_C_default_ave_norm_sum[1] = event_κ_1_C_default_ave[2] ./ event_κ_1_C_default_ave[2]
event_κ_1_C_default_ave_norm_sum[2] = event_κ_1_C_default_ave[3] ./ event_κ_1_C_default_ave[2]
event_κ_1_C_default_ave_norm_sum[3] = (event_κ_1_C_default_ave[3]+event_κ_1_C_default_ave[4]) ./ event_κ_1_C_default_ave[2]

# On labour supply (singles vs. couples)
# Singles
event_κ_S_labor = zeros(num_events_κ_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_κ_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_S_labor[counter,:] = parameters.n_grid[panel_S_n[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_S_labor_ave = sum(event_κ_S_labor,dims=1)./counter

event_κ_S_labor_ave_norm = event_κ_S_labor_ave ./ event_κ_S_labor_ave[2]

# Couples
event_κ_1_C_labor_1 = zeros(num_events_κ_1_C,event_window+3)
event_κ_1_C_labor_2 = zeros(num_events_κ_1_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_κ_1_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_1_C_labor_1[counter,:] = parameters.n_grid[panel_C_1_n[hh_i,(period_i-2):(period_i+event_window)]]
        event_κ_1_C_labor_2[counter,:] = parameters.n_grid[panel_C_2_n[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_1_C_labor_1_ave = sum(event_κ_1_C_labor_1,dims=1)./counter
event_κ_1_C_labor_2_ave = sum(event_κ_1_C_labor_2,dims=1)./counter

event_κ_1_C_labor_ave_sum = (event_κ_1_C_labor_1_ave + event_κ_1_C_labor_2_ave) ./ 2

event_κ_1_C_labor_1_ave_norm = event_κ_1_C_labor_1_ave ./ event_κ_1_C_labor_1_ave[2]
event_κ_1_C_labor_2_ave_norm = event_κ_1_C_labor_2_ave ./ event_κ_1_C_labor_2_ave[2]

event_κ_1_C_labor_ave_sum_norm = event_κ_1_C_labor_ave_sum ./ event_κ_1_C_labor_ave_sum[2]

# Effect of default on labor supply
# Singles


#=
# Effect of partial default on type score
event_PD = findall(panel_default .== 2)
event_PD_score = []

for event_i in 1:length(event_PD)
    hh_i = event_PD[event_i][1]
    period_i = event_PD[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            append!(event_PD_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
        end
    end
end

event_PD_score = reshape(event_PD_score, (3,:))
event_PD_score_mean = sum(event_PD_score,dims=2) ./ size(event_PD_score)[2]

# Effect of partial default on type score conditional on high earners
event_PD_high_e = findall(panel_default .== 2)
event_PD_high_e_score = []

for event_i in 1:length(event_PD_high_e)
    hh_i = event_PD_high_e[event_i][1]
    period_i = event_PD_high_e[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            if shock_e[hh_i,period_i] == 3
                append!(event_PD_high_e_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
            end
        end
    end
end

event_PD_high_e_score = reshape(event_PD_high_e_score, (3,:))
event_PD_high_e_score_mean = sum(event_PD_high_e_score,dims=2) ./ size(event_PD_high_e_score)[2]

# Effect of partial default on type score conditional on HHs with both bank and payday loans
event_PD_both_loans = findall(panel_default .== 2)
event_PD_both_loans_score = []

for event_i in 1:length(event_PD_both_loans)
    hh_i = event_PD_both_loans[event_i][1]
    period_i = event_PD_both_loans[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            if (panel_asset[hh_i,period_i] .< findfirst(parameters.asset_ind[:,1] .== parameters.a_ind_zero)) .& (isodd.(panel_asset[hh_i,period_i]))
                append!(event_PD_both_loans_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
            end
        end
    end
end

event_PD_both_loans_score = reshape(event_PD_both_loans_score, (3,:))
event_PD_both_loans_score_mean = sum(event_PD_both_loans_score,dims=2) ./ size(event_PD_both_loans_score)[2]

# Effect of full default on type score
event_FD = findall(panel_default .== 1)
event_FD_score = []

for event_i in 1:length(event_FD)
    hh_i = event_FD[event_i][1]
    period_i = event_FD[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            append!(event_FD_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
        end
    end
end

event_FD_score = reshape(event_FD_score, (3,:))
event_FD_score_mean = sum(event_FD_score,dims=2) ./ size(event_FD_score)[2]

# Effect on type score of switching from saving to borrowing
event_switch_score = []
asset_zero = findlast(parameters.asset_ind[:,1] .== parameters.a_ind_zero)

for hh_i in 1:num_hh
    for period_i in 1:size(panel_asset)[2]-1
        if (panel_asset[hh_i,period_i] > asset_zero) & (panel_asset[hh_i,period_i+1] < asset_zero-1)
            if all(shock_ρ[hh_i,period_i:period_i+1] .== 1)
                append!(event_switch_score, parameters.s_grid[panel_score[hh_i,period_i:period_i+1]])
            end
        end
    end
end

event_switch_score = reshape(event_switch_score, (2,:))
event_switch_score_mean = sum(event_switch_score,dims=2) ./ size(event_switch_score)[2]

# Effect on type score of switching from high e to low e
event_switch_earnings_score = []

for hh_i in 1:num_hh
    for period_i in 1:size(panel_asset)[2]-1
        if (shock_e[hh_i,period_i] - shock_e[hh_i,period_i+1]) > 0
            if all(shock_ρ[hh_i,period_i:period_i+1] .== 1)
                append!(event_switch_earnings_score, parameters.s_grid[panel_score[hh_i,period_i:period_i+1]])
            end
        end
    end
end

event_switch_earnings_score = reshape(event_switch_earnings_score, (2,:))
event_switch_earnings_score_mean = sum(event_switch_earnings_score,dims=2) ./ size(event_switch_earnings_score)[2]

# Effect on type score using both loans to borrow vs. using only bank loans (cond. on total borrowing amount and HH state)
event_window = 1

#=
β_i_target = 2
z_i_target = 1
e_i_target = 1
# asset_i_target = parameters.a_ind_zero*parameters.b_size
asset_i_target = 99
s_i_target = 15
=#
e_i_target = 2
tb_i_target = parameters.a_ind_zero-2

# use both loans
a_both_i = tb_i_target + 1
asset_both_i = (a_both_i-1)*parameters.b_size + 1
event_both_loans = findall(panel_asset .== asset_both_i)

# use only bank loans
a_bank_i = tb_i_target
asset_bank_i = a_bank_i*parameters.b_size
event_bank_loans = findall(panel_asset .== asset_bank_i)

# event containers
event_both_loans_score = []
event_bank_loans_score = []

for event_i in 1:length(event_both_loans)

    hh_i = event_both_loans[event_i][1]
    period_i = event_both_loans[event_i][2]
    period_i = period_i - 1

    if event_window < period_i < (period_all-event_window)
        #=
        β_i = shock_β[hh_i,period_i-1]
        z_i = shock_z[hh_i,period_i-1]
        e_i = shock_e[hh_i,period_i-1]
        asset_i = panel_asset[hh_i,period_i-1]
        s_i = panel_score[hh_i,period_i-1]
        =#
        # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target)  && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

        e_i = shock_e[hh_i,period_i]

        if (e_i == e_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
            append!(event_both_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
        end
    end
end

event_both_loans_score = reshape(event_both_loans_score,2*event_window+1,:)

event_both_loans_score_ave = sum(event_both_loans_score,dims=2)/size(event_both_loans_score)[2]

for event_i in 1:length(event_bank_loans)
    hh_i = event_bank_loans[event_i][1]
    period_i = event_bank_loans[event_i][2]
    period_i = period_i - 1

    if event_window < period_i < (period_all-event_window)

        #=
        β_i = shock_β[hh_i,period_i-1]
        z_i = shock_z[hh_i,period_i-1]
        e_i = shock_e[hh_i,period_i-1]
        asset_i = panel_asset[hh_i,period_i-1]
        s_i = panel_score[hh_i,period_i-1]
        =#

        # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

        e_i = shock_e[hh_i,period_i]

        if (e_i == e_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
            append!(event_bank_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
        end
    end
end

event_bank_loans_score = reshape(event_bank_loans_score,2*event_window+1,:)

event_bank_loans_score_ave = sum(event_bank_loans_score,dims=2)/size(event_bank_loans_score)[2]

################################################################################
# Across all possible tb_i

#e_i_target = 2

# event containers
event_both_loans_score = []
event_bank_loans_score = []

for tb_i_target in 1:(parameters.a_size_neg-1)

    # use both loans
    a_both_i = tb_i_target + 1
    asset_both_i = (a_both_i-1)*parameters.b_size + 1
    event_both_loans = findall(panel_asset .== asset_both_i)

    for event_i in 1:length(event_both_loans)

        hh_i = event_both_loans[event_i][1]
        period_i = event_both_loans[event_i][2]
        period_i = period_i - 1

        if event_window < period_i < (period_all-event_window)
            #=
            β_i = shock_β[hh_i,period_i-1]
            z_i = shock_z[hh_i,period_i-1]
            e_i = shock_e[hh_i,period_i-1]
            asset_i = panel_asset[hh_i,period_i-1]
            s_i = panel_score[hh_i,period_i-1]
            =#
            # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target)  && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

            # e_i = shock_e[hh_i,period_i]

            if all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
                append!(event_both_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
            end
        end
    end
end

event_both_loans_score = reshape(event_both_loans_score,2*event_window+1,:)

event_both_loans_score_ave = sum(event_both_loans_score,dims=2)/size(event_both_loans_score)[2]

for tb_i_target in 1:(parameters.a_size_neg-1)

    # use only bank loans
    a_bank_i = tb_i_target
    asset_bank_i = a_bank_i*parameters.b_size
    event_bank_loans = findall(panel_asset .== asset_bank_i)

    for event_i in 1:length(event_bank_loans)

        hh_i = event_bank_loans[event_i][1]
        period_i = event_bank_loans[event_i][2]
        period_i = period_i - 1

        if event_window < period_i < (period_all-event_window)

            #=
            β_i = shock_β[hh_i,period_i-1]
            z_i = shock_z[hh_i,period_i-1]
            e_i = shock_e[hh_i,period_i-1]
            asset_i = panel_asset[hh_i,period_i-1]
            s_i = panel_score[hh_i,period_i-1]
            =#

            # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

            # e_i = shock_e[hh_i,period_i]

            if all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
                append!(event_bank_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
            end
        end
    end
end

event_bank_loans_score = reshape(event_bank_loans_score,2*event_window+1,:)

event_bank_loans_score_ave = sum(event_bank_loans_score,dims=2)/size(event_bank_loans_score)[2]


################################################################################

using Plots
plot(event_both_loans_score_ave, lc = :blue, legend = :topleft)
plot!(event_bank_loans_score_ave, lc = :red)

mid_point = event_window+1

both_normalized = ((event_both_loans_score_ave.-event_both_loans_score_ave[mid_point])./event_both_loans_score_ave[mid_point])*100
bank_normalized = ((event_bank_loans_score_ave.-event_bank_loans_score_ave[mid_point])./event_bank_loans_score_ave[mid_point])*100
plot(both_normalized, lc = :blue, legend=:topleft)
plot!(bank_normalized, lc = :red)

both_normalized = ((event_both_loans_score_ave.-event_both_loans_score_ave[mid_point]))
bank_normalized = ((event_bank_loans_score_ave.-event_bank_loans_score_ave[mid_point]))
plot(both_normalized, lc = :blue, label = "both", legend=:topleft)
plot!(bank_normalized, lc = :red, label = "bank")








# reference
# σ = ones(action_size, β_size, z_size, e_size, asset_size, s_size) ./ action_size
# Q_s = zeros(s_size, action_a_size, e_size, a_size, s_size)
# μ = ones(β_size, z_size, e_size, asset_size, s_size)

# containers
# effect_both = zeros(parameters.a_size_neg-1,parameters.β_size,parameters.z_size,parameters.e_size,parameters.asset_size,parameters.s_size)
# effect_bank = zeros(parameters.a_size_neg-1,parameters.β_size,parameters.z_size,parameters.e_size,parameters.asset_size,parameters.s_size)
effect_diff = zeros(parameters.a_size_neg-1,parameters.β_size,parameters.z_size,parameters.e_size,parameters.asset_size,parameters.s_size)

test_num = 0.0
test_den = 0.0

test_2_num = 0.0
test_2_den = 0.0

for s_i in 1:parameters.s_size
    for asset_i in 1:parameters.asset_size, e_i in 1:parameters.e_size, z_i in 1:parameters.z_size, β_i in 1:parameters.β_size

        @inbounds @views a_i, b_i = parameters.asset_ind[asset_i,:]

        for tb_i in 1:(parameters.a_size_neg-1)
        # for tb_i in (parameters.a_size_neg-2):(parameters.a_size_neg-1)

            # use both loans
            a_p_i_both = tb_i + 1
            action_a_i_both = 1 + a_p_i_both
            action_i_both = 1 + parameters.a_size + (a_p_i_both-1)*parameters.b_size + 1
            @inbounds @views s_p_both = sum(parameters.s_grid.*Q_s[:,action_a_i_both,e_i,a_i,s_i])
            # effect_both[tb_i,β_i,z_i,e_i,asset_i,s_i] = s_p_both

            # use only bank loans
            a_p_i_bank = tb_i
            action_a_i_bank = 1 + a_p_i_bank
            action_i_bank = 1 + parameters.a_size + a_p_i_bank*parameters.b_size
            @inbounds @views s_p_bank = sum(parameters.s_grid.*Q_s[:,action_a_i_bank,e_i,a_i,s_i])
            # effect_bank[tb_i,β_i,z_i,e_i,asset_i,s_i] = s_p_bank

            if (σ[action_i_both,β_i,z_i,e_i,asset_i,s_i] > 0.0) && (σ[action_i_bank,β_i,z_i,e_i,asset_i,s_i] > 0.0)
                @inbounds effect_diff[tb_i,β_i,z_i,e_i,asset_i,s_i] = ((s_p_both-s_p_bank)/s_p_bank)*100

                test_num += ((s_p_both-s_p_bank)/s_p_bank)*σ[action_i_bank,β_i,z_i,e_i,asset_i,s_i]*μ[β_i,z_i,e_i,asset_i,s_i]
                test_den += σ[action_i_bank,β_i,z_i,e_i,asset_i,s_i]*μ[β_i,z_i,e_i,asset_i,s_i]

                test_2_num += ((s_p_both-s_p_bank)/s_p_bank)
                test_2_den += 1
            end
        end
    end
end

test_num/test_den*100

test_2_num/test_2_den*100

effect_tb = zeros(parameters.a_size_neg-1)

for tb_i in 1:(parameters.a_size_neg-1)
    effect_diff_tb = effect_diff[tb_i,:,:,:,:,:]
    effect_tb[tb_i] = sum(effect_diff_tb .* μ)
end

for s_i in 1:parameters.s_size
    for asset_i in 1:parameters.asset_size, e_i in 1:parameters.e_size, z_i in 1:parameters.z_size, β_i in 1:parameters.β_size

        for tb_i in 1:(parameters.a_size_neg-1)

        end
    end
end
=#
