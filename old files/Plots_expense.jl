# Plots

using Plots
using LaTeXStrings
using Measures

path = "C:/Users/JanSun/Dropbox/Bankruptcy-Family/Results/24/"

# Pricing functions
# Singles
plot_q_S_e = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,1],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a'", title="\$ \\textrm{Pricing Schedule for Singles } q_S(a',z)\$")
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,2], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,3], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,4], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,5], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_q_S_e, string(path,"plot_q_S_e.pdf"))

# Couples
plot_q_C_e = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,1,1],legend=:bottomright,label="\$ z_1,z_2=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a'", title="\$ \\textrm{Pricing Schedule for Couples } q_C(a',z_1,z_2)\$")
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,2,2], label="\$ z_1,z_2=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,3,3], label="\$ z_1,z_2=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,4,4], label="\$ z_1,z_2=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,5,5], label="\$ z_1,z_2=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_q_C_e, string(path,"plot_q_C_e.pdf"))

# Across marital status
z_ind = 3
plot_q_S_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_q_S_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,z_ind],legend=:topleft,label="Singles", lw = 3, lc= :blue, xlabel = L"a'", title="\$ \\textrm{Pricing Schedule across Marital Status } q_S(a',z=$(round(parameters.z_grid[z_ind],digits=2))) \\textrm{ vs. } q_C(a',z=$(round(parameters.z_grid[z_ind],digits=2)),z=$(round(parameters.z_grid[z_ind],digits=2)))\$")
plot_q_S_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,z_ind,z_ind], label="Couples",lw = 3, lc = :red)

savefig(plot_q_S_C, string(path,"plot_q_S_C.pdf"))

# Labor supply
# Singles
# Across assets
η_ind = 2
κ_ind = 1
plot_n_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(n_S_i[:,1,η_ind,κ_ind])],legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Labor supply choice for Singles } n_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,2,η_ind])], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(n_S_i[:,3,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,4,η_ind])], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(n_S_i[:,5,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_n_S, string(path,"plot_n_S.pdf"))

# Couples


# Asset choices
# Singles
# Across assets
η_ind = 2
κ_ind = 1
plot_a_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,1,η_ind,κ_ind])],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Asset choice for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_a_S = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,3,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,5,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S, string(path,"plot_a_S.pdf"))

# Couples
η_ind = 2
κ_ind = 1
plot_a_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,1,1,η_ind,η_ind,κ_ind])],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Asset choice for Couples } a_C(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,3,3,η_ind,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,5,5,η_ind,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_C, string(path,"plot_a_C.pdf"))

# Default decision
# Singles
η_ind = 2
κ_ind = 1
plot_d_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,1,η_ind,κ_ind],legend=:left,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Default Decision for Singles } d_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\kappa=$(round(parameters.κ_grid[κ_ind],digits=2))) \$")
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,3,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,5,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_d_S, string(path,"plot_d_S.pdf"))

# Couples
η_ind = 2
κ_ind = 1
plot_d_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,1,1,η_ind,η_ind,κ_ind],legend=:left,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Default Decision for Couples } d_C(a,z,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\kappa=$(round(parameters.κ_grid[κ_ind],digits=2))) \$")
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,3,3,η_ind,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,5,5,η_ind,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_d_C, string(path,"plot_d_C.pdf"))

############################
#       Moments            #
############################
plot_moment_borrow = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_moment_borrow = plot!(parameters.z_grid,fraction_S_loan_users_z, lw = 3, lc= :blue, xlabel = L"z", title="\$ \\textrm{Fraction of Borrowers Across } z \$", legend=:topright, label="Singles", ylabel = L"\%")
plot_d_C = plot!(parameters.z_grid,fraction_C_loan_users_z_reduced, label="Couples",lw = 3, lc = :red)

savefig(plot_moment_borrow, string(path,"plot_moment_borrow.pdf"))

############################
#       Event Studies      #
############################

x = [-1, 0, 1, 2]

# Z event
# Default probability
plot_event_default = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_default = plot!(x,event_S_default_ave_norm[2:5], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Fraction of Households Defaulting around Event } z_3 \\textrm{ to } z_1 \$", legend=:topleft, label="Singles", ylim=(0.0, 9.0), yticks = (1.0:2.0:9.0))
plot_event_default = plot!(x,event_C_default_ave_norm[2:5], label="Couples",lw = 3, lc = :red)

savefig(plot_event_default, string(path,"plot_event_default.pdf"))

# Labor
plot_event_labor = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_labor = plot!(x,event_S_labor_ave_norm[2:5], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Labor Supply at Event } z_3 \\textrm{ to } z_1 \$", legend=:right, label="Singles")
plot_event_labor = plot!(x,event_C_labor_1_ave_norm[2:5], label="Couples - Affected",lw = 3, lc = :red)
plot_event_labor = plot!(x,event_C_labor_2_ave_norm[2:5], label="Couples - Unaffected",lw = 3, lc = :green)

savefig(plot_event_labor, string(path,"plot_event_labor.pdf"))

# η event
plot_event_η_default = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_η_default = plot!(x,event_η_S_default_ave_norm[2:5], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Fraction of Households Defaulting around Event } \\eta_3 \$", legend=:topleft, label="Singles", ylim=(0.0, 5.0), yticks = (1.0:1.0:5.0))
plot_event_η_default = plot!(x,event_η_C_default_ave_norm[2:5], label="Couples",lw = 3, lc = :red)

savefig(plot_event_η_default, string(path,"plot_event_eta_default.pdf"))

# κ event
plot_event_κ_default = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_κ_default = plot!(x,event_κ_S_default_ave_norm[2:5], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Fraction of Households Defaulting around Event } \\kappa_3 \$", legend=:topleft, label="Singles", ylim=(0.0, 2.0), yticks = (0.0:0.2:2.0))
plot_event_κ_default = plot!(x,event_κ_C_default_ave_norm[2:5], label="Couples",lw = 3, lc = :red)

savefig(plot_event_κ_default, string(path,"plot_event_kappa_default.pdf"))
