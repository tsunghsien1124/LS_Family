# Plots

using Plots
using LaTeXStrings
using Measures

path = "C:/Users/JanSun/Dropbox/Bankruptcy-Family/Results/5/"

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
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,1,1],legend=:topleft,label="\$ z_1,z_2=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a'", title="\$ \\textrm{Pricing Schedule for Couples } q_C(a',z_1,z_2)\$")
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
plot_n_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(n_S_i[:,1,η_ind])],legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Labor supply choice for Singles } n_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,2,η_ind])], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(n_S_i[:,3,η_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,4,η_ind])], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(n_S_i[:,5,η_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_n_S, string(path,"plot_n_S.pdf"))

# Couples


# Asset choices
# Singles
# Across assets
η_ind = 2
plot_a_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,1,η_ind])],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Asset choice for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_a_S = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,3,η_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,5,η_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S, string(path,"plot_a_S.pdf"))

# Couples
η_ind = 2
plot_a_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,1,1,η_ind,η_ind])],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Asset choice for Couples } a_C(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,3,3,η_ind,η_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,5,5,η_ind,η_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_C, string(path,"plot_a_C.pdf"))

# Default decision
# Singles
η_ind = 2
plot_d_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,1,η_ind],legend=:left,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Default Decision for Singles } d_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,3,η_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,5,η_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_d_S, string(path,"plot_d_S.pdf"))

# Couples
η_ind = 2
plot_d_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,1,1,η_ind,η_ind],legend=:left,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Default Decision for Couples } d_C(a,z,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,3,3,η_ind,η_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,5,5,η_ind,η_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_d_C, string(path,"plot_d_C.pdf"))
