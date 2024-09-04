# Plots
using Plots
using LaTeXStrings
using Measures

using Interpolations
using Dierckx

path = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value/"
path = "C:/Users/JanSun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value/"

path = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_kappa_0/"
path = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_no_bankruptcy/"

a_grid_plot = round.(parameters.a_grid, digits=2)
z_grid_plot = round.(parameters.z_grid, digits=2)
η_grid_plot = round.(parameters.η_grid, digits=2)

#====================#
# Pricing functions  #
#====================#
age = 5
gender = 2
# Singles
plot_q_S_e = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,1,age,gender],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1,gender],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a'", title=L"\textrm{Pricing\ Schedule\ for\ Singles\ } q_S(a',z)")
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,3,age,gender], label="\$ z=$(round(parameters.z_grid[3,gender],digits=2)) \$",lw = 3, lc = :red)
plot_q_S_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_S[1:parameters.a_ind_zero,5,age,gender], label="\$ z=$(round(parameters.z_grid[5,gender],digits=2)) \$",lw = 3, lc = :green)

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
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,1,1,age],legend=:topleft,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a'", title="\$ \\textrm{Pricing Schedule for Couples } q_C(a',z_1,z_2)\$")
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,2,2,age], label="\$ (z_1,z_2)=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_q_C_e = plot!(parameters.a_grid[1:parameters.a_ind_zero],q_C[1:parameters.a_ind_zero,3,3,age], label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)

savefig(plot_q_C_e, string(path,"plot_q_C_e.pdf"))

# Across marital status
z_ind = 2
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

#=================#
# Value Function  #
#=================#
η_ind = 2
κ_ind = 1

# Singles
plot_W_S_e = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_W_S_e = plot!(parameters.a_grid,W_S[:,1,η_ind,κ_ind],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title=L"\textrm{Value\ Function\ for\ Singles\ } W_S(a,z,\eta,\kappa)")
plot_W_S_e = plot!(parameters.a_grid,W_S[:,2,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_W_S_e = plot!(parameters.a_grid,W_S[:,3,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)

#===============#
# Labor supply  #
#===============#

# Singles

# Across z
σ_S_temp = sum(σ_S,dims=1)
a_p = 1.5
a_ind = parameters.a_ind_zero
z_ind = 2
η_ind = 2

plot_CH_figure_1_lr_S_z = plot(box = :on, size = [800, 500],
                        xtickfont = font(12, "Computer Modern", :black),
                        ytickfont = font(12, "Computer Modern", :black),
                        legendfont = font(14, "Computer Modern", :black),
                        guidefont = font(14, "Computer Modern", :black),
                        titlefont = font(16, "Computer Modern", :black),
                        margin = 4mm
                        )
plot_CH_figure_1_lr_S_z = plot!(parameters.n_grid,
                         σ_S_temp[1,:,a_ind,1,η_ind],
                         title="\$ \\textrm{\\ Choice\\ Probability\\ } \\sigma^{(R,\\cdot,n)} (a=$(a_grid_plot[a_ind]),z,\\eta=$(η_grid_plot[η_ind])) \$",
                         xlabel="\$ \\textrm{Labor\\ choice\\ } n \$",
                         # ylabel=L"\sigma^{(R,b',\cdot)}",
                         label="\$ z=$(z_grid_plot[1]) \$",
                         linecolor = :blue,
                         legend = :topleft,
                         # linestyle = :dot,
                         lw=3)

plot_CH_figure_1_lr_S_z = plot!(parameters.n_grid,
                         σ_S_temp[1,:,a_ind,2,η_ind],
                         label="\$ z=$(z_grid_plot[2]) \$",
                         linecolor = :red,
                         lw=3)

 plot_CH_figure_1_lr_S_z = plot!(parameters.n_grid,
                          σ_S_temp[1,:,a_ind,3,η_ind],
                          label="\$ z=$(z_grid_plot[3]) \$",
                          linecolor = :black,
                          lw=3)

# Old stuff

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
plot_n_S = plot!(parameters.a_grid,n_S_i[:,1,η_ind,κ_ind],legend=:bottomleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Labor supply choice for Singles } n_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,2,η_ind])], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_n_S = plot!(parameters.a_grid,n_S_i[:,3,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,4,η_ind])], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_n_S = plot!(parameters.a_grid,n_S_i[:,5,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_n_S, string(path,"plot_n_S.pdf"))

# Interpolate
η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_S_z_1 = vcat(n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_S_z_3 = vcat(n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_S_z_5 = vcat(n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

plot_n_S_itp = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_n_S_itp = plot!(parameters.a_grid,n_S_z_1,legend=:bottomleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Labor supply choice for Singles } n_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\kappa=$(round(parameters.κ_grid[κ_ind],digits=2))) \$",ylim=(0.2,0.8))
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,2,η_ind])], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_n_S_itp = plot!(parameters.a_grid,n_S_z_3, label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,4,η_ind])], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_n_S_itp = plot!(parameters.a_grid,n_S_z_5, label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_n_S_itp, string(path,"plot_n_S_itp.pdf"))

# Couples
η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1

z_1_ind = 1
z_2_ind = 1
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_C_z_1 = vcat(n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]))

z_1_ind = 3
z_2_ind = 3
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_C_z_3 = vcat(n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]))

z_1_ind = 5
z_2_ind = 5
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_C_z_5 = vcat(n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]))

plot_n_C_itp = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_n_C_itp = plot!(parameters.a_grid,n_C_z_1,legend=:bottomleft,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Labor supply for Couples } n_{C,1}(a,z_1,z_2,\\eta_1=$(round(parameters.η_grid[η_1_ind],digits=2)),\\eta_2=$(round(parameters.η_grid[η_2_ind],digits=2)),\\kappa_1=$(round(parameters.κ_grid[κ_1_ind],digits=2)),\\kappa_2=$(round(parameters.κ_grid[κ_2_ind],digits=2))) \$", ylim=(0.2,0.8))
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,2,η_ind])], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_n_C_itp = plot!(parameters.a_grid,n_C_z_3, label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,4,η_ind])], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_n_C_itp = plot!(parameters.a_grid,n_C_z_5, label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_n_C_itp, string(path,"plot_n_C_itp.pdf"))

# Excess hours worked, Porbably still wrong
η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_S_z_1 = vcat(n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_S_z_3 = vcat(n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_S_z_5 = vcat(n_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1

z_1_ind = 1
z_2_ind = 1
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_C_z_1 = vcat(n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]))

z_1_ind = 3
z_2_ind = 3
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_C_z_3 = vcat(n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]))

z_1_ind = 5
z_2_ind = 5
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind]
spl_S = Spline1D(x,y,k=2,s=0.01)
n_C_z_5 = vcat(n_C_1_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]))

n_z_1 = n_S_z_1 .- n_C_z_1
n_z_3 = n_S_z_3 .- n_C_z_3
n_z_5 = n_S_z_5 .- n_C_z_5

plot_n_C_itp = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_n_C_itp = plot!(parameters.a_grid,n_z_1,legend=:bottomright,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Labor supply for Couples } n_{C,1}(a,z_1,z_2,\\eta_1=$(round(parameters.η_grid[η_1_ind],digits=2)),\\eta_2=$(round(parameters.η_grid[η_2_ind],digits=2)),\\kappa_1=$(round(parameters.κ_grid[κ_1_ind],digits=2)),\\kappa_2=$(round(parameters.κ_grid[κ_2_ind],digits=2))) \$")
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,2,η_ind])], label="\$ z=$(round(parameters.z_grid[2],digits=2)) \$",lw = 3, lc = :red)
plot_n_C_itp = plot!(parameters.a_grid,n_z_3, label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
# plot_n_S = plot!(parameters.a_grid,parameters.n_grid[Int.(variables.n_S_i[:,4,η_ind])], label="\$ z=$(round(parameters.z_grid[4],digits=2)) \$",lw = 3, lc = :purple)
plot_n_C_itp = plot!(parameters.a_grid,n_z_5, label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_n_C_itp, string(path,"plot_n_C_itp.pdf"))

#================#
# Asset choices  #
#================#
# Singles

# Across z
σ_S_temp = sum(σ_S,dims=2)
a_p = 1.5
a_ind = parameters.a_ind_zero
z_ind = 2
η_ind = 2

plot_CH_figure_1_lr_S_z = plot(box = :on, size = [800, 500],
                        xtickfont = font(12, "Computer Modern", :black),
                        ytickfont = font(12, "Computer Modern", :black),
                        legendfont = font(14, "Computer Modern", :black),
                        guidefont = font(14, "Computer Modern", :black),
                        titlefont = font(16, "Computer Modern", :black),
                        margin = 4mm
                        )
plot_CH_figure_1_lr_S_z = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_S_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,a_ind,1,η_ind],
                         title="\$ \\textrm{\\ Choice\\ Probability\\ } \\sigma_S^{(R,a',\\cdot)} (a=$(a_grid_plot[a_ind]),z,\\eta=$(η_grid_plot[η_ind])) \$",
                         xlabel="\$ \\textrm{Asset\\ choice\\ } a' \$",
                         # ylabel=L"\sigma^{(R,b',\cdot)}",
                         label="\$ z=$(z_grid_plot[1]) \$",
                         linecolor = :blue,
                         legend = :topleft,
                         # linestyle = :dot,
                         lw=3)
plot_CH_figure_1_lr_S_z = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_S_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,a_ind,2,η_ind],
                         label="\$ z=$(z_grid_plot[2]) \$",
                         linecolor = :red,
                         lw=3)
 plot_CH_figure_1_lr_S_z = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                          σ_S_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,a_ind,3,η_ind],
                          label="\$ z=$(z_grid_plot[3]) \$",
                          linecolor = :black,
                          lw=3)

# Across η
plot_CH_figure_1_lr_S_eta = plot(box = :on, size = [800, 500],
                        xtickfont = font(12, "Computer Modern", :black),
                        ytickfont = font(12, "Computer Modern", :black),
                        legendfont = font(14, "Computer Modern", :black),
                        guidefont = font(14, "Computer Modern", :black),
                        titlefont = font(16, "Computer Modern", :black),
                        margin = 4mm
                        )
plot_CH_figure_1_lr_S_eta = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_S_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,a_ind,z_ind,1],
                         title="\$ \\textrm{\\ Choice\\ Probability\\ } \\sigma_S^{(R,a',\\cdot)} (a=$(a_grid_plot[a_ind]),z=$(z_grid_plot[z_ind]),\\eta) \$",
                         xlabel="\$ \\textrm{Asset\\ choice\\ } a' \$",
                         # ylabel=L"\sigma^{(R,b',\cdot)}",
                         label="\$ \\eta=$(η_grid_plot[1]) \$",
                         linecolor = :blue,
                         legend = :topleft,
                         # linestyle = :dot,
                         lw=3)
plot_CH_figure_1_lr_S_eta = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_S_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,a_ind,z_ind,2],
                         label="\$ \\eta=$(η_grid_plot[2]) \$",
                         linecolor = :red,
                         lw=3)
 plot_CH_figure_1_lr_S_eta = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                          σ_S_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,a_ind,z_ind,3],
                          label="\$ \\eta=$(η_grid_plot[3]) \$",
                          linecolor = :black,
                          lw=3)

# Couples

# Across z
σ_C_temp = sum(σ_C,dims=[2 3])
a_p = 1.5
a_ind = parameters.a_ind_zero
z_1_ind = 2
z_2_ind = 2
η_1_ind = 2
η_2_ind = 2

plot_CH_figure_1_lr_C_z = plot(box = :on, size = [800, 500],
                        xtickfont = font(12, "Computer Modern", :black),
                        ytickfont = font(12, "Computer Modern", :black),
                        legendfont = font(14, "Computer Modern", :black),
                        guidefont = font(14, "Computer Modern", :black),
                        titlefont = font(16, "Computer Modern", :black),
                        margin = 4mm
                        )
plot_CH_figure_1_lr_C_z = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_C_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,1,a_ind,1,1,η_1_ind,η_2_ind],
                         title="\$ \\textrm{\\ Choice\\ Probability\\ } \\sigma_C^{(R,a',\\cdot)} (a=$(a_grid_plot[a_ind]),z_1,z_2,\\eta_1=$(η_grid_plot[η_1_ind]),\\eta_2=$(η_grid_plot[η_2_ind])) \$",
                         xlabel="\$ \\textrm{Asset\\ choice\\ } a' \$",
                         # ylabel=L"\sigma^{(R,b',\cdot)}",
                         label="\$ z_1,z_2=$(z_grid_plot[1]) \$",
                         linecolor = :blue,
                         legend = :topleft,
                         # linestyle = :dot,
                         lw=3)
plot_CH_figure_1_lr_C_z = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_C_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,1,a_ind,2,2,η_1_ind,η_2_ind],
                         label="\$ z_1,z_2=$(z_grid_plot[2]) \$",
                         linecolor = :red,
                         lw=3)
 plot_CH_figure_1_lr_C_z = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                          σ_C_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,1,a_ind,3,3,η_1_ind,η_2_ind],
                          label="\$ z_1,z_2=$(z_grid_plot[3]) \$",
                          linecolor = :black,
                          lw=3)

# Across η
plot_CH_figure_1_lr_C_eta = plot(box = :on, size = [800, 500],
                        xtickfont = font(12, "Computer Modern", :black),
                        ytickfont = font(12, "Computer Modern", :black),
                        legendfont = font(14, "Computer Modern", :black),
                        guidefont = font(14, "Computer Modern", :black),
                        titlefont = font(16, "Computer Modern", :black),
                        margin = 4mm
                        )
plot_CH_figure_1_lr_C_eta = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_C_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,1,a_ind,z_1_ind,z_2_ind,1,1],
                         title="\$ \\textrm{\\ Choice\\ Probability\\ } \\sigma_C^{(R,a',\\cdot)} (a=$(a_grid_plot[a_ind]),z_1=$(z_grid_plot[z_1_ind]),z_2=$(z_grid_plot[z_2_ind]),\\eta_1,\\eta_2) \$",
                         xlabel="\$ \\textrm{Asset\\ choice\\ } a' \$",
                         # ylabel=L"\sigma^{(R,b',\cdot)}",
                         label="\$ \\eta_1,\\eta_2=$(η_grid_plot[1]) \$",
                         linecolor = :blue,
                         legend = :topleft,
                         # linestyle = :dot,
                         lw=3)
plot_CH_figure_1_lr_C_eta = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                         σ_C_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,1,a_ind,z_1_ind,z_2_ind,2,2],
                         label="\$ \\eta_1,\\eta_2=$(η_grid_plot[2]) \$",
                         linecolor = :red,
                         lw=3)
 plot_CH_figure_1_lr_C_eta = plot!(parameters.a_grid[1:minimum(findall(parameters.a_grid .>= a_p))],
                          σ_C_temp[2:minimum(findall(parameters.a_grid .>= a_p))+1,1,1,a_ind,z_1_ind,z_2_ind,3,3],
                          label="\$ \\eta_1,\\eta_2=$(η_grid_plot[3]) \$",
                          linecolor = :black,
                          lw=3)

# Old stuff

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

# Net savings
η_ind = 2
κ_ind = 1
plot_a_S_net = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_net = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,1,η_ind,κ_ind])].-parameters.a_grid,legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Net Asset Savings for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) - a\$")
plot_a_S_net = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,3,η_ind,κ_ind])].-parameters.a_grid, label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_net = plot!(parameters.a_grid,parameters.a_grid[Int.(a_S_i[:,5,η_ind,κ_ind])].-parameters.a_grid, label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_net, string(path,"plot_a_S_net.pdf"))

# Interpolate
η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_1 = vcat(parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 2], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_3 = vcat(parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 2], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_5 = vcat(parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 2], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

plot_a_S_net_itp = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_net_itp = plot!(parameters.a_grid,a_S_z_1,legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Net Asset Savings for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) - a \$",ylim=(-0.7, 1.7) )
plot_a_S_net_itp = plot!(parameters.a_grid,a_S_z_3, label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_net_itp = plot!(parameters.a_grid,a_S_z_5, label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_net_itp, string(path,"plot_a_S_net_itp.pdf"))

# Without default
Nan_grid = zeros(parameters.a_size)
Nan_grid .= NaN

η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_1 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_3 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_5 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]))

plot_a_S_net_itp_no_d = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_net_itp_no_d = plot!(parameters.a_grid,a_S_z_1,legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Net Asset Savings for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) - a \$",ylim=(-0.7, 1.0) )
plot_a_S_net_itp_no_d = plot!(parameters.a_grid,a_S_z_3, label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_net_itp_no_d = plot!(parameters.a_grid,a_S_z_5, label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_net_itp_no_d, string(path,"plot_a_S_net_itp_no_d.pdf"))

# Interpolate a/2, This is probably not correct yet!!!
η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_1 = vcat((parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 2])./2, spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_3 = vcat((parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 2])./2, spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_5 = vcat((parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 2])./2, spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

plot_a_S_net_itp_a_2 = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_net_itp_a_2 = plot!(parameters.a_grid,a_S_z_1,legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Net Asset Savings for Singles } a_S(a/2,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) - a \$",ylim=(-0.7, 1.7) )
plot_a_S_net_itp_a_2 = plot!(parameters.a_grid,a_S_z_3, label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_net_itp_a_2 = plot!(parameters.a_grid,a_S_z_5, label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_net_itp_a_2, string(path,"plot_a_S_net_itp_a_2.pdf"))

# a/2, no d, This is probably not correct yet!!!
Nan_grid = zeros(parameters.a_size)
Nan_grid .= NaN

η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_1 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_3 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])].-parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_5 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

plot_a_S_net_itp_a_2_no_d = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_net_itp_a_2_no_d = plot!(parameters.a_grid,a_S_z_1,legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Net Asset Savings for Singles } a_S(a/2,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) - a \$",ylim=(-0.7, 1.7) )
plot_a_S_net_itp_a_2_no_d = plot!(parameters.a_grid,a_S_z_3, label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_net_itp_a_2_no_d = plot!(parameters.a_grid,a_S_z_5, label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_net_itp_a_2_no_d, string(path,"plot_a_S_net_itp_a_2_no_d.pdf"))

# Only for positive assets
η_ind = 2
κ_ind = 1
plot_a_S_pos = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_pos = plot!(parameters.a_grid[parameters.a_ind_zero:end],parameters.a_grid[Int.(a_S_i[parameters.a_ind_zero:end,1,η_ind,κ_ind])],legend=:topleft,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Asset choice for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) \$")
plot_a_S_pos = plot!(parameters.a_grid[parameters.a_ind_zero:end],parameters.a_grid[Int.(a_S_i[parameters.a_ind_zero:end,3,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_pos = plot!(parameters.a_grid[parameters.a_ind_zero:end],parameters.a_grid[Int.(a_S_i[parameters.a_ind_zero:end,5,η_ind,κ_ind])], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_pos, string(path,"plot_a_S_pos.pdf"))

# Net savings
η_ind = 2
κ_ind = 1
plot_a_S_pos_net = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_S_pos_net = plot!(parameters.a_grid[parameters.a_ind_zero:end],parameters.a_grid[Int.(a_S_i[parameters.a_ind_zero:end,1,η_ind,κ_ind])].-parameters.a_grid[parameters.a_ind_zero:end],legend=:topright,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Net Asset Savings for Singles } a_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2))) - a \$")
plot_a_S_pos_net = plot!(parameters.a_grid[parameters.a_ind_zero:end],parameters.a_grid[Int.(a_S_i[parameters.a_ind_zero:end,3,η_ind,κ_ind])].-parameters.a_grid[parameters.a_ind_zero:end], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_S_pos_net = plot!(parameters.a_grid[parameters.a_ind_zero:end],parameters.a_grid[Int.(a_S_i[parameters.a_ind_zero:end,5,η_ind,κ_ind])].-parameters.a_grid[parameters.a_ind_zero:end], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_S_pos_net, string(path,"plot_a_S_pos_net.pdf"))

# # Couples
η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1
plot_a_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,1,1,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])],legend=:topleft,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Asset choice for Couples } a_C(a,z_1,z_2,\\eta_1=$(round(parameters.η_grid[η_1_ind],digits=2)),\\eta_2=$(round(parameters.η_grid[η_2_ind],digits=2))) \$")
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,3,3,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_C = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,5,5,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_C, string(path,"plot_a_C.pdf"))

# Net savings
η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1
plot_a_C_net = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_C_net = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,1,1,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid,legend=:topright,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{ Net Asset Savings for Couples } a_C(a,z_1,z_2,\\eta_1=$(round(parameters.η_grid[η_1_ind],digits=2)),\\eta_2=$(round(parameters.η_grid[η_2_ind],digits=2))) - a \$")
plot_a_C_net = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,3,3,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid, label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_C_net = plot!(parameters.a_grid,parameters.a_grid[Int.(a_C_i[:,5,5,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid, label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_C_net, string(path,"plot_a_C_net.pdf"))

# Interpolated
η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1

z_1_ind = 1
z_2_ind = 1
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_1 = vcat(parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

z_1_ind = 3
z_2_ind = 3
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_3 = vcat(parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

z_1_ind = 5
z_2_ind = 5
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_5 = vcat(parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

plot_a_C_net_itp = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_C_net_itp = plot!(parameters.a_grid,a_C_z_1,legend=:topright,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{ Net Asset Savings for Couples } a_C(a,z_1,z_2,\\eta_1=$(round(parameters.η_grid[η_1_ind],digits=2)),\\eta_2=$(round(parameters.η_grid[η_2_ind],digits=2))) - a \$",ylim=(-0.7, 1.7))
plot_a_C_net_itp = plot!(parameters.a_grid,a_C_z_3, label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_C_net_itp = plot!(parameters.a_grid,a_C_z_5, label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_C_net_itp, string(path,"plot_a_C_net_itp.pdf"))

# No d
Nan_grid = zeros(parameters.a_size)
Nan_grid .= NaN

η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1

z_1_ind = 1
z_2_ind = 1
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_1 = vcat(Nan_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

z_1_ind = 3
z_2_ind = 3
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_3 = vcat(Nan_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

z_1_ind = 5
z_2_ind = 5
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])].-parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_5 = vcat(Nan_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

plot_a_C_net_itp_no_d = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_C_net_itp_no_d = plot!(parameters.a_grid,a_C_z_1,legend=:topright,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{ Net Asset Savings for Couples } a_C(a,z_1,z_2,\\eta_1=$(round(parameters.η_grid[η_1_ind],digits=2)),\\eta_2=$(round(parameters.η_grid[η_2_ind],digits=2))) - a \$",ylim=(-0.7, 1.0))
plot_a_C_net_itp_no_d = plot!(parameters.a_grid,a_C_z_3, label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_C_net_itp_no_d = plot!(parameters.a_grid,a_C_z_5, label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_C_net_itp_no_d, string(path,"plot_a_C_net_itp_no_d.pdf"))

# Excess Savings
Nan_grid = zeros(parameters.a_size)
Nan_grid .= NaN

η_ind = 2
κ_ind = 1

z_ind = 1
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_1 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 3
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_3 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

z_ind = 5
x = parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]
y = parameters.a_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 1,z_ind,η_ind,κ_ind])]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_S_z_5 = vcat(Nan_grid[Int.(a_S_i[d_S_i[:,z_ind,η_ind,κ_ind] .== 2,z_ind,η_ind,κ_ind])], spl_S(parameters.a_grid[d_S_i[:,z_ind,η_ind,κ_ind] .== 1]./2))

η_1_ind = 2
η_2_ind = 2
κ_1_ind = 1
κ_2_ind = 1

z_1_ind = 1
z_2_ind = 1
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_1 = vcat(Nan_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

z_1_ind = 3
z_2_ind = 3
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_3 = vcat(Nan_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

z_1_ind = 5
z_2_ind = 5
x = parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1]
y = parameters.a_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 1,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])]
spl_S = Spline1D(x,y,k=3,s=0.1)
a_C_z_5 = vcat(Nan_grid[Int.(a_C_i[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind] .== 2,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind])], spl_S(parameters.a_grid[d_C_i[:,z_1_ind,z_2_ind,η_1_ind,η_2_ind,κ_1_ind,κ_2_ind].== 1]))

a_z_1 = 2*a_S_z_1 .- a_C_z_1
a_z_3 = 2*a_S_z_3 .- a_C_z_3
a_z_5 = 2*a_S_z_5 .- a_C_z_5

plot_a_net_itp_no_d = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_a_net_itp_no_d = plot!(parameters.a_grid,a_z_1,legend=:topleft,label="\$ (z_1,z_2)=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", title="\$ \\textrm{Excess Savings for Singles vs. Couples } 2*a_S(a/2,\\cdot) - a_C(a,\\cdot) \$")
plot_a_net_itp_no_d = plot!(parameters.a_grid,a_z_3, label="\$ (z_1,z_2)=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_a_net_itp_no_d = plot!(parameters.a_grid,a_z_5, label="\$ (z_1,z_2)=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_a_net_itp_no_d, string(path,"plot_a_net_itp_no_d.pdf"))

#==================#
# Default decision #
#==================#
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
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,1,η_ind,κ_ind],legend=:left,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, xlabel = L"a", yticks = ((1:2)), title="\$ \\textrm{Default Decision for Singles } d_S(a,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\kappa=$(round(parameters.κ_grid[κ_ind],digits=2))) \$")
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,3,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_d_S = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_S_i[1:parameters.a_ind_zero,5,η_ind,κ_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_d_S, string(path,"plot_d_S.pdf"))

# Couples
η_ind = 2
κ_1_ind = 1
κ_2_ind = 1
plot_d_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,1,1,η_ind,η_ind,κ_1_ind,κ_2_ind],legend=:left,label="\$ z=$(round(parameters.z_grid[1],digits=2)) \$", lw = 3, lc= :blue, yticks = ((1:2)), xlabel = L"a", title="\$ \\textrm{Default Decision for Couples } d_C(a,z,z,\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\eta=$(round(parameters.η_grid[η_ind],digits=2)),\\kappa_1=$(round(parameters.κ_grid[κ_1_ind],digits=2)),\\kappa_2=$(round(parameters.κ_grid[κ_2_ind],digits=2))) \$")
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,3,3,η_ind,η_ind,κ_1_ind,κ_2_ind], label="\$ z=$(round(parameters.z_grid[3],digits=2)) \$",lw = 3, lc = :green)
plot_d_C = plot!(parameters.a_grid[1:parameters.a_ind_zero],d_C_i[1:parameters.a_ind_zero,5,5,η_ind,η_ind,κ_1_ind,κ_2_ind], label="\$ z=$(round(parameters.z_grid[5],digits=2)) \$",lw = 3, lc = :black)

savefig(plot_d_C, string(path,"plot_d_C.pdf"))

############################
#       Moments            #
############################
# Borrowing
plot_moment_borrow = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_moment_borrow = plot!(parameters.z_grid,fraction_S_loan_users_z, lw = 3, lc= :blue, xlabel = L"z", title="\$ \\textrm{Fraction of Borrowers Across } z \$", legend=:topright, label="Singles", ylabel = L"\%")
plot_moment_borrow = plot!(parameters.z_grid,fraction_C_loan_users_z_reduced, label="Couples",lw = 3, lc = :red)

savefig(plot_moment_borrow, string(path,"plot_moment_borrow.pdf"))

# Defaulting
plot_moment_default = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_moment_default = plot!(parameters.z_grid,fraction_S_default_z, lw = 3, lc= :blue, xlabel = L"z", title="\$ \\textrm{Fraction of Defaulters Across } z \$", legend=:topright, label="Singles", ylabel = L"\%")
plot_moment_default = plot!(parameters.z_grid,fraction_C_default_z_reduced, label="Couples",lw = 3, lc = :red)

savefig(plot_moment_default, string(path,"plot_moment_default.pdf"))

############################
#       Event Studies      #
############################

x = [-2, -1, 0, 1, 2]

# Divorce event
# Consumption response
plot_divorce_event_consumption = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_consumption = plot!(x,divorce_event_consumption, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Mean Per Capita Consumption Around Divorce Event}", legend=:topright, xticks = (-2:1:2), label = "Benchmark")
plot_divorce_event_consumption = plot!(x,divorce_event_consumption_κ_0, lw = 3, lc= :red, label = "No divorce cost shock")

savefig(plot_divorce_event_consumption, string(path,"plot_divorce_event_consumption.pdf"))

# Asset response
plot_divorce_event_assets = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_assets = plot!(x,divorce_event_assets, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Mean Per Capita Assets Around Divorce Event}", legend=:topleft, xticks = (-2:1:2), label = "Benchmark")
plot_divorce_event_assets = plot!(x,divorce_event_assets_κ_0, lw = 3, lc= :red, label = "No divorce cost shock")

savefig(plot_divorce_event_assets, string(path,"plot_divorce_event_assets.pdf"))

# Asset response (cond. on no default)
plot_divorce_event_assets_no_default = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_assets_no_default = plot!(x,divorce_event_assets_no_default, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Asset\ Dynamics\ Around\ Divorce\ Event\ (cond.\ on\ no\ default)}", legend=:none, xticks = (-2:1:2))

savefig(plot_divorce_event_assets_no_default, string(path,"plot_divorce_event_assets_no_default.pdf"))

# Fraction of borrowers
plot_divorce_event_fraction_borrow = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_fraction_borrow = plot!(x,divorce_event_fraction_borrow, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Fraction\ of\ Borrowers\ Around\ Divorce\ Event}", legend=:topright, xticks = (-2:1:2), label = "Benchmark")
plot_divorce_event_fraction_borrow = plot!(x,divorce_event_fraction_borrow_κ_0, lw = 3, lc= :red, label = "No divorce cost shock")

savefig(plot_divorce_event_fraction_borrow, string(path,"plot_divorce_event_fraction_borrow.pdf"))

# Fraction of defaulters
plot_divorce_event_default = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_default = plot!(x,divorce_event_default, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Fraction\ of\ Defaulters\ Around\ Divorce\ Event}", legend=:topleft, xticks = (-2:1:2), label = "Benchmark")
plot_divorce_event_default = plot!(x,divorce_event_default_κ_0, lw = 3, lc= :red, label = "No divorce cost shock")

savefig(plot_divorce_event_default, string(path,"plot_divorce_event_default.pdf"))

# Fraction of defaulters (only κ = 0)
plot_divorce_event_default = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_default = plot!(x,divorce_event_default_κ_0, lw = 3, lc= :red, xlabel = "Event Time", title=L" \textrm{Fraction\ of\ Defaulters\ Around\ Divorce\ Event}", legend=:topleft, xticks = (-2:1:2), label = "No divorce cost shock")

savefig(plot_divorce_event_default, string(path,"plot_divorce_event_default_kappa_0.pdf"))

# Mean labor supply
plot_divorce_event_labor = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )
plot_divorce_event_labor = plot!(x,divorce_event_labor, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Mean\ Labor\ Supply\ Around\ Divorce\ Event}", legend=:topright, xticks = (-2:1:2), label = "Benchmark")
plot_divorce_event_labor = plot!(x,divorce_event_labor_κ_0, lw = 3, lc= :red, label = "No divorce cost shock")

savefig(plot_divorce_event_labor, string(path,"plot_divorce_event_labor.pdf"))


# Old code

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
plot_event_default = plot!(x,event_S_default_ave_norm[2:4], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Fraction of Households Defaulting around Event } z_3 \\textrm{ to } z_2 \$", legend=:topleft, label="Singles", ylim=(0.0, 5.0), xticks = (-1:1:1))
plot_event_default = plot!(x,event_C_default_ave_norm[2:4], label="Couples",lw = 3, lc = :red)

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
plot_event_labor = plot!(x,event_S_labor_ave_norm[2:4], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Average Labor Supply at Event } z_3 \\textrm{ to } z_2 \$", legend=:right, label="Singles", xticks = (-1:1:1))
plot_event_labor = plot!(x,event_C_labor_1_ave_norm[2:4], label="Couples - Affected Individual",lw = 3, lc = :red)
plot_event_labor = plot!(x,event_C_labor_2_ave_norm[2:4], label="Couples - Unaffected Individual",lw = 3, lc = :green)

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
plot_event_η_default = plot!(x,event_η_S_default_ave_norm[2:4], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Fraction of Households Defaulting around Event } \\eta_1 \$", legend=:topleft, label="Singles", ylim=(0.0, 5.0), yticks = (1.0:1.0:5.0))
plot_event_η_default = plot!(x,event_η_C_default_ave_norm[2:4], label="Couples",lw = 3, lc = :red)

savefig(plot_event_η_default, string(path,"plot_event_eta_default.pdf"))

# κ event
plot_event_κ_1_default = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_κ_1_default = plot!(x,event_κ_S_default_ave_norm[2:4], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Number of Households Defaulting around Event } \\kappa_2 \$", legend=:topleft, label="Singles", xticks = (-1:1:1))
plot_event_κ_1_default = plot!(x,event_κ_1_C_default_ave_norm[2:4], label="Couples",lw = 3, lc = :red)

savefig(plot_event_κ_1_default, string(path,"plot_event_kappa_1_default.pdf"))

# Sum version
plot_event_κ_1_default_sum = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_κ_1_default_sum = plot!(x,event_κ_S_default_ave_norm_sum, lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Cumulative Number of Households Defaulting after Event } \\kappa_2 \$", legend=:topleft, label="Singles", xticks = (-1:1:1))
plot_event_κ_1_default_sum = plot!(x,event_κ_1_C_default_ave_norm_sum, label="Couples",lw = 3, lc = :red)

savefig(plot_event_κ_1_default_sum, string(path,"plot_event_kappa_1_default_sum.pdf"))

# Labor supply
plot_event_κ_1_labor = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )
plot_event_κ_1_labor = plot!(x,event_κ_S_labor_ave_norm[2:4], lw = 3, lc= :blue, xlabel = "Event Time", title="\$ \\textrm{Average Labor Supply around Event } \\kappa_2 \$", legend=:left, label="Singles", xticks = (-1:1:1))
# plot_event_κ_1_labor = plot!(x,event_κ_1_C_labor_1_ave_norm[2:4], label="Couples - Affected Indi.",lw = 3, lc = :red)
# plot_event_κ_1_labor = plot!(x,event_κ_1_C_labor_2_ave_norm[2:4], label="Couples - Unaffected Indi.",lw = 3, lc = :green)
# plot_event_κ_1_labor = plot!(x,event_κ_1_C_labor_ave_sum_norm[2:4], label="Couples - Average",lw = 3, lc = :purple, linestyle = :dot)
plot_event_κ_1_labor = plot!(x,event_κ_1_C_labor_ave_sum_norm[2:4], label="Couples - Average",lw = 3, lc = :red)

savefig(plot_event_κ_1_labor, string(path,"plot_event_kappa_1_labor_alt.pdf"))

#====================#
# Welfare comparison #
#====================#

x_vals = 1.0:0.01:9.0

plot_x = collect(x_vals*0.1)

# welfare_S_raw = [-9.79282635128806, -9.86595036527491, -9.88555344299473, -9.90025658013086, -9.89285463174075, -9.89740892191023, -9.88357314525559, -9.85474162555193, -9.82647475291722]

welfare_S_raw = [-9.79282635128806, -9.86595036527491, -9.88555344299473, -9.90025658013086, (-9.90025658013086-9.89740892191023)/2, -9.89740892191023, -9.88357314525559, -9.85474162555193, -9.82647475291722]

welfare_C_raw = [-8.84545796737139, -8.90072281472212, -8.90349720645915, -8.89221393533201, -8.88184980748625, -8.86759249770615, -8.84999396956581, -8.82882616008967, -8.80431733228915]

# welfare_S_norm = zeros(length(welfare_S_raw))
#
# for i in 1: length(welfare_S_norm)
#     welfare_S_norm[i] = abs(welfare_S_raw[i]-welfare_S_raw[1])/
# end

itp_S = interpolate(welfare_S_raw, BSpline(Quadratic(Flat(OnCell()))))

itp_C = interpolate(welfare_C_raw, BSpline(Quadratic(Flat(OnCell()))))

# plot(welfare_S_raw)
#
# plot(welfare_S_norm)

# x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# nodes = (x,)
# itp_S=interpolate(nodes, welfare_S_raw, Gridded(Linear()))

# Singles
plot_y_S = itp_S(x_vals)

κ_best = argmax(plot_y_S)

κ_best_ind = findall((1:1:length(x_vals)) .== κ_best)

plot_welfare_S = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_S = plot!(plot_x,plot_y_S,lw = 3, lc= :blue, xlabel = L"\phi", title="\$ \\textrm{Ex-Ante Welfare of Newborn Across Garnishment Rates } \\phi \$", legend=:topright, label = "")

plot_welfare_S = plot!(plot_x[[κ_best]], plot_y_S[[κ_best]], seriestype = :scatter, color = :red, label = "Optimal Rate",
                        markersize = 8, markershapes = :auto, markerstrokecolor = :auto)

savefig(plot_welfare_S, string(path,"plot_welfare_S.pdf"))

# Couples
plot_y_C = itp_C(x_vals)

κ_best = argmax(plot_y_C)

κ_best_ind = findall((1:1:length(x_vals)) .== κ_best)

plot_welfare_C = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_C = plot!(plot_x,plot_y_C,lw = 3, lc= :blue, xlabel = L"\phi", title="\$ \\textrm{Ex-Ante Welfare of Newborn Across Garnishment Rates } \\phi \$", legend=:bottomright, label = "")

plot_welfare_C = plot!(plot_x[[κ_best]], plot_y_C[[κ_best]], seriestype = :scatter, color = :red, label = "Optimal Rate",
                        markersize = 8, markershapes = :auto, markerstrokecolor = :auto)

savefig(plot_welfare_C, string(path,"plot_welfare_C.pdf"))

# Prices across phi

# prices_S_raw = [52.7820731268112, 30.2531517608838, 30.0314152638029, 18.3667083113091, 13.2851584356158, 10.8230897442509, 7.91400517925884, 7.2623386933655, 6.91283138533016]

prices_S_raw = [52.7820731268112, 30.2531517608838, (30.2531517608838+18.3667083113091)/2, 18.3667083113091, 13.2851584356158, 10.8230897442509, 7.91400517925884, 7.2623386933655, 6.91283138533016]

prices_C_raw = [56.6928905254406, 19.8190066022181, 11.0735813774202, 8.23323585552571, 7.36173960144399, 6.96484312276455, 6.83853411034032, 6.75293984293297, 6.68199144756651]

itp_prices_S = interpolate(prices_S_raw, BSpline(Quadratic(Flat(OnCell()))))

itp_prices_C = interpolate(prices_C_raw, BSpline(Quadratic(Flat(OnCell()))))

# Singles
plot_y_prices_S = itp_prices_S(x_vals)

# Couples
plot_y_prices_C = itp_prices_C(x_vals)

plot_prices_ϕ = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )

plot_prices_ϕ = plot!(plot_x,plot_y_prices_S,lw = 3, lc= :blue, ylabel = "%", xlabel = L"\phi", title="\$ \\textrm{Average Interest Rate Across Garnishment Rates } \\phi \$", legend=:topright, label = "Singles")
plot_prices_ϕ = plot!(plot_x,plot_y_prices_C,lw = 3, lc= :red, label = "Couples")

savefig(plot_prices_ϕ, string(path,"plot_prices_phi.pdf"))

# Borrowing across ϕ

# borrow_S_raw = [1.40933902568953, 2.04847034162571, 10.7291109533009, 4.92958742990452, 13.5652197212657, 11.7886716611168, 15.4272574295209, 27.2029413714096, 29.7972620794199]

borrow_S_raw = [1.40933902568953, 2.04847034162571, 10.7291109533009, (10.7291109533009+13.5652197212657)/2, 13.5652197212657, (13.5652197212657+15.4272574295209)/2, 15.4272574295209, 27.2029413714096, 29.7972620794199]

borrow_C_raw = [5.57850305195526, 7.42551823902954, 12.6641460394166, 22.8114354959008, 28.8167514801474, 39.062662808388, 45.5651821961841, 50.5794520148242, 53.82422679938]

itp_borrow_S = interpolate(borrow_S_raw, BSpline(Quadratic(Flat(OnCell()))))

itp_borrow_C = interpolate(borrow_C_raw, BSpline(Quadratic(Flat(OnCell()))))

# Singles
plot_y_borrow_S = itp_borrow_S(x_vals)

# Couples
plot_y_borrow_C = itp_borrow_C(x_vals)

plot_borrow_ϕ = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )

plot_borrow_ϕ = plot!(plot_x,plot_y_borrow_S,lw = 3, lc= :blue, ylabel = "%", xlabel = L"\phi", title="\$ \\textrm{Fraction of Households Borrowing Across Garnishment Rates } \\phi \$", legend=:topleft, label = "Singles")
plot_borrow_ϕ = plot!(plot_x,plot_y_borrow_C,lw = 3, lc= :red, label = "Couples")

savefig(plot_borrow_ϕ, string(path,"plot_borrow_phi.pdf"))

# Default rate across phi
# default_S_raw = [7.00168933532748, 3.01759817970426, 2.38889202802058, 0.928658387029074, 0.930978678125751, 0.451685755400498, 0.17831802563844, 0.151069159078516, 0.0686076182527847]

default_S_raw = [7.00168933532748, 3.01759817970426, 2.38889202802058, (2.38889202802058+0.930978678125751)/2, 0.930978678125751, 0.451685755400498, 0.17831802563844, 0.151069159078516, 0.0686076182527847]

default_C_raw = [9.11020109970099, 1.95832062222291, 0.730242143599917, 0.344575203627214, 0.187619528777399, 0.108891538769045, 0.0732991279748198, 0.0408761574631638, 0.00773180619707612, ]

itp_default_S = interpolate(default_S_raw, BSpline(Quadratic(Flat(OnCell()))))

itp_default_C = interpolate(default_C_raw, BSpline(Quadratic(Flat(OnCell()))))

# Singles
plot_y_default_S = itp_default_S(x_vals)

# Couples
plot_y_default_C = itp_default_C(x_vals)

plot_default_ϕ = plot(box = :on, size = [800, 500],
                xtickfont = font(12, "Computer Modern", :black),
                ytickfont = font(12, "Computer Modern", :black),
                legendfont = font(12, "Computer Modern", :black),
                guidefont = font(14, "Computer Modern", :black),
                titlefont = font(14, "Computer Modern", :black),
                margin = 4mm
                )

plot_default_ϕ = plot!(plot_x,plot_y_default_S,lw = 3, lc= :blue, ylabel = "%", xlabel = L"\phi", title="\$ \\textrm{Default Rates Across Garnishment Rates } \\phi \$", legend=:topright, label = "Singles")
plot_default_ϕ = plot!(plot_x,plot_y_default_C,lw = 3, lc= :red, label = "Couples")

savefig(plot_default_ϕ, string(path,"plot_default_phi.pdf"))

#########################
#       Compute CEV     #
#########################
value_S = exp.((1-parameters.β*parameters.ρ)*(V_S.-V_S_bench)).-1
value_C = exp.((1-parameters.β*parameters.ρ)*(V_C.-V_C_bench)).-1

# Weighted
cev_S_weighted = sum(μ_S_bench.*value_S)
cev_C_weighted = sum(μ_C_bench.*value_C)

# Unweighted
cev_S_unweighted = sum(value_S)/(parameters.a_size*parameters.z_size*parameters.η_size*parameters.κ_size)
cev_C_unweighted = sum(value_C)/(parameters.a_size*parameters.z_size*parameters.z_size*parameters.η_size*parameters.η_size*parameters.κ_size*parameters.κ_size)

cev = [cev_S_weighted
       cev_C_weighted
       cev_S_unweighted
       cev_C_unweighted
       ]

#############################
# Compute Deadweight Loss   #
#############################
# Singles
deadweight_S = 0.0

for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size, a_i in 1:parameters.a_size
    if d_S_i[a_i,z_i,η_i,κ_i] == 2
        deadweight_S += μ_S[a_i,z_i,η_i,κ_i]*parameters.z_grid[z_i]*parameters.η_grid[η_i]*(n_S_i[a_i,z_i,η_i,κ_i]^parameters.θ)*parameters.ϕ
    end
end

# Couples
deadweight_C = 0.0

for κ_2_i in 1:parameters.κ_size, κ_1_i in 1:parameters.κ_size, η_2_i in 1:parameters.η_size, η_1_i in 1:parameters.η_size, z_2_i in 1:parameters.z_size, z_1_i in 1:parameters.z_size, a_i in 1:parameters.a_size
    if d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 2
        deadweight_C += μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]*(parameters.z_grid[z_1_i]*parameters.η_grid[η_1_i]*(n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]^parameters.θ)+parameters.z_grid[z_2_i]*parameters.η_grid[η_2_i]*(n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]^parameters.θ))*parameters.ϕ
    end
end

# With assets
# Singles
deadweight_S_plus_a = 0.0

for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size, a_i in 1:parameters.a_size
    if d_S_i[a_i,z_i,η_i,κ_i] == 2
        deadweight_S_plus_a += μ_S[a_i,z_i,η_i,κ_i]*parameters.z_grid[z_i]*parameters.η_grid[η_i]*(n_S_i[a_i,z_i,η_i,κ_i]^parameters.θ)*parameters.ϕ
        if a_i > parameters.a_ind_zero
            deadweight_S_plus_a += μ_S[a_i,z_i,η_i,κ_i]*parameters.a_grid[a_i]
        end
    end
end

# Couples
deadweight_C_plus_a = 0.0

for κ_2_i in 1:parameters.κ_size, κ_1_i in 1:parameters.κ_size, η_2_i in 1:parameters.η_size, η_1_i in 1:parameters.η_size, z_2_i in 1:parameters.z_size, z_1_i in 1:parameters.z_size, a_i in 1:parameters.a_size
    if d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 2
        deadweight_C_plus_a += μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]*(parameters.z_grid[z_1_i]*parameters.η_grid[η_1_i]*(n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]^parameters.θ)+parameters.z_grid[z_2_i]*parameters.η_grid[η_2_i]*(n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]^parameters.θ))*parameters.ϕ
        if a_i > parameters.a_ind_zero
            deadweight_C_plus_a += μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]*parameters.a_grid[a_i]
        end
    end
end

# Only a
# Singles
deadweight_S_only_a = 0.0

for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size, a_i in 1:parameters.a_size
    if d_S_i[a_i,z_i,η_i,κ_i] == 2
        if a_i > parameters.a_ind_zero
            deadweight_S_only_a += μ_S[a_i,z_i,η_i,κ_i]*parameters.a_grid[a_i]
        end
    end
end

# Couples
deadweight_C_only_a = 0.0

for κ_2_i in 1:parameters.κ_size, κ_1_i in 1:parameters.κ_size, η_2_i in 1:parameters.η_size, η_1_i in 1:parameters.η_size, z_2_i in 1:parameters.z_size, z_1_i in 1:parameters.z_size, a_i in 1:parameters.a_size
    if d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 2
        if a_i > parameters.a_ind_zero
            deadweight_C_only_a += μ_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]*parameters.a_grid[a_i]
        end
    end
end

deadweight = [deadweight_S_plus_a
       deadweight_C_plus_a
       deadweight_S_only_a
       deadweight_C_only_a
       ]



#####################################
# Plots for Macro Seminar Fall 2021 #
#####################################

# Policy Experiments

plot_x = [0.1, 0.3, 0.5, 0.7, 0.9]

# Changes in default
policy_d_S = [6.997277778, 0.655166667, 0.232388889, 0.066611111, 0.019333333]
policy_d_C = [4.84162963, 0.749111111, 0.17762963, 0.028962963, 0]

plot_policy_default = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_default = plot!(plot_x,policy_d_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Fraction\ of\ Defaulters\ Across\ Bankruptcy\ Cost\ } \phi", legend=:right, label = "Singles")

plot_policy_default = plot!(plot_x, policy_d_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_default, string(path,"plot_policy_default.pdf"))

# Changes in loan users
policy_borrow_S = [10.83766667, 14.48255556, 25.13283333, 35.28405556, 45.97777778]
policy_borrow_C = [20.12851852, 25.20622222, 30.674, 39.94096296, 53.66822222]

plot_policy_borrow = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_borrow = plot!(plot_x,policy_borrow_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Fraction\ of\ Borrowers\ Across\ Bankruptcy\ Cost\ } \phi", legend=:bottomright, label = "Singles")

plot_policy_borrow = plot!(plot_x, policy_borrow_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_borrow, string(path,"plot_policy_borrow.pdf"))

# Changes in DTI
policy_dti_S = [65.42256485, 14.09838696, 22.63501976, 39.38662311, 75.4768926]
policy_dti_C = [14.22719331, 14.01895842, 18.99013601, 24.98382649, 39.9808697]

plot_policy_dti = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_dti = plot!(plot_x, policy_dti_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Average\ Debt\ to\ Income\ Ratio\ Across\ Bankruptcy\ Cost\ } \phi", legend=:bottomright, label = "Singles")

plot_policy_dti = plot!(plot_x, policy_dti_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_dti, string(path,"plot_policy_dti.pdf"))

# Changes in interest rate
policy_rate_S = [2.021972283, 0.116810369, 0.076294305, 0.068675069, 0.067105298]
policy_rate_C = [0.543969688, 0.184162179, 0.110963218, 0.075666465, 0.066681907]

plot_policy_rate = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_rate = plot!(plot_x, policy_rate_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Average\ Interest\ Rate\ Across\ Bankruptcy\ Cost\ } \phi", legend=:right, label = "Singles")

plot_policy_rate = plot!(plot_x, policy_rate_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_rate, string(path,"plot_policy_rate.pdf"))

# Changes in consumption
policy_c_S = [0.851735076, 0.856093352, 0.852494276, 0.84777381, 0.841281192]
policy_c_C = [0.810902787, 0.814507541, 0.819187991, 0.826079097, 0.838089409]

plot_policy_consumption = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_consumption = plot!(plot_x,policy_c_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Average\ Consumption\ Across\ Bankruptcy\ Cost\ } \phi", legend=:left, label = "Singles")

plot_policy_consumption = plot!(plot_x, policy_c_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_consumption, string(path,"plot_policy_consumption.pdf"))

# Changes in labor
policy_n_S = [0.533896389, 0.535724444, 0.5358575, 0.537590556, 0.544417222]
policy_n_C = [0.590597685, 0.591539352, 0.59104713, 0.588921111, 0.584859907]

plot_policy_labor = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_labor = plot!(plot_x,policy_n_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Average\ Labor\ Supply\ Across\ Bankruptcy\ Cost\ } \phi", legend=:right, label = "Singles")

plot_policy_labor = plot!(plot_x, policy_n_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_labor, string(path,"plot_policy_labor.pdf"))

# Changes in assets
policy_a_S = [1.228242942, 1.274335481, 1.15501661, 0.985225252, 0.667629784]
policy_a_C = [0.873336059, 0.834638787, 0.706548, 0.497023982, 0.102819128]

plot_policy_assets = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_assets = plot!(plot_x,policy_a_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Average\ Assets\ Across\ Bankruptcy\ Cost\ } \phi", legend=:topright, label = "Singles")

plot_policy_assets = plot!(plot_x, policy_a_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_assets, string(path,"plot_policy_assets.pdf"))

# Changes in consumption insurance (productivity)
policy_insurance_prod_S = [0.376954508, 0.346369974, 0.35174969, 0.341467748, 0.321297676]
policy_insurance_prod_C = [0.239100681, 0.231056393, 0.225572269, 0.220476983, 0.202029541]

plot_policy_insurance_prod = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_insurance_prod = plot!(plot_x,policy_insurance_prod_S,lw = 3, lc= :blue, xlabel = L"\phi", ylabel = L"\mu", title=L"\textrm{Consumption\ Insurance\ Across\ Bankruptcy\ Cost\ } \phi", legend=:left, label = "Singles")

plot_policy_insurance_prod = plot!(plot_x, policy_insurance_prod_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_insurance_prod, string(path,"plot_policy_insurance_prod.pdf"))

# Changes in consumption insurance (earnings)
policy_insurance_earnings_S = [0.312198865, 0.285529431, 0.288852239, 0.27896562, 0.257896666]
policy_insurance_earnings_C = [0.309165635, 0.287611245, 0.274677122, 0.266010572, 0.247983308]

plot_policy_insurance_earnings = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_insurance_earnings = plot!(plot_x,policy_insurance_earnings_S,lw = 3, lc= :blue, xlabel = L"\phi", ylabel = L"\mu", title=L"\textrm{Consumption\ Insurance\ Across\ Bankruptcy\ Cost\ } \phi", legend=:topright, label = "Singles")

plot_policy_insurance_earnings = plot!(plot_x, policy_insurance_earnings_C, lw = 3, lc= :red, label = "Couples")

savefig(plot_policy_insurance_earnings, string(path,"plot_policy_insurance_earnings.pdf"))

# Changes in welfare
policy_welfare_S = [-6.016652935, -6.019134579, -5.922971013, -5.802455958, -5.625277122]
policy_welfare_C = [-5.622189723, -5.613810471, -5.548428793, -5.445516106, -5.275635615]
policy_welfare_C_ψ_0_3 = [-5.971878753, -6.008920073, -5.942155029, -5.823904327, -5.640980826]
policy_welfare_C_ψ_0_5 = [-6.069933074, -6.136519088, -6.06261795, -5.937470515, -5.74949073]

plot_policy_welfare = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_welfare = plot!(plot_x,policy_welfare_S,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Ex-Ante\ Welfare\ of\ Newborns\ Across\ Bankruptcy\ Cost\ } \phi", legend=:topleft, label = "Singles")

plot_policy_welfare = plot!(plot_x, policy_welfare_C, lw = 3, lc= :red, label = L"\textrm{Couples,\ }\psi=0.1")
plot_policy_welfare = plot!(plot_x, policy_welfare_C_ψ_0_3, lw = 3, lc= :red, label =  L"\textrm{Couples,\ }\psi=0.3", linestyle = :dash)
plot_policy_welfare = plot!(plot_x, policy_welfare_C_ψ_0_5, lw = 3, lc= :red, label = L"\textrm{Couples,\ }\psi=0.5", linestyle = :dot)

savefig(plot_policy_welfare, string(path,"plot_policy_welfare.pdf"))

# Changes in welfare (normalized)
policy_welfare_S_norm = (policy_welfare_S[1] .- policy_welfare_S)  ./ policy_welfare_S[1]
policy_welfare_C_norm = (policy_welfare_C[1] .- policy_welfare_C) ./ policy_welfare_C[1]
policy_welfare_C_ψ_0_3_norm = (policy_welfare_C_ψ_0_3[1] .- policy_welfare_C_ψ_0_3) ./ policy_welfare_C_ψ_0_3[1]
policy_welfare_C_ψ_0_5_norm = (policy_welfare_C_ψ_0_5[1] .- policy_welfare_C_ψ_0_5) ./ policy_welfare_C_ψ_0_5[1]

# policy_welfare_S_norm = -(policy_welfare_S[1] .- policy_welfare_S)
# policy_welfare_C_norm = -(policy_welfare_C[1] .- policy_welfare_C)
# policy_welfare_C_ψ_0_3_norm = -(policy_welfare_C_ψ_0_3[1] .- policy_welfare_C_ψ_0_3)
# policy_welfare_C_ψ_0_5_norm = -(policy_welfare_C_ψ_0_5[1] .- policy_welfare_C_ψ_0_5)

plot_policy_welfare_norm = plot(box = :on, size = [800, 500],
                xtickfont = font(16, "Computer Modern", :black),
                ytickfont = font(16, "Computer Modern", :black),
                legendfont = font(16, "Computer Modern", :black),
                guidefont = font(18, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_policy_welfare_norm = plot!(plot_x,policy_welfare_S_norm,lw = 3, lc= :blue, xlabel = L"\phi", title=L"\textrm{Ex-Ante\ Welfare\ of\ Newborns\ Across\ Bankruptcy\ Cost\ } \phi", legend=:topleft, label = "Singles")
plot_policy_welfare_norm = plot!(plot_x, policy_welfare_C_norm, lw = 3, lc= :red, label = L"\textrm{Couples,\ }\psi=0.1")
plot_policy_welfare_norm = plot!(plot_x, policy_welfare_C_ψ_0_3_norm, lw = 3, lc= :red, label = L"\textrm{Couples,\ }\psi=0.3", linestyle = :dash)
plot_policy_welfare_norm = plot!(plot_x, policy_welfare_C_ψ_0_5_norm, lw = 3, lc= :red, label = L"\textrm{Couples,\ }\psi=0.5", linestyle = :dot)

savefig(plot_policy_welfare_norm, string(path,"plot_policy_welfare_norm.pdf"))
