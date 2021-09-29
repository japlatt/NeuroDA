using OrdinaryDiffEq
using Interpolations
using Noise
using DelimitedFiles
using Plots
using Distributions

include("../neuroda.jl")
using .NeuroDA

path_to_stim_file = "NaKL_Twin/current_time.txt"

stim_values = readdlm(path_to_stim_file, Float64)
stim = LinearInterpolation(stim_values[:, 1], stim_values[:, 2])
Iinj(t) = stim(t)

function NaKL(du, u, p, t)
    V, m, h, n = u
    Cm, g_Na, g_K, g_L, E_Na, E_K, E_L, vm, vh, vn, dvm, dvh, dvn, tm0, tm1, th0, th1, tn0, tn1 = p
    du[1] = -1/Cm * (g_Na*m^3*h*(V - E_Na) + g_K*n^4*(V - E_K) + g_L*(V - E_L) - Iinj(t))
    du[2] = (0.5*(1+tanh((V - vm)/dvm)) - m)/(tm0 + tm1*(1 - tanh((V - vm)/dvm)^2))
    du[3] = (0.5*(1+tanh((V - vh)/dvh)) - h)/(th0 + th1*(1 - tanh((V - vh)/dvh)^2))
    du[4] = (0.5*(1+tanh((V - vn)/dvn)) - n)/(tn0 + tn1*(1 - tanh((V - vn)/dvn)^2))
end 

remake_data = true
if remake_data
    Lidx = [1]
    u0 = [-70.0, 0.1, 0.1, 0.1]
    p = (1.0, 120.0, 20.0, 0.3, 50.0, -77.0, -54.4, -40.0, -60.0, -55.0, 15.0, -15.0, 30.0, 0.1, 0.4, 1.0, 7.0, 1.0, 5.0)
    tspan = (0.0, 330.0)
    dt = 0.025
    ϵ = 2.0
    make_data_dynamics = ODEProblem(NaKL, u0, tspan, p)
    save_path = "NaKL_Twin/voltage.txt"
    make_data(make_data_dynamics, dt, ϵ, Lidx, save_path = save_path, plot_data=true)
end

D = 4
Np = 19
start = 1
num_pts = 4000
num_pts_rmse = 1000
dt = 0.025
path_to_obs = "NaKL_Twin/voltage.txt"
lower_bounds = [-71.0, 0.0, 0.0, 0.0, 0.5, 50.0, 5.0, 0.1, 0.0, -100.0, -60.0,
                -60.0, -70.0, -70.0, 11.0, -20.0, 25.0, 0.05, 0.1, 0.1, 1.0, 0.1, 2.0]
upper_bounds = [100.0, 1.0, 1.0, 1.0, 2.0, 200.0, 40.0, 1.0, 100.0, -50.0, -50.0,
                -30.0, -40.0, -40.0, 27.0, -8.0, 39.0, 0.25, 1.0, 5.0, 15.0, 5.0, 12.0]
obs_vars = [1]
rand_p = [rand(truncated(Normal(mean(x), (x[2]-x[1])/4), x[1], x[2])) for x in collect(zip(lower_bounds, upper_bounds))]
dynamics = ODEProblem(NaKL, rand(Float64, D), (start*dt, (start+num_pts)*dt), rand_p)

config = neuroda(D, Np, start, num_pts, dt, path_to_obs,
                 lower_bounds, upper_bounds, dynamics, obs_vars)

# init_guess(config, 500, 150000, 1e-10)

u0 = [-67.66508841714855, 0.07335992430322484, 0.08615476172098958, 0.0, 1.1364278830898669, 160.62357835287756, 39.9999999564207, 0.166998390984229, 58.75229845028077, -79.90359498268285, -53.413560264307314, -30.884407714018987, -56.215884061924896, -50.8090810189477, 21.396737053684692, -20.0, 36.100555704790345, 0.06825456417406059, 0.15946916541110218, 0.09999999999999999, 5.007508089262189, 2.3931449265510194, 2.0]

# xmin = run_neuroda(config, u0; maxtime=360.0, popsize=-1, ϵ=0.7)

# plot_data_sim(config, xmin)