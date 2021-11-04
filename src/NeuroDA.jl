module NeuroDA

module CostFunctions

using DiffEqBase
using Statistics
using StatsBase

export Loss, find_spike_times

struct Loss{T, D <: Vector} <: DiffEqBase.DECostFunction
    t::D
    data::T
    spike_timings::D
    ϵ::Float64
    fit_rmse::Int64
    spike_thresh::Float64
end

function (f::Loss)(sol::DiffEqBase.AbstractEnsembleSolution)
    f.(sol.u)
end

function (f::Loss)(sol::DiffEqBase.DESolution)
    data = matrixize(f.data)
    spike_timings_data = f.spike_timings

    if sol isa DiffEqBase.AbstractEnsembleSolution
        failure = any(s.retcode !== :Success && s.retcode !== :Terminated for s in sol)
    else
        failure = sol.retcode !== :Success && sol.retcode !== :Terminated
    end
    failure && return Inf
    
    RMSE = rmsd(data[:, 1:f.fit_rmse], sol[:, 1:f.fit_rmse]; normalize=true)

    spike_timings_sol = find_spike_times(sol.t, hcat(sol.u[:, 1]...)'[:, 1], f.spike_thresh)

    d = bivariate_spike_distance(spike_timings_data, spike_timings_sol, sol.t[1], sol.t[end], 50)
    
    sumd = sum(d)/length(d)
    f.ϵ*sumd+(1-f.ϵ)*RMSE
end


function find_spike_times(time, voltage, spike_thresh = 0.0)
    dt = time[2]-time[1]
    a = Float64[]
    first = 1
    bool_spike = voltage .> spike_thresh
    if all(bool_spike .== 0) 
        return Float64[]
    end
    spike_times = time[bool_spike]
    for i in 1:1:length(spike_times)-1
        if (spike_times[i+1] - spike_times[i]) - dt > dt/2 || i == length(spike_times)-1
            push!(a, mean(spike_times[first:i]))
            first = i+1
        end
    end
    return a
end

function bivariate_spike_distance(t11, t22, ti, te, N)
    t = LinRange(ti+(te-ti)/N, te, N)
    d = zeros(length(t))

    t1 = zeros(length(t11)+2)
    t2 = zeros(length(t22)+2)
    
    t1[1] = ti
    t1[2:end-1] = t11
    t1[end] = te

    t2[1] = ti
    t2[2:end-1] = t22
    t2[end] = te

    corner_spikes = zeros((N,5))

    ibegin_t1 = 1
    ibegin_t2 = 1
    corner_spikes[:,1] = t
    for (itc, tc) in enumerate(t)
        corner_spikes[itc,2:3], ibegin_t1 = find_corner_spikes(tc, t1, ibegin_t1, ti, te)
        corner_spikes[itc,4:5], ibegin_t2 = find_corner_spikes(tc, t2, ibegin_t2, ti, te)
    end

    xisi = zeros((N,2))
    xisi[:,1] = corner_spikes[:,3] - corner_spikes[:,2]
    xisi[:,2] = corner_spikes[:,5] - corner_spikes[:,4]
    norm_xisi = sum(xisi,dims=2).^2

    dp1 = minimum(abs.(repeat(t2,1,N)' - repeat(reshape(corner_spikes[:,2],(N,1)), 1,length(t2))),dims=2)
    df1 = minimum(abs.(repeat(t2,1,N)' - repeat(reshape(corner_spikes[:,3],(N,1)), 1, length(t2))),dims=2)

    # And the smallest distance between the spikes in t1 and the corner spikes of t2
    dp2 = minimum(abs.(repeat(t1,1,N)' - repeat(reshape(corner_spikes[:,4],(N,1)), 1,length(t1))),dims=2)
    df2 = minimum(abs.(repeat(t1,1,N)' - repeat(reshape(corner_spikes[:,5],(N,1)), 1, length(t1))),dims=2)

    xp1 = t - corner_spikes[:,2]
    xf1 = corner_spikes[:,3] - t 
    xp2 = t - corner_spikes[:,4]
    xf2 = corner_spikes[:,5] - t

    S1 = (dp1 .* xf1 .+ df1 .* xp1)./xisi[:,1]
    S2 = (dp2 .* xf2 .+ df2 .* xp2)./xisi[:,2]

    d = (S1 .* xisi[:,2] .+ S2 .* xisi[:,1]) ./ (norm_xisi/2.0)

    return d
end

function find_corner_spikes(t, train, ibegin, ti, te)
    tprev = (ibegin == 1) ? ti : train[ibegin-1]
    
    for (idts, ts) in enumerate(train[ibegin:end])
        if ts >= t
            return [tprev, ts], idts+ibegin-1
        end
        tprev = ts
    end
    return [train[end],te], length(train[ibegin:end])+ibegin-1
end

matrixize(x) = typeof(x) <: Vector ? reshape(x,1,length(x)) : x


end


using DifferentialEquations
using DiffEqParamEstim
using Distributions
using Random
using CMAEvolutionStrategy
using Plots
using DelimitedFiles
using NLopt
using Noise

using .CostFunctions

export neuroda, run_neuroda, init_guess, plot_data_sim, make_data

"""
    neuroda(D, Np, start, num_pts, dt, path_to_obs,
            lower_bounds, upper_bounds, dynamics, obs_vars)

Create a neuroda object.

# Arguments
- `D`: the number of state variables.
- `Np`: the number of parameters.
- `start`: starting point of estimation data.
- `num_pts`: number of points from start for estimation.
- `dt`: time step for the data.
- `path_to_obs`: path to the observation file.
- `lower_bounds`: array of lower bounds, length(D+Np).
- `upper_bounds`: array of upper bounds, length(D+Np).
- `dynamics`: definition of the model.  See DifferentialEquations.jl for structure
- `obs_vars`: list of observed variables e.g., [1, 3, 5]
"""
struct neuroda
    D::Int
    Np::Int
    start::Int
    num_pts::Int
    dt::Float64
    lower_bounds::Array{Float64,1}
    upper_bounds::Array{Float64,1}
    data_t::Array{Float64,1}
    data_u::AbstractArray
    dynamics::ODEProblem
    obs_vars::Array{Int64,1}
    path_to_obs::String
    function neuroda(D, Np, start, num_pts, dt, path_to_obs,
                     lower_bounds, upper_bounds, dynamics, obs_vars)
        data_t, data_u = load_data(path_to_obs, start, num_pts; plot_data=true)
        @assert all(lower_bounds .< upper_bounds)
        new(D, Np, start, num_pts, dt, lower_bounds, upper_bounds, data_t, data_u, dynamics,
            obs_vars, path_to_obs)
    end
end

"""
    run_neuroda(config::neuroda, init_guess::Array{Float64,1}, num_pts_rmse::Int64, <keyword arguments>)

Run the data assimilation routine.

# Arguments
- `config`: configuration for the da problem.
- `init_guess`: initial guess for the D+Np initial conditions + parameters.
- `num_pts_rmse`: Number of points to apply the rmse cost function.
- `maxtime` : Amount of time (seconds) to spend on the optimization.
- `popsize` : population size for the CMAES algorith. default -> 4 + floor(3*log(D+Np))
- `ϵ` : cost = ϵ*spike_cost + (1-ϵ)*RMSE.
- `spike_thresh`: threshold in voltage for the spiking threshold
"""
function run_neuroda(config::neuroda, init_guess::Array{Float64,1}, num_pts_rmse::Int64;
                     maxtime=360.0, popsize=-1, ϵ=0.7, spike_thresh=0.0, σᵪ = 2)
    @assert num_pts_rmse < config.num_pts

    lower_bounds = config.lower_bounds
    upper_bounds = config.upper_bounds
    data_t = config.data_t
    data_u = config.data_u
    D = config.D
    Np = config.Np

    function untransform(θ::Array{Float64, 1})
        lower_bounds + (upper_bounds - lower_bounds).*θ/10
    end
    
    function transform(θ::Array{Float64, 1})
        clamp.(10*(θ - lower_bounds)./(upper_bounds - lower_bounds), 0.0, 10.0)
    end
    
    function problem_new_parameters_ic(ens_prob::EnsembleProblem, θ)
        θp = mapslices(untransform, θ, dims = 1)
        prob = ens_prob.prob
        u0_new = θp[1:D, :]
        p_new = θp[D+1:Np+D, :]
        function prob_func(prob,i,repeat)
            remake(prob,u0=u0_new[:, i], p = p_new[:, i])
        end
        EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
    end

    function problem_new_parameters_ic(prob::ODEProblem, θ)
        u0_new = θ[1:D]
        p_new = θ[D+1:Np+D]
        remake(prob, u0 = u0_new, p = p_new)
    end

    if popsize == -1 
        popsize = 4 + floor(Int, 3*log(config.D+config.Np))
    end

    ens_prob = EnsembleProblem(config.dynamics)

    spike_time_data = find_spike_times(collect(data_t), data_u[:, 1])
    obj = build_loss_objective(ens_prob, Tsit5(),
                               CostFunctions.Loss(data_t, collect(data_u'),
                                                  spike_time_data, ϵ, num_pts_rmse,
                                                  spike_thresh),
                               saveat=data_t,
                               prob_generator = problem_new_parameters_ic,
                               ensemblealg = EnsembleThreads(),
                               trajectories = popsize,
                               save_idxs = config.obs_vars,
                               abstol = 1e-6, reltol = 1e-4
                               )

    result = minimize(obj.cost_function, transform(init_guess), σᵪ;
                        lower = zeros(Np+D),
                        upper = 10.0*ones(Np+D),
                        popsize = popsize,
                        parallel_evaluation = true,
                        multi_threading = false,
                        verbosity = 1,
                        seed = rand(UInt),
                        xtol = 1e-7,
                        maxtime = maxtime)
    xmin = untransform(xbest(result))
    return xmin
end

function init_guess(config::neuroda, num_pts, num_it = 10000, tol = 1e-5, initial_θ = Float64[])
    data_t = config.data_t[1:num_pts]
    data_u = config.data_u[1:num_pts, 1:end]

    p = plot(data_t, data_u, layout = (length(config.obs_vars), 1))
    display(p)

    data_u = transpose(data_u)

    if isempty(initial_θ)
        initial_θ = [rand(truncated(Normal(mean(x), (x[2]-x[1])/4), x[1], x[2])) for x in collect(zip(config.lower_bounds, config.upper_bounds))]
        initial_θ[1] = data_u[1]
    end
    prob_local = remake(config.dynamics, tspan = (data_t[1], data_t[end]))
    local_opt(prob_local, data_t, data_u, initial_θ, config, num_it, tol)
end

function plot_data_sim(config::neuroda, xmin; plot_obsvar_only=true, num_pred=0)
    @assert num_pred >= 0
    if num_pred == 0
        data_u = config.data_u
        data_t = config.data_t
    else
        data_t, data_u = load_data(config.path_to_obs, config.start, config.num_pts+num_pred)
    end

    dynamics = remake(config.dynamics, u0 = xmin[1:config.D],
                      p = xmin[config.D+1:end], tspan = (data_t[1], data_t[end]))
    sol = solve(dynamics, Vern9(), abstol = 1e-10, reltol = 1e-10)

    if plot_obsvar_only
        layout = (length(config.obs_vars), 1)
        p = plot(data_t, data_u, ls = :dash, label="Data", layout = layout)
        plot!(sol, label = "xmin", layout = layout, vars = config.obs_vars,
              lw = 2, alpha=0.7)
    else
        layout = (length(config.D), 1)
        p = plot(sol, label = "xmin", layout = layout, lw = 2)
        for i in config.obs_vars
            plot!(p[i], data_t, data_u[:, i], ls = :dash, label="Data",
                  alpha = 0.7)
        end
    end

    display(p)
end

function make_data(dynamics::ODEProblem, dt, ϵ, obs_var; save_path = "data.txt", plot_data=false)
    tspan = dynamics.tspan
    time = tspan[1]:dt:tspan[2]
    data_sol = solve(dynamics, Vern9(), abstol = 1e-12, reltol = 1e-12)
    data = transpose(convert(Array, data_sol(time))[obs_var, :])

    add_gauss!(data, ϵ)

    if plot_data
        p = plot(time, data, ls = :dash, label="Data",
                 layout = (length(obs_var), 1))
        plot!(data_sol, vars = obs_var, label = "True",
              layout = (length(obs_var), 1))
        display(p)
    end
    writedlm(save_path, hcat(time, data))
    return time, data
end

function local_opt(prob_local, data_t, data_u, initial_θ, config, num_it = 10000, tol = 1e-5)
    D = config.D
    Np = config.Np
    obs_vars = config.obs_vars
    lower_bounds = config.lower_bounds
    upper_bounds = config.upper_bounds
    function problem_new_parameters_ic(prob::ODEProblem,θ::Array{Float64, 1})
        u0_new = θ[1:D]
        p_new = θ[D+1:Np+D]
        remake(prob, u0 = u0_new, p = p_new)
    end
    obj = build_loss_objective(prob_local, Tsit5(), 
                                L2Loss(data_t, data_u), 
                                saveat=data_t, reltol=1e-4, abstol=1e-6,
                                mpg_autodiff = true,
                                autodiff_prototype = zeros(D+Np),
                                prob_generator = problem_new_parameters_ic,
                                save_idxs = obs_vars)
    opt = NLopt.Opt(:LN_BOBYQA, D+Np)

    NLopt.lower_bounds!(opt, lower_bounds)
    NLopt.upper_bounds!(opt, upper_bounds)
    NLopt.min_objective!(opt, obj.cost_function2)
    NLopt.xtol_rel!(opt, tol)
    NLopt.maxeval!(opt, num_it)

    println("Starting local optimization")
    (minf,minx,ret) = NLopt.optimize(opt, initial_θ)
    println(ret)

    prob_local_min = remake(prob_local, u0 = minx[1:D], p = minx[D+1:D+Np])
    sol_local_min = solve(prob_local_min, abstol=1e-6, reltol=1e-4)

    p = plot(data_t, transpose(data_u), ls = :dash, label="Data",
             layout = (length(obs_vars), 1))
    plot!(sol_local_min, vars = obs_vars, label = "Fit",
          layout = (length(obs_vars), 1))
    display(p)
    minx
end

function load_data(file_name, start, num_pts; plot_data=false)
    input_data = readdlm(file_name, Float64)
    if start == 0
        start+=1
        num_pts+=1
    end
    time = input_data[start:num_pts+start, 1]
    data = input_data[start:num_pts+start, 2:end]

    if plot_data
        num_plots = size(data)[2]
        p = plot(time, data, layout = (1, num_plots), legend = false)
        display(p)
    end
    return time, data
end

end