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

    #print corner_spikes
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