# NeuroDA

NeuroDA is an optimization package that seeks to find the parameters of a detailed neuron model given voltage data.

## Installation Instructions

NeuroDA runs using [Julia](https://julialang.org/).  To install first install julia as per instructions on the website.  Then in terminal type "julia" to enter the REPL.  Once there type "]" to enter the package manager and type "add https://github.com/japlatt/NeuroDA.git" which will install NeuroDA and dependencies.

Once installed you can get the latest version by going to the REPL (julia in terminal).  Then "]", "up NeuroDA" which will fetch the latest from github.

## Usage
Make sure to add "using NeuroDA" to import the functions from the package.

Julia needs to be run in the REPL, start the REPL by typing "julia -t NUM_THREADS" into the terminal (replace NUM_THREADS with a number).  Workflow would be:

1. julia -t 4
2. include("my_neuroda_program.jl")
my_neuroda_program.jl will include all the definitions but will not necessarily call the whole program.
3. run init_guess(config::neuroda, num_pts, num_it, tol, initial_θ) as many times as needed to get a good initial guess
4. run run_neuroda(config::neuroda, init_guess::Array{Float64,1}, num_pts_rmse::Int64; maxtime, popsize, ϵ, spike_thresh, σᵪ) with the correct initial guess


### Stimulus
You will first want to define a stimulus (if you have one) as a continuous function in time.  To do this read in the stimulus and time.  For example:

```Julia
using Interpolations
using DelimitedFiles

path_to_stim_file = "current_time.txt"

stim_values = readdlm(path_to_stim_file, Float64)
stim = LinearInterpolation(stim_values[:, 1], stim_values[:, 2])
Iinj(t) = stim(t)
```

### Define Dynamics
The next step is to define the dynamics as an ODE.  The dynamics function needs to be defined as an odefunction as described [here](https://diffeq.sciml.ai/stable/),

For instance NaKL would be
```Julia
function NaKL(du, u, p, t)
    V, m, h, n = u
    Cm, g_Na, g_K, g_L, E_Na, E_K, E_L, vm, vh, vn, dvm, dvh, dvn, tm0, tm1, th0, th1, tn0, tn1 = p
    du[1] = -1/Cm * (g_Na*m^3*h*(V - E_Na) + g_K*n^4*(V - E_K) + g_L*(V - E_L) - Iinj(t))
    du[2] = (0.5*(1+tanh((V - vm)/dvm)) - m)/(tm0 + tm1*(1 - tanh((V - vm)/dvm)^2))
    du[3] = (0.5*(1+tanh((V - vh)/dvh)) - h)/(th0 + th1*(1 - tanh((V - vh)/dvh)^2))
    du[4] = (0.5*(1+tanh((V - vn)/dvn)) - n)/(tn0 + tn1*(1 - tanh((V - vn)/dvn)^2))
end
```

See that Iinj(t) is now a function referencing the previously defined stimulus.

### config

One configures the optimization problem by initializing a "neuroda object."  This specifies all the options for the problem you are trying to solve.

The observation file must be in the following format:
[time, obs1,...,obsN]

dynamics will be an ODEProblem defined [here](https://diffeq.sciml.ai/stable/).  For example:
```Julia
# This is just a placeholder
init = rand(Float64, D)

tspan = (start*dt, (start+num_pts)*dt)

# random initialization within the bounds
randp = [rand(truncated(Normal(mean(x), (x[2]-x[1])/4), x[1], x[2])) for x in collect(zip(lower_bounds, upper_bounds))]

# create the ODEProblem
dynamics = ODEProblem(NaKL, init, tspan, rand_p)
```

obs_vars: remember julia is one indexed!

```Julia
'''
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
'''
```

### Initial Guess
A good initial guess can be important.  One way to get this is to run a local optimization routine over only the first spike.  The function to do this is
```Julia
init_guess(config::neuroda, num_pts, num_it = 10000, tol = 1e-5, initial_θ = Float64[])
```
where config is the above configuration file and num_pts is the number of pts from the beginning of the data to the end of the first spike.  This can give a reasonable initial guess to feed to the global routine although it is not strictly necessary.


### Run

Now just call run_neuroda by passing in your neuroda config and an initial guess.  Other parameters for spiking neuron data are ϵ and num_pts_rmse.  Often I will want to only closely fit the shape of the first spike and then fit the spike timings for the rest of the data.  num_pts_rmse will tell the algorithm how many points from t = 0 you want to fit using the rmse, the rest is spike distance.  ϵ governs the weight between rmse and spike timing.  ϵ = 0 means all the weight is on emse while ϵ = 1 only counts spike timing.

```Julia
'''
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
'''
```

### Plot 

run plot_data_sim(config, xmin, num_pred=NUM_PRED) with the config object and the xmin returned from run_neuroda to see the estimation and prediction.

### Make Data

For convenience in twin experiments there's make_data(dynamics::ODEProblem, dt, ϵ, obs_var; save_path = "data.txt", plot_data=false) that takes in an ODEFunction and creates data with time step dt and gaussian noise ϵ.  Only observed variables are saved.

