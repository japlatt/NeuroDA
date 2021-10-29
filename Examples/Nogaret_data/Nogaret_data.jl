using OrdinaryDiffEq
using Interpolations
using Noise
using DelimitedFiles
using Plots
using Distributions

# include("/home/jplatt/Documents/DA/NeuroDA/neuroda.jl")
using NeuroDA

path_to_stim_file = "datafiles/N2_stim.txt"

stim_values = readdlm(path_to_stim_file, Float64)
stim = LinearInterpolation(stim_values[:, 1], stim_values[:, 2])
Iapp(t)::Float64 = stim(t)

const Vtemp = 13.

function albeta(VV, V, dV, dVt, aV4, t0, τϵ, delta)
    # calculation of alpha and beta variable given the membrane voltage VV */
    #    int j;
    #    double thetai,thetait,tauj,thetai2,thetai1;
        alpha = zeros(14)
        beta = zeros(14)
        for j in 2:D::Int64
            thetai = (VV-V[j])/dV[j];
            thetait = (VV-V[j])/dVt[j];
            if j==6 || j==8 # A2 and K2 tau_h */
                tauj = t0[j]+delta[j]+0.5*(1-tanh(1000*(VV-V[j]-aV4[j])))*(τϵ[j]*(1-tanh(thetait)*tanh(thetait))-delta[j]);
            else
                if (j == 11) #T tau_h */
                    thetai2 = (VV-V[j])/dVt[j]
                    thetai1 = (VV-V[j])/aV4[j]
                    tauj = t0[j]+τϵ[j]*(1+tanh(thetai2))*(1-tanh(thetai1))*(1-tanh(1000*(VV-V[j]))*tanh(thetai2+thetai1))/(1+tanh(thetai2)^2)
                else
                    tauj = t0[j]+τϵ[j]*(1 - tanh(thetait)*tanh(thetait));
                end
            end
            alpha[j] = 0.5*(1+tanh(thetai))/tauj
            beta[j] = 0.5*(1-tanh(thetai))/tauj
        end
        return alpha, beta
    end
    
function Nogaret_N2(dyydx, yy, p, t)
    Ccap,gNa,gNaP,ENa,gA1,gA2,gK2,gC,EK,gL,EL,gou,gin,gh,Iarea,rlt = @view p[1:16]
    
    V1, V7, V8 = 0.0, -43., -59.
    V2, V3, V4, V5, V6, V9, V10, V11, V12, V13, V14 = @view p[17:27]
    V = [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14]
    
    dV1, dV7, dV8 = 0.0, +34., -21.
    dV2, dV3, dV4, dV5, dV6, dV9, dV10, dV11, dV12, dV13, dV14 = @view p[28:38]
    dV = [dV1, dV2, dV3, dV4, dV5, dV6, dV7, dV8, dV9, dV10, dV11, dV12, dV13, dV14]
    
    dVt1, dVt7, dVt8 = 0.0, +32., +25.
    dVt2, dVt3, dVt4, dVt5, dVt6, dVt9, dVt10, dVt11, dVt12, dVt13, dVt14 = @view p[39:49]
    dVt = [dVt1, dVt2, dVt3, dVt4, dVt5, dVt6, dVt7, dVt8, dVt9, dVt10, dVt11, dVt12, dVt13, dVt14]
    
    t01, t07, t08 = 0.0, 9.9, 50.
    t02, t03, t04, t05, t06, t09, t010, t011, t012, t013, t014 = @view p[50:60]
    t0 = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t010, t011, t012, t013, t014]
    
    τϵ1, τϵ7, τϵ8 = 0.0,  66., 530.
    τϵ2, τϵ3, τϵ4, τϵ5,  τϵ6, τϵ9, τϵ10, τϵ11,τϵ12, τϵ13, τϵ14 = @view p[61:71]
    τϵ = [τϵ1, τϵ2, τϵ3, τϵ4, τϵ5, τϵ6, τϵ7, τϵ8, τϵ9, τϵ10, τϵ11,τϵ12, τϵ13, τϵ14]
    
    delta = Float64[0, 0, 0, 0, 0, 0, 0, 450., 0, 0, 10., 0, 0, 0]
    aV4 = Float64[0, 0, 0, 0, 0, 0.790126, 0, 0, 0, 0, 5, 0, 0, 0]
    
    delta[6] = p[72]
    aV4[6], aV4[11] = p[73], p[74]
    
    alpha, beta = albeta(yy[1], V, dV, dVt, aV4, t0, τϵ, delta)
    
    yy1 = yy[1]/Vtemp;
    yy2 = 1+yy1/51 #Computes the (exp(x)-1)/x function to avoid divergence */
    for k in 50:-1:2
        yy2 = 1. + (yy1/k)*yy2
    end

    dyydx[1] = (gNa*yy[2]^3*yy[3]+gNaP*yy[4])*(ENa-yy[1]);
    dyydx[1] += (gA1*yy[13]^4+gA2*yy[5]^4*yy[6]+gK2*yy[7]^4*yy[8]+gC*yy[12])*(EK-yy[1]);
    dyydx[1] += (rlt*yy[9]^2+yy[10]^2*yy[11])*Vtemp*(gou-gin*exp(yy1))/yy2;
    dyydx[1] += gL*(EL-yy[1])+gh*yy[14]*(-43-yy[1])+(Iapp(t)-yy[1]/1550)/Iarea #was yy[0]/2260
    dyydx[1] /= Ccap;
    for k in 2:D::Int64 
        dyydx[k] = alpha[k]*(1-yy[k])-beta[k]*yy[k] 
    end
end

remake_data = false
if remake_data
    Lidx = [1]

    ys = zeros(14)
    ys[1]=-66.15
    ys[2]=1.;  ys[3]=0.;  ys[4]=0.28;  ys[5]=0.679;  ys[6]=0.524;  ys[7]=0.;  ys[8]=0.;
    ys[9]=0.;  ys[10]=0.636;  ys[11]=1.;  ys[12]=0.178;  ys[13]=0.00663;  ys[14]=0.;
    tstart = 0.
    tend = 165.

    Ccap = 0.3 #Capacitance of the nerve fibre - uF/cm^2 */
    gNa = 120. # Sodium conductance - mS/cm^2 */
    gNaP = 0. # Persistent sodium conductance - mS/cm^2 */
    ENa = 50. # Sodium action potential - mV */
    gA1 = 0.7865198 # Potassium conductance of A1 channel - mS/cm^2 */
    gA2 = 1.138630 #  Potassium conductance of A2 channel - mS/cm^2 */
    gK2 = 0. # Potassium conductance of K2 channel - mS/cm^2 */
    gC = 0. # Calcium activated potassium current - mS/cm^2 */
    EK = -90. # Potassium action potential - mV */
    gL = 0.01970969 # Leakage conductance - mS/cm^2 */
    EL = -84.18847 # Leakage action potential - mV */
    gou = 0.01164139 # Conductivity of inward Ca current (outward ions coming in) - mS/cm^2 */
    gin = 1e-4 # Conductivity of outward Ca current (inward ions coming out) - mS/cm^2 */
    gh = 0.1 # Conductivity of h current ~ 0.03 - 0.37ms/cm2 */
    rlt = 5.758333 # ratio between the L and T current conductivity */
    Iarea = 0.151255 # Surface area of injected current - x10^{-3}cm^2 */
    Vtemp = 13. # kT/Ze : temperature equivalent voltage for Ca ions (2+) - mV */

    # # Threshold of gate variable - mV ; i=1...12
    V = [-27.682, -75.5238, -60.997, -65.12, -74.9259, -51.4117, -80., -80.55488, 7., -56.03685, -75.]
    # # Voltage width of gate sigmoid - mV ; i=1...12 
    dV = [ +27.470, -23.8346, +25., +39.8276, -25.6701, +23.8666, +29., -5., 26., 23.78036, -10.]

    dVt = [+5., 5., +33.095, +5., -5., +15.3232, +57., 21.83786, 36., 31.79630, 20.]
    # # Asymptotic value of tau - ms ; i=1...12 */
    t0 = [0.05118, 0.431129, 0.2666 , 0.417067, 156.7156, 0.09, 0.9, 52.836, 0.1, 4.086303, 5]
    # # Width of tau function - ms ; i=1...12 */
    τϵ = [0.0120 , 0.105836, 0.06917, 0.5, 210, 29, 34.857, 64.2225, 6.4, 22.53280, 1000];

    p = [Ccap,gNa,gNaP,ENa,gA1,gA2,gK2,gC,EK,gL,EL,gou,gin,gh,Iarea,rlt, V..., dV..., dVt..., t0..., τϵ..., 0.0, 0.790126, 5.];

    tspan = (0.0, 330.0)
    dt = 0.025
    ϵ = 2.0
    make_data_dynamics = ODEProblem(Nogaret_N2, ys, (tstart, tend), p)
    save_path = "datafiles/N2_twin.txt"
    make_data(make_data_dynamics, dt, ϵ, Lidx, save_path = save_path, plot_data=true)
end

D = 14
Np = 74

lower_init = zeros(D); lower_init[1] = -70.0
upper_init = ones(D);  upper_init[1] = -60.0

lower = [0.3, 70.,  0.,  40., 0.,   0.,  0.,  0.,  -95., 0.01, -120., 0.0, 0.0,  0.03, 0.05, 0.01]
upper = [0.4, 160., 20., 54., 130., 80., 80., 12., -60., 0.6,  -40.,  10., 10.,  0.37, 0.3,  100.]

Vlow = -100*ones(11)
Vup  = zeros(11); Vup[9] = 40

dVlow = [5.0,  -60., 5.0,   5.0,  -60.0, 5.0,  5.,   -60., 5.,   5.,   -60.]
dVup  = [100., -10., 100.0, 100., -10.0, 100., 100., 0.0, 100., 100., 0.0]

dVtlow = [+1.0, 1.0, 1.0, 1.0, -50, 1.0, 5.0,  5.0, 5.0, 5.0, 5.0]
dVtup  = [+50., 50., 50., 50., -1., 50., 100., 50., 50., 50., 50.]

t0low = [0.02, 0.02, 0.02, 0.02, 1.0,  0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
t0up  = [0.7,  0.7,  0.7,  0.7,  200., 1.0,  10.0, 150., 0.7,  20.,  25.]

τϵlow = [0.001, 0.001, 0.001, 0.001, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0];
τϵup  = [0.1,   1.0,   1.0,   1.0,   500., 100., 100., 100., 100., 100., 5000.];

delta_avlow = [0.0,   0.0,  0.0]
delta_avup  = [2000., 10.0, 50.]


start = 1200
num_pts = 45000
num_pts_rmse = 30000
dt = 0.02
path_to_obs = "datafiles/N2_volt.txt"

lower_bounds = [lower_init...,lower..., Vlow..., dVlow..., dVtlow..., t0low..., τϵlow..., delta_avlow...]
upper_bounds = [upper_init...,upper..., Vup..., dVup..., dVtup..., t0up..., τϵup..., delta_avup...]
obs_vars = [1]
rand_p = [rand(truncated(Normal(mean(x), (x[2]-x[1])/4), x[1], x[2])) for x in collect(zip(lower_bounds, upper_bounds))]
dynamics = ODEProblem(Nogaret_N2, rand(Float64, D), (start*dt, (start+num_pts)*dt), rand_p)

config = neuroda(D, Np, start, num_pts, dt, path_to_obs,
                 lower_bounds, upper_bounds, dynamics, obs_vars)

# u0 = init_guess(config, 1200, 25000, 1e-20, u0)

u0 = [-60.03044592275348, 0.08305051640996845, 0.5918820125394436, 0.032502774912230074, 0.9578577736901883, 0.04629529548544211, 0.1675458523007507, 0.06815189950067448, 0.09895678173358906, 0.2003161027660228, 0.7115347072627934, 0.9971046249079425, 0.6582655111851267, 0.7738910314659763, 0.37403393323286555, 128.7125227679851, 3.925093829794128, 40.60251287161679, 64.14212008853946, 4.085067971208834, 2.268414575279405, 0.9514461544647169, -68.2418546160535, 0.13308287441542851, -82.40881670953456, 2.8619409224298624, 3.2865709173213964, 0.2895222220871655, 0.10684980590621901, 2.485005626815731, -0.2610237585670916, -10.742950780708355, -4.7530098146850435, -20.27566204176513, -97.3512696327712, -0.3013152008829678, -13.839895998185526, -32.75937772286731, 39.50938475935925, -20.43755093308482, -95.95201212448666, 19.690295874743235, -10.061640021377151, 5.018312848123933, 46.35845677704906, -11.465981758511944, 13.087683601605352, 44.91768060943197, -28.23981471786049, 39.682665219133305, 10.195079366843393, -50.7820666625916, 14.280138172456926, 4.81929837187022, 9.157953243504348, 24.132618208419498, -40.68443776612459, 47.85059217227679, 99.21390311447914, 32.36151060026927, 17.21139677770007, 7.412800946071803, 13.024026920750519, 0.021079535334036316, 0.11775571257450962, 0.1736662154446968, 0.35456539197699743, 68.16362760178238, 0.5686994765004793, 1.3823333525736219, 131.45402162035006, 0.025149146603503758, 0.7274220789506627, 2.8855462370111296, 0.07509428779616947, 0.6259117113684444, 0.18477840159008044, 0.0012061077999847235, 168.65001844490962, 2.2348869547461847, 1.085183248067053, 75.35123886928726, 2.6165872514534434, 48.38131869945162, 2599.227461043693, 1947.2733522335632, 5.299747491546443, 49.39468787002162]

plot_data_sim(config, u0)

# xmin = run_neuroda(config, u0, num_pts_rmse;
#                     maxtime=360.0, popsize=150, 
#                     ϵ=0.7, spike_thresh=15.0,
#                     σᵪ = 2)

# plot_data_sim(config, xmin)