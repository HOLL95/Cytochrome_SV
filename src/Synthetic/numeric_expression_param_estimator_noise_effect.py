from single_e_anal import analytical_electron
import sys
sys.path.append('..')
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt
import pints
from pints.plot import trace
freq_range=[5*(10**x) for x in np.arange(1,2)]
freq_range=[50]
noises=np.linspace(-3, -1, 20)
noises=[x/100 for x in [0.5, 1, 2, 3, 4, 5]]

num_oscillations=[10, 50, 100, 200, 300]
time_ends=[10.0, 5.0,0.7, 0.07, 0.015, 0.015]
num_freqs=len(freq_range)
T=(273+25)
F=96485.3328959
R=8.314459848
plt.figure(num=None, figsize=(9,10), dpi=120, facecolor='w', edgecolor='k')
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))
def chain_appender(chains, burn=5000):
    first_chain=chains[0, burn:]
    for i in range(1, len(chains[:, 0])):
        first_chain=np.append(first_chain, chains[i, burn:])
    return first_chain
d_E_vals=[3*((R*T)/F)]
desired_harms=list(range(2, 5))
estimate_array=np.zeros(len(noises))
estimate_stds=np.zeros(len(noises))
num_noise_repeats=1
big_array=[]
for lcv_0 in range(0, len(noises)):
    noise_repeat_array=np.zeros(num_noise_repeats)
    for noise_repeats in range(0, num_noise_repeats):
        for lcv_1 in range(0, num_freqs):
            peak_heights=np.zeros((2, len(desired_harms)))
            E_ins=[0,0]
            for lcv_2 in range(0, len(d_E_vals)):
                estart=E_ins[lcv_2]-d_E_vals[lcv_2]
                erev=E_ins[lcv_2]+d_E_vals[lcv_2]
                param_list={
                    'E_start':  estart, #(starting dc voltage - V)
                    'E_reverse':  erev ,   #  (reverse dc voltage - V)
                    'omega':freq_range[lcv_1],#8.88480830076,  #    (frequency Hz)
                    'd_E': d_E_vals[lcv_2],   #(ac voltage amplitude - V) freq_range[j],#
                    'original_omega':freq_range[lcv_1],
                      #       (scan rate s^-1)
                    'area': 0.03, #(electrode surface area cm^2)
                    'Ru': 0.0,  #     (uncompensated resistance ohms)
                    'Cdl': 1e-6*0, #(capacitance parameters)
                    'CdlE1': 0,#0.000653657774506,
                    'CdlE2': 0,#0.000245772700637,
                    'CdlE3': 0,#1.10053945995e-06,
                    'gamma': 9.5e-12,
                    'original_gamma':9.5e-12,      # (surface coverage per unit area)
                    'E_0': 1*((R*T)/F),      #       (reversible potential V)
                    'k_0': 1, #(reaction rate s-1)
                    'alpha': 0.55,
                    'sampling_freq' : (1.0/400),
                    'phase' : 0,
                    'time_end':-1,
                    "num_peaks":25
                }
                simulation_options={
                    "no_transient":False,
                    "numerical_debugging": False,
                    "experimental_fitting":False,
                    "dispersion":False,
                    "dispersion_bins":None,
                    "test": False,
                    "method": "sinusoidal",
                    "phase_only":True,
                    "likelihood":"timeseries",
                    "numerical_method": "Brent minimisation",
                    "label": "MCMC",
                    "optim_list":[]
                }
                other_values={
                    "filter_val": 0.5,
                    "harmonic_range":list(range(3, 8)),
                    "bounds_val":20000,
                }
                param_bounds={
                    'E_0':[param_list['E_start'],param_list['E_reverse']],
                    'omega':[0.1,1e5],#8.88480830076,  #    (frequency Hz)
                    'Ru': [0, 1000],  #     (uncompensated resistance ohms)
                    'Cdl': [0,1e-1], #(capacitance parameters)
                    'CdlE1': [-0.05,0.15],#0.000653657774506,
                    'CdlE2': [-0.01,0.01],#0.000245772700637,
                    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
                    'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
                    'k_0': [0.1, 1e6], #(reaction rate s-1)
                    'alpha': [0.4, 0.6],
                    "alpha_mean":[0.4, 0.65],
                    "alpha_std":[1e-3, 0.3],
                    "k0_shape":[0,1],
                    "k0_scale":[0,1e4],
                    "k0_range":[1e2, 1e4],
                    'phase' : [0, 2*math.pi],
                }
                E_ins[lcv_2]=(estart+erev)/2
                anal=analytical_electron(param_list, 0.0)
                time_range=np.linspace(0,param_list["num_peaks"]/param_list["omega"], int(param_list["num_peaks"]*(1/param_list["sampling_freq"])))
                non_dim_time_range=time_range/anal.nd_param.c_T0
                time_len=len(time_range)
                nondim_i=anal.nd_param.c_I0
                harmonic_range=np.arange(1,10,1)
                numerical=single_electron(None, param_list, simulation_options, other_values, param_bounds)
                #synthetic_data=numerical.i_nondim(numerical.test_vals([], "timeseries"))
                numerical_time=numerical.t_nondim(numerical.time_vec)
                numerical.def_optim_list(["E_0", "k_0","alpha"])
                true_params=[param_list[x] for x in numerical.optim_list]
                i_val=numerical.test_vals(true_params, "timeseries")
                #interped_anal=np.interp(numerical_time, time_range, dim_i)
                harms=harmonics(desired_harms, param_list["omega"], 0.1)
                error=noises[lcv_0]*max(i_val)
                noisy_i=numerical.add_noise(i_val, error)
                mcmc_problem=pints.SingleOutputProblem(numerical, numerical.time_vec, noisy_i)
                updated_lb=[param_bounds[x][0] for x in numerical.optim_list]+[0.1*error]
                updated_ub=[param_bounds[x][1] for x in numerical.optim_list]+[10*error]
                updated_b=[updated_lb, updated_ub]
                updated_b=np.sort(updated_b, axis=0)
                for i in range(0, len(true_params)):
                    print(true_params[i], updated_b[0][i], updated_b[1][i])
                log_liklihood=pints.GaussianLogLikelihood(mcmc_problem)
                log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
                log_posterior=pints.LogPosterior(log_liklihood, log_prior)
                print(true_params)
                mcmc_parameters=true_params+[error]
                xs=[mcmc_parameters,
                    mcmc_parameters,
                    mcmc_parameters
                    ]
                print(xs)
                mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)
                mcmc.set_parallel(True)
                mcmc.set_max_iterations(1000)
                chains=mcmc.run()

                single_chains=chain_appender(chains, 0)
                print(single_chains)
                trace(chains)
                plt.show()
    big_array.append({str(error): single_chains})
f=open("Numerical_anal_comp_MCMC", "wb")
np.save(f, big_array)
f.close()
