import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import sys
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
import time
from single_e_class_unified import single_electron
from scipy.integrate import odeint
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/PSV"
files=os.listdir(data_loc)
experimental_dict={}
param_file=open(data_loc+"/PSV_params", "r")
useful_params=dict(zip(["max", "min", "Amp[0]", "Freq[0]"], ["E_reverse", "E_start", "d_E", "original_omega"]))
dec_amount=32
for line in param_file:
    split_line=line.split()
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]

for i in range(1, 2):
    file_name="PSV_Cyt_{0}_cv_".format(i)
    current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
        'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
        'omega':9.015120071612014, #8.88480830076,  #    (frequency Hz)
        "original_omega":9.015120071612014,
        'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 1.0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        "CdlE3":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 10, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "E0_skew":0.2,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :3*math.pi/2,
        "time_end": -1,
        'num_peaks': 30,
    }
    print(param_list)
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["original_omega"])
    simulation_options={
        "no_transient":time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[20],
        "GH_quadrature":True,
        "test": False,
        "method": "sinusoidal",
        "phase_only":False,
        "likelihood":likelihood_options[1],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }

    other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(4,100,1)),
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":20000,
    }
    param_bounds={
        'E_0':[-0.1, 0.1],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 3e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,8e-4], #(capacitance parameters)
        'CdlE1': [-0.1,0.1],#0.000653657774506,
        'CdlE2': [-0.05,0.05],#0.000245772700637,
        'CdlE3': [-0.05,0.05],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [0, 1e4], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.1, -0.04],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [math.pi, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    print(current_results[0], current_results[-1])
    voltage_results=cyt.other_values["experiment_voltage"]
    plt.plot(voltage_results, current_results)
    plt.show()
    h_class=harmonics(list(range(2, 12)), 1, 0.05)
    #h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, xaxis=voltage_results)
    fft=one_tail(np.fft.fft(current_results))
    hann_fft=one_tail(np.fft.fft(np.multiply(np.hanning(len(current_results)),current_results)))
    f=one_tail(np.fft.fftfreq(len(time_results), time_results[1]-time_results[0]))
    predicted_cap_params=[0.00016107247651709253, 0.0032886486609914056, 0.0009172547160104724]
    cap_param_list=["Cdl", "CdlE1", "CdlE2"]
    #for i in range(0, len(cap_param_list)):
    #    cyt.param_bounds[cap_param_list[i]]=[predicted_cap_params[i]*0.75, predicted_cap_params[i]*1.25]
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
    inferred_params4=[-0.08064031224498379, 0.020906827217859487, 63.01964378454537, 112.80693965100555, 0.0007989280702348236, -0.008716352699352406, 0.0012650098345725197, -2.60067208995296e-05, 2.9678863151432805e-11, 9.014741538924035, 5.49549816495015, 5.193383660743178, 0.5999999698409333]


    true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)


    cmaes_test=cyt.test_vals(inferred_params4, "timeseries")
    #plt.plot(cmaes_test)
    #plt.show()
    h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, data_time_series=cmaes_test,   xaxis=voltage_results, alpha_increment=0.3)

    plt.plot(fourier_arg)
    plt.plot(cyt.top_hat_filter(cmaes_test))
    plt.show()
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase"])
    cyt.dim_dict["alpha"]=0.6
    cyt.simulation_options["label"]="MCMC"
    if simulation_options["likelihood"]=="timeseries":
        MCMC_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
    elif simulation_options["likelihood"]=="fourier":
        dummy_times=np.linspace(0, 1, len(fourier_arg))
        MCMC_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
        plt.plot(fourier_arg)
        plt.show()
    error=np.std(np.abs(np.subtract(fourier_arg, cyt.top_hat_filter(cmaes_test))))

    updated_lb=[param_bounds[x][0] for x in cyt.optim_list]+[0.1*error]
    updated_ub=[param_bounds[x][1] for x in cyt.optim_list]+[10*error]
    updated_b=[updated_lb, updated_ub]
    updated_b=np.sort(updated_b, axis=0)
    log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
    log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
    log_posterior=pints.LogPosterior(log_liklihood, log_prior)
    mcmc_parameters=inferred_params4[:-1]

    mcmc_parameters=np.append(mcmc_parameters, error)
    xs=[mcmc_parameters,
        mcmc_parameters,
        mcmc_parameters
        ]
    mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)
    mcmc.set_parallel(True)
    mcmc.set_max_iterations(10000)
    chains=mcmc.run()
    #plot.trace(chains)
    #plt.show()
    f=open("alice_cyt_1_MCMC", "wb")
    np.save(f, chains)
    f.close()
