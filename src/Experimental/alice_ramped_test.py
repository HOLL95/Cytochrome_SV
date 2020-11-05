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
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/FTACV"
files=os.listdir(data_loc)
experimental_dict={}
param_file=open(data_loc+"/FTACV_params", "r")
useful_params=dict(zip(["max", "start", "Amp[0]", "freq", "rate"], ["E_reverse", "E_start", "d_E", "omega", "v"]))
dec_amount=64
for line in param_file:
    split_line=line.split()
    print(split_line)
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]

for i in range(1, 3):
    file_name="FTACV_Cyt_{0}_cv_".format(i)
    current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  -0.33898177567074794, #(starting dc voltage - V)
        'E_reverse':0.26049326614698887,
        'omega':8.959294996508683, #8.88480830076,  #    (frequency Hz)
        "v":0.022316752195354346,
        'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        'phase' :6.283185307179562,
        "time_end": -1,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["omega"])
    simulation_options={
        "no_transient":False,#time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":16,
        "test": False,
        "method": "ramped",
        "phase_only":False,
        "likelihood":likelihood_options[0],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }

    other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(4,9,1)),
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":200,
    }
    param_bounds={
        'E_0':[-0.1, 0.0],
        "E_start":[0.9*param_list["E_start"], 1.1*param_list["E_start"]],
        "E_reverse":[0.9*param_list["E_reverse"], 1.1*param_list["E_reverse"]],
        "v":[0.9*param_list["v"], 1.1*param_list["v"]],
        'omega':[0.8*param_list['omega'],1.2*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.05,0.05],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[param_list["E_reverse"],param_list["E_start"]],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [0, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    print(current_results[0], current_results[-1])
    voltage_results=cyt.other_values["experiment_voltage"]
    h_class=harmonics(other_values["harmonic_range"], param_list["omega"]*cyt.nd_param.c_T0, 0.05)
    h_class.plot_harmonics(time_results, experimental_time_series=current_results, hanning=True, plot_func=abs)
    volts=cyt.define_voltages()
    plt.plot(cyt.time_vec, cyt.e_nondim(volts)*1e3)
    plt.plot(time_results, cyt.e_nondim(voltage_results)[cyt.time_idx]*1e3)
    plt.show()

    cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])
    inferred_params=[-0.3353782900811744, 330.32351080703046, 1.782826365614962e-11, 0.00012080435738781963, 0.0025267358958767356, 0.0006879410370079902, 1.9057794524924306e-11, param_list["omega"], 0, 0.4092580171166158]
    inferred_params=[0.0970288663866738, 362.510776540853, 9248.005110326583, 4.9897881784094635e-03, 0.09999999902298948*0, -0.005266814814752714, 2.9148497981974626e-11, param_list["omega"], 0, 0.5987156826440815]
    ramped_inferred=[-0.04485376873500503, 293.2567587982391, 146.0113118472105, 0.0001576519851347672, 0.006105674536299788, 0.0012649370988525588, 2.2215281961212185e-11, 8.959294996508683, 6.147649245979944, 0.5372803774088237]
    inferred_params=[-0.021495031150668878, 17.570527719697008, 1949.3033882011057, 0.0001576519851347672, 0.006105674536299788, 0.0012649370988525588, 2.2215281961212185e-10, 8.959294996508683, 6.147649245979944, 0.5372803774088237]
    cmaes_test=cyt.test_vals(inferred_params, "timeseries")
    #w0 = [current_results[0],0, voltage_results[0]]
    #wsol = odeint(cyt.current_ode_sys, w0, time_results)
    #adaptive_current=wsol[:,0]
    #adaptive_potential=wsol[:,2]
    #adaptive_theta=wsol[:, 1]
    cyt.simulation_options["method"]="dcv"
    dcv_volt=cyt.e_nondim(cyt.define_voltages()[cyt.time_idx])
    cyt.simulation_options["method"]="ramped"
    h_class.plot_harmonics(time_results, experimental_time_series=current_results, simulated_time_series=cmaes_test, hanning=True, plot_func=abs, xaxis=dcv_volt)
    cyt.simulation_options["label"]="cmaes"
    cyt.simulation_options["test"]=False
    cyt.simulation_options["voltage_only"]=False
    true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)
    if simulation_options["likelihood"]=="timeseries":
        cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
    elif simulation_options["likelihood"]=="fourier":
        dummy_times=np.linspace(0, 1, len(fourier_arg))
        cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
    score = pints.SumOfSquaresError(cmaes_problem)
    CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
    x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(not cyt.simulation_options["test"])
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    #plt.subplot(1,2,1)
    plt.plot(time_results, cmaes_time)
    plt.plot(time_results, voltage_results, alpha=0.5)
    plt.show()
    h_class.plot_harmonics(time_results, experimental_time_series=current_results, simulated_time_series=cmaes_time, hanning=True, plot_func=abs)
