import numpy as np
try:
    import matplotlib.pyplot as plt
    plot=True
    from harmonics_plotter import harmonics
except:
    print("No plotting for ben_rama")
    plot=False
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="1"
freq="_9_"
dec_amount=8
for file in files:
    if scan in file and freq in file:

        if "current" in file:
            current_data=np.loadtxt(data_loc+"/"+file)
        elif "voltage" in file:
            voltage_data=np.loadtxt(data_loc+"/"+file)
try:
    current_results=current_data[0::dec_amount,1]
    time_results=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")

param_list={
    "E_0":0.2,
    'E_start':  min(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #(starting dc voltage - V)
    'E_reverse':max(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]),
    'omega':8.94,#8.88480830076,  #    (frequency Hz)
    "original_omega":8.94,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' : 3*(math.pi/2),
    "time_end": None,
    'num_peaks': 30
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(4,12,1)),
    "experiment_time": time_results,
    "experiment_current": current_results,
    "experiment_voltage":voltage_results,
    "bounds_val":20,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
    'k_0': [50, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[0.2, 0.3],
    "E0_std": [1e-5,  0.1],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
    "noise":[0, 100]
}
noramp_fit=single_electron(None, param_list, simulation_options, other_values, param_bounds)
print(noramp_fit.nd_param.c_I0)
noramp_fit.define_boundaries(param_bounds)
time_results=noramp_fit.other_values["experiment_time"]
current_results=noramp_fit.other_values["experiment_current"]
voltage_results=noramp_fit.other_values["experiment_voltage"]
noramp_fit.dim_dict["noise"]=0
noramp_fit.dim_dict["phase"]=3*math.pi/2
print(len(current_results))
#noramp_fit.def_optim_list(["E_0","k0_shape", "k0_scale","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
noramp_fit.simulation_options["dispersion_bins"]=[5,5]
noramp_fit.simulation_options["GH_quadrature"]=True
noramp_fit.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
vals=[-0.2, 100, 100,1e-5, 0,0, 5.609826904007704e-11,8.94086695405389, 3*math.pi/2, 3*math.pi/2, 0.40000750225452375]
true_signal=noramp_fit.test_vals(vals, "timeseries")
test_data=noramp_fit.add_noise(true_signal, 0*0.05*max(true_signal))
#noramp_fit.simulation_options["alpha_dispersion"]="uniform"
#noramp_fit.def_optim_list(["Ru","Cdl","CdlE1", "CdlE2",'omega',"phase","cap_phase"])
#noramp_fit.dim_dict["gamma"]=0
true_data=test_data
plt.plot(voltage_results, true_data)
plt.show()
fourier_arg=noramp_fit.top_hat_filter(test_data)
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(noramp_fit, time_results, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(noramp_fit, dummy_times, fourier_arg)
score = pints.SumOfSquaresError(cmaes_problem)#[4.56725844e-01, 4.44532637e-05, 2.98665132e-01, 2.96752050e-01, 3.03459391e-01]#
CMAES_boundaries=pints.RectangularBoundaries(list([np.zeros(len(noramp_fit.optim_list))]), list([np.ones(len(noramp_fit.optim_list))]))
noramp_fit.simulation_options["label"]="cmaes"
noramp_fit.simulation_options["test"]=False
#noramp_fit.simulation_options["test"]=True
num_runs=5
param_mat=np.zeros((num_runs,len(noramp_fit.optim_list)))
score_vec=np.ones(num_runs)*1e6

for i in range(0, num_runs):
    x0=abs(np.random.rand(noramp_fit.n_parameters()))#noramp_fit.change_norm_group(gc4_3_low_ru, "norm")
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(not noramp_fit.simulation_options["test"])
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=noramp_fit.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=noramp_fit.test_vals(cmaes_results, likelihood="timeseries", test=False)
    #plt.subplot(1,2,1)
    plt.plot(voltage_results, cmaes_time)
    plt.plot(voltage_results, true_signal)
    plt.plot(voltage_results, test_data, alpha=0.5)
    #plt.subplot(1,2,2)
    #plt.plot(time_results, noramp_fit.define_voltages()[noramp_fit.time_idx:])
    #plt.plot(time_results, voltage_results)
    plt.show()
    #cmaes_fourier=noramp_fit.test_vals(cmaes_results, likelihood="fourier", test=False)
    param_mat[i,:]=cmaes_results
    score_vec[i]=found_value
    print("Finish?")
    #i, o, e = select.select( [sys.stdin], [], [], 5)
    #if len(i) != 0:
    #    break
