import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
sys.path.append("..")
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from scipy.integrate import odeint
from single_e_class_unified import single_electron
from scipy.optimize import curve_fit
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/DCV"
files=os.listdir(data_loc)
experimental_dict={}
file_name="/dcV_Cjx-183D_WT_pH_7_3_3"
dcv_file=np.loadtxt(data_loc+file_name, skiprows=2)
dcv_file_time=dcv_file[:,0]
dcv_file_voltage=dcv_file[:,1]
dcv_file_current=dcv_file[:,2]
param_list={
    "E_0":0.2,
    'E_start':  -0.39, #(starting dc voltage - V)
    'E_reverse':0.3,
    'omega':8.94, #8.88480830076,  #    (frequency Hz)
    "v":30*1e-3,
    'd_E': 0,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 0, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    "Cdlinv":0,
    'CdlE1inv': 0,#0.000653657774506,
    'CdlE2inv': 0,#0.000245772700637,
    "CdlE3inv":0,
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
    'phase' :0,
    "time_end": -1,
    'num_peaks': 30,
    'Q':1,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[8],
    "GH_quadrature":True,
    "test": False,
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":range(0, 1),
    "experiment_time": dcv_file_time,
    "experiment_current": dcv_file_current,
    "experiment_voltage":dcv_file_voltage,
    "bounds_val":200,
}
param_bounds={
    'E_0':[-0.06,-0.04],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e5],  #     (uncompensated resistance ohms)
    "Q":[0,1],
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'Cdlinv': [1e-5, 1e-3], #(capacitance parameters)
    'CdlE1inv': [-0.1,0.1],#0.000653657774506,
    'CdlE2inv': [-0.1,0.1],#0.000245772700637,
    'CdlE3inv': [-0.1,0.1],#1.10053945995e-06,
    'gamma': [0.8*param_list["original_gamma"],10*param_list["original_gamma"]],
    'k_0': [5, 1e4], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[-0.06,-0.04],
    "E0_std": [0.045,  0.05],
    "alpha_mean":[0.4, 0.5],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
}
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
plt.plot(voltage_results, current_results)
plt.show()
PSV_optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"]
true_data=current_results

error=0.01*max(true_data)
cyt.param_bounds["Q"][1]=5*error
cyt.def_optim_list(["E0_mean", "E0_std", "k_0", "Ru", "Cdl", "gamma", "alpha", "Q"])
cyt.simulation_options["GH_quadrature"]=True
cyt.simulation_options["numerical_method"]="Kalman_simulate"
cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)

cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=False

score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
num_runs=5
for i in range(0, num_runs):
    x0=abs(np.random.rand(len(cyt.optim_list)))#cyt.change_norm_group(inferred_params, "norm")
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    plt.plot(voltage_results, cmaes_time)
    plt.plot(voltage_results, true_data)
    plt.show()
    cyt.simulation_options["Kalman_capacitance"]=True
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    plt.plot(voltage_results, cyt.farad_current)
    plt.plot(voltage_results, cyt.pred_cap)
    plt.plot(voltage_results, current_results)
    plt.show()
#[-0.04950345403273563, 0.415104966434246, 5.000000006505078, 499.999929294756, 1.4495126487366279e-15, 4.7671138947314476e-11, 0.5163700064087133, 0.2624943522428564]
set_1=[-0.04955459994332252, 0.024409418663214232, 9275.479999845043, 99999.99712409446, 7.607122198928626e-16, 4.447804239083664e-11, 0.4651366220351519, 0.2624943521183878]
set_2=[-0.051771148281119825, 0.029350351522082246, 6669.073576729849, 99999.93040678784, 3.609098295197487e-15, 1.698026851872925e-11, 0.5930110730782626, 0.21646041040994204]
set_3=[-0.05215086968395874, 0.01842669773591587, 5.000000443751222, 99999.97439365527, 3.469009482232843e-14, 1.0179800910532819e-11, 0.599995808518657, 0.20877245570182038]
