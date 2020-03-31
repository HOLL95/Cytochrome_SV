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
dec_amount=32
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
        "E_0":0.2, #Midpoint potnetial (V)
        "E0_mean": 0.2,
        "E0_std":0.01,
        'E_start': min(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #Sinusoidal input minimum (or starting) potential (V)
        'E_reverse': max(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #Sinusoidal input maximum potential (V)
        'omega':8.94,   #frequency Hz
        "original_omega":8.94, #Nondimensionalising value for frequency (Hz)
        'd_E': 300e-3,   #ac voltage amplitude - V
        'area': 0.07, #electrode surface area cm^2
        'Ru': 100.0,  #     uncompensated resistance ohms
        'Cdl': 1e-5, #capacitance parameters
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,   # surface coverage per unit area
        "original_gamma":1e-10,        # Nondimensionalising cvalue for surface coverage
        "k0_shape":0,
        "k0_scale":0,
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5, #(Symmetry factor)
        'phase' : 3*(math.pi/2),#Phase of the input potential
        "cap_phase":3*(math.pi/2),
        'sampling_freq' : (1.0/800),
        "noise":0
    }
likelihood_options=["timeseries", "fourier"]
simulation_options={
        "no_transient":2/param_list["omega"],
        "experimental_fitting":True,
        "method": "sinusoidal",
        "likelihood":"timeseries",
        "phase_only":False,
        "dispersion_bins":[16],
        "dispersion_distributions":["lognormal"],
        "label": "cmaes",
        "GH_quadrature": False,
        "optim_list":[],
    }

other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(1,9,1)),
        "experiment_current":current_results,
        "experiment_time":time_results,
        "experiment_voltage":voltage_results,
        "num_peaks": 30,
        "bounds_val":20,
    }
param_bounds={
    'E_0':[param_list["E_start"], param_list["E_reverse"]],#[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.15,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-12,1e-10],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0, 2*math.pi],
    "E0_mean":[param_list["E_start"], param_list["E_reverse"]],
    "E0_std": [1e-5,  0.2],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_scale":[0,1e4],
    "k0_shape":[0, 1],
    'phase' : [0, 2*math.pi],
    "noise":[0, 100]
}
cyt=single_electron(file_name=None, dim_parameter_dictionary=param_list, simulation_options=simulation_options, other_values=other_values, param_bounds=param_bounds)
nd_current=cyt.other_values["experiment_current"]
nd_voltage=cyt.other_values["experiment_voltage"]
nd_time=cyt.other_values["experiment_time"]
print(param_list["E_start"],param_list["E_reverse"] )
cyt.def_optim_list(["E_0","k_0","Cdl","CdlE1", "CdlE2","Ru", "omega", "gamma", "alpha", "phase", "cap_phase"])

vals=[-0.2, 100, 1e-4, 0, 0, 100.6737816215855, 8.94086695405389, 5.609826904007704e-11, 0.40000750225452375, 3*math.pi/2, 3*math.pi/2]
test_data=cyt.test_vals(vals, "timeseries")
plt.plot(nd_voltage, test_data)
plt.show()
filtered=cyt.top_hat_filter(nd_current)
#plt.plot(nd_time, filtered)
#plt.plot(nd_voltage, nd_current, alpha=0.7)
#plt.legend()
#plt.show()
harms=harmonics(cyt.simulation_options["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.05)
data_harmonics=harms.generate_harmonics(nd_time, nd_current)
#for i in range(0, len(data_harmonics)):
#    plt.subplot(len(data_harmonics), 1, i+1)
#    plt.plot(nd_voltage, data_harmonics[i,:])
#plt.show()
#cyt.def_optim_list(["E0_mean", "E0_std","k_0","Cdl","CdlE1", "CdlE2","CdlE3","Ru", "omega", "gamma", "alpha", "phase", "cap_phase"])

true_data=test_data
fourier_arg=cyt.top_hat_filter(test_data)
plt.plot(fourier_arg)
plt.legend()
plt.show()
cyt.secret_data_fourier=fourier_arg
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(cyt, nd_time, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
score = pints.SumOfSquaresError(cmaes_problem)#[4.56725844e-01, 4.44532637e-05, 2.98665132e-01, 2.96752050e-01, 3.03459391e-01]#
CMAES_boundaries=pints.RectangularBoundaries(list([np.zeros(len(cyt.optim_list))]), list([np.ones(len(cyt.optim_list))]))
cyt.simulation_options["test"]=False
num_runs=20
param_mat=np.zeros((num_runs,len(cyt.optim_list)))
score_vec=np.ones(num_runs)*1e6
voltages=cyt.define_voltages(transient=True)
for i in range(0, num_runs):
    x0=abs(np.random.rand(cyt.n_parameters()))
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    paralell=False#not cyt.simulation_options["test"]
    print(paralell)
    cmaes_fitting.set_parallel(paralell)
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    if plot==True:
        if cyt.simulation_options["likelihood"]=="timeseries":
            plt.subplot(1,2,1)
            plt.plot(nd_voltage, cmaes_time)
            plt.plot(nd_voltage, true_data, alpha=0.7)
            plt.subplot(1,2,2)
            plt.plot(nd_time, cmaes_time)
            plt.plot(nd_time, true_data, alpha=0.7)
            plt.show()
        elif cyt.simulation_options["likelihood"]=="fourier":
            cyt.test_vals(cmaes_results, likelihood="fourier", test=True)
            harms=harmonics(cyt.simulation_options["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.05)
            data_harmonics=harms.generate_harmonics(nd_time, true_data)
            sim_harmonics=harms.generate_harmonics(nd_time, cmaes_time)
            for i in range(0, len(data_harmonics)):
                plt.subplot(len(data_harmonics), 1, i+1)
                plt.plot(nd_time, sim_harmonics[i,:])
                plt.plot(nd_time, data_harmonics[i,:], alpha=0.7)
            plt.show()
