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
    "E_0":0.2,
    'E_start': -0.3679743 , #(starting dc voltage - V)
    'E_reverse': 0.2320027,
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
    'sampling_freq' : (1.0/400),
    'phase' : 3*(math.pi/2),
}
solver_list=["Brent minimisation"]
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
    "likelihood":"fourier",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[],

}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),
    "experiment_time": time_results,
    "experiment_current": current_results,
    "experiment_voltage":voltage_results,
    "bounds_val":20,
    "num_peaks":30,
}
param_bounds={
    'E_0':[ -0.35, -0.35+(300e-3*2)],#[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-4], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-12,1e-10],
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
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
print(cyt.nd_param.c_I0)
nd_time=cyt.other_values["experiment_time"]
nd_current=cyt.other_values["experiment_current"]
nd_voltage=cyt.other_values["experiment_voltage"]
cyt.dim_dict["noise"]=0
cyt.dim_dict["phase"]=3*math.pi/2

#cyt.def_optim_list(["E_0","k0_shape", "k0_scale","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
cyt.simulation_options["dispersion_bins"]=[7,7]
cyt.simulation_options["GH_quadrature"]=True
vals=[-0.2, 100, 100,1e-4, 0, 0, 8.94086695405389, 5.609826904007704e-11, 3*math.pi/2, 3*math.pi/2, 0.40000750225452375]
cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","omega","gamma","cap_phase","phase", "alpha"])
test_data=cyt.test_vals(vals, "timeseries")
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
    """
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
        """
