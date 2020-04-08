import numpy as np
import matplotlib.pyplot as plt
import sys
plot=True
from harmonics_plotter import harmonics
import os
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
    'E_start':  -500e-3, #(starting dc voltage - V)
    'E_reverse':100e-3,
    'omega':8.94, #8.88480830076,  #    (frequency Hz)
    "original_omega":8.94,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
    'phase' :0.1,
    "time_end": None,
    'num_peaks': 30,
    "noise_00":None,
    "noise_01":None,
    "noise_10":None,
    "noise_11":None,
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
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),
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
    'k_0': [0.1, 1e3], #(reaction rate s-1)
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
    "noise":[0, 100],
    "noise_00":[-1e10, 1e10],
    "noise_01":[-1e10, 1e10],
    "noise_10":[-1e10, 1e10],
    "noise_11":[-1e10, 1e10],
}
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
print(cyt.nd_param.c_I0)
cyt.define_boundaries(param_bounds)
time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
cyt.dim_dict["noise"]=0
cyt.dim_dict["phase"]=3*math.pi/2
print(len(current_results))
#cyt.def_optim_list(["E_0","k0_shape", "k0_scale","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
cyt.simulation_options["dispersion_bins"]=[5]
cyt.simulation_options["GH_quadrature"]=True
cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])

reduced_list=["E_0","k_0","Ru","gamma","omega","cap_phase","phase", "alpha"]
vals=[-0.2115664620575202,  50.70216501235601, 1.2299591730542196e-08, 1e-5, 0.02019028489306987, 0.0021412236896496805, 7.777463350920317e-11, 8.940744445900455, 4.371233348216213, 3.393780500934783, 0.5310264089857374]
true_signal=cyt.test_vals(vals, "timeseries")
f_true=cyt.test_vals(vals, "fourier")
test_data=cyt.add_noise(true_signal, 0.005*max(true_signal))
true_data=test_data
#true_data=current_results
fourier_arg=cyt.top_hat_filter(true_data)

cov=np.cov(true_data)
test_fourier=cyt.test_vals(vals, "fourier", test=False)
harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
data_harmonics=harms.generate_harmonics(time_results,(current_results))
syn_harmonics=harms.generate_harmonics(time_results, (test_data))

"""
fig, ax=plt.subplots(len(data_harmonics), 1)
for i in range(0, len(data_harmonics)):

    ax[i].plot(time_results, (syn_harmonics[i,:]))
    ax[i].plot(time_results, (data_harmonics[i,:]), alpha=0.7)
    ax[i].plot(time_results, np.subtract(data_harmonics[i,:],syn_harmonics[i,:]), alpha=0.7)
    ax2=ax[i].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel(other_values["harmonic_range"][i])
plt.show()"""


#cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase", "phase", "alpha"])#, "noise_00", "noise_01", "noise_10", "noise_11"
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
score = pints.GaussianLogLikelihood(cmaes_problem)#[4.56725844e-01, 4.44532637e-05, 2.98665132e-01, 2.96752050e-01, 3.03459391e-01]#
"""x0=np.random.rand(cyt.n_parameters())
x0_vals=[-0.4398655594202501, 4.718608302867817, 25.780060141795115, 0.0009607039629253518, 0.014366036483479602, -0.008964590734026388, 9.65856012375879e-11, 8.623237722997326, 5.412009394070987, 5.309059493022305, 0.4193561932667581]


x0_vals=vals


signal=cyt.test_vals(x0_vals, "fourier")
error=(fourier_arg-signal)
error=np.fft.fft(error)
#plt.plot(np.real(error))
print(score.__call__(np.append(x0_vals, [np.std(np.real(error)), 0, 0, np.std(np.imag(error))])))"""






CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list)+1)), list(np.ones(len(cyt.optim_list)+1)))
cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=False
#cyt.simulation_options["test"]=True
num_runs=5
param_mat=np.zeros((num_runs,len(cyt.optim_list)))
score_vec=np.ones(num_runs)*1e6

for i in range(0, num_runs):
    x0=abs(np.random.rand(cyt.n_parameters()+1))#cyt.change_norm_group(gc4_3_low_ru, "norm")
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(False)
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:-1], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    #plt.subplot(1,2,1)
    #plt.plot(voltage_results, cmaes_time)
    #plt.plot(voltage_results, true_data, alpha=0.5)
    #plt.subplot(1,2,2)
    #plt.plot(time_results, cyt.define_voltages()[cyt.time_idx:])
    #plt.plot(time_results, voltage_results)
    #plt.show()
    #cmaes_fourier=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
    #param_mat[i,:]=cmaes_results
    #score_vec[i]=found_value
    #print("Finish?")
    #i, o, e = select.select( [sys.stdin], [], [], 5)
    #if len(i) != 0:
    #    break
