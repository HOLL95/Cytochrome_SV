import numpy as np
import matplotlib.pyplot as plt
plot=True
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/Ramped"
files=os.listdir(data_loc)
scan="_1_"
freq="_9Hz"
dec_amount=1
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
    'omega':8.88,#8.88480830076,  #    (frequency Hz)
    "v":    22.35174e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "original_gamma":1e-10,        # (surface coverage per unit area)
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
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "ramped",
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
    "bounds_val":200,
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
time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
voltages=cyt.define_voltages(transient=True)
plt.plot(cyt.e_nondim(voltage_results), current_results)
plt.show()
harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
data_harmonics=harms.generate_harmonics(time_results,(current_results))
cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha", "noise_00", "noise_01", "noise_10", "noise_11"])

vals=[-0.22,1000.37267320998559, 900.9999999999768, 0.00001226316755359932*0, 0.010515263489958092*0, -0.0014258396718951558*0, 1.037611264615161e-11, 8.880736406988611, 0, 0.40000000000002656]
syn_time=cyt.test_vals(vals, "timeseries")
syn_harmonics=harms.generate_harmonics(time_results,(syn_time))
plt.plot(syn_time)
plt.show()
fig, ax=plt.subplots(len(data_harmonics), 1)
quake=500
for i in range(0, len(data_harmonics)):
    ax[i].plot(time_results[quake:-quake], (data_harmonics[i,:][quake:-quake]),  label="Data")
    ax[i].plot(time_results[quake:-quake], syn_harmonics[i,:][quake:-quake], alpha=0.7, label="Simulated")
    ax2=ax[i].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
    if i==0:
        ax[i].legend(loc="upper right")
    if i==len(data_harmonics)-1:
        ax[i].set_xlabel("Nondim voltage")
    if i==3:
        ax[i].set_ylabel("Nondim current")
plt.show()
