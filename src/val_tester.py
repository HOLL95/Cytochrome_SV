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
        "E_0":0.2, #Midpoint potnetial (V)
        "E0_mean": 0.2,
        "E0_std":0.01,
        'E_start': min(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #Sinusoidal input minimum (or starting) potential (V)
        'E_reverse': max(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #Sinusoidal input maximum potential (V)
        'omega':8.94,   #frequency Hz
        "original_omega":8.94, #Nondimensionalising value for frequency (Hz)
        'd_E': 300e-3,   #ac voltage amplitude - V
        'area': 0.125, #electrode surface area cm^2
        'Ru': 100.0,  #     uncompensated resistance ohms
        'Cdl': 1e-5, #capacitance parameters
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-11,   # surface coverage per unit area
        "original_gamma":1e-11,        # Nondimensionalising cvalue for surface coverage
        "k0_shape":0,
        "k0_scale":0,
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5, #(Symmetry factor)
        'phase' : 3*(math.pi/2),#Phase of the input potential
        "cap_phase":3*(math.pi/2),
        'sampling_freq' : (1.0/400),
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
        "harmonic_range":list(range(3,9,1)),
        "experiment_current":current_results,
        "experiment_time":time_results,
        "experiment_voltage":voltage_results,
        "num_peaks":20,
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
    "E0_mean":[param_list["E_start"], 0],
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
cyt.def_optim_list(["E_0", "k_0", "Cdl","CdlE1", "CdlE2","CdlE3","Ru", "omega", "gamma", "alpha", "phase", "cap_phase"])
inf_params=[-0.26742665869741566, 50.00000000019233, 5.8652622271648164e-05, 0.009842939588946886, 0.0005954905413801843, 7.701265437505891e-06, 8.206747435756358e-08, 8.940664700078155, 1.7888689475894705e-11, 0.5999999999956317, 4.9913707115353425, 4.313702716197615]
cmaes=cyt.test_vals(inf_params, "timeseries")
plt.subplot(1,2,1)
plt.plot(nd_voltage, cmaes)
plt.plot(nd_voltage, nd_current, alpha=0.7)
plt.xlabel("ND voltage")
plt.ylabel("ND current")
plt.subplot(1,2,2)
where_idx=tuple(np.where((nd_time>2) & (nd_time<4)))
plt.plot(nd_time[where_idx], cmaes[where_idx])
plt.plot(nd_time[where_idx], nd_current[where_idx], alpha=0.7)
plt.xlabel("ND time")
plt.ylabel("ND current")
plt.show()
