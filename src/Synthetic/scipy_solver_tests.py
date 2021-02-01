import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
plot=True
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from scipy_solver_class import scipy_funcs
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="3"
freq="_9_"
dec_amount=8
for file in files:
    if scan in file and freq in file:

        if "current" in file:
            current_data=np.loadtxt(data_loc+"/"+file)
        elif "voltage" in file:
            voltage_data=np.loadtxt(data_loc+"/"+file)
try:
    current_results1=current_data[0::dec_amount,1]
    time_results1=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results1=voltage_data[0::dec_amount,1]

except:
    raise ValueError("No voltage file of that scan and frequency found")
harm_range=list(range(3,8,2))
num_harms=len(harm_range)+1


def flatten(something):
    if isinstance(something, (list, tuple, set, range)):
        for sub in something:
            yield from flatten(sub)
    elif isinstance(something, dict):
        for key in something.keys():
            yield from flatten(something[key])
    else:
        yield something
regime="reversible"
regime="irreversible"
k0_vals=[10, 100, 1000, 10000, 5e5, 1e6]
num_freqs=6



param_list={
    "E_0":-0.25,
    'E_start':  -0.3402878, #(starting dc voltage - V)
    'E_reverse': 0.2610671,
    'omega':9.014741538924035,#8.88480830076,  #    (frequency Hz)
    "original_omega":9.014741538924035,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 0.0,  #     (uncompensated resistance ohms)
    'Cdl': 8e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 100, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":-0.25,
    "E0_std": 0.05,
    "k_1":0,
    "k_2":0,
    "E0_skew":0,
    "k0_shape":0.5,
    "k0_scale":100,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/2400),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 5,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[15],
    "GH_quadrature":False,
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
    "harmonic_range":list(range(2, 11)),
    "experiment_time": None,
    "experiment_current": None,
    "experiment_voltage":None,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1000],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "k_1":[0,100],
    "k_2":[0,100],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[param_list['E_start'],param_list['E_reverse']],
    "E0_std": [1e-4,  0.2],
    "E0_skew":[-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [0, 2*math.pi],
}
k2_val=80
table_params=["E_start", "E_reverse", "E_0","k_0","Ru","Cdl","gamma", "alpha", "v", "omega", "phase","d_E","sampling_freq", "area"]
real_params=[[-0.08064031224498379,0.5, 65, 112.80693965100555, 1e-6,  2.9678863151432805e-11, 9.014741538924035, 3*math.pi/2, 3*math.pi/2, 0.5, k2_val, k2_val],
            [-0.08064031224498379, 65, 112.80693965100555, 1e-6,  2.9678863151432805e-11, 9.014741538924035, 3*math.pi/2, 3*math.pi/2, 0.5, k2_val, k2_val],
            [-0.08064031224498379, 0.04, 65, 112.80693965100555, 1e-6,  2.9678863151432805e-11, 9.014741538924035, 3*math.pi/2, 3*math.pi/2, 0.5, k2_val, k2_val]]
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)

#DO RAMPED MOTHERFUCKER

freqs=np.linspace(9.014907846043297,100, 10)
cdls=[1e-6, 1e-5, 1e-4]
num_vars=3
var_list=np.linspace(0.9, 1.1, num_vars)
harms=harmonics(other_values["harmonic_range"], 1, 0.05)

phase_params=["phase", "cap_phase"]
cyt.simulation_options["numerical_method"]="scipy"
cyt.simulation_options["scipy_type"]="EC"
optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","gamma","omega","cap_phase","phase", "alpha", "k_1", "k_2"]
cyt.def_optim_list(optim_list)
e0_disp_test=cyt.test_vals(real_params[2], "timeseries")
optim_list=["E_0","k_0","Ru","Cdl","gamma","omega","cap_phase","phase", "alpha", "k_1", "k_2"]
cyt.def_optim_list(optim_list)
test_no_disp=cyt.test_vals(real_params[1], "timeseries")
cyt.simulation_options["scipy_type"]="EC"
optim_list=["E_0","k0_shape", "k0_scale","Ru","Cdl","gamma","omega","cap_phase","phase", "alpha", "k_1", "k_2"]
cyt.def_optim_list(optim_list)

test=cyt.test_vals(real_params[0], "timeseries")

cyt.simulation_options["numerical_method"]="Brent minimisation"
test_brent=cyt.test_vals(real_params[0], "timeseries")
volts=cyt.define_voltages()[cyt.time_idx]
plt.subplot(1,2,1)
#plt.plot(volts, test_brent, label="Pure Faradaic $k^0$ dispersion")
plt.plot(volts, test_no_disp, label="EC model, no dispersion")
plt.plot(volts, test, label="EC model $k^0$ dispersion")

plt.plot(volts, e0_disp_test, label="EC model, E0_disp")
plt.xlabel("Nondimensional voltage")
plt.ylabel("Nondimensional current")
plt.legend()
plt.subplot(1,2,2)
values, weights=cyt.return_distributions(1000)
plt.plot(np.multiply(values, 1/cyt.nd_param.c_T0), weights)
plt.xlabel("$k^0$")
plt.ylabel("Weights")
plt.show()










    #ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
