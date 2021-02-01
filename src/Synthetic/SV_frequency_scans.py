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
for i in range(0, len(k0_vals)):
    k0_val=k0_vals[i]
    k0_log=np.log10(k0_val)
    freq_range=[10**x for x in np.linspace(0, 4, num_freqs)]
    for j in range(0, num_freqs):


        param_list={
            "E_0":-0.25,
            'E_start':  -0.5, #(starting dc voltage - V)
            'E_reverse':0.1,
            'omega':freq_range[j],#8.88480830076,  #    (frequency Hz)
            "original_omega":freq_range[j],
            'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
            'area': 0.07, #(electrode surface area cm^2)
            'Ru': 0.0,  #     (uncompensated resistance ohms)
            'Cdl': 8e-4, #(capacitance parameters)
            'CdlE1': 0,#0.000653657774506,
            'CdlE2': 0,#0.000245772700637,
            "CdlE3":0,
            'gamma': 3e-11,
            "original_gamma":3e-11,        # (surface coverage per unit area)
            'k_0': k0_val, #(reaction rate s-1)
            'alpha': 0.5,
            "E0_mean":-0.25,
            "E0_std": 0.05,
            "E0_skew":0,
            "k0_shape":0.5,
            "k0_scale":k0_val,
            "cap_phase":3*math.pi/2,
            "alpha_mean":0.5,
            "alpha_std":1e-3,
            'sampling_freq' : (1.0/400),
            'phase' :3*math.pi/2,
            "time_end": None,
            'num_peaks': 30,
        }
        solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
        likelihood_options=["timeseries", "fourier"]
        time_start=3/(param_list["omega"])
        simulation_options={
            "no_transient":time_start,
            "numerical_debugging": False,
            "experimental_fitting":False,
            "dispersion":False,
            "dispersion_bins":[16],
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
            "harmonic_range":list(range(4, 8)),
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
        table_params=["E_start", "E_reverse", "E_0","k_0","Ru","Cdl","gamma", "alpha", "v", "omega", "phase","d_E","sampling_freq", "area"]
        cyt_params=[-0.07688974263309035, 0.027377978478251054, 124.68226016812127, 82.65176202767852, 0.0007178487287654843, -0.005485275534002848, 0.0012629554419994436, -1.721924369159128e-05, 1.999999965112904e-11, 9.014907846043297, 5.269613328541364, 4.96139021671786, 0.599999897192592]
        cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
        copy_params=copy.deepcopy(cyt_params)
        current=cyt.test_vals([-0.25, k0_val], "timeseries")
        current=cyt.add_noise(current, 0.01*max(current))
        no_disp=cyt.i_nondim(current)
        times=cyt.t_nondim(cyt.time_vec)
        harms=harmonics(cyt.other_values["harmonic_range"], 1, 0.5)
        plot_harmonics,amplitudes=harms.generate_harmonics(cyt.time_vec, current, hanning=False, get_amplitudes=True)
        plt.subplot(2, 3, i+1)
        volts=cyt.e_nondim(cyt.define_voltages()[cyt.time_idx])
        plt.plot(volts, plot_harmonics[0, :], label="$\Omega=$"+str(param_list["omega"]))
        #plt.semilogy(cyt.other_values["harmonic_range"], amplitudes, label="$\Omega=$"+str(param_list["omega"]))
    plt.title(k0_val)
    #plt.legend()
    plt.xlabel("Potential (V)")
    plt.ylabel("Current (A)")
plt.show()






    #ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
