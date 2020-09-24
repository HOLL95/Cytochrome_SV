import numpy as np
import matplotlib.pyplot as plt
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
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/SV"
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
figure=multiplot(3, 2, **{"distribution_position":list(range(0,3)), "num_harmonics":num_harms, "orientation":"landscape",  "plot_width":5, "row_spacing":1,"col_spacing":2, "plot_height":1})
for j in range(2, 4):
    figure.axes_dict["row3"][j].set_axis_off()

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
for regime in ["reversible", "irreversible"]:
    figure=multiplot(3, 2, **{"distribution_position":list(range(0,3)), "num_harmonics":num_harms, "orientation":"landscape",  "plot_width":5, "row_spacing":1,"col_spacing":2, "plot_height":1})
    for j in range(2, 4):
        figure.axes_dict["row3"][j].set_axis_off()
    if regime=="reversible":
        scale_scan_vals=np.flip([50, 100, 500])
        k0_val=100
    elif regime=="irreversible":
        scale_scan_vals=np.flip([0.5, 1, 2])
        k0_val=1
    params={"E_0":{"E0_mean":[-0.2, -0.25, -0.3],
                    "E0_std":[0.03,0.05 , 0.07],
                    "E0_skew":[-5, 0, 5]},
            "k_0":{"k0_shape":([0.25, 0.5, 0.65]),
                    "k0_scale":scale_scan_vals}}
    flattened_list=list(flatten(params))
    total_length=len(flattened_list)
    unit_dict={
        "E_0": "V",
        'E_start': "V", #(starting dc voltage - V)
        'E_reverse': "V",
        'omega':"Hz",#8.88480830076,  #    (frequency Hz)
        'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
        'v': '$s^{-1}$',   #       (scan rate s^-1)
        'area': '$cm^{2}$', #(electrode surface area cm^2)
        'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
        'Cdl': "F", #(capacitance parameters)
        'CdlE1': "",#0.000653657774506,
        'CdlE2': "",#0.000245772700637,
        'CdlE3': "",#1.10053945995e-06,
        'gamma': 'mol cm^{-2}$',
        'k_0': '$s^{-1}$', #(reaction rate s-1)
        'alpha': "",
        'E0_skew':"",
        "E0_mean":"V",
        "E0_std": "V",
        "k0_shape":"",
        "k0_loc":"",
        "k0_scale":"",
        "cap_phase":"rads",
        'phase' : "rads",
        "alpha_mean": "",
        "alpha_std": "",
        "":"",
        "noise":"",
        "error":"$\\mu A$",
    }
    fancy_names={
        "E_0": '$E^0$',
        'E_start': '$E_{start}$', #(starting dc voltage - V)
        'E_reverse': '$E_{reverse}$',
        'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
        'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
        'v': "v",   #       (scan rate s^-1)
        'area': "Area", #(electrode surface area cm^2)
        'Ru': "Ru",  #     (uncompensated resistance ohms)
        'Cdl': "$C_{dl}$", #(capacitance parameters)
        'CdlE1': "$C_{dlE1}$",#0.000653657774506,
        'CdlE2': "$C_{dlE2}$",#0.000245772700637,
        'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
        'gamma': '$\\Gamma',
        'E0_skew':"$E^0 \\kappa$",
        'k_0': '$k_0$', #(reaction rate s-1)
        'alpha': "$\\alpha$",
        "E0_mean":"$E^0 \\mu$",
        "E0_std": "$E^0 \\sigma$",
        "cap_phase":"C$_{dl}$ phase",
        "k0_shape":"$\\log (k^0) \\sigma$",
        "k0_scale":"$\\log (k^0) \\mu$",
        "alpha_mean": "$\\alpha\\mu$",
        "alpha_std": "$\\alpha\\sigma$",
        'phase' : "Phase",
        "":"Experiment",
        "noise":"$\sigma$",
        "error":"RMSE",
    }
    param_list={
        "E_0":-0.25,
        'E_start':  -0.5, #(starting dc voltage - V)
        'E_reverse':0.1,
        'omega':8.940960632790196,#8.88480830076,  #    (frequency Hz)
        "original_omega":8.940960632790196,
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 0.0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': k0_val, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":-0.25,
        "E0_std": 0.05,
        "E0_skew":0,
        "k0_shape":0.25,
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
    time_start=1/(param_list["omega"])
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
        "harmonic_range":list(range(3, 8)),
        "experiment_time": time_results1,
        "experiment_current": current_results1,
        "experiment_voltage":voltage_results1,
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

    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    
