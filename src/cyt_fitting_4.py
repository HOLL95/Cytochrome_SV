import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from harmonics_plotter import harmonics
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
    current_results1=current_data[0::dec_amount,1]
    time_results1=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results1=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")
rs=[5, 50, 500, 2000, 50000, 500000]
estart=min(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4])
ereverse=max(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4])
for i in range(0, len(rs)):
    param_list={
        "E_0":(estart+ereverse)/2,
        'E_start':  min(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]), #(starting dc voltage - V)
        'E_reverse':max(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]),
        'omega':8.94,#8.88480830076,  #    (frequency Hz)
        "original_omega":8.94,
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': rs[i],  #     (uncompensated resistance ohms)
        'Cdl': 7.719E-5, #(capacitance parameters)
        'CdlE1':1.889E-3,#0.000653657774506,
        'CdlE2': -3.359E-4,#0.000245772700637,
        "CdlE3":0,
        'gamma': 9.897E-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 101.97, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/200),
        'phase' : 3*(math.pi/2),
        "time_end": -1,
        'num_peaks': 40
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
        "multi_output": True,
        "label": "MCMC",
        "optim_list":[]
    }
    other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(3,9,1)),
        "experiment_time": time_results1,
        "experiment_current": current_results1,
        "experiment_voltage":voltage_results1,
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
        "noise":[0, 100]
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    print(cyt.nd_param.c_I0)
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    voltage_results=cyt.other_values["experiment_voltage"]
    cyt.dim_dict["noise"]=0
    cyt.dim_dict["phase"]=3*math.pi/2
    cyt.simulation_options["phase_only"]=True
    cyt.simulation_options["numerical_method"]="pybamm"
    test=cyt.test_vals([], "timeseries")
    #cyt.simulation_options["adaptive_ru"]=True
    #test2=cyt.test_vals([], "timeseries")
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = time_results[-1]
    numpoints = len(current_results)
    volts=cyt.define_voltages()
    w0 = [0,0, volts[0]]

    # Call the ODE solver.

    wsol = odeint(cyt.current_ode_sys, w0, time_results,
                  atol=abserr, rtol=relerr)
    adaptive_current=wsol[:,0]
    adaptive_potential=wsol[:,2]
    adaptive_theta=wsol[:, 1]
    plt.subplot(2, 3, i+1)
    plt.plot(time_results, adaptive_current[cyt.time_idx], label="Scipy")
    plt.plot(time_results, test, label="Brent")
    plt.title("$R_u="+str(rs[i])+" \\Omega$")
    plt.xlabel("Nondim voltage")
    plt.ylabel("Nondim current")
    plt.legend()
    """test2=cyt.test_vals([], "timeseries")
    plt.plot(voltage_results, test2)"""
plt.show()
