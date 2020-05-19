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
harm_range=list(range(1,8,1))
values=[
        [-0.2494100028589219, 0.064016493582288162, 9151.980166228543, 275.62973801523026,.00010021922824160558, -0.0007532772365652918, 1.6169579148649083e-05, 4.6107253848269156e-11, 8.884813341771366,0,0.400000000000158],
        [-0.2494100028589219, 0.050016493582288162, 9151.980166228543, 275.62973801523026,0.00010021922824160558, -0.0007532772365652918, 1.6169579148649083e-05, 4.6107253848269156e-11, 8.884813341771366,0,0.400000000000158],
]
fig, ax=plt.subplots(len(harm_range), len(values))
for j in range(1, 4):
    scan="_"+str(j)+"_"
    freq="_9Hz"
    dec_amount=1
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






    for q in range(0, len(values)):
        param_list={
            "E_0":0.2,
            'E_start':  -500e-3, #(starting dc voltage - V)
            'E_reverse':100e-3,
            'omega':8.884813341771366,#8.88480830076,  #    (frequency Hz)
            "v":    22.35174e-3,
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
            "harmonic_range":harm_range,
            "experiment_time": time_results1,
            "experiment_current": current_results1,
            "experiment_voltage":voltage_results1,
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

        harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
        data_harmonics=harms.generate_harmonics(time_results,(current_results), np.real)
        cyt.simulation_options["dispersion_bins"]=[20]
        cyt.simulation_options["GH_quadrature"]=True
        print(len(cyt.optim_list),len(values[q]))
        cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])
        if len(cyt.optim_list)==len(values[q])+1:
            cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])


        vals=[-0.2085010348585462, 0.05769719441256009, 300.88915054231003, 621.7895281614545, 0.00038185721998072834*0, 0.14858346584854376, 0.005464884000805036, 1.995887932170653e-11, 8.88, 3.481574064188825, 0.5990602813196874]
        vals=[-0.2120092471414607, 0.0005478233474769771, 1000.21497065365581, 431.92918718571053, 0.00044100528598203375*0, 0.14367030986553458, 0.005163770653874243, 9.999387751676114e-11, 8.941077023434541,  3.4597619059667073, 0.5997147084901965]
        syn_time=cyt.test_vals(values[q], "timeseries")
        syn_harmonics=harms.generate_harmonics(time_results,(syn_time), np.real)
        cyt.simulation_options["method"]="dcv"
        dcv_volt=cyt.e_nondim(cyt.define_voltages())
        cyt.simulation_options["method"]="ramped"


        quake=500
        for i in range(0, len(data_harmonics)):
            ax[i][q].plot(dcv_volt[quake:-quake], abs(data_harmonics[i,:][quake:-quake]),  label="Data")
            ax[i][q].plot(dcv_volt[quake:-quake], abs(syn_harmonics[i,:][quake:-quake]), alpha=0.7, label="Simulated")
            ax2=ax[i][q].twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
            if i==0:
                ax[i][q].legend(loc="upper right")
            if i==len(data_harmonics)-1:
                ax[i][q].set_xlabel("Nondim time")
            if i==3:
                ax[i][q].set_ylabel("Nondim current")

    plt.show()
plt.plot(current_results)
plt.plot(syn_time*-1, alpha=0.7)
plt.show()
