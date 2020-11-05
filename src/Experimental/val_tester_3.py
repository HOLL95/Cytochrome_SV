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
    current_results1=current_data[0::dec_amount,1]
    time_results1=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results1=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")
    #["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"]
cdl_vals=[0.00010640637373095699, 0.005028640673736776, 0.0008598525708093628, 8.940630374330569, 4.341216122575276]

cdl_params=["Cdl","CdlE1", "CdlE2", "omega", "cap_phase"]
farad_params=["E0_mean", "E0_std","k_0","Ru", "gamma", "alpha", "phase"]
all_params=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"]


#3-6
[-0.3679764998754323, 0.1422464146023959, 44.75877502784956, 468.18385860454373, 4.443350510344139e-10, 0.5215371585216145, 5.348741619902399]
#4-7
[-0.36797649999999954, 0.13517071991295526, 58.50763441820108, 439.66475714149743, 4.038205467075284e-10, 0.5999999999999999, 5.29245733802453]
#5-8
[-0.3679764999999972, 0.2664620206528828, 170.1500626179889, 520.9423347508204, 6.794048135978194e-11, 0.5303371419890249, 5.1923202141405715]
#6-10
[-0.34417347144925503, 0.49704242685318817, 2934.2937667223364, 997.0654667064832, 4.332998056574655e-10, 0.5252375323827561, 6.283179321960487]
values=[[-0.3679764998754323, 0.1422464146023959, 44.75877502784956, 468.18385860454373, 4.443350510344139e-10, 0.5215371585216145, 5.348741619902399],
        [-0.36797649999999954, 0.13517071991295526, 58.50763441820108, 439.66475714149743, 4.038205467075284e-10, 0.5999999999999999, 5.29245733802453],
        [-0.3679764999999972, 0.2664620206528828, 170.1500626179889, 520.9423347508204, 6.794048135978194e-11, 0.5303371419890249, 5.1923202141405715],
        [-0.34417347144925503, 0.49704242685318817, 2934.2937667223364, 997.0654667064832, 4.332998056574655e-10, 0.5252375323827561, 6.283179321960487]
]
values=[[-0.3679764999999972, 0.2664620206528828, 170.1500626179889, 520.9423347508204, 6.794048135978194e-11, 0.5303371419890249, 5.1923202141405715],
        [-0.19641403413160682, 0.10616120327006115, 2.4686855470912716, 149.0272756934875, 3.06537508868333e-10, 0.5999999999999999, 4.059111342177294],
        [-0.003808583576515645, 0.4999999999705347, 3784.6157375283487, 631.453871630171, 1.0033710827129637e-10, 0.5999999760343172, 5.184808475268891],
        [-0.003079888857984825, 0.4999999999999977, 9999.999999999765, 770.4830056213029, 1.3371001423223723e-10, 0.5999999999934313, 5.300647862588599]


]


harm_range=list(range(1,9,1))
fig, ax=plt.subplots(len(harm_range), len(values))
count=-1
for q in range(0, len(values)):
    count+=1
    param_list={
        "E_0":0.2,
        'E_start':  min(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]), #(starting dc voltage - V)
        'E_reverse':max(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]),
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
        "harmonic_range":harm_range,
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
    cyt.simulation_options["dispersion_bins"]=[10]
    cyt.simulation_options["GH_quadrature"]=True
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])

    reduced_list=["E_0","k_0","Ru","gamma","omega","cap_phase","phase", "alpha"]
    if len(values[count])==len(farad_params):
        new_list=np.zeros(len(all_params))
        joint_dict=dict(zip(farad_params, values[count]))
        cdl_dict=dict(zip(cdl_params, cdl_vals))
        joint_dict.update(cdl_dict)
        for i in range(0, len(all_params)):
            new_list[i]=joint_dict[all_params[i]]
        values[count]=new_list
    elif len(values[count])==len(all_params)-1:
        cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    print("~"*20,list(values[count]))
    true_signal=cyt.test_vals(values[count], "timeseries")
    #cyt.test_vals(values[count], "fourier", test=True)
    #test_data=cyt.add_noise(true_signal, 0.005*max(true_signal))
    true_data=current_results
    #true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)

    cov=np.cov(true_data)
    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
    data_harmonics=harms.generate_harmonics(time_results,(current_results))
    syn_harmonics=harms.generate_harmonics(time_results, (true_signal))
    #plt.plot(voltage_results, current_results)
    #plt.plot(voltage_results, true_signal, alpha=0.7)
    #plt.show()
    for i in range(0, len(data_harmonics)):



        ax[i][count].plot(voltage_results, (data_harmonics[i,:]),  label="Data")
        ax[i][count].plot(voltage_results, (syn_harmonics[i,:]), label="Simulation", alpha=0.7)
        #ax[i].plot(voltage_results, np.subtract(data_harmonics[i,:],syn_harmonics[i,:]), alpha=0.7, label="Residual")
        ax2=ax[i][count].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
        if i==0:
            ax[i][count].legend(loc="upper right")
        if i==len(data_harmonics)-1:
            ax[i][count].set_xlabel("Nondim voltage")
        else:
            ax[i][count].set_xticks([])
        if i==3:
            ax[i][count].set_ylabel("Nondim current")
plt.show()
