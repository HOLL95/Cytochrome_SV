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
    current_results1=current_data[0::dec_amount,1]
    time_results1=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results1=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")
values=values=[[-0.2120092471414607, 0.000005478233474769771, 116.21497065365581, 431.92918718571053, 0.00044100528598203375, 0.14367030986553458, 0.005163770653874243, 9.999387751676114e-11, 8.881077023434541,0, 0.5997147084901965],\
                [-0.2485010348585462, 0.06702142523298088, 500.88915054231003, 876.7895281614545, 0.000108185721998072834*0, 0.14858346584854376*0, 0.005464884000805036*0, 1.1040282150229229e-10, 8.88129543205022, 0, 0.5990602813196874],
                [-0.2661180439669948, 0.06702142523298088, 81251.01912987458, 876.5733127765511, 0.000108185721998072834*0, 0.14858346584854376*0, 0.005464884000805036*0,1.1040282150229229e-10,  8.88129543205022, 0,0.41024786661899215]

]
harm_range=list(range(3,9,1))
#fig, ax=plt.subplots(len(harm_range), len(values))

for q in range(0, len(values)):
    param_list={
        "E_0":0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
        'omega':8.881077023434541,#8.88480830076,  #    (frequency Hz)
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
        "E0_skew":0,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :0.1,
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
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    voltage_results=cyt.other_values["experiment_voltage"]
    voltages=cyt.define_voltages(transient=True)

    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
    data_harmonics=harms.generate_harmonics(time_results,(current_results))
    cyt.simulation_options["dispersion_bins"]=[16]
    cyt.simulation_options["GH_quadrature"]=False
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])



    """
    quake=1000
    for i in range(0, len(data_harmonics)):
        ax.plot(dcv_volt[quake:-quake], abs(data_harmonics[i,:][quake:-quake]),  label="Data")
        ax.plot(dcv_volt[quake:-quake], abs(syn_harmonics[i,:][quake:-quake]), alpha=0.7, label="Simulated")
        ax2=ax.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
        if i==0:
            ax.legend(loc="upper right")
        if i==len(data_harmonics)-1:
            ax.set_xlabel("Nondim time")
        if i==3:
            ax.set_ylabel("Nondim current")

plt.show()
"""
cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])



true_data=current_results
fourier_arg=cyt.top_hat_filter(true_data)
cyt.secret_data_fourier=fourier_arg
cyt.secret_data_time_series=true_data

#cyt.def_optim_list(["Ru","Cdl","CdlE1", "CdlE2","omega","cap_phase"])
#cyt.dim_dict["gamma"]=0

cdl_params=["Cdl","CdlE1", "CdlE2","omega","cap_phase", "phase"]
#for z in range(0, len(cdl_params)):
#	cyt.param_bounds[cdl_params[z]]=[0.75*cdl_vals[z], 1.25*cdl_vals[z]]
#cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","gamma","omega", "phase", "alpha"])
cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
vals=[
        [-0.23935261365191968, 0.00020486942554457825, -1.3936849491275964, 131.1311416295316, 999.9996074395051, 5.383823665743347e-05, -0.02076353199575397, 0.007029381215910355, 2.173426672774541e-11, 8.884884754617062, 1.965008426077372, 1.3878109329295476, 0.5460636485738265],
        [-0.23935261365191968, 0.010000001833497703, -9.99771847635053, 53.93220379977755, 999.9996074395051, 5.383823665743347e-05, -0.02076353199575397, 0.007029381215910355, 2.173426672774541e-11, 8.884884754617062, 1.965008426077372, 1.3878109329295476, 0.5460636485738265],
        [-0.23935261365191968, 0.010000001833497703, -9.99771847635053, 53.93220379977755, 999.9996074395051, 5.383823665743347e-05, -0.02076353199575397, 0.007029381215910355, 2.173426672774541e-11, 8.884884754617062, 1.965008426077372, 1.3878109329295476, 0.5460636485738265],
        ]
vals=[

        [-0.22729161723898927, 0.02274019351354671, -8.618005726466912, 50.36805568061325, 1214.3687528084954, 2.470942807529444e-05, -0.02407633543515114, 0.00999999999998086, 2.2237135661359653e-11, 8.884829963320826, 6.2831853070106085, 1.3199558755592415e-09, 0.5402705909466587]
        ]
vals=[
    [-0.24475850404009977, 0.020813753114607054, 120.58250154877311, 0.423398159178522, 0.0006022876040908104, -0.025876564530892432, -0.017483429836173528, 1.496476957450661e-11, 8.884847353877376, 3.241957939541493, 1.6380317139124516, 0.4003811412321984]
]
vals=[
    [-0.25549546467502104, 0.024599690122827058, 320.1269038302346, 425.092729031765, 0.0007580642617042125, 0.06264702347418366, 0.008437167809256962, 8.159994053140338e-11, 8.884808603546047, 4.526127180837634, 3.797180059461594, 0.5999995579028523],


    ]
vals=[
    [-0.15000001621515757, 0.02267500146549557, 329.02327044966836, 499.9302524363667, 0.0012101637063185937, 0.025735070587578207, 0.00012489571678975242, 9.999925872986152e-11, 8.884792799115775, 2.403679949352115, 1.6470415159900251, 0.5491541073325497],
    [-0.14997643957614473, 0.02593397924135194, 999.9758566864258, 272.60391462492834, 0.0009015717901376583, 0.03145637922101943, 0.006447730866059417, 5.7415551946709094e-11, 8.884804160189987, 6.283021107040668, 2.044493717397712, 0.5999999945631227],
    [-0.15236973167412687, 0.01830005679704452, 145.04966192816798, 888.5967590058907, 0.00019674908439381512, 0.05532229642696043, 0.0009387422901490639, 2.270338213969965e-11, 8.884788899781832, 2.9516383282220935, 2.4485463003746806, 0.4000000086361465]

]
cyt.simulation_options["method"]="dcv"
dcv_volt=cyt.e_nondim(cyt.define_voltages())
cyt.simulation_options["method"]="ramped"

for q in range(0, len(vals)):
    syn_time=cyt.test_vals(vals[q], "timeseries")
    syn_harmonics=harms.generate_harmonics(time_results,(syn_time))
    data_harmonics=harms.generate_harmonics(time_results,(current_results))



    for i in range(0, len(data_harmonics)):
        plt.subplot(len(data_harmonics),1,i+1)
        ax=plt.gca()

        ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
        if q==0:
            ax.plot(time_results, abs(data_harmonics[i,:]),  alpha=0.7, label="Exp")
        ax2=ax.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
        if i==0:
            ax.legend(loc="upper right")
        if i==len(data_harmonics)-1:
            ax.set_xlabel("Nondim time")
        if i==3:
            ax.set_ylabel("Nondim current")

plt.show()
