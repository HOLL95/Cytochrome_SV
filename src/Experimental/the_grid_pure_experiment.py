import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import sys
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import math
import copy
import pints
import time
from single_e_class_unified import single_electron
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import itertools
harm_range=list(range(3, 8))
num_harms=len(harm_range)
#plt.show()
SV_param_list={
    "E_0":-0.2,
    'E_start':  -0.34024, #(starting dc voltage - V)
    'E_reverse':0.2610531,
    'omega':9.015120071612014, #8.88480830076,  #    (frequency Hz)
    "original_omega":9.015120071612014,
    'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl':0, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":3*math.pi/2,
    'sampling_freq' : (1.0/400),
    'phase' :3*math.pi/2,
    "time_end": -1,
    'num_peaks': 30,
}
ramped_param_list=copy.deepcopy(SV_param_list)
directory=os.getcwd()
dir_list=directory.split("/")
exp_loc=("/").join(dir_list[:-2])+"/Experiment_data/"
del ramped_param_list["original_omega"]
del ramped_param_list["num_peaks"]
dcv_param_list=copy.deepcopy(ramped_param_list)
ramped_dif_params={'E_start':  -0.33898177567074794, 'E_reverse':0.26049326614698887,'omega':8.959294996508683,"v":0.022316752195354346,'d_E': 150*1e-3,"phase":0}
dcv_dif_params={'E_start':  -0.39, 'E_reverse':0.3,'omega':8.94,"v":30*1e-3,'d_E': 0,"phase":0}
dcv_keys=dcv_dif_params.keys()
for key in ramped_dif_params.keys():
    ramped_param_list[key]=ramped_dif_params[key]
    if key in dcv_keys:
        dcv_param_list[key]=dcv_dif_params[key]
SV_vals=[[-0.007311799334716082, 0.06011218461168426, 100.47324948026552, 49.49286763913283, 0.00011151822964084257, 0.00033546831322842086, 0.009999941792246102,0, 2.999999994953957e-11, 9.015005507706533, 6.283185267250843, 1.9462916200793343, 0.4847309255453626],
        [-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.37095896576959e-11, 9.01499164308653, 4.7220768639743085, 4.554136092744141, 0.5999999989106146]
]
experiment_dict={
                "ramped":{"file_loc":"Alice_2_11_20/FTACV", "filename":"FTACV_Cyt_1_cv_","plot_loc":2, "decimation":32, "method":"ramped", "params":ramped_param_list, "transient":1/SV_param_list["omega"], "bounds":20,"harm_range":list(range(3, 8)),
                            "values":[-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333*0, -0.004699728058449013*0, 2.1898117688472174e-11, 8.959294458587753,0, 0.5592126258301378],
                            "param_list":["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"],
                    },
                "SV":{"file_loc":"Alice_2_11_20/PSV", "filename":"PSV_Cyt_1_cv_","plot_loc":4, "decimation":32, "method":"sinusoidal", "params":SV_param_list, "transient":1/ramped_param_list["omega"],"bounds":20000,"harm_range":list(range(4, 9)),
                    "values":SV_vals[1],#[-0.07963988256472493, 0.043023349442016245, 20.550878790990733, 581.5147052074157, 8.806259570340898e-05, -0.045583168350011485, -0.00011159862236990309, 0.00018619134662841048, 2.9947102043021914e-11, 9.014976375142606, 5.699844468024501, 5.18463541959069, 0.5999994350046962],
                    "param_list":["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"],
                    },
                "DCV_1":{"file_loc":"Alice_2_11_20/DCV", "filename":"dcV_Cjx-183D_WT_pH_7_1_3","plot_loc":1, "method":"dcv", "params":dcv_param_list, "transient":False,"bounds":2000,"harm_range":list(range(4, 9)),
                    "values":[-0.051002951188454194, 1.0000168463482572e-05, 0.5871952270775241, 3.4006592554802374e-07, 3.107639506959287e-11, 0.4521221354084581],
                    "param_list":["E0_mean", "E0_std","k_0","Ru","gamma", "alpha"],
                    },
                "DCV_2":{"file_loc":"Alice_2_11_20/DCV", "filename":"dcV_Cjx-183D_WT_pH_7_2_3","plot_loc":3, "method":"dcv", "params":dcv_param_list, "transient":False,"bounds":2000,"harm_range":list(range(4, 9)),
                    "values":[-0.049697063448006555, 0.0037230268328498823, 0.3527926751864493, 4.773196671248675e-10, 1.196896571851427e-11, 0.5513930594926286],
                    "param_list":["E0_mean", "E0_std","k_0","Ru","gamma", "alpha"],
                    },
                }

def empty(arg):
    return arg
def poly_3(x, a, b, c, d):
    return (a*x**3)+(b*x**2)+c*x+d
unchanged_params=["omega"]
plot_keys=["DCV_1", "ramped", "DCV_2", "SV"]
harmonic_files=["ramped", "SV"]
master_simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[16],
    "GH_quadrature":True,
    "test": False,
    "method": None,
    "phase_only":False,
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[]
}

master_other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "experiment_time": None,
    "experiment_current": None,
    "experiment_voltage":None,
    "bounds_val":200000,
}
master_param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[5,15],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-12,1e-9],
    'k_0': [0, 2e2], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[-0.08, 0.04],
    "E0_std": [1e-4,  0.1],
    'phase' : [math.pi, 2*math.pi],
}
counter=1
for experiment_type in plot_keys:
    data_loc=exp_loc+experiment_dict[experiment_type]["file_loc"]
    file_name=experiment_dict[experiment_type]["filename"]
    params=experiment_dict[experiment_type]["params"]
    if experiment_type in harmonic_files:
        dec_amount=experiment_dict[experiment_type]["decimation"]
        current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
        voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
        volt_data=voltage_data_file[0::dec_amount, 1]
        time_data=current_data_file[0::dec_amount, 0]
        current_data=current_data_file[0::dec_amount, 1]
        del current_data_file
        del voltage_data_file
    else:
        dcv_file=np.loadtxt(data_loc+"/"+file_name, skiprows=2)
        time_data=dcv_file[:,0]
        volt_data=dcv_file[:,1]
        current_data=dcv_file[:,2]
    experiment_other_vals=copy.deepcopy(master_other_values)
    experiment_other_vals["experiment_current"]=current_data
    experiment_other_vals["experiment_time"]=time_data
    experiment_other_vals["experiment_voltage"]=volt_data
    experiment_simulation_options=copy.deepcopy(master_simulation_options)
    experiment_simulation_options["method"]=experiment_dict[experiment_type]["method"]
    experiment_simulation_options["no_transient"]=experiment_dict[experiment_type]["transient"]
    experiment_other_vals["bounds_val"]=experiment_dict[experiment_type]["bounds"]
    experiment_class=single_electron(None, params, experiment_simulation_options, experiment_other_vals, master_param_bounds)
    time_results=experiment_class.t_nondim(experiment_class.other_values["experiment_time"])
    current_results=experiment_class.i_nondim(experiment_class.other_values["experiment_current"])*1e6
    voltage_results=experiment_class.e_nondim(experiment_class.other_values["experiment_voltage"])
    exp_optim_list=experiment_dict[experiment_type]["param_list"]
    experiment_class.def_optim_list(exp_optim_list)
    plt.subplot(1, 4, counter)
    counter+=1
    print(plot_keys)
    if experiment_type!="ramped":
        plt.plot(voltage_results, current_results)
        plt.xlabel("Potential (V)")
    else:
        plt.plot(time_results, current_results)
        plt.xlabel("Time (s)")
    plt.ylabel("Current $(\\mu A)$")

plt.show()
