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
figure=multiplot(4,4, **{"harmonic_position":[1,3], "num_harmonics":num_harms, "orientation":"landscape",  "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":2, "plot_height":1, "harmonic_spacing":1})
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
SV_vals=[[-0.007311799334716082, 0.06011218461168426, 100.47324948026552, 49.49286763913283, 0.00011151822964084257, 0.00033546831322842086, 0.009999941792246102,0, 2.999999994953957e-11, 9.015005507706533, 6.283185267250843, 1.9462916200793343, 0.4847309255453626]
]
experiment_dict={
                "ramped":{"file_loc":"Alice_2_11_20/FTACV", "filename":"FTACV_Cyt_1_cv_","plot_loc":2, "decimation":32, "method":"ramped", "params":ramped_param_list, "transient":1/SV_param_list["omega"], "bounds":20,"harm_range":list(range(3, 8)),
                            "values":[-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333*0, -0.004699728058449013*0, 2.1898117688472174e-11, 8.959294458587753,0, 0.5592126258301378],
                            "param_list":["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"],
                    },
                "SV":{"file_loc":"Alice_2_11_20/PSV", "filename":"PSV_Cyt_1_cv_","plot_loc":4, "decimation":32, "method":"sinusoidal", "params":SV_param_list, "transient":1/ramped_param_list["omega"],"bounds":20000,"harm_range":list(range(4, 9)),
                    "values":SV_vals[0],#[-0.07963988256472493, 0.043023349442016245, 20.550878790990733, 581.5147052074157, 8.806259570340898e-05, -0.045583168350011485, -0.00011159862236990309, 0.00018619134662841048, 2.9947102043021914e-11, 9.014976375142606, 5.699844468024501, 5.18463541959069, 0.5999994350046962],
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
    "dispersion_bins":[8],
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
    "bounds_val":20000,
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
    axes_row=figure.axes_dict["row"+str(experiment_dict[experiment_type]["plot_loc"])]
    exp_optim_list=experiment_dict[experiment_type]["param_list"]
    experiment_class.def_optim_list(exp_optim_list)


    if experiment_type not in harmonic_files:
        first_idx=10#len(voltage_results)//15
        current=[]
        middle_idx=list(voltage_results).index(max(voltage_results))
        idx_1=[first_idx, middle_idx+20]
        idx_2=[middle_idx, -first_idx]
        func=poly_3
        interesting_section=[[-0.15, 0.12], [-0.15, 0.08]]
        subtract_current=np.zeros(len(current_results))
        fitted_curves=np.zeros(len(current_results))
        nondim_v=voltage_results
        for q in range(0, 2):
            current_half=current_results[idx_1[q]:idx_2[q]]
            time_half=time_results[idx_1[q]:idx_2[q]]
            volt_half=voltage_results[idx_1[q]:idx_2[q]]
            noise_idx=np.where((volt_half<interesting_section[q][0]) | (volt_half>interesting_section[q][1]))
            noise_voltages=volt_half[noise_idx]
            noise_current=current_half[noise_idx]
            noise_times=time_half[noise_idx]
            popt, pcov = curve_fit(func, noise_times, noise_current)
            fitted_curve=[func(t, *popt) for t in time_half]
            subtract_current[idx_1[q]:idx_2[q]]=np.subtract(current_half, fitted_curve)
            fitted_curves[idx_1[q]:idx_2[q]]=fitted_curve
    counter=0
    for i in range(0, len(plot_keys)):

            parameter_vals=experiment_dict[plot_keys[i]]["values"]
            parameter_names=experiment_dict[plot_keys[i]]["param_list"]
            simulation_parameters=np.zeros(len(exp_optim_list))
            for j in range(0, len(exp_optim_list)):
                if exp_optim_list[j] in unchanged_params:
                    simulation_parameters[j]=experiment_dict[experiment_type]["params"][exp_optim_list[j]]
                elif exp_optim_list[j] in parameter_names:
                    simulation_parameters[j]=parameter_vals[parameter_names.index(exp_optim_list[j])]
                else:
                    simulation_parameters[j]=experiment_dict[experiment_type]["params"][exp_optim_list[j]]
            print(experiment_class.bounds_val)
            print(experiment_class.optim_list)
            print(list(simulation_parameters))
            numerical_current=experiment_class.i_nondim(experiment_class.test_vals(simulation_parameters, "timeseries"))*1e6
            if experiment_type in harmonic_files:
                if experiment_type=="ramped":
                    xaxis=time_results
                    hanning=True
                    func=abs
                    xlabel="Time(s)"
                elif experiment_type=="SV":
                    xaxis=voltage_results
                    hanning=False
                    func=empty
                    xlabel="Potential(mV)"
                harms=harmonics(experiment_dict[experiment_type]["harm_range"], experiment_class.dim_dict["omega"],0.1)
                exp_harmonics=harms.generate_harmonics(time_results, current_results, hanning=hanning)
                numerical_harmonics=harms.generate_harmonics(time_results, numerical_current, hanning=hanning)
                #harms.plot_harmonics(times=time_results, experimental_time_series=current_results, data_time_series=numerical_current, xaxis=voltage_results)
                for q in range(0, len(exp_harmonics)):
                    print("COUNTER", counter)
                    axes_row[counter].plot(xaxis, func(numerical_harmonics[q,:]))
                    twinx=axes_row[counter].twinx()
                    twinx.set_ylabel(experiment_dict[experiment_type]["harm_range"][q], rotation=0)
                    twinx.set_yticks([])
                    axes_row[counter].plot(xaxis, func(exp_harmonics[q,:]), alpha=0.8)
                    if q==len(exp_harmonics)-1:
                        axes_row[counter].set_xlabel(xlabel)
                    if i==0:
                        if q==len(exp_harmonics)//2:
                            axes_row[counter].set_ylabel("Current($\\mu A$)")
                    counter+=1
            else:
                axes_row[i].plot(voltage_results, numerical_current, label="Sim")
                axes_row[i].plot(voltage_results, subtract_current, alpha=0.8, label="Exp")
                axes_row[i].set_xlabel("Potential(mV)")

                if experiment_type=="DCV_1":
                    axes_row[i].set_title(plot_keys[i]+ " params")
                    axes_row[i].legend(loc="upper left")
                if i==0:
                    axes_row[i].set_ylabel("Current($\\mu A$)")
plt.subplots_adjust(top=0.965,
                    bottom=0.055,
                    left=0.065,
                    right=0.98,
                    hspace=0.2,
                    wspace=0.2)
plt.show()
