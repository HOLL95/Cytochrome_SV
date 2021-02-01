import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
sys.path.append("..")
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from scipy.integrate import odeint
from single_e_class_unified import single_electron
from scipy.optimize import curve_fit
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/DCV"
files=os.listdir(data_loc)
experimental_dict={}

dcv_params=[[-0.050472074915102805, 0.015228603477284797, 2.8611150044327795e-11],[-0.051750488603603075, 0.012282371135025802, 9.614862575901893e-12],[-0.056141230116644396, 1.0000012977576324e-05, 4.71910546612281e-12]]
dcv_params=[[-0.050472074915102805, 0.015228603477284797, 2.8611150044327795e-11],[-0.08064031224498379, 0.020906827217859487, 2.9678863151432805e-11],[-0.056141230116644396, 1.0000012977576324e-05, 4.71910546612281e-12]]
fig, ax=plt.subplots(1, len(dcv_params))
for i in range(1, 4):
    file_name="/dcV_Cjx-183D_WT_pH_7_{0}_3".format(i)
    dcv_file=np.loadtxt(data_loc+file_name, skiprows=2)
    dcv_file_time=dcv_file[:,0]
    dcv_file_voltage=dcv_file[:,1]
    dcv_file_current=dcv_file[:,2]
    param_list={
        "E_0":0.2,
        'E_start':  -0.39, #(starting dc voltage - V)
        'E_reverse':0.3,
        'omega':8.94, #8.88480830076,  #    (frequency Hz)
        "v":30*1e-3,
        'd_E': 0,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 0,  #     (uncompensated resistance ohms)
        'Cdl': 0, #(capacitance parameters)
        'CdlE1': 0,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        "CdlE3":0,
        "Cdlinv":0,
        'CdlE1inv': 0,#0.000653657774506,
        'CdlE2inv': 0,#0.000245772700637,
        "CdlE3inv":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 1000, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/200),
        'phase' :0,
        "time_end": -1,
        'num_peaks': 30,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    simulation_options={
        "no_transient":False,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[10],
        "GH_quadrature":True,
        "test": False,
        "method": "dcv",
        "phase_only":False,
        "likelihood":likelihood_options[0],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }

    other_values={
        "filter_val": 0.5,
        "harmonic_range":range(0, 1),
        "experiment_time": dcv_file_time,
        "experiment_current": dcv_file_current,
        "experiment_voltage":dcv_file_voltage,
        "bounds_val":200,
    }
    param_bounds={
        'E_0':[-0.1,0.0],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e6],  #     (uncompensated resistance ohms)
        'Cdl': [1e-5,1e-3], #(capacitance parameters)
        'CdlE1': [-0.1,0.1],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'Cdlinv': [1e-5, 1e-3], #(capacitance parameters)
        'CdlE1inv': [-0.1,0.1],#0.000653657774506,
        'CdlE2inv': [-0.1,0.1],#0.000245772700637,
        'CdlE3inv': [-0.1,0.1],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [50, 1e4], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.1,0.1],
        "E0_std": [1e-5,  0.04],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        "k0_range":[1e2, 1e4],
        'phase' : [math.pi, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    def poly_2(x, a, b, c):
        return (a*x**2)+b*x+c
    def poly_3(x, a, b, c, d):
        return (a*x**3)+(b*x**2)+c*x+d
    def poly_4(x, a, b, c, d, e):
        return (a*x**4)+(b*x**3)+(c*x**2)+d*x+e
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    voltage_results=cyt.other_values["experiment_voltage"]
    middle_idx=list(voltage_results).index(max(voltage_results))
    first_idx=10#len(voltage_results)//15
    current=[]
    idx_1=[first_idx, middle_idx+20]
    idx_2=[middle_idx, -first_idx]
    func=poly_3
    interesting_section=[[-0.15, 0.12], [-0.15, 0.08]]
    subtract_current=np.zeros(len(current_results))
    fitted_curves=np.zeros(len(current_results))
    nondim_v=cyt.e_nondim(voltage_results)









    #inferred_params=DCV_inferred#[-4.97691836e-02,  1.71478268e-02,  3.04770512e+03,  2.63521136e+05, 4.12128630e-11,  4.80069036e-01]
    counter=1
    def i_calc(current, time_vec, time):
        return np.interp([time],time_vec, current)

    true_data=subtract_current
    cyt.def_optim_list(["E0_mean", "E0_std", "gamma", "Cdl"])


    axes=ax[i-1]
    for CDL_val in [5e-5,1e-4, 5e-4]:
        test=cyt.test_vals(dcv_params[i-1]+[CDL_val], "timeseries")
        axes.plot(voltage_results, test, label=CDL_val)
    axes.plot(voltage_results, current_results)
    axes.legend()
plt.show()
