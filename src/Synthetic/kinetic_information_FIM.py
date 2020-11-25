import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from mpl_toolkits.mplot3d import Axes3D
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import scipy.integrate as integ
import scipy.optimize as opt
import math
import copy
import pints
from single_e_class_unified import single_electron
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import itertools
from matplotlib import cm
import seaborn
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="3"
freq="_9_"
dec_amount=8
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
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 1e-5*0,#0.000653657774506,
    'CdlE2': 1e-5*0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 100, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":-0.25,
    "E0_std": 0.05,
    "E0_skew":0,
    "k0_shape":0.25,
    "k0_scale":1000,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 15,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
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
def RMSE(series1, series2):
    return np.sqrt(np.mean(np.power(np.subtract(series1, series2),2)))
def approx_error(series1, series2):
     return np.mean(np.divide(abs(np.subtract(series1, series2)),series1))*100
def information_calc(val_dict, param_of_interest, current_class,other_values, delta):
    disp_params=["E0_mean","E0_std","k0_scale", "k0_shape", "alpha_mean", "alpha_std"]
    dim_dict=current_class.dim_dict
    sim_options=copy.deepcopy(current_class.simulation_options)
    for param in val_dict.keys():
        dim_dict[param]=val_dict[param]
        if param=="omega":
            sim_options["no_transient"]=1/val_dict[param]
            dim_dict["original_omega"]=val_dict[param]
    optim_list=[x for x in val_dict.keys() if x in disp_params]
    if param_of_interest not in optim_list:
        optim_list+=[param_of_interest]

    new_class=single_electron(None, dim_dict, sim_options, other_values, current_class.param_bounds)
    new_class.def_optim_list(optim_list)
    params=[val_dict[key] for key in new_class.optim_list]

    interest_position=new_class.optim_list.index(param_of_interest)
    change_val=delta*dim_dict[param_of_interest]
    params[interest_position]=dim_dict[param_of_interest]+change_val

    #volts=new_class.define_voltages()[new_class.time_idx]

    upper_param=(new_class.test_vals(params, "timeseries"))
    params[interest_position]=dim_dict[param_of_interest]-change_val
    normal_param=(new_class.test_vals(params, "timeseries"))
    derivative=np.divide(abs(np.subtract(upper_param, normal_param)), 2*change_val)
    #plt.plot(upper_param)
    #plt.plot(normal_param)
    #plt.plot(volts, np.subtract(upper_param, normal_param))
    #plt.show()
    #total_information=integ.trapz(np.power(derivative,2))
    return derivative, new_class.time_vec[new_class.time_idx], max(normal_param)
def k0_diff_calc(val_dict,  current_class,other_values):
    disp_params=["E0_mean","E0_std","k0_scale", "k0_shape", "alpha_mean", "alpha_std"]
    dim_dict=current_class.dim_dict
    sim_options=copy.deepcopy(current_class.simulation_options)
    for param in val_dict.keys():
        dim_dict[param]=val_dict[param]
        if param=="omega":
            sim_options["no_transient"]=1/val_dict[param]
            dim_dict["original_omega"]=val_dict[param]

    new_class=single_electron(None, dim_dict, sim_options, other_values, current_class.param_bounds)
    new_class.def_optim_list(["k_0"])
    no_disp=(new_class.test_vals([dim_dict["k0_scale"]], "timeseries"))
    print(new_class.dim_dict["k_0"])
    new_class.def_optim_list(["k0_shape", "k0_scale"])
    volts=new_class.define_voltages()[new_class.time_idx]
    disp=(new_class.test_vals([dim_dict["k0_shape"], dim_dict["k0_scale"]], "timeseries"))
    volts2=new_class.define_voltages()[new_class.time_idx]
    print(new_class.dim_dict["k0_shape"], new_class.dim_dict["k0_scale"])


    return (RMSE(disp, no_disp))
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
cyt.def_optim_list(["Cdl"])
cyt.simulation_options["dispersion_bins"]=[16]

cyt.def_optim_list(["k0_shape", "k0_scale"])
voltage=cyt.define_voltages()[cyt.time_idx]
times=cyt.time_vec[cyt.time_idx]
frequencies=np.arange(10, 100, 1)
amplitudes=np.arange(0.3, 0.31)
k0_exps=np.linspace(-1, 4, 10)
k0_vals=[10**x for x in k0_exps]#3z\13ave
ru_vals=copy.deepcopy(k0_vals)
#k0_vals=np.linspace(0.1, 0.8, 5)
copied_vals=copy.deepcopy(other_values)
parameters=["k_0", "Ru"]
val_dict=dict(zip(parameters, [param_list[x] for x in parameters]))
import time
val_dict["d_E"]=param_list["d_E"]
sensitivity_matrix=np.zeros((len(cyt.time_vec[cyt.time_idx]), len(parameters)))
counter=1
num_freqs=20
k0_infs=np.zeros(num_freqs)
k0_ru_infs=np.zeros(num_freqs)
heatmap_k0=np.zeros((len(k0_vals), len(ru_vals)))
heatmap_k0_ru=np.zeros((len(k0_vals), len(ru_vals)))
for k0 in range(0, len(k0_vals)):
    val_dict["k_0"]=k0_vals[k0]
    freqs=np.array([10**x for x in np.linspace(k0_exps[k0]-2,k0_exps[k0]+2, num_freqs)])
    for ru in range(0, len(ru_vals)):
        val_dict["Ru"]=ru_vals[ru]
        for z in range(0, num_freqs):
            val_dict["omega"]=freqs[z]
            for i in range(0, len(parameters)):
                    sensitivities, times, max_c=information_calc(val_dict, parameters[i], cyt,copy.deepcopy(other_values), 1e-2)
                    sensitivity_matrix[:,i]=np.interp(cyt.time_vec[cyt.time_idx], times, sensitivities)
            FIM=np.matmul(sensitivity_matrix.transpose(), sensitivity_matrix)
            k0_infs[z]=FIM[0,0]/max_c
            k0_ru_infs[z]=FIM[0, 1]
        #plt.subplot(1,2,1)
        #plt.loglog(freqs, k0_infs)
        #plt.subplot(1,2,2)
        #plt.loglog(freqs, k0_ru_infs)
        #plt.show()
        heatmap_k0[k0][ru]=freqs[np.where(k0_infs==max(k0_infs))]
        heatmap_k0_ru[k0][ru]=freqs[np.where(k0_ru_infs==max(k0_ru_infs))]
    #print(FIM)

            #infs[z]=information_calc(val_dict, "k0_scale", cyt, copied_vals, 1e-6)#/information_calc(val_dict, "Ru", cyt, copied_vals, 1e-6)
            #information_calc(val_dict, "k0_scale", cyt, copied_vals, 1e-6)#/information_calc(val_dict, "Ru", cyt, copied_vals, 1e-6)

print(heatmap_k0)
np.save("FIM_K0_check", heatmap_k0)
#np.save("FIM_K0_ru", heatmap_k0_ru)
