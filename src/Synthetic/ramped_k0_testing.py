import numpy as np
import matplotlib.pyplot as plt
plot=True
import sys
sys.path.append("..")
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
from matplotlib.ticker import FormatStrFormatter
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Ramped"
files=os.listdir(data_loc)
harm_range=list(range(3,8,2))
num_harms=len(harm_range)+1
scan="_1_"
freq="_9Hz"
dec_amount=1
def flatten(something):
    if isinstance(something, (list, tuple, set, range)):
        for sub in something:
            yield from flatten(sub)
    elif isinstance(something, dict):
        for key in something.keys():
            yield from flatten(something[key])
    else:
        yield something


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

regime="irreversible"
for regime in ["irreversible", "reversible"]:

    if regime=="reversible":
        scale_scan_vals=np.flip([50, 100, 500])
        k0_val=100
    elif regime=="irreversible":
        scale_scan_vals=np.flip([0.5, 1, 2])
        k0_val=1
    params={"E_0":{"E0_mean":[-0.2, -0.25, -0.3],
                    "E0_std":[0.03,0.05 , 0.07],
                    "E0_skew":[5,-5, 0,]},
            "k_0":{"k0_shape":([0.25, 0.5, 0.65]),
                    "k0_scale":scale_scan_vals}}
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
    flattened_list=list(flatten(params))
    total_length=len(flattened_list)
    param_list={
        "E_0":-0.25,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
        'omega':8.881077023434541,#8.88480830076,  #    (frequency Hz)
        "v":    22.35174e-3,
        'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        "k0_shape":0.75,
        "k0_scale":k0_val,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/200),
        'phase' :0.0,
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
        "dispersion_bins":4,
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
    table_params=["E_start", "E_reverse", "E_0","k_0","Ru","Cdl","gamma", "alpha", "v", "omega", "phase","d_E","sampling_freq", "area", "E0_mean", "E0_std", "E0_skew", "k0_shape", "k0_scale"]

    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    print([cyt.dim_dict[x] if x in cyt.dim_dict else 0 for x in table_params])
    original_params=["E_0","k0_shape", "k0_scale"]
    table_params=["E_0","k0_scale", "k0_shape","Ru","Cdl","CdlE1","gamma","phase", "alpha"]
    print([param_list[x] for x in table_params])
    cyt.def_optim_list(["E_0", "k_0"])
    no_disp=cyt.test_vals([param_list["E_0"], param_list["k_0"]], "timeseries")
    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"], 0.1)
    time_results=cyt.t_nondim(cyt.other_values["experiment_time"])
    current_results=cyt.i_nondim(cyt.other_values["experiment_current"])
    voltage_results=cyt.e_nondim(cyt.other_values["experiment_voltage"])
    no_disp_harmonics=harms.generate_harmonics(time_results, cyt.i_nondim(no_disp))
    cyt.simulation_options["dispersion_bins"]=[20]
    cyt.simulation_options["GH_quadrature"]=False
    def e0_re_dim(arg):
        return np.multiply(arg, cyt.nd_param.c_E0)
    def k0_re_dim(arg):
        return np.divide(arg, cyt.nd_param.c_T0)
    plot_locs={"E_0":{"E0_mean":{"row":"row1", "col":0, "col_end":False, "func":e0_re_dim},
                    "E0_std":{"row":"row2", "col":0, "col_end":False, "func":e0_re_dim},
                    "E0_skew":{"row":"row3", "col":0, "col_end":True, "func":e0_re_dim}},
            "k_0":{"k0_shape":{"row":"row2", "col":1, "col_end":True, "func":k0_re_dim},
                    "k0_scale":{"row":"row1", "col":1, "col_end":False, "func":k0_re_dim}}}

    for parameter in original_params:
        disped_params=list(params[parameter].keys())
        updated_optim_list=copy.deepcopy(original_params)
        cyt.simulation_options["dispersion_test"]=True
        cyt.def_optim_list(updated_optim_list)
        for disp_param in disped_params:
            sim_params=[param_list[key] for key in updated_optim_list]
            for i in range(0, len(params[parameter][disp_param])):
                #sim_params[param_loc]=params[parameter][disp_param][i]
                syn_time=cyt.test_vals(sim_params, "timeseries")

                #plt.plot(syn_time)

                print(sim_params)
                #plt.show()
                #print("E0_std", cyt.dim_dict["E0_std"])
                syn_harmonics=harms.generate_harmonics(time_results, cyt.i_nondim(syn_time), hanning=True)
                no_disp_harms=harms.generate_harmonics(time_results, cyt.i_nondim(no_disp), hanning=True)
                #plt.subplot(2,2,j)
                #plt.plot(syn_time, alpha=0.5)
                func=plot_locs[parameter][disp_param]["func"]
                values, weights=cyt.return_distributions(1000)
                values=func(values)
                for q in range(0, len(cyt.disp_test)):
                    print(q)
                    current_harm=harms.generate_harmonics(time_results, cyt.i_nondim(cyt.disp_test[q]), hanning=True)
                    for z in range(0, len(current_harm)):
                        plt.subplot(len(current_harm)+1,1, z+1)
                        if q==0:
                            plt.plot(time_results, abs(syn_harmonics[z, :]))
                            plt.plot(time_results, abs(no_disp_harms[z, :]))
                        plt.plot(time_results, abs(current_harm[z, :]))
                    if q==0:
                        plt.subplot(len(current_harm)+1,1, len(current_harm)+1)
                        plt.plot(cyt.values, cyt.weights)
                plt.show()









    #ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
