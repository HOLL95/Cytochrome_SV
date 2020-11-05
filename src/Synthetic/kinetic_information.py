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
k0_val=100
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
    'num_peaks': 20,
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
    volts=new_class.define_voltages()[new_class.time_idx]
    upper_param=(new_class.test_vals(params, "timeseries"))
    params[interest_position]=dim_dict[param_of_interest]-change_val
    normal_param=(new_class.test_vals(params, "timeseries"))
    derivative=np.divide(np.subtract(upper_param, normal_param), 2*change_val)
    total_information=integ.trapz(np.power(derivative,2))
    return total_information
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
k0_exps=range(-1, 4)
k0_vals=[10**x for x in k0_exps]#3z\13ave
ru_vals=copy.deepcopy(k0_vals)
#k0_vals=np.linspace(0.1, 0.8, 5)
copied_vals=copy.deepcopy(other_values)
param_combs=list(itertools.product(frequencies, amplitudes))
kinetic_information=np.zeros((len(k0_vals), len(ru_vals)))
val_dict={}
import time
val_dict["d_E"]=param_list["d_E"]
const_params={"k0_scale":{"range":k0_vals, "other":{"param":"k0_shape", "val":0.5}}, "k0_shape":{"range":np.arange(0.2, 1.0, 0.2), "other":{"param":"k0_scale", "val":100}}}
counter=1
for param in const_params.keys():
    for i in range(0, len(const_params[param]["range"])):
        val_dict[param]=const_params[param]["range"][i]
        val_dict[const_params[param]["other"]["param"]]=const_params[param]["other"]["val"]
        val_dict["Ru"]=100
        if param=="k0_scale":
            freq_min=k0_exps[i]-2
            freq_max=k0_exps[i]+2
            freqs=np.linspace(freq_min, freq_max, 100)
            freqs=np.power(10, freqs)
        else:
            freqs=[10**x for x in np.linspace(0,4, 100)]
            print(freqs, "FREQS")
        errors=np.zeros(len(freqs))
        for z in range(0, len(freqs)):
            val_dict["omega"]=freqs[z]

            #infs[z]=information_calc(val_dict, "k0_scale", cyt, copied_vals, 1e-6)#/information_calc(val_dict, "Ru", cyt, copied_vals, 1e-6)
            #information_calc(val_dict, "k0_scale", cyt, copied_vals, 1e-6)#/information_calc(val_dict, "Ru", cyt, copied_vals, 1e-6)
            errors[z]=k0_diff_calc(val_dict, cyt, other_values)

        #plt.show()
        plt.subplot(1,2,counter)
        plt.plot(np.log10(freqs), errors, label=fancy_names[param]+"="+str(round(const_params[param]["range"][i],2)))
    plt.legend(loc="upper right")
    plt.xlabel("$\\log_{10}$(Frequency)")
    plt.ylabel("Dimensionless RMSE")
    counter+=1
fig=plt.gcf()
fig.set_size_inches((7, 4.5))
save_path="Scale_information.png"
#plt.show()
plt.subplots_adjust(top=0.88,
                    bottom=0.11,
                    left=0.1,
                    right=0.985,
                    hspace=0.2,
                    wspace=0.2)
plt.show()
fig.savefig(save_path, dpi=500)

"""try:
    kinetic_information[i][j]=freqs[np.where(infs==max(infs))]
except:
    fs=freqs[np.where(infs==max(infs))]
    kinetic_information[i][j]=fs[0]
        #plt.plot(volt, current)
    #plt.show()
#file= open("Scan_results.txt", "w")"""
np.save("Scan_results.txt", kinetic_information)
ax=seaborn.heatmap(kinetic_information, xticklabels=ru_vals, yticklabels=k0_vals, cbar=True)
ax.invert_yaxis()
ax.set_xlabel("Ru")
ax.set_ylabel("k0")
plt.show()
X,Y=np.meshgrid(amplitudes, frequencies)

fig = plt.figure()
ax = fig.gca(projection='3d')


surf = ax.plot_surface(X, Y, area_ratios, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#print(opt.minimize(frequency_min, x0=10,method = 'Nelder-Mead', args=(cyt, 1e-3, 10000, copy.deepcopy(other_values))) )
plt.show()

highest_cdl_val=-1
cdl_ints=range(-7, highest_cdl_val)
spacing=20
cdl_powers=[0]*(spacing*(len(cdl_ints)-1)+len(cdl_ints))
for i in range(0, len(cdl_ints)):
    cdl_powers[i*(spacing+1)]=cdl_ints[i]
    if i>0:
        cdl_powers[((i-1)*(spacing+1))+1:(i*(spacing+1))]=np.linspace(cdl_ints[i-1], cdl_ints[i], spacing+2)[1:-1]


cdl_vals=[10**x for x in cdl_powers]
area=np.zeros(len(cdl_vals))


for i in range(0, len(cdl_vals)):
    current_cdl=cyt.test_vals([cdl_vals[i]], "timeseries")

    if np.log10(cdl_vals[i])%1==0:

        plt.subplot(1,2,1)
        plt.plot(voltage, current_cdl, label=np.log10(cdl_vals[i]))
    area[i]=integ.simps(abs(current_cdl), times)
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.log10(cdl_vals), np.divide(cdl_0_current_area, area))
plt.legend()
plt.show()
