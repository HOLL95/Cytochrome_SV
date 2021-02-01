import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
plot=True
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import matplotlib
from matplotlib.ticker import FormatStrFormatter
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="3"
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
regime="reversible"
regime="irreversible"
k0_vals=[10, 100, 1000, 10000, 5e5, 1e6]
num_freqs=6



param_list={
    "E_0":-0.25,
    'E_start':  -0.3402878, #(starting dc voltage - V)
    'E_reverse': 0.2610671,
    'omega':1,#8.88480830076,  #    (frequency Hz)
    "original_omega":1,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 0.0,  #     (uncompensated resistance ohms)
    'Cdl': 8e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 100, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":-0.25,
    "E0_std": 0.05,
    "E0_skew":0,
    "k0_shape":0.5,
    "k0_scale":100,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 30,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=3/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[10],
    "GH_quadrature":True,
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
    "harmonic_range":list(range(2, 11)),
    "experiment_time": None,
    "experiment_current": None,
    "experiment_voltage":None,
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
real_params=[[-0.05897077320642114, 0.012624609618582739, 569.8770986714504, 137.37161854510956, 0.0008889805390117373, 5.2754243128435144e-11, 9.014645134635346, 4.449177598843859, 5.024539591886025, 0.5819447228966595],
[-0.08064031224498379, 0.020906827217859487, 63.01964378454537, 112.80693965100555, 0.0007989280702348236,  2.9678863151432805e-11, 9.014741538924035, 5.49549816495015, 5.193383660743178, 0.5999999698409333],
[-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06,  1.37095896576959e-11, 9.01499164308653, 4.7220768639743085, 4.554136092744141, 0.5999999989106146]]



freqs=np.linspace(9.014907846043297,100, 10)
cdls=[1e-6, 1e-5, 1e-4]
num_vars=1
var_list=np.linspace(1.0, 1.5, num_vars)
harms=harmonics(other_values["harmonic_range"], 1, 0.05)
optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","gamma","omega","cap_phase","phase", "alpha"]
fig_list=[]
ax_list=[]
for v in range(0, harms.num_harmonics):
    fig, ax=plt.subplots(2, 5)
    fig_list.append(fig)
    ax_list.append(ax)

for j in range(0, len(real_params)):
    for m in range(0, len(optim_list)):
        master_params=real_params[j]
        for r in range(0, num_vars):
            cyt_params=copy.deepcopy(master_params)
            idx=optim_list.index(optim_list[m])
            if optim_list[idx]=="omega":
                continue
            cyt_params[idx]=master_params[idx]*var_list[r]
            print(list(cyt_params))
            amps=np.zeros((harms.num_harmonics, len(freqs)))
            for i in range(0, len(freqs)):
                param_list["omega"]=freqs[i]
                param_list["original_omega"]=freqs[i]
                simulation_options["no_transient"]=False
                cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
                cyt.def_optim_list(optim_list)
                copy_params=copy.deepcopy(cyt_params)
                copy_params[cyt.optim_list.index("omega")]=freqs[i]
                current=cyt.test_vals(copy_params, "timeseries")
                volts=cyt.e_nondim(cyt.define_voltages()[cyt.time_idx])
                times=cyt.t_nondim(cyt.time_vec[cyt.time_idx])
                no_disp=cyt.i_nondim(current)
                times=cyt.t_nondim(cyt.time_vec)

                #plt.plot(cyt.time_vec[cyt.time_idx], current)
                #plt.show()
                plot_harmonics, amplitude=harms.generate_harmonics(cyt.time_vec[cyt.time_idx], current, hanning=True, return_amps=True)
                amps[:, i]=amplitude
                #plt.subplot(2, 3, j+1)
                #plt.plot(volts, np.real(plot_harmonics[3, :]), label="$\Omega=$"+str(param_list["omega"]))
                #plt.xlabel("Potential (V)")
                #plt.ylabel("Current (A)")
            #for q in range(0, harms.num_harmonics):
            for k in range(0, harms.num_harmonics):

                ax=ax_list[k]
                if abs(cyt_params[idx])<1e-2:
                    legend_string="{:.2e}".format(cyt_params[idx])
                else:
                    legend_string=str(round(cyt_params[idx], 3))
                ax[m//5,m%5].plot(freqs, amps[k, :], label=optim_list[m]+"="+legend_string)
                ax[m//5,m%5].legend()
                ax[m//5,m%5].set_title(harms.harmonics[k])
    for k in range(0, harms.num_harmonics):
        fig=fig_list[k]
        fig.set_size_inches(16, 8)
        fig.subplots_adjust(top=0.96,
        bottom=0.075,
        left=0.035,
        right=0.97,
        hspace=0.22,
        wspace=0.26)
plt.show()
        #fig.savefig("Harmonic_amplitude_"+str(k+other_values["harmonic_range"][0])+"_parameter_scans.png", dpi=300)








    #ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
