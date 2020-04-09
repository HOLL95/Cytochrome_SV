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
values=[[-0.21990071721861593, 44.01096292456005, 119.48426204727059, 0.0002451477915335569, 0.13729895076202486, 0.007928163599045154, 9.239820340868298e-11, 8.941077048962455, 4.7123345102082865, 3.813219214475715, 0.5740881278500752],\
        [-0.2515873899481767, 380.850371209114, 190.56427390559483, 0.0009996714551589862, 0.05283450443740578, 2.4559146124050746e-05, 9.979263862879792e-11, 8.94052872191747, 6.228679431228626, 3.5117346885780156, 0.4454217075669103],\
        [-0.3737835519192576, 110.89381450977515, 192.9584698382905, 0.000999061235805586, 0.05059781896924842, 6.770899887461636e-05, 9.971258525476312e-11, 8.940476323669214, 1.5710812221704191, 4.285970106715876, 0.4074546974273283],\
        [-0.3035803306837689, 133.5644652314799, 70.93960743168643, 0.0008576399748363899, 0.14120750241196917, 0.00908999058237603, 8.384891653172269e-11, 8.940392636062402, 6.028883772290079, 5.154957131517447, 0.5699823467206875]]

values=[[-0.2120092471414607, 0.0005478233474769771, 116.21497065365581, 431.92918718571053, 0.00044100528598203375, 0.14367030986553458, 0.005163770653874243, 9.999387751676114e-11, 8.941077023434541, 4.4700719867926875, 3.4597619059667073, 0.5997147084901965],\
        [-0.30978826058698744, 0.0011747969813993733, 83.58943751970683, 351.84551152031054, 0.000767528492853802, 0.0828273845158425, 0.0016183047215945039, 9.889421336332628e-11, 8.940528233151985, 1.5707987979911695, 4.022718048224439, 0.597523000771728],\
        [-0.2085010348585462, 0.0001769719441256009, 93.88915054231003, 621.7895281614545, 0.00038185721998072834, 0.14858346584854376, 0.005464884000805036, 9.995887932170653e-11, 8.94129543205022, 4.488924821745932, 3.481574064188825, 0.5990602813196874],\
        [-0.18408651864872588, 0.03810377947655846, 37.99182466615228, 206.07566180326032, 0.000917892619282633, 0.14267120449901333, 0.005115346249619759, 9.999947573178033e-10, 8.94051615668972, 4.564635581685883, 3.67191067357323, 0.5971094383283885]

]

#4-6
#[-0.278363899435654, 0.0008259325710375285, 735.6449871442546, 44.333552877469735, 0.0006071998304525408, 0.14954569406064083, 0.00929097560844948, 7.83905579413785e-11, 8.940528737796546, 1.664609172135028, 3.550241253861702, 0.4626203617983563]
#[-0.30978826058698744, 0.0011747969813993733, 83.58943751970683, 351.84551152031054, 0.000767528492853802, 0.0828273845158425, 0.0016183047215945039, 9.889421336332628e-11, 8.940528233151985, 1.5707987979911695, 4.022718048224439, 0.597523000771728]
#3-5
#[-0.2120092471414607, 0.0005478233474769771, 116.21497065365581, 431.92918718571053, 0.00044100528598203375, 0.14367030986553458, 0.005163770653874243, 9.999387751676114e-11, 8.941077023434541, 4.4700719867926875, 3.4597619059667073, 0.5997147084901965]
#[-0.45584844089513743, 0.035885073716691786, 906.663024596196, 927.7587043294853, 0.0004503215938428236, 0.14607330978980265, 0.0055799382065847055, 9.144044134173687e-11, 8.941077298537868, 4.854835346057383, 3.690283369322451, 0.58811927979759]
#3-6
#[-0.2085010348585462, 0.0001769719441256009, 93.88915054231003, 621.7895281614545, 0.00038185721998072834, 0.14858346584854376, 0.005464884000805036, 9.995887932170653e-11, 8.94129543205022, 4.488924821745932, 3.481574064188825, 0.5990602813196874]
#CDL_only
#[0.00035368666405915036, 0.058740340674938746, 0.002138640810685391, 8.94063571073963, 4.332415066777621, 4.785295526373359]

harm_range=list(range(1,9,1))
fig, ax=plt.subplots(len(harm_range), len(values))
count=-1
for q in range(0, len(values)):
    count+=1
    param_list={
        "E_0":0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
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
    vals=[-0.2115664620575202,  50.70216501235601, 1.2299591730542196e-08, 1e-5, 0.02019028489306987, 0.0021412236896496805, 7.777463350920317e-11, 8.940744445900455, 4.371233348216213, 3.393780500934783, 0.5310264089857374]
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
