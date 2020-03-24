import numpy as np
try:
    import matplotlib.pyplot as plt
    plot=True
    from harmonics_plotter import harmonics
except:
    print("No plotting for ben_rama")
    plot=False
import os
import sys
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
dec_amount=16
for file in files:
    if scan in file and freq in file:

        if "current" in file:
            current_data=np.loadtxt(data_loc+"/"+file)
        elif "voltage" in file:
            voltage_data=np.loadtxt(data_loc+"/"+file)
try:
    current_results=current_data[0::dec_amount,1]
    time_results=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")

param_list={
        "E_0":0.2, #Midpoint potnetial (V)
        "E0_mean": 0.2,
        "E0_std":0.01,
        'E_start': min(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #Sinusoidal input minimum (or starting) potential (V)
        'E_reverse': max(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #Sinusoidal input maximum potential (V)
        'omega':8.94,   #frequency Hz
        "original_omega":8.94, #Nondimensionalising value for frequency (Hz)
        'd_E': 300e-3,   #ac voltage amplitude - V
        'area': 0.07, #electrode surface area cm^2
        'Ru': 100.0,  #     uncompensated resistance ohms
        'Cdl': 1e-5, #capacitance parameters
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-11,   # surface coverage per unit area
        "original_gamma":1e-11,        # Nondimensionalising cvalue for surface coverage
        "k0_shape":0,
        "k0_scale":0,
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5, #(Symmetry factor)
        'phase' : 3*(math.pi/2),#Phase of the input potential
        "cap_phase":3*(math.pi/2),
        'sampling_freq' : (1.0/400),
        "noise":0
    }
likelihood_options=["timeseries", "fourier"]
simulation_options={
        "no_transient":2/param_list["omega"],
        "experimental_fitting":True,
        "method": "sinusoidal",
        "likelihood":"fourier",
        "phase_only":False,
        "dispersion_bins":[16],
        "dispersion_distributions":["lognormal"],
        "label": "cmaes",
        "GH_quadrature": False,
        "optim_list":[],
    }
other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(4,10,1)),
        "experiment_current":current_results,
        "experiment_time":time_results,
        "experiment_voltage":voltage_results,
        "num_peaks":20,
    }
param_bounds={
    'E_0':[param_list["E_start"], param_list["E_reverse"]],#[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.15,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-12,1e-10],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0, 2*math.pi],
    "E0_mean":[param_list["E_start"], param_list["E_reverse"]],
    "E0_std": [1e-5,  0.2],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_scale":[0,1e4],
    "k0_shape":[0, 1],
    'phase' : [0, 2*math.pi],
    "noise":[0, 100]
}
cyt=single_electron(file_name=None, dim_parameter_dictionary=param_list, simulation_options=simulation_options, other_values=other_values, param_bounds=param_bounds)
nd_current=cyt.other_values["experiment_current"]
nd_voltage=cyt.other_values["experiment_voltage"]
nd_time=cyt.other_values["experiment_time"]
harms=harmonics(cyt.simulation_options["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.05)
data_harmonics=harms.generate_harmonics(nd_time, nd_current)
for i in range(0, len(data_harmonics)):
    plt.subplot(len(data_harmonics), 1, i+1)
    plt.plot(nd_voltage, data_harmonics[i,:])
plt.show()
plots="fourier"
cyt.def_optim_list(["E_0", "k_0", "Cdl","CdlE1", "CdlE2","CdlE3","Ru", "omega", "gamma", "alpha", "phase", "cap_phase"])
inf_params=[-0.26742665869741566, 50.00000000019233, 5.8652622271648164e-05, 0.009842939588946886, 0.0005954905413801843, 7.701265437505891e-06, 8.206747435756358e-08, 8.940664700078155, 1.7888689475894705e-11, 0.5999999999956317, 4.9913707115353425, 4.313702716197615]
inf_params_abs_fourier=[0.23203819989502655, 113.61318675353178, 0.00033665335590685874, -0.046617605673274984, -0.0012233226052263957, -0.0039373338896817826, 547.0164875315204, 8.941340842776235, 4.461696674088344e-11, 0.4000000011094948, 5.661175597154945, 1.8227255650448815]
inf_params_real_fourier=[0.010371794868859863, 0.7884306376225872, 0.0004858260741909631, -0.01607176909161101, -0.009858004231351813, -0.003997582231756558, 427.5422029353303, 8.93812581548564, 6.057934985536461e-11, 0.5633922027962549, 2.848857972309427, 5.914666146907428]
inf_params_real_4_9=[0.004246414426657297, 222.15842581845718, 0.00034622727986807365, 0.05706405306789489, 0.009932158556052817, 0.0006991843978110864, 383.80108765562017, 8.940090752161254, 5.3891932358446315e-11, 0.4906835504630738, 1.6944550110849772, 0.002061114786534785]
inf_params_real_4_10=[-0.030475811789263785, 38.55022465755111, 0.0002691859443380048, 0.0704733052072592, 0.009999926825132751, -0.009984786488729934, 865.5493476284323, 8.94018995145225, 5.734235422126767e-11, 0.4000000107761078, 1.3834607073832788, 4.470012024698083]
inf_params_real_4_10=[-0.0720588953506438, 28.768337502923615, 0.00035314588527364055, -0.11615588233295393, -0.0059141046634961095, -0.004857233358735673, 982.0201585811727, 8.940079221729626, 8.09266030189611e-11, 0.5954989868164128, 0.017300505531537013, 1.1724011100110385]
cmaes=cyt.test_vals(inf_params_real_4_10, "fourier", test=True)
cmaes_time=cyt.test_vals(inf_params_real_4_10, "timeseries", test=False)
freqs=np.fft.fftfreq(len(nd_time), nd_time[1]-nd_time[0])
if plots=="fourier":
    harms=harmonics(cyt.simulation_options["harmonic_range"], 8.94*cyt.nd_param.c_T0, 0.05)
    data_harmonics=harms.generate_harmonics(nd_time, nd_current)
    sim_harmonics=harms.generate_harmonics(nd_time, cmaes_time)
    for i in range(0, len(data_harmonics)):
        plt.subplot(len(data_harmonics), 1, i+1)
    #   plt.plot(nd_voltage, sim_harmonics[i,:])
        plt.plot(nd_voltage, data_harmonics[i,:], alpha=0.7)
    plt.show()
if plots=="timeseries":
    plt.subplot(1,2,1)
    plt.plot(nd_voltage, cmaes)
    plt.plot(nd_voltage, nd_current, alpha=0.7)
    plt.xlabel("ND voltage")
    plt.ylabel("ND current")
    plt.subplot(1,2,2)
    where_idx=tuple(np.where((nd_time>2) & (nd_time<4)))
    plt.plot(nd_time[where_idx], cmaes[where_idx])
    plt.plot(nd_time[where_idx], nd_current[where_idx], alpha=0.7)
    plt.xlabel("ND time")
    plt.ylabel("ND current")
    plt.show()
