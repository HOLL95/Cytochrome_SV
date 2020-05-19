import numpy as np
import matplotlib.pyplot as plt
import sys
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from single_e_class_unified import single_electron
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/DCV/22_Aug/"
files=os.listdir(data_loc)
number="2"
blank_nos=[str(x) for x in range(1, 7)]
for letter in ["A"]:
    for file in files:
        if ".ids" not in file:
            if "183"+letter in file and number in file:
                dcv_file=np.loadtxt(data_loc+file, skiprows=1)
                plt.plot(dcv_file[:,1], dcv_file[:, 2])
    plt.show()
current_results1=np.array(dcv_file[:, 2])
voltage_results1=np.array(dcv_file[:, 1])
time_results1=np.array(dcv_file[:, 0])
for letter in ["blank"]:
    for number in blank_nos:
        for file in files:
            if ".ids" not in file:
                if letter in file and number in file:
                    blank_file=np.loadtxt(data_loc+file, skiprows=1)
                    blank_current=blank_file[:, 2]
                    blank_voltage=blank_file[:, 1]
                    blank_time=blank_file[:, 0]


plt.show()
len_current=len(current_results1)
new_current=copy.deepcopy(current_results1)
new_voltage=copy.deepcopy(voltage_results1)
first_current=new_current[:len_current//2]
second_current=new_current[len_current//2:]
first_voltage=new_voltage[:len_current//2]
second_voltage=new_voltage[len_current//2:]
first_time=time_results1[:len_current//2]
second_time=time_results1[len_current//2:]

subtracted_idx_1=tuple(np.where((first_voltage<-0.25) | (first_voltage>-0.07)))
subtracted_idx_2=tuple(np.where((second_voltage<-0.44) | (second_voltage>-0.20)))
#new_current=np.interp(time_results1, time_results1[subtracted_idx], current_results1[subtracted_idx])

#first_current[subtracted_idx_1]=0
#second_current[subtracted_idx_2]=0
interp_first=np.interp(first_time, first_time[subtracted_idx_1], first_current[subtracted_idx_1])
interp_second=np.interp(second_time, second_time[subtracted_idx_2], second_current[subtracted_idx_2])
#plt.plot(first_voltage, interp_first)
#plt.plot(second_voltage, interp_second)
total_interp=np.append(interp_first, interp_second)
plt.plot(voltage_results1, np.subtract(current_results1, total_interp))
plt.plot(voltage_results1, current_results1)
plt.show()

#current_results1=current_results1-blank_current
param_list={
    "E_0":0.2,
    'E_start':  min(voltage_results1), #(starting dc voltage - V)
    'E_reverse':max(voltage_results1),
    'omega':8.94, #8.88480830076,  #    (frequency Hz)
    "v":0.03,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    "Cdlinv":0,
    'CdlE1inv': 0,#0.000653657774506,
    'CdlE2inv': 0,#0.000245772700637,
    "CdlE3inv":0,
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
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["omega"])
simulation_options={
    "no_transient":0.15,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":16,
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
    "harmonic_range":list(range(3,9,1)),
    "experiment_time": time_results1,
    "experiment_current": np.subtract(current_results1, total_interp),
    "experiment_voltage":voltage_results1,
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.3,-0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1,1],#0.000653657774506,
    'CdlE2': [-1,1],#0.000245772700637,
    'CdlE3': [-1,1],#1.10053945995e-06,
    'Cdlinv': [0,100],
    'CdlE1inv': [-10,10],#0.000653657774506,
    'CdlE2inv': [-2,2],#0.000245772700637,
    'CdlE3inv': [-2,2],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
    'k_0': [0, 1e4], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[param_list['E_start'],param_list['E_reverse']],
    "E0_std": [1e-5,  0.5],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
}
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)

time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
volts=cyt.define_voltages()
plt.plot(volts)
plt.plot(voltage_results)
plt.show()

subtracted_params=[-0.2388971165773926, 0.016113247615555718, 0.13852598242255806, 7.056740959347867e-06, 2.0199596465623735e-17, 1.8868271004500844e-10, 0.5501855054699475]

cyt.simulation_options["dispersion_bins"]=[50]
cyt.simulation_options["GH_quadrature"]=True
cyt.def_optim_list([ "E0_mean", "E0_std", "k_0","Ru","Cdl","gamma", "alpha"])
cyt.simulation_options["test"]=False
cyt.simulation_options["label"]="cmaes"
results=cyt.test_vals(subtracted_params, "timeseries")
plt.plot(cyt.e_nondim(voltage_results), cyt.i_nondim(results), label="Simulation")
plt.plot(cyt.e_nondim(voltage_results), cyt.i_nondim(current_results), label="Background subtracted data")
#plt.show()




print("--"*20, cyt.values, cyt.weights)

other_current=np.zeros(len(time_results))
z=np.ones(len(other_current)//2)
z=np.append(z, z*-1)
z=np.append(z, -1)

for i in range(0, len(cyt.values)):
    cyt.dim_dict["E_0"]=cyt.e_nondim(cyt.values[i])
    c_current=cyt.Armstrong_dcv_current(cyt.t_nondim(time_results), cyt.e_nondim(voltage_results))
    c_current=np.multiply(c_current, z)
    other_current=np.add(other_current, np.multiply(c_current, cyt.weights[i]))
plt.plot(cyt.e_nondim(voltage_results), other_current, label="Armstrong current")
plt.xlabel("Voltage(V)")
plt.ylabel("Current(A)")
plt.legend()
plt.show()

cmaes_problem=pints.SingleOutputProblem(cyt, time_results, current_results)
score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
#x0=cyt.change_norm_group(ifft_vals, "norm")
cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
cmaes_fitting.set_parallel(False)
found_parameters, found_value=cmaes_fitting.run()
print(found_parameters)
cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
print(list(cmaes_results))
cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)

plt.plot(voltage_results, cmaes_time)
plt.plot(voltage_results, current_results, alpha=0.5)

plt.show()
