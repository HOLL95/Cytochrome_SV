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
current_results1=dcv_file[:, 2]
voltage_results1=dcv_file[:, 1]
time_results1=dcv_file[:, 0]
for letter in ["blank"]:
    for number in blank_nos:
        for file in files:
            if ".ids" not in file:
                if letter in file and number in file:
                    blank_file=np.loadtxt(data_loc+file, skiprows=1)
                    blank_current=blank_file[:, 2]
                    blank_voltage=blank_file[:, 1]
                    blank_time=blank_file[:, 0]
                    plt.plot(blank_voltage, current_results1-blank_current, label=number)
plt.plot(voltage_results1, blank_current)
plt.legend()
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
    "no_transient":0.1,
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
    "experiment_current": current_results1,
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
    'k_0': [0.1, 1e4], #(reaction rate s-1)
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
plt.plot(voltage_results, current_results)
plt.show()
volts=cyt.define_voltages()
plt.plot(volts)
plt.plot(voltage_results)
plt.show()
cyt.simulation_options["dispersion_bins"]=[16]
cyt.simulation_options["GH_quadrature"]=True
cyt.def_optim_list([ "E_0", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv","gamma", "alpha"])
norm_vals=[ "E_0", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv","gamma", "alpha"]
curr_best=[-0.12138673699417057, 5653.711865172047, 6.251920538521568e-07, 0.0005113312954312908, 0.032688180215297846, -0.0016959746570033296, -4.363761285119949e-05, 5.118104508670231, -0.2856860853954224, -0.012643633142239707, -0.0001955648458022985, 6.441193562150172e-12, 0.5250534731453991]
blank_sub=[0.09306177584874675, 1.9344951154832872, 7.4238908166570345, 0.0001868507457614954, 0.03418077288539134, -0.003700461869470928, -8.222260002632709e-05, 1.8219980821049462, -0.4487186289315357, -0.007048236265024821, 0.00020170449845746674, 1.4283135211180592e-12, 0.5569545398118998]
with_params=[-0.25819776867643385,  0.06004627245093131, 83951.6420075274, 6.251920538521568e-07, 0.0005113312954312908, 0.032688180215297846, -0.0016959746570033296, -4.363761285119949e-05, 5.118104508670231, -0.2856860853954224, -0.012643633142239707, -0.0001955648458022985, 2.0464993756449906e-10, 0.5250534731453991]
with_params_sub=[-0.25819776867643385,  0.06004627245093131, 83951.6420075274, 7.4238908166570345, 0.0001868507457614954, 0.03418077288539134, -0.003700461869470928, -8.222260002632709e-05, 1.8219980821049462, -0.4487186289315357, -0.007048236265024821, 0.00020170449845746674, 2.0464993756449906e-10, 0.5569545398118998]

plot_results=[current_results1,current_results1, current_results1-blank_current, current_results1-blank_current]
vals=[curr_best,with_params, blank_sub,  with_params_sub]

"""for i in range(0, len(vals)):
    if len(vals[i])>len(norm_vals):
        cyt.def_optim_list([ "E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv","gamma", "alpha"])
    else:
        cyt.def_optim_list(norm_vals)
    test=cyt.test_vals(vals[i], "timeseries")
    plt.subplot(2,2,i+1)
    if i==1:
        plt.ylim([-2, 2])
    if i>1:
        plt.ylim([-1, 1])
    plt.plot(voltage_results1, cyt.i_nondim(test)*1e6)
    plt.plot(voltage_results1, plot_results[i]*1e6)
    plt.xlabel("voltage(V)")
    plt.ylabel("Current($\\mu$A)")
plt.show()"""
cdl_vals=[0.0005113312954312908, 0.032688180215297846, -0.0016959746570033296, -4.363761285119949e-05, 5.118104508670231, -0.2856860853954224, -0.012643633142239707, -0.0001955648458022985]
cdl_params=["Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv"]
for i in range(0, len(cdl_vals)):
    cyt.dim_dict[cdl_params[i]]=cdl_vals[i]
cyt.def_optim_list([ "E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv","gamma", "alpha"])
cyt.simulation_options["test"]=False
cyt.simulation_options["label"]="cmaes"
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
