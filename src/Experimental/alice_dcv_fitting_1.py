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
file_name="/dcV_Cjx-183D_WT_pH_7_2_3"
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
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 0, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    "Cdlinv":0,
    'CdlE1inv': 0,#0.000653657774506,
    'CdlE2inv': 0,#0.000245772700637,
    "CdlE3inv":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/50),
    'phase' :0.1,
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
    "dispersion_bins":[5],
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
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1,0.0],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e5],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'Cdlinv': [0,10], #(capacitance parameters)
    'CdlE1inv': [-5,5],#0.000653657774506,
    'CdlE2inv': [-5,5],#0.000245772700637,
    'CdlE3inv': [-5,5],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
    'k_0': [15, 1e4], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[-0.1,0.1],
    "E0_std": [1e-5,  0.5],
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
plt.plot(nondim_v, current_results)
for i in range(0, 2):

    current_half=current_results[idx_1[i]:idx_2[i]]
    time_half=time_results[idx_1[i]:idx_2[i]]
    volt_half=cyt.e_nondim(voltage_results[idx_1[i]:idx_2[i]])
    noise_idx=np.where((volt_half<interesting_section[i][0]) | (volt_half>interesting_section[i][1]))
    noise_voltages=volt_half[noise_idx]
    noise_current=current_half[noise_idx]
    noise_times=time_half[noise_idx]
    popt, pcov = curve_fit(func, noise_times, noise_current)
    fitted_curve=[func(t, *popt) for t in time_half]
    subtract_current[idx_1[i]:idx_2[i]]=np.subtract(current_half, fitted_curve)
    fitted_curves[idx_1[i]:idx_2[i]]=fitted_curve
    #plt.plot(volt_half, fitted_curve, color="red")
    #plt.plot(noise_voltages, noise_current)

cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","gamma", "alpha"])
PSV_optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"]
ramped_optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"]
PSV_params=[-0.021495031150668878, 17.570527719697008, 1949.3033882011057, 0.00020134013002972008, 0.00411080805846069, 0.0006879411270873684, 2.3340827906876003e-10, 9.014898777848584, 2.5763763013646175, 4.359308770046119, 0.4000000157539228]
PSV_params=[-0.06999997557877347, 1.245269093253469, 701.0390205847876, 0.00012081970706517015, 0.0024664965342477527, 0.0011465681607091556, 1.7085728483891467e-09, 9.014958257811347, 6.014524412936475, 5.676241758957758, 0.4683583542196279]
PSV_params=[-0.06999999976247695, 0.054653156208964916, 0.423123012427712, 361.21853877194775, 0.0005089406894416405, -0.055068050329032774, 0.0022654775318873364, 7.263427801734963e-11, 9.01488754940552, 1.5710075181470322, 5.2472698323265075, 0.4837659989252462]
ramped_params=[-0.04485376873500503, 293.2567587982391, 146.0113118472105, 0.0001576519851347672, 0.006105674536299788, 0.0012649370988525588, 2.2215281961212185e-11, 8.959294996508683, 6.147649245979944,0, 0.5372803774088237]
ramped_params=[-0.06158660103168602, 0.02767157845943783, 86.53789947453802, 40.56259593918775, 0.0009310670277647225, 0.030236335577448786, -0.0002525820042452911, 2.2619461371744093e-11, 8.959288930500506, 5.644994973337578, 0.5170561841197072]

DCV_inferred=[-4.98656884e-02,  6.57255463e-01,  1.53468787e-08,  4.00488302e-11,4.16446219e-01]
DCV_inferred=[-4.97750674e-02,  1.21728678e-02,  6.45936238e-01,  2.44965964e-08,4.08829719e-11 , 4.00000000e-01]

inferred_params=[PSV_params[PSV_optim_list.index(x)] for x in cyt.optim_list]
inferred_params=[ramped_params[ramped_optim_list.index(x)] for x in cyt.optim_list]
#inferred_params=DCV_inferred#[-4.97691836e-02,  1.71478268e-02,  3.04770512e+03,  2.63521136e+05, 4.12128630e-11,  4.80069036e-01]

test=cyt.test_vals(inferred_params, "timeseries")
#plt.plot(nondim_v, test)
plt.plot(nondim_v, subtract_current)
plt.show()
plt.show()
true_data=subtract_current
cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)

cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=False
score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
num_runs=5
for i in range(0, num_runs):
    x0=cyt.change_norm_group(inferred_params, "norm")
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(not cyt.simulation_options["test"])
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    plt.plot(voltage_results, cmaes_time)
    plt.plot(voltage_results, subtract_current)
    plt.show()
