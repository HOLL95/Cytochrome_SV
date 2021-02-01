import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import sys
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
import time
from single_e_class_unified import single_electron
from scipy.integrate import odeint
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/FTACV"
files=os.listdir(data_loc)
experimental_dict={}
param_file=open(data_loc+"/FTACV_params", "r")
useful_params=dict(zip(["max", "start", "Amp[0]", "freq", "rate"], ["E_reverse", "E_start", "d_E", "omega", "v"]))
dec_amount=64
for line in param_file:
    split_line=line.split()
    print(split_line)
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]

for i in range(1, 3):
    file_name="FTACV_Cyt_{0}_cv_".format(i)
    current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  -0.33898177567074794, #(starting dc voltage - V)
        'E_reverse':0.26049326614698887,
        'omega':8.959320552342025, #8.88480830076,  #    (frequency Hz)
        "v":0.022316752195354346,
        'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        "E0_skew":0.2,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :6.283185307179562,
        "time_end": -1,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["omega"])
    simulation_options={
        "no_transient":False,#time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[16],
        "GH_quadrature":True,
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
        "harmonic_range":list(range(4,8,1)),
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":200,
    }
    param_bounds={
        'E_0':[-0.1, 0.0],
        "E_start":[0.9*param_list["E_start"], 1.1*param_list["E_start"]],
        "E_reverse":[0.9*param_list["E_reverse"], 1.1*param_list["E_reverse"]],
        "v":[0.9*param_list["v"], 1.1*param_list["v"]],
        'omega':[0.8*param_list['omega'],1.2*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.05,0.05],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.1, 0.0],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [0, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    print(current_results[0], current_results[-1])
    voltage_results=cyt.other_values["experiment_voltage"]
    h_class=harmonics(other_values["harmonic_range"], param_list["omega"]*cyt.nd_param.c_T0, 0.05)

    volts=cyt.define_voltages()


    cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])
    inferred_params=[-0.3353782900811744, 330.32351080703046, 1.782826365614962e-11, 0.00012080435738781963, 0.0025267358958767356, 0.0006879410370079902, 1.9057794524924306e-11, param_list["omega"], 0, 0.4092580171166158]
    inferred_params=[0.0970288663866738, 362.510776540853, 9248.005110326583, 4.9897881784094635e-03, 0.09999999902298948*0, -0.005266814814752714, 2.9148497981974626e-11, param_list["omega"], 0, 0.5987156826440815]
    ramped_inferred=[-0.04485376873500503, 293.2567587982391, 146.0113118472105, 0.0001576519851347672, 0.006105674536299788, 0.0012649370988525588, 2.2215281961212185e-11, 8.959294996508683, 6.147649245979944, 0.5372803774088237]
    inferred_params=[-0.021495031150668878, 17.570527719697008, 1949.3033882011057, 0.0001576519851347672, 0.006105674536299788, 0.0012649370988525588, 2.2215281961212185e-10, 8.959294996508683, 6.147649245979944, 0.5372803774088237]
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","omega","phase", "alpha"])
    ramped_inferred=[-0.05174906751135161, 0.016887979832412262, 366.94064660027357, 212.25658687481226, 0.0018474547411381252, 0.0343738492662911*0, -0.002837054011354272*0, 8.367546164744082e-11, 8.982298908479827, 1.5595328057012567, 0.46514102280787095]
    #ramped_inferred=[-0.06158660103168602, 0.02767157845943783, 86.53789947453802, 40.56259593918775, 0.0009310670277647225, 0.030236335577448786, -0.0002525820042452911, 2.2619461371744093e-11, 8.959288930500506, 5.644994973337578, 0.5170561841197072]
    ramped_inferred_4=[-0.06399623913220037, 0.028829876305729137, 72.32019727498674, 5.1441749230098965, 0.00019999999896173226, -0.02526547219292212, 0.00999999933066467, 2.0738300680573824e-11, 8.959294007054194, 3.8121510094580344, 0.5790715806838499]
    ramped_inferred_4=[-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333, -0.004699728058449013, 2.1898117688472174e-11, 8.959294458587753, 0.9281447797610709, 0.5592126258301378]
    ramped_inferred_4=[-0.07963988256472493, 0.023023349442016245, 20.550878790990733, 58.15147052074157, 8.806259570340898e-05, -0.045583168350011485*0, -0.00011159862236990309*0, 2.9947102043021914e-11, 8.959294458587753, 0, 0.5999994350046962]
    ramped_inferred_4=[-0.07273249049030611, 0.05175073250299541, 6999.792928467961, 34.93194847564691, 1.1959203102235415e-05, -0.014686358116624373, 0.08214545405649867, -0.0005651537062958054, 1.436464498385903e-11, 8.959294458587753, 0, 0.5019999790379509]
    ramped_inferred_5=[-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.37095896576959e-11, 8.959294458587753, 0, 0.5999999989106146]
    #ramped_inferred_4=[-0.05744624908677006, 0.02788517403038323, 164.40170561479968, 268.7042102662674, 0.00019999999918567714, -0.01781717221653123, -0.0010262335333251996, 2.4414183646021532e-11, 8.959320552342025, 6.283184732454485, 0.40000000000073965]
    #ramped_inferred=[-0.07963988256472493, 0.043023349442016245, 20.550878790990733, 581.5147052074157, 8.806259570340898e-05, -0.045583168350011485, -0.00011159862236990309, 2.9947102043021914e-11, 8.959320552342025, 0, 0.5999994350046962]
    cmaes_test=cyt.test_vals(ramped_inferred_4, "timeseries")
    cmaes_test_4=cyt.test_vals(ramped_inferred_5, "timeseries")
    #w0 = [current_results[0],0, voltage_results[0]]
    #wsol = odeint(cyt.current_ode_sys, w0, time_results)
    #adaptive_current=wsol[:,0]
    #adaptive_potential=wsol[:,2]
    #adaptive_theta=wsol[:, 1]
    cyt.simulation_options["method"]="dcv"
    dcv_volt=cyt.e_nondim(cyt.define_voltages()[cyt.time_idx])
    cyt.simulation_options["method"]="ramped"
    h_class.plot_harmonics(time_results, experimental_time_series=current_results, fitted_alpha_time_series=cmaes_test, fixed_alpha_time_series=cmaes_test_4, hanning=True, alpha_increment=0.2, plot_func=abs)
    plt.plot(cmaes_test)
    plt.plot(current_results)
    plt.show()
    cyt.simulation_options["label"]="cmaes"
    cyt.simulation_options["test"]=False
    cyt.simulation_options["voltage_only"]=False
    true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)
    if simulation_options["likelihood"]=="timeseries":
        cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
    elif simulation_options["likelihood"]=="fourier":
        dummy_times=np.linspace(0, 1, len(fourier_arg))
        cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
    score = pints.SumOfSquaresError(cmaes_problem)
    CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
    x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(not cyt.simulation_options["test"])
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    #plt.subplot(1,2,1)
    plt.plot(time_results, cmaes_time)
    plt.plot(time_results, voltage_results, alpha=0.5)
    plt.show()
    h_class.plot_harmonics(time_results, experimental_time_series=current_results, simulated_time_series=cmaes_time, hanning=True, plot_func=abs)
    plt.plot(cyt.top_hat_filter(cmaes_time))
    plt.plot(fourier_arg)
    plt.show()
