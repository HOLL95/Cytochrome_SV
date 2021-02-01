import sys
sys.path.append("..")
import numpy as np
from multiplotter import multiplot
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
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/PSV"
files=os.listdir(data_loc)
experimental_dict={}
param_file=open(data_loc+"/PSV_params", "r")
harms=list(range(3,11))
useful_params=dict(zip(["max", "min", "Amp[0]", "Freq[0]"], ["E_reverse", "E_start", "d_E", "original_omega"]))
#figure=multiplot(1,2, **{"harmonic_position":[1], "num_harmonics":len(harms), "orientation":"portrait",  "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":1, "plot_height":1, "harmonic_spacing":1})

#plt.show()
dec_amount=8
for line in param_file:
    split_line=line.split()
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]
def single_oscillation_plot(times, data, colour, label="", alpha=1, ax=None):
    end_time=int(times[-1]//1)-1
    start_time=3
    if ax==None:
        ax=plt.subplots()
    for i in range(start_time, end_time):
        data_plot=data[np.where((times>=i) & (times<(i+1)))]
        time_plot=np.linspace(0, 1, len(data_plot))
        if i==start_time:
            ax.plot(time_plot, data_plot, color=colour, label=label, alpha=alpha)
        else:
            ax.plot(time_plot, data_plot, color=colour, alpha=alpha)
for i in range(1, 2):
    file_name="PSV_Cyt_{0}_cv_".format(i)
    current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":0,
        'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
        'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
        'omega':9.015120071612014, #8.88480830076,  #    (frequency Hz)
        "original_omega":9.015120071612014,
        'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 1.0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-4, #(capacitance parameters)
        'CdlE1': 1e-3,#0.000653657774506,
        'CdlE2': 1e-4,#0.000245772700637,
        "CdlE3":0,
        'gamma': 1e-11*0,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":-0.1,
        "E0_std": 0.01,
        "E0_skew":0.2,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :3*math.pi/2,
        "time_end": -1,
        'num_peaks': 50,
    }
    print(param_list)
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["original_omega"])
    simulation_options={
        "no_transient":time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[8],
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
        "harmonic_range":list(range(4,100,1)),
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":200000,
    }
    param_bounds={
        'E_0':[-0.1, 0.1],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e2],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.1,0.1],#0.000653657774506,
        'CdlE2': [-0.05,0.05],#0.000245772700637,
        'CdlE3': [-0.05,0.05],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],3*param_list["original_gamma"]],
        'k_0': [0, 2e2], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.08, 0.04],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [math.pi, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    print(current_results[0], current_results[-1])
    voltage_results=cyt.other_values["experiment_voltage"]
    h_class=harmonics(harms , 1, 0.05)
    #h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, xaxis=voltage_results)
    fft=one_tail(np.fft.fft(current_results))
    hann_fft=one_tail(np.fft.fft(np.multiply(np.hanning(len(current_results)),current_results)))
    f=one_tail(np.fft.fftfreq(len(time_results), time_results[1]-time_results[0]))
    predicted_cap_params=[0.00016107247651709253, 0.0032886486609914056, 0.0009172547160104724]
    cap_param_list=["Cdl", "CdlE1", "CdlE2"]
    #for i in range(0, len(cap_param_list)):
    #    cyt.param_bounds[cap_param_list[i]]=[predicted_cap_params[i]*0.75, predicted_cap_params[i]*1.25]
    cyt.def_optim_list(["E0_mean","E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","omega","cap_phase","phase", "alpha"])

    #inferred_params4=[-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.37095896576959e-11, 9.01499164308653, 4.7220768639743085, 4.554136092744141, 0.5999999989106146]
    inferred_params4=[param_list[x] for x in cyt.optim_list]




    true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)
    colours=plt.rcParams['axes.prop_cycle'].by_key()['color']

    #single_oscillation_plot(time_results, fourier_arg, colours[0], label="Experiment Data", ax=figure.axes_dict["col1"][0])

    normal=cyt.test_vals(inferred_params4, "timeseries")
    plt.plot(voltage_results, normal)
    plt.show()
    noisy_current=cyt.add_noise(normal, 0.01*max(normal))
    cyt.dim_dict["Q"]=0.01*max(normal)
    cyt.secret_data_time_series=noisy_current
    cyt.simulation_options["numerical_method"]="Kalman_simulate"
    cyt.simulation_options["Kalman_capacitance"]==True
    #inferred_params4[cyt.optim_list.index("cap_phase")]=3.5*math.pi/2
    kalman_pred=cyt.test_vals(inferred_params4, "timeseries")
    #plt.plot(voltage_results, kalman_pred)
    #plt.plot(voltage_results, cyt.pred_cap)
    #plt.plot(voltage_results, cyt.farad_current)
    plt.show()

    if simulation_options["likelihood"]=="timeseries":
        cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
    elif simulation_options["likelihood"]=="fourier":
        dummy_times=np.linspace(0, 1, len(fourier_arg))
        cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
        plt.plot(fourier_arg)
        plt.show()
    cyt.simulation_options["label"]="cmaes"
    cyt.simulation_options["test"]=False
    score = pints.SumOfSquaresError(cmaes_problem)
    CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
    num_runs=20
    for i in range(0, num_runs):
        x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
        print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
        cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
        cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
        cmaes_fitting.set_parallel(not cyt.simulation_options["test"])
        found_parameters, found_value=cmaes_fitting.run()
        print(found_parameters)
        cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
        print(list(cmaes_results))
        cmaes_time=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
        plt.plot(cmaes_time)
        plt.plot(fourier_arg)
        plt.show()
        #plt.subplot(1,2,1)
        #plt.plot(voltage_results, cmaes_time)
        #plt.plot(voltage_results, true_data, alpha=0.5)
        #plt.show()
        #h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, data_time_series=cmaes_time,  xaxis=voltage_results)
        #plt.subplot(1,2,2)
        #plt.plot(time_results, cyt.define_voltages()[cyt.time_idx:])
        #plt.plot(time_results, voltage_results)
