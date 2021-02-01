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
from scipy.optimize import curve_fit
import cma

directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Nick_1_2_2021/"
experimental_dict={}

useful_params=dict(zip(["max", "min", "Amp[0]", "Freq[0]"], ["E_reverse", "E_start", "d_E", "original_omega"]))

dec_list=[8, 4]+([2]*9)
"""for line in param_file:
    split_line=line.split()
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])"""
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
frequencies=[9*x for x in range(1, 12)]
true_freqs=[9.02, 18.18, 27.12, 36.36, 45.30, 54.24, 63.18, 72.12, 81.06, 90.00, 99.54]
#TRUEFTACV=8.85

class oscillator:
    def __init__(self, times, data):
        self.data=data
        self.times=times
    def damped_osc(self, A,lamb,frequency,phase):
        #print(A, lamb, frequency, phase)
        x=self.times
        cosine=np.cos(np.add(np.multiply(x, frequency), phase))
        exponent=np.exp(np.multiply(-lamb, x))
        predicition=A*exponent*cosine
        return RMSE(predicition, self.data)
    def RMSE(self, y, y_data):
        return np.mean(np.sqrt(np.square(np.subtract(y, y_data))))
#fig, axes=plt.subplots(3,4)
c_counter=0
for i in np.flip(range(0, len(frequencies))):

    for j in range(1, 2):
        file_name="Cyt_{0}_hz_{1}_cv_".format(frequencies[i], j)

        dec_amount=dec_list[i]
        current_data_file=np.loadtxt(data_loc+"/text_files/"+file_name+"current")
        voltage_data_file=np.loadtxt(data_loc+"/text_files/"+file_name+"voltage")
        param_file_name='{0} Hz diff range PSV method'.format(frequencies[i])
        param_file=open(data_loc+"/input_params/"+param_file_name, "r")
        for line in param_file:
            if "Freq[0]" in line:
                start_idx=line.index("]")
                input_freq=float(line[start_idx+1:])
                break
        volt_data=voltage_data_file[0::dec_amount, 1]

        param_list={
            "E_0":-0.2,
            'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
            'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
            'omega':input_freq, #8.88480830076,  #    (frequency Hz)
            "original_omega":input_freq,
            'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
            'phase' :3*math.pi/2,
            "time_end": -1,
            'num_peaks': 30,
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
            "harmonic_range":list(range(4,100,1)),
            "experiment_time": current_data_file[0::dec_amount, 0],
            "experiment_current": current_data_file[0::dec_amount, 1],
            "experiment_voltage":volt_data,
            "bounds_val":20000,
        }
        param_bounds={
            'E_0':[-0.1, 0.1],
            'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
            'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
            'Cdl': [0,1e-5], #(capacitance parameters)
            'CdlE1': [-0.1,0.1],#0.000653657774506,
            'CdlE2': [-0.1,0.1],#0.000245772700637,
            'CdlE3': [-0.05,0.05],#1.10053945995e-06,
            'gamma': [0.1*param_list["original_gamma"],2*param_list["original_gamma"]],
            'k_0': [50, 1e4], #(reaction rate s-1)
            'alpha': [0.498, 0.502],
            "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/3],
            "E0_mean":[-0.1, 0.1],
            "E0_std": [1e-4,  0.1],
            "E0_skew": [-10, 10],
            "alpha_mean":[0.4, 0.65],
            "alpha_std":[1e-3, 0.3],
            "k0_shape":[0,1],
            "k0_scale":[0,1e4],
            'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/3],
        }
        cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        del current_data_file
        del voltage_data_file
        cyt.define_boundaries(param_bounds)
        time_results=cyt.other_values["experiment_time"]
        current_results=cyt.other_values["experiment_current"]
        voltage_results=cyt.e_nondim(cyt.other_values["experiment_voltage"])
        start_harm=1
        harms=list(range(start_harm,11))
        h_class=harmonics(harms , 1, 0.05)
        h, amps=h_class.generate_harmonics(time_results, cyt.i_nondim(current_results), return_amps=True)
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
        cyt.dim_dict["alpha"]=0.5
        true_data=current_results
        fourier_arg=cyt.top_hat_filter(true_data)
        if simulation_options["likelihood"]=="timeseries":
            cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
        elif simulation_options["likelihood"]=="fourier":
            dummy_times=np.linspace(0, 1, len(fourier_arg))
            cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
        cyt.simulation_options["label"]="cmaes"
        cyt.simulation_options["test"]=False
        score = pints.SumOfSquaresError(cmaes_problem)
        CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
        num_runs=10
        for i in range(0, num_runs):
            x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
            print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
            cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
            cmaes_fitting.set_parallel(True)
            found_parameters, found_value=cmaes_fitting.run()
            cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
            cmaes_time=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
            print(list(cmaes_results))
