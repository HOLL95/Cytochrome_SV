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
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Nick_19_1_21"
files=os.listdir(data_loc)
experimental_dict={}

useful_params=dict(zip(["max", "min", "Amp[0]", "Freq[0]"], ["E_reverse", "E_start", "d_E", "original_omega"]))
#figure=multiplot(1,2, **{"harmonic_position":[1], "num_harmonics":len(harms), "orientation":"portrait",  "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":1, "plot_height":1, "harmonic_spacing":1})

#plt.show()
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
scipy_fits=[[ 2.10478843e-01,  5.19380498e-01,  1.37553765e+00,  3.22296415e-01],
 [ 7.71963752e-01,  7.64167986e-01,  1.38944381e+00,  2.33822854e-01,],
 [ 9.25838344e+00,  1.34488491e+00,  1.24634837e+00,  6.72720572e-01,],
 [ 1.43753479e+00,  1.19661466e+00, -4.75286072e-05,  6.39750773e+00,],
 [ 7.82488808e-03,  2.31955215e-01, -1.58859121e+00,  1.10915529e+01,],
 [ 4.71441621e+00,  1.56365986e+00, -7.53438642e-03,  4.13235661e+00,],
 [ 1.19741262e-01,  5.79721068e-01, -4.16848632e-01,  3.91952412e+00,],
 [ 6.15329950e-02,  5.06062773e-01, -6.70200223e-01,  5.60299509e+00,],
 [ 6.31550540e-02,  5.41423017e-01,  7.79505878e-01, -1.40758029e-01,],
 [ 5.91568137e-02,  5.62671594e-01,  7.94444526e-01, -3.32788135e-01,],
 [ 7.96153198e-02,  6.53561197e-01,  8.50795833e-01, -8.25921962e-01,]]
class oscillator:
    def __init__(self, times, data, bounds):
        self.data=data
        self.times=times
        self.bounds=bounds
    def normalise(self, norm, boundaries):
        return  (norm-boundaries[0])/(boundaries[1]-boundaries[0])
    def un_normalise(self, norm, boundaries):
        return (norm*(boundaries[1]-boundaries[0]))+boundaries[0]
    def damped_osc(self,x, A,lamb,frequency,phase):
        #print(A, lamb, frequency, phase)
        cosine=np.cos(np.add(np.multiply(x, frequency), phase))
        exponent=np.exp(np.multiply(-lamb, x))
        prediction=A*exponent*cosine
        return prediction

    def CMAES_simulate(self, parameters):
        sim_params=np.zeros(len(parameters))
        #print(list(parameters))
        for i in range(0, len(parameters)):
            #print(parameters[i], self.bounds[i])
            sim_params[i]=self.un_normalise(parameters[i], self.bounds[i])
        #print(list(sim_params))
        sim=self.damped_osc(self.times, *sim_params)

        #plt.plot(sim)
        #plt.plot(self.data)
        #plt.show()
        return self.RMSE(sim, self.data)
    def RMSE(self, y, y_data):
        return np.mean(np.sqrt(np.square(np.subtract(y, y_data))))
fig, axes=plt.subplots(3,4)
c_counter=0
for i in range(0, len(frequencies)):

    for j in range(1, 2):
        file_name="Cyt_{0}Hz_{1}_cv_".format(frequencies[i], j)
        dec_amount=dec_list[i]
        current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
        voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
        volt_data=voltage_data_file[0::dec_amount, 1]

        param_list={
            "E_0":-0.2,
            'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
            'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
            'omega':true_freqs[i], #8.88480830076,  #    (frequency Hz)
            "original_omega":true_freqs[i],
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
        voltage_results=cyt.other_values["experiment_voltage"]
        start_harm=3
        harms=list(range(start_harm,20))
        h_class=harmonics(harms , 1, 0.05)
        h, amps=h_class.generate_harmonics(time_results, cyt.i_nondim(current_results), return_amps=True)
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        boundaries=[[0, 50], [0, 5], [0, 2], [0, 2*math.pi]]
        cmaes_osc=oscillator(h_class.harmonics, amps, boundaries)
        initial_position=[cmaes_osc.normalise(scipy_fits[i][x], boundaries[x]) for x in range(0, len(boundaries))]
        es = cma.CMAEvolutionStrategy(abs(np.random.rand(len(boundaries))), 0.5, {"bounds":[[0 for x in range(0, len(boundaries))], [1 for x in range(0, len(boundaries))]], "tolfun":1e-6, "tolx":1e-6})
        es.optimize(cmaes_osc.CMAES_simulate)
        result=[cmaes_osc.un_normalise(es.result[0][x], boundaries[x]) for x in range(0, len(boundaries))]
        print(result)
        generate_times=np.linspace(h_class.harmonics[0], h_class.harmonics[-1], 100)
        generated_oscillator=cmaes_osc.damped_osc(generate_times, *result)
        row=i//4
        col=i%4
        axes[row, col].plot(generate_times, generated_oscillator, label=frequencies[i], color=cycle[c_counter%len(cycle)])
        axes[row, col].plot(h_class.harmonics, amps, label=frequencies[i], color=cycle[(c_counter+1)%len(cycle)])
        fourier_idx=np.where((h_class.f>4) & (h_class.f<h_class.harmonics[-1]+1))
        axes[row, col].plot(h_class.f[fourier_idx], np.real(h_class.Y[fourier_idx]), color=cycle[c_counter%len(cycle)])
        c_counter+=1
        #plt.legend()
print(scipy_fits)
plt.legend()
plt.show()
#e
