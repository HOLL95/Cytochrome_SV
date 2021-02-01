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
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/DCV"
files=os.listdir(data_loc)
experimental_dict={}
file_name="/dcV_Cjx-183D_WT_pH_7_1_3"
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
    'gamma': 1e-11,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/800),
    'phase' :0,
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
    "bounds_val":200,
}
param_bounds={
    'E_0':[-0.1,0.0],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e6],  #     (uncompensated resistance ohms)
    'Cdl': [1e-5,1e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'Cdlinv': [1e-5, 1e-3], #(capacitance parameters)
    'CdlE1inv': [-0.1,0.1],#0.000653657774506,
    'CdlE2inv': [-0.1,0.1],#0.000245772700637,
    'CdlE3inv': [-0.1,0.1],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
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
#plt.plot(nondim_v, current_results)
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
cyt.def_optim_list(["k_0", "Ru", "Cdl","CdlE1", "CdlE2","CdlE3"])

volts=cyt.define_voltages()
#plt.plot(noisy_current)
tr=cyt.nd_param.nd_param_dict["E_reverse"]-cyt.nd_param.nd_param_dict["E_start"]
u=np.ones(len(cyt.time_vec))
u[np.where(cyt.time_vec>tr)]*=-1
#plt.plot(u)
#plt.show()


counter=0
for resistances in [10, 100, 1000, 10000]:
    counter+=1
    DCV_inferred=[100, resistances, 1e-3, 1e-2, 1e-3, 1e-5    ]
    current=cyt.test_vals(DCV_inferred, "timeseries")
    error=0.02*max(current)
    noisy_current=cyt.add_noise(current, error )
    cyt.secret_data_time_series=noisy_current
    for z in range(0, 2):
        plt.subplot(2, 4, counter+(4*z))

        #plt.scatter(cyt.time_vec[time_start:time_idx], noisy_current[time_start:time_idx],  label="Data", s=2, color="black")
        for ru_scale in np.flip([error, 1e-2,5e-2,  0.5]):
                #DCV_inferred[1]=true_ru*ru_scale

                q=ru_scale
                pred_cap, kalman_plot= cyt.kalman_pure_capacitance(cyt.secret_data_time_series, q)
                if z==1:
                    kalman_plot=kalman_plot*cyt.kalman_u
                psi_s="$\\Phi_s$= "
                if ru_scale==error:
                    plt.plot(cyt.time_vec, kalman_plot, linestyle="--", label=psi_s+"True error")
                else:
                    plt.plot(cyt.time_vec,kalman_plot,  label=psi_s+"{:.1e}".format(ru_scale))
                plt.ylabel("Nondim current")
                plt.xlabel("Nondim time")
                plt.title("R="+str(resistances))
                #cyt.test_vals(DCV_inferred, "timeseries")
        if z==0:
            plt.plot(cyt.time_vec,noisy_current, alpha=0.7, label="I")
            plt.legend(loc="upper right", frameon=False)
        else:
            plt.plot(cyt.time_vec,pred_cap, alpha=0.7, label="I(t+1)/ $\\frac{dE(t+1)}{dt}-R_u \\frac{I(t+1)-I(t)}{dt}$")
            if resistances ==1000:
                plt.legend(loc="upper left", frameon=False)
            else:
                plt.legend(loc="lower left", frameon=False)
plt.show()
