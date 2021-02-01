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
inferred_params=[[-0.04955459994332252, 0.024409418663214232, 9275.479999845043, 99999.99712409446, 7.607122198928626e-16, 4.447804239083664e-11, 0.4651366220351519, 0.2624943521183878],
[-0.051771148281119825, 0.029350351522082246, 6669.073576729849, 99999.93040678784, 3.609098295197487e-15, 1.698026851872925e-11, 0.5930110730782626, 0.21646041040994204],
[-0.05215086968395874, 0.01842669773591587, 5.000000443751222, 99999.97439365527, 3.469009482232843e-14, 1.0179800910532819e-11, 0.599995808518657, 0.20877245570182038],]
for i in range(1, 4):
    file_name="/dcV_Cjx-183D_WT_pH_7_{0}_3".format(i)
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
        'sampling_freq' : (1.0/200),
        'phase' :0,
        "time_end": -1,
        'num_peaks': 30,
        'Q':1,
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
        'E_0':[-0.06,-0.04],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e5],  #     (uncompensated resistance ohms)
        "Q":[0,1],
        'Cdl': [0,1e-3], #(capacitance parameters)
        'CdlE1': [-0.1,0.1],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'Cdlinv': [1e-5, 1e-3], #(capacitance parameters)
        'CdlE1inv': [-0.1,0.1],#0.000653657774506,
        'CdlE2inv': [-0.1,0.1],#0.000245772700637,
        'CdlE3inv': [-0.1,0.1],#1.10053945995e-06,
        'gamma': [0.8*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [5, 1e4], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[param_list["E_start"]*1.1,0.9*param_list["E_reverse"]],
        "E0_std": [0.01,  0.08],
        "alpha_mean":[0.4, 0.5],
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
    #plt.plot(volts)
    #plt.plot(voltage_results)
    #plt.show()
    #plt.plot(voltage_results, current_results)
    #plt.show()
    PSV_optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"]
    true_data=current_results

    error=0.01*max(true_data)
    cyt.param_bounds["Q"][1]=5*error
    cyt.def_optim_list(["E0_mean", "E0_std", "k_0", "Ru", "Cdl", "gamma", "alpha"])
    cyt.simulation_options["GH_quadrature"]=True
    cyt.simulation_options["numerical_method"]="Kalman_simulate"

    cyt.simulation_options["label"]="MCMC"
    cyt.simulation_options["test"]=False
    cyt.secret_data_time_series=current_results
    param_vals=inferred_params[i-1][:-1]
    cyt.dim_dict["Q"]=inferred_params[i-1][-1]
    cmaes_time=cyt.test_vals(param_vals, "timeseries")
    error=np.std(np.abs(np.subtract(cmaes_time, current_results)))
    mcmc_problem=pints.SingleOutputProblem(cyt, cyt.time_vec, current_results)


    error=np.std(abs(np.subtract(cmaes_time, current_results)))
    #plt.plot(voltage_results, cmaes_time)
    #plt.plot(voltage_results, current_results, alpha=0.5)
    #plt.show()


    updated_lb=np.append([cyt.param_bounds[key][0] for key in cyt.optim_list], 0.01*error)
    updated_ub=np.append([cyt.param_bounds[key][1] for key in cyt.optim_list], 10*error)
    #updated_lb=np.append([x*0.65 for x in param_vals],0.01*error)
    #updated_ub=np.append([x*1.35 for x in param_vals], 10*error)
    updated_b=[updated_lb, updated_ub]
    updated_b=np.sort(updated_b, axis=0)
    #error=1
    log_liklihood=pints.GaussianLogLikelihood(mcmc_problem)
    print(vars(log_liklihood))

    log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
    log_posterior=pints.LogPosterior(log_liklihood, log_prior)
    mcmc_parameters=param_vals
    mcmc_parameters=np.append(mcmc_parameters, error)
    xs=[mcmc_parameters,
        mcmc_parameters,
        mcmc_parameters
        ]
    cyt.simulation_options["label"]="MCMC"
    cyt.simulation_options["test"]=False
    num_runs=5
    scores=np.ones(num_runs)*10
    skews=np.ones(num_runs)*10
    for j in range(0, num_runs):
        current_min=min(scores)
        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)
        #alpha_index=cyt.optim_list.index("alpha")
        alpha_chain=[]
        mcmc.set_parallel(True)
        mcmc.set_max_iterations(10000)
        chains=mcmc.run()
        rhat_mean=np.mean(pints.rhat_all_params(chains[:, 5000:, :]))
        #for q in range(0, 2):
        #    alpha_chain=np.append(alpha_chain, chains[q, 30000:, alpha_index])
        #alpha_skew=stat.skew(alpha_chain)
        """
        if alpha_skew<-0.05:
            Electrode_save="Yellow"
            run2="MCMC_runs/omega_nondim/high_skew"
            save_file=file+"_MCMC_run9"
            filepath=("/").join([dir_path, "Inferred_params", Electrode_save, run2])
            if abs(alpha_skew)<min([abs(x) for x in skews]) and rhat_mean<1.1:
                f=open(filepath+"/"+save_file, "wb")
                np.save(f, chains)
                f.close()
        """


        #k_rhat=pints.rhat_all_params(chains[:, 20000:, :])[2]
        #pints.plot.trace(chains)
        #plt.show()
        filepath=os.getcwd()+"/MCMC/DCV"
        save_file=file_name+"_kalman_MCMC_2"
        if rhat_mean<1.08:
            f=open(filepath+"/"+save_file, "wb")
            np.save(f, chains)
            f.close()
            break
        elif rhat_mean<min(scores):
            f=open(filepath+"/"+save_file, "wb")
            np.save(f, chains)
            f.close()
        scores[j]=rhat_mean


#[-0.04950345403273563, 0.415104966434246, 5.000000006505078, 499.999929294756, 1.4495126487366279e-15, 4.7671138947314476e-11, 0.5163700064087133, 0.2624943522428564]
