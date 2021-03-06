import numpy as np
import matplotlib.pyplot as plt
plot=True
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/Ramped"
files=os.listdir(data_loc)
scan="_1_"
freq="_9Hz"
dec_amount=1
for file in files:
    if scan in file and freq in file:

        if "current" in file:
            current_data=np.loadtxt(data_loc+"/"+file)
        elif "voltage" in file:
            voltage_data=np.loadtxt(data_loc+"/"+file)
try:
    current_results1=current_data[0::dec_amount,1]
    time_results1=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results1=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")
values=values=[[-0.2120092471414607, 0.000005478233474769771, 116.21497065365581, 431.92918718571053, 0.00044100528598203375, 0.14367030986553458, 0.005163770653874243, 9.999387751676114e-11, 8.881077023434541,0, 0.5997147084901965],\
                [-0.2485010348585462, 0.06702142523298088, 500.88915054231003, 876.7895281614545, 0.000108185721998072834*0, 0.14858346584854376*0, 0.005464884000805036*0, 1.1040282150229229e-10, 8.88129543205022, 0, 0.5990602813196874],
                [-0.2661180439669948, 0.06702142523298088, 81251.01912987458, 876.5733127765511, 0.000108185721998072834*0, 0.14858346584854376*0, 0.005464884000805036*0,1.1040282150229229e-10,  8.88129543205022, 0,0.41024786661899215]

]

harm_range=list(range(3,9,1))
fig, ax=plt.subplots(len(harm_range), 1)

for q in range(0, len(values)):
    param_list={
        "E_0":0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
        'omega':8.881077023434541,#8.88480830076,  #    (frequency Hz)
        "v":    22.35174e-3,
        'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :0.1,
        "time_end": None,
        'num_peaks': 30,
        "noise_00":None,
        "noise_01":None,
        "noise_10":None,
        "noise_11":None,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=1/(param_list["omega"])
    simulation_options={
        "no_transient":time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":16,
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
        "harmonic_range":harm_range,
        "experiment_time": time_results1,
        "experiment_current": current_results1,
        "experiment_voltage":voltage_results1,
        "bounds_val":20000,
    }
    param_bounds={
        'E_0':[param_list['E_start'],param_list['E_reverse']],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1000],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.05,0.15],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[param_list['E_start'],param_list['E_reverse']],
        "E0_std": [0.01,  0.2],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        "k0_range":[1e2, 1e4],
        'phase' : [0, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    voltage_results=cyt.other_values["experiment_voltage"]
    voltages=cyt.define_voltages(transient=True)

    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
    data_harmonics=harms.generate_harmonics(time_results,(current_results))
    cyt.simulation_options["dispersion_bins"]=[25]
    cyt.simulation_options["GH_quadrature"]=True
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])

    vals=[-0.2085010348585462, 0.05769719441256009, 300.88915054231003, 621.7895281614545, 0.00038185721998072834*0, 0.14858346584854376, 0.005464884000805036, 1.995887932170653e-11, 8.88, 3.481574064188825, 0.5990602813196874]
    v=[[-0.2458983674007203, 0.014480430628464849, 999.9998329296884, 592.2880684094663, 5.3494907175470974e-05, -0.0499999997770132, 0.004212513848717608, 2.0694222664584297e-11, 8.884843597198367, 6.283185294727796, 2.3248235120878205, 0.5999999797204587]]
    #v=[[-0.3703299403301354, 0.027303197192336336, 46.6953215655466, 2.296457060438854e-06, 1.2909122014549365e-05, -0.012191460118923324, 0.009999999983033919, 5.836886994157381e-10, 8.884879455769067, 5.734336325697864, 1.9686527574398272, 0.5705934663538055]]
    v=[[-0.2516469751843067, 0.010000000010342883, 39.38483515098593, 156.92469814473787, 0.00018838294267892265, -0.0499999716987813, -0.0003664820084007588, 2.0050279612708814e-11, 8.88488526687, 2.945862447788219, 0.4772330344837204, 0.538083321236119]
    v=[[-0.3046986975646577, 0.1692280773575256, 59.16536815195476, 350.99885744498613, 0.00015306443130899493, 0.020370243736069882, 0.0013754384753472126, 1.8517396403606723e-10, 8.940944195940126, 5.055745799391211, 5.414311229328655, 0.5999999984388575]]
    syn_time=cyt.test_vals(v[0], "timeseries")
    syn_harmonics=harms.generate_harmonics(time_results,(syn_time))
    cyt.simulation_options["method"]="dcv"
    dcv_volt=cyt.e_nondim(cyt.define_voltages())
    cyt.simulation_options["method"]="ramped"


    quake=1000
    for i in range(0, len(data_harmonics)):
        ax[i].plot(time_results[quake:-quake], abs(data_harmonics[i,:][quake:-quake]),  label="Data")
        ax[i].plot(time_results[quake:-quake], abs(syn_harmonics[i,:][quake:-quake]), alpha=0.7, label="Simulated")
        ax2=ax[i].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
        if i==0:
            ax[i].legend(loc="upper right")
        if i==len(data_harmonics)-1:
            ax[i].set_xlabel("Nondim time")
        if i==3:
            ax[i].set_ylabel("Nondim current")

plt.show()

cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])

v=[-0.2485010348585462, 500.88915054231003, 621.7895281614545,1e-4, 0.014858346584854376, 0.005464884000805036, 5.995887932170653e-11, 8.88129543205022, 0, 0.5990602813196874]

syn_time=cyt.test_vals(v, "timeseries")
syn_time=cyt.add_noise(syn_time, 0.005*max(syn_time))
true_data=current_results
fourier_arg=cyt.top_hat_filter(true_data)
cyt.secret_data_fourier=fourier_arg
cyt.secret_data_time_series=true_data

#cyt.def_optim_list(["Ru","Cdl","CdlE1", "CdlE2","omega","cap_phase"])
#cyt.dim_dict["gamma"]=0
cdl_vals=[0.00010542644711031235, -0.0027839326890568145, -0.00010519225659381785, 8.884813331158371, 6.185678568714874, 2.5648205339871604]
cdl_params=["Cdl","CdlE1", "CdlE2","omega","cap_phase", "phase"]
#for z in range(0, len(cdl_params)):
#	cyt.param_bounds[cdl_params[z]]=[0.75*cdl_vals[z], 1.25*cdl_vals[z]]
#cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","gamma","omega", "phase", "alpha"])
cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
score = pints.SumOfSquaresError(cmaes_problem)




CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=False
#cyt.simulation_options["test"]=True
num_runs=10
param_mat=np.zeros((num_runs,len(cyt.optim_list)))
score_vec=np.ones(num_runs)*1e6
v=[-0.24892008541902, 1.4420634387553057e-05, 103.80638917478434, 251.5314717200847, 0.0012702287310144848, -0.029785256916493207, -0.0006004534921610592, 7.13462751341829e-11, 8.884862786981506, 3.317186895889136, 5.272982852527357, 0.4000001145710844]
v=[-0.2458983674007203, 0.014480430628464849, 999.9998329296884, 592.2880684094663, 5.3494907175470974e-05, -0.0499999997770132, 0.004212513848717608, 2.0694222664584297e-11, 8.884843597198367, 6.283185294727796, 2.3248235120878205, 0.5999999797204587]
#v=[-0.3703299403301354, 0.027303197192336336, 46.6953215655466, 2.296457060438854e-06, 1.2909122014549365e-05, -0.012191460118923324, 0.009999999983033919, 5.836886994157381e-10, 8.884879455769067, 5.734336325697864, 1.9686527574398272, 0.5705934663538055]
v=[-0.25042015555905595, 0.02035228986359169, 6.32515685058267, 143.79266867485552, 3.782234923470199e-05, 0.14999999997656382, 0.005425044700463438, 5.839539506584178e-11, 8.88518640798856, 6.283185307178574, 0.9656195843883533, 0.5015226281103404]
v=[-0.2516469751843067, 0.010000000010342883, 39.38483515098593, 156.92469814473787, 0.00018838294267892265, -0.0499999716987813, -0.0003664820084007588, 2.0050279612708814e-11, 8.88488526687, 2.945862447788219, 0.4772330344837204, 0.538083321236119]
v=[-0.3046986975646577, 0.1692280773575256, 59.16536815195476, 350.99885744498613, 0.00015306443130899493, 0.020370243736069882, 0.0013754384753472126, 1.8517396403606723e-10, 8.940944195940126, 5.055745799391211, 5.414311229328655, 0.5999999984388575]]
syn_fourier=cyt.test_vals(v, "fourier")
print("+"*40,np.sum(abs(np.subtract(fourier_arg, syn_fourier))))
plt.plot(syn_fourier)
plt.plot(fourier_arg)
plt.show()

for i in range(0, num_runs):
    x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
    x0=cyt.change_norm_group(v, "norm")
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()
    print(found_parameters)
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    print(list(cmaes_results))
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    #plt.subplot(1,2,1)
    plt.plot(time_results, cmaes_time)
    plt.plot(time_results, true_data, alpha=0.5)
    #plt.subplot(1,2,2)
    #plt.plot(time_results, cyt.define_voltages()[cyt.time_idx:])
    #plt.plot(time_results, voltage_results)
    plt.show()
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="fourier", test=True)
    #cmaes_fourier=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
    #param_mat[i,:]=cmaes_results
    #score_vec[i]=found_value
    #print("Finish?")
    #i, o, e = select.select( [sys.stdin], [], [], 5)
    #if len(i) != 0:
    #    break
