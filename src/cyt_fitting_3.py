import numpy as np
import matplotlib.pyplot as plt
import sys
plot=True
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from single_e_class_unified import single_electron
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="1"
freq="_9_"
dec_amount=8
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
values=[[-0.21990071721861593, 44.01096292456005, 119.48426204727059, 0.0002451477915335569, 0.13729895076202486, 0.007928163599045154, 9.239820340868298e-11, 8.941077048962455, 4.7123345102082865, 3.813219214475715, 0.5740881278500752],\
        [-0.2515873899481767, 380.850371209114, 190.56427390559483, 0.0009996714551589862, 0.05283450443740578, 2.4559146124050746e-05, 9.979263862879792e-11, 8.94052872191747, 6.228679431228626, 3.5117346885780156, 0.4454217075669103],\
        [-0.3737835519192576, 110.89381450977515, 192.9584698382905, 0.000999061235805586, 0.05059781896924842, 6.770899887461636e-05, 9.971258525476312e-11, 8.940476323669214, 1.5710812221704191, 4.285970106715876, 0.4074546974273283],\
        [-0.3035803306837689, 133.5644652314799, 70.93960743168643, 0.0008576399748363899, 0.14120750241196917, 0.00908999058237603, 8.384891653172269e-11, 8.940392636062402, 6.028883772290079, 5.154957131517447, 0.5699823467206875]]
count=-1
plt.plot(voltage_results1)
plt.show()
for harmonic in range(3, 7):
    count+=1
    print("~"*30, "harmonic=", harmonic,"-", harmonic+4 ,"~"*30)
    param_list={
        "E_0":0.2,
        'E_start':  min(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]), #(starting dc voltage - V)
        'E_reverse':max(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]),
        'omega':8.94, #8.88480830076,  #    (frequency Hz)
        "original_omega":8.94,
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["omega"])
    simulation_options={
        "no_transient":time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":16,
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
        "harmonic_range":list(range(3,9,1)),
        "experiment_time": time_results1,
        "experiment_current": current_results1,
        "experiment_voltage":voltage_results1,
        "bounds_val":20,
    }
    param_bounds={
        'E_0':[param_list['E_start'],param_list['E_reverse']],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,1e-2], #(capacitance parameters)
        'CdlE1': [-0.05,0.15],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
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
        "noise":[0, 100],
        "noise_00":[-1e10, 1e10],
        "noise_01":[-1e10, 1e10],
        "noise_10":[-1e10, 1e10],
        "noise_11":[-1e10, 1e10],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    print(cyt.nd_param.c_I0)
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    voltage_results=cyt.other_values["experiment_voltage"]
    #plt.plot(time_results, current_results)
    cyt.dim_dict["noise"]=0
    cyt.dim_dict["phase"]=3*math.pi/2
    print(len(current_results))
    #cyt.def_optim_list(["E_0","k0_shape", "k0_scale","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    cyt.simulation_options["dispersion_bins"]=[5]
    cyt.simulation_options["GH_quadrature"]=True
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])


    reduced_list=["E_0","k_0","Ru","gamma","omega","cap_pha100se","phase", "alpha"]
    vals=[-0.2499999999795545, 0.3997722688641264, 9999.999440245034, 114.8382192838832, 0.00013334402533556777, 0.1077568991954742, 0.0028577794838729777, 1.192053173965407e-10, 8.938971787355355, 1.570796326795171, 4.738470849368598, 0.5999999996633747]
    vals=[-0.3152122174013743, 0.28357512333528784, 464.8367255826205, 20.645888957700034, 2.6724941197529202e-05, -0.04999999999850869, -0.005846400851874563, 1.3189947035043904e-10, 8.937882789293578, 1.5707963267988925, 4.743664540414751, 0.40000000000583846]
    vals=[-0.003808583576515645, 0.4999999999705347, 3784.6157375283487, 631.453871630171, 0.00010640637373095699, 0.005028640673736776, 0.0008598525708093628, 1.0033710827129637e-10, 8.940630374330569, 4.341216122575276, 5.184808475268891, 0.5999999760343172]



    ifft_vals=[-0.3152122174013743, 0.28357512333528784, 464.8367255826205, 20.645888957700034, 2.6724941197529202e-05, -0.04999999999850869, -0.005846400851874563, 1.3189947035043904e-10, 8.937882789293578, 1.5707963267988925, 4.743664540414751, 0.40000000000583846]
    true_signal=cyt.test_vals(vals, "timeseries")

    f_true=cyt.test_vals(vals, "fourier")
    #test_data=cyt.add_noise(true_signal, 0.005*max(true_signal))
    true_data=true_signal

    #true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)
    voltages=cyt.define_voltages(transient=True)

    #true_data=np.real(np.fft.ifft(fourier_arg))#[:len(fourier_arg)//2]
    cyt.secret_data_time_series=true_data
    plt.plot(true_data)
    plt.show()
    test_fourier=cyt.test_vals(vals, "fourier", test=False)
    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
    data_harmonics=harms.generate_harmonics(time_results,(current_results))
    syn_harmonics=harms.generate_harmonics(time_results, true_data)
    voltages=cyt.define_voltages(transient=True)

    """
    fig, ax=plt.subplots(len(data_harmonics), 1)
    for i in range(0, len(data_harmonics)):

        ax[i].plot(time_results, (syn_harmonics[i,:]))
        ax[i].plot(time_results, (data_harmonics[i,:]), alpha=0.7)
        ax[i].plot(time_results, np.subtract(data_harmonics[i,:],syn_harmonics[i,:]), alpha=0.7)
        ax2=ax[i].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(other_values["harmonic_range"][i])
    plt.show()"""

    #cyt.time_idx=tuple(np.where((voltage_results<-2) & (voltage_results>-12.5)))
    cdl_vals=[0.00010640637373095699, 0.005028640673736776, 0.0008598525708093628, 8.940630374330569, 4.341216122575276]
    cdl_params=["Cdl","CdlE1", "CdlE2", "omega", "cap_phase"]
    for z in range(0, len(cdl_vals)):
        cyt.dim_dict[cdl_params[z]]=cdl_vals[z]
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru", "gamma", "alpha", "phase"])
    num_runs=8
    param_mat=np.zeros((num_runs,len(cyt.optim_list)))
    score_vec=np.ones(num_runs)*1e6
    values=[[-0.3679764998754323, 0.1422464146023959, 44.75877502784956, 468.18385860454373, 4.443350510344139e-10, 0.5215371585216145, 5.348741619902399],
            [-0.36797649999999954, 0.13517071991295526, 58.50763441820108, 439.66475714149743, 4.038205467075284e-10, 0.5999999999999999, 5.29245733802453],
            [-0.3679764999999972, 0.2664620206528828, 170.1500626179889, 520.9423347508204, 6.794048135978194e-11, 0.5303371419890249, 5.1923202141405715],
            [-0.34417347144925503, 0.49704242685318817, 2934.2937667223364, 997.0654667064832, 4.332998056574655e-10, 0.5252375323827561, 6.283179321960487]
    ]
    values=[[-0.2594100028589219, 0.06416493582288162, 9151.980166228543, 275.62973801523026, 4.6107253848269156e-11, 0.400000000000158, 4.886504948153826]
    ]

    for h in range(0, len(values)):
        test=cyt.test_vals(values[h], "fourier")
        true_signal=cyt.test_vals(values[h], "timeseries")
        data_harmonics=harms.generate_harmonics(time_results,(current_results))
        syn_harmonics=harms.generate_harmonics(time_results, (true_signal))
        for i in range(0, len(data_harmonics)):
            plt.subplot(len(data_harmonics),1, i+1)
            plt.plot(voltage_results, (data_harmonics[i,:]),  label="Data")
            plt.plot(voltage_results, (syn_harmonics[i,:]), label="Simulation", alpha=0.7)
            #ax[i].plot(voltage_results, np.subtract(data_harmonics[i,:],syn_harmonics[i,:]), alpha=0.7, label="Residual")
            if i==0:
                plt.legend(loc="upper right")
            if i==len(data_harmonics)-1:
                plt.xlabel("Nondim voltage")
            else:
                plt.xticks([])
            if i==3:
                plt.ylabel("Nondim current")
    plt.show()
    values=[[-0.2594100028589219, 0.06416493582288162, 9151.980166228543, 275.62973801523026, 4.6107253848269156e-11*0, 0.400000000000158, 4.886504948153826]
    ]
    #values=[[-0.2594100028589219, 9151.980166228543, 275.62973801523026, 4.6107253848269156e-11, 0.400000000000158, 4.886504948153826]
    #]
    for z in range(0, len(cdl_vals)):
        cyt.dim_dict[cdl_params[z]]=cdl_vals[z]
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru", "gamma","alpha", "phase"])
    true_signal=cyt.test_vals(values[0], "timeseries")
    plt.plot(voltage_results, true_signal)
    plt.plot(voltage_results,current_results, alpha=0.7)
    plt.show()
    #, "noise_00", "noise_01", "noise_10", "noise_11"
    true_data=true_signal
    fourier_arg=cyt.top_hat_filter(true_data)
    plt.plot(voltage_results, true_data)
    plt.show()

    #cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru", "gamma", "alpha", "phase"])
    #cyt.dim_dict["Cdl"]=0
    if simulation_options["likelihood"]=="timeseries":
        cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
    elif simulation_options["likelihood"]=="fourier":
        dummy_times=np.linspace(0, 1, len(fourier_arg))
        cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
    score = pints.SumOfSquaresError(cmaes_problem)#[4.56725844e-01, 4.44532637e-05, 2.98665132e-01, 2.96752050e-01, 3.03459391e-01]#




    CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
    cyt.simulation_options["label"]="cmaes"
    cyt.simulation_options["test"]=False
    #cyt.simulation_options["test"]=True


    for i in range(0, num_runs):
        x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
        #x0=cyt.change_norm_group(ifft_vals, "norm")
        print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
        cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
        cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
        cmaes_fitting.set_parallel(False)
        found_parameters, found_value=cmaes_fitting.run()
        print(found_parameters)
        cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
        print(list(cmaes_results))
        cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
        #plt.subplot(1,2,1)
        plt.plot(voltage_results, cmaes_time)
        plt.plot(voltage_results, true_data, alpha=0.5)
        #plt.subplot(1,2,2)
        #plt.plot(time_results, cyt.define_voltages()[cyt.time_idx:])
        #plt.plot(time_results, voltage_results)
        plt.show()
        #cmaes_fourier=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
        #param_mat[i,:]=cmaes_results
        #score_vec[i]=found_value
        #print("Finish?")
        #i, o, e = select.select( [sys.stdin], [], [], 5)
        #if len(i) != 0:
        #    break
