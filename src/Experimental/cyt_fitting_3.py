import numpy as np
import matplotlib.pyplot as plt
import sys
plot=True
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
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="3"
freq="_9_"
dec_amount=32
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
        "E0_skew":0.2,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :0.1,
        "time_end": -1,
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
        "bounds_val":20000,
    }
    param_bounds={
        'E_0':[-0.3,-0.2],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.01,0.01],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.3,-0.2],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        "k0_range":[1e2, 1e4],
        'phase' : [math.pi, 2*math.pi],
        }
    print(param_list["E_start"],param_list["E_reverse"] )
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    print(cyt.nd_param.c_I0)
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    voltage_results=cyt.other_values["experiment_voltage"]
    start=time.time()
    cyt.test_vals([], "timeseries")
    print("TIME", time.time()-start, len(voltage_results))
    #plt.plot(time_results, current_results)
    cyt.dim_dict["noise"]=0
    cyt.dim_dict["phase"]=3*math.pi/2
    print(len(current_results))
    #cyt.def_optim_list(["E_0","k0_shape", "k0_scale","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    cyt.simulation_options["dispersion_bins"]=[2]
    cyt.simulation_options["GH_quadrature"]=False
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    reduced_list=["E_0","k_0","Ru","gamma","omega","cap_phase","phase", "alpha"]
    vals=[-0.2499999999795545, 0.3997722688641264, 9999.999440245034, 114.8382192838832, 0.00013334402533556777, 0.1077568991954742, 0.0028577794838729777, 1.192053173965407e-10, 8.938971787355355, 1.570796326795171, 4.738470849368598, 0.5999999996633747]
    vals=[-0.3152122174013743, 0.28357512333528784, 464.8367255826205, 20.645888957700034, 2.6724941197529202e-05, -0.04999999999850869, -0.005846400851874563, 1.3189947035043904e-10, 8.937882789293578, 1.5707963267988925, 4.743664540414751, 0.40000000000583846]
    vals=[-0.003808583576515645, 0.4999999999705347, 3784.6157375283487, 631.453871630171, 0.00010640637373095699, 0.005028640673736776, 0.0008598525708093628, 1.0033710827129637e-10, 8.940630374330569, 4.341216122575276, 5.184808475268891, 0.5999999760343172]



    ifft_vals=[-0.3152122174013743, 0.28357512333528784, 464.8367255826205, 20.645888957700034, 2.6724941197529202e-05, -0.04999999999850869, -0.005846400851874563, 1.3189947035043904e-10, 8.937882789293578, 1.5707963267988925, 4.743664540414751, 0.40000000000583846]
    true_signal=cyt.test_vals(vals, "timeseries")

    f_true=cyt.test_vals(vals, "fourier")
    #test_data=cyt.add_noise(true_signal, 0.005*max(true_signal))
    true_data=current_results

    #true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)
    voltages=cyt.define_voltages(transient=True)

    #true_data=np.real(np.fft.ifft(fourier_arg))#[:len(fourier_arg)//2]
    cyt.secret_data_time_series=true_data
    plt.plot(true_data)
    plt.show()
    test_fourier=cyt.test_vals(vals, "fourier", test=False)
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
    values=[[-0.2594100028589219, 0.06416493582288162, 9151.980166228543, 275.62973801523026, 4.6107253848269156e-11, 0.400000000000158, 4.886504948153826, 8.94],
            [-0.25819776867643385, 0.06004627245093131, 83951.6420075274, 220.83537164564285, 4.0464993756449906e-11, 0.4000000006829789, 4.832264870422034, 8.940990936103178]
    ]
    values=[[-0.2359932087530336, 0.010000000000157562, 122.50460094880125, 132.31031250108373, 0.0004086696281652576, -0.007088616668708109, 0.00037653873425454176, 2.9774189204094853e-11, 8.940950662362539, 5.087825757462735, 5.201924195489391, 0.5515478449176727]]
    #values=[[-0.2689898442882075, 0.1464604686665382, 45.27669426676901, 143.7146851754486, 0.0009043681474193589, 0.015041808481931207, 0.0005136942641310593, 5.116896389239807e-10, 8.940952127189066, 5.47854765415177, 5.804140291549392, 0.599999948777021]]
    #values=[[-0.34907222078847766, 0.16728103291804883, 36.20766537971544, 309.6008045093056, 0.0004684603907108233, -0.044714993569560824, -0.0003666268602918951, 1.7016016387164703e-10, 8.940709137470366, 1.570796326795193, 5.108518361948413, 0.5999999999999984]]
    #values=[[-0.3046986975646577, 0.1692280773575256, 59.16536815195476, 350.99885744498613, 0.00015306443130899493, 0.020370243736069882, 0.0013754384753472126, 1.8517396403606723e-10, 8.940944195940126, 5.055745799391211, 5.414311229328655, 0.5999999984388575]]
    values=[[-0.23316743616752747, 3.918646030384116e-05, 125.32922208435807, 9000.19806606006952, 0.0002308962063433851, -0.03321102983786629, -0.00044615696059807815, 2.1474804559923774e-11, 8.940960632790196, 3.6013722357683, 4.9024678072287475, 0.47377445163491094]]
    #cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru", "gamma", "alpha", "phase"])
    #cyt.dim_dict["Cdl"]=0
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl", "CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    cyt.simulation_options["numerical_method"]="Brent minimisation"

    """for h in range(0, len(values)):
        test=cyt.test_vals(values[h], "fourier")
        true_signal=cyt.test_vals(values[h], "timeseries")
        data_harmonics=harms.generate_harmonics(time_results,(current_results))
        syn_harmonics=harms.generate_harmonics(time_results, (true_signal))
        for i in range(0, len(data_harmonics)):
            plt.subplot(len(data_harmonics),1, i+1)
            plt.plot(voltage_results, (data_harmonics[i,:]),  label="Data")
            plt.plot(voltage_results, (syn_harmonics[i,:]), label="Simulation", alpha=0.7)
            if i==3:
                plt.ylabel("Nondim current")

            #ax[i].plot(voltage_results, np.subtract(data_harmonics[i,:],syn_harmonics[i,:]), alpha=0.7, label="Residual")
            if i==0:
                plt.legend(loc="upper right")
            if i==len(data_harmonics)-1:
                plt.xlabel("Nondim voltage")
            else:
                plt.xticks([])
            ax=plt.gca()
            ax1=ax.twinx()
            ax1.set_ylabel(other_values["harmonic_range"][i], rotation=0)
            ax1.set_yticks([])
    plt.show()
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = time_results[-1]
    numpoints = len(current_results)

    w0 = [current_results[0],0, voltage_results[0]]

    # Call the ODE solver.
    wsol = odeint(cyt.current_ode_sys, w0, time_results,
                  atol=abserr, rtol=relerr)

    adaptive_current=wsol[:,0]
    adaptive_potential=wsol[:,2]
    adaptive_theta=wsol[:, 1]#cyt.calc_theta(wsol[:,0])
    true_signal=cyt.test_vals(values[0], "timeseries")
    plt.plot(voltage_results, true_signal)
    plt.plot(voltage_results, adaptive_current, alpha=0.7)
    plt.show()
    true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)
    plt.plot(time_results, true_signal)
    plt.plot(time_results, voltage_results)
    plt.show()
    #cyt.def_optim_list(["Cdl","CdlE1", "CdlE2","omega","cap_phase"])
    #cyt.dim_dict["gamma"]=0"""
    cyt.def_optim_list(["E0_mean", "E0_std","E0_skew","k_0","Ru","Cdl", "CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    #cyt.def_optim_list(["E_0","k_0","Ru","Cdl", "CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    cyt.dim_dict["Cdl"]=0.00010640637373095699
    cyt.dim_dict["CdlE1"]=0.005028640673736776
    cyt.dim_dict["CdlE2"]=0.0008598525708093628

    if simulation_options["likelihood"]=="timeseries":
        cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
    elif simulation_options["likelihood"]=="fourier":
        dummy_times=np.linspace(0, 1, len(fourier_arg))
        cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
    score = pints.SumOfSquaresError(cmaes_problem)#[4.56725844e-01, 4.44532637e-05, 2.98665132e-01, 2.96752050e-01, 3.03459391e-01]#
    vals=[-0.2999999987915772, 75.0304099214542, 1129.641670283357, 9.999999991897702e-11, 8.941022396967469, 1.8700190844073654, 4.7959402030923215, 0.5999999948253767]
    vals=[-0.29999998164350944, 0.010000041471497887, 10.774171620544701, 97.8726770656541, 0.0019999969180225312, 0.009999998431337425, 0.0003819160490711026, 5.675257231479832e-11, 8.939591092801802, 1.5707964495605589, 5.217487625792245, 0.40273190589632335]
    vals=[-0.2228073641220668, 0.00010603430631268799, 449.1343851598667, 376.3477261551556, 0.0005920303558656501, -0.004353832825229434, 0.001689359566633972, 9.999919753517283e-11, 8.941107666337048, 5.889206808272153, 5.7192890464726895, 0.5999992778697664]
    vals=[-0.2579076671924837, 0.00015728683492038188, 52.68097510680315, 68.50707882828115, 0.000733377715139665, 0.016236691146013005, -7.820740568357382e-05, 2.536139328581548e-11, 8.940938684529883, 1.5709392594136768, 4.915384770995644, 0.5382906064246701]
    vals=[-0.25787767177157567, 52.76186403679123, 68.39497579798049, 0.00010640637373095699, 0.005028640673736776, 0.0008598525708093628, 2.5340812914467008e-11, 8.940938900479116, 1.570839499336432, 4.915380377308822, 0.5385210166244453]
    vals=[-0.23439783739718922, 42.00841759206751, 267.99933752556836, 0.00011704701110398429, 0.004525778371071013, 0.000945837827855062, 3.270236305815754e-11, 8.941036628944167, 2.7688108758782226, 5.038258046998925, 0.5868493852130298]
    vals=[-0.23823115534306044, 0.010853855815583322, 0,54.83121527567214, 247.1060593274587, 0.0001170470107128401, 0.0045257789042600595, 0.0009458378264908692, 3.0265369303073404e-11, 8.940744052716612, 2.74961385519821, 4.984323211104551, 0.5786042890230194]
    vals=[-0.23822574830472312, 0.01082804568798319,0, 54.80815485794722, 247.10661114567895, 0.00011704701099294868, 0.004525776663764095, 0.0009458378023780387, 3.0263501981253515e-11, 8.940742898846258, 2.7496174184849864, 4.984421155873804, 0.5786256352781749]
    vals=[-0.2355296901348899, 0.00020486942554457825, -1.3936849491275964, 131.1311416295316, 450.4761351603096, 0.00011470185276081948, 0.0025143210192612882, 0.001289738847946819, 2.922684768919658e-11, 8.941016507562189, 5.046194168331546, 5.1882186609272765, 0.5527421473090354]
    test_time=cyt.test_vals(vals, "timeseries")
    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)
    data_harmonics=harms.generate_harmonics(time_results,(current_results))
    syn_harmonics=harms.generate_harmonics(time_results, test_time)
    voltages=cyt.define_voltages(transient=True)

    fig, ax=plt.subplots(len(data_harmonics), 1)

    for i in range(0, len(data_harmonics)):
      ax[i].plot(voltage_results, (syn_harmonics[i,:]), label="Sim")
      ax[i].plot(voltage_results, (data_harmonics[i,:]), alpha=0.7, label="Exp")
      ax2=ax[i].twinx()
      ax2.set_yticks([])
      ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
      if i==0:
          ax[i].legend(loc="upper right")
          ax[i].set_title("Sinusoidal skew fit")
      if i==len(data_harmonics)-1:
          ax[i].set_xlabel("Nondim time")
      if i==3:
          ax[i].set_ylabel("Nondim voltage")
    plt.show()


    CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
    cyt.simulation_options["label"]="cmaes"
    cyt.simulation_options["test"]=False
    #cyt.simulation_options["test"]=True

    cyt.simulation_options["adaptive_ru"]=False
    cyt.simulation_options["numerical_method"]="Brent minimisation"
    for i in range(0, num_runs):
        x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
        #x0=cyt.change_norm_group(ifft_vals, "norm")
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

"""
 1.83336803367885298e-01
 1.20146299821343061e-01
 9.96787480415790794e-01
 1.82670404298622135e-01
 3.79449285057558081e-02
 5.63430193348374344e-05
 5.27523739734607755e-01
 5.01107506208392062e-01
"""
