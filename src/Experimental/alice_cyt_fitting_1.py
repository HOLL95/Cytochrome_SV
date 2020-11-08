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
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/PSV"
files=os.listdir(data_loc)
experimental_dict={}
param_file=open(data_loc+"/PSV_params", "r")
useful_params=dict(zip(["max", "min", "Amp[0]", "Freq[0]"], ["E_reverse", "E_start", "d_E", "original_omega"]))
dec_amount=32
for line in param_file:
    split_line=line.split()
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]

for i in range(1, 6):
    file_name="PSV_Cyt_{0}_cv_".format(i)
    current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
        'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
        'omega':9.015120071612014, #8.88480830076,  #    (frequency Hz)
        "original_omega":9.015120071612014,
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
        "harmonic_range":list(range(2,100,1)),
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
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
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
    h_class=harmonics(list(range(2, 12)), 1, 0.05)
    #h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, xaxis=voltage_results)
    fft=one_tail(np.fft.fft(current_results))
    hann_fft=one_tail(np.fft.fft(np.multiply(np.hanning(len(current_results)),current_results)))
    f=one_tail(np.fft.fftfreq(len(time_results), time_results[1]-time_results[0]))
    predicted_cap_params=[0.00016107247651709253, 0.0032886486609914056, 0.0009172547160104724]
    cap_param_list=["Cdl", "CdlE1", "CdlE2"]
    #for i in range(0, len(cap_param_list)):
    #    cyt.param_bounds[cap_param_list[i]]=[predicted_cap_params[i]*0.75, predicted_cap_params[i]*1.25]
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
    inferred_params=[0.09914947692931006, 999.9469476990267, 807.042914431112, 0.0001208124331619577, 0.004110808011450815, 0.0011465660548728757, 1.8241387286724777e-10, 9.015925747638107, 2.344303255727037, 3.3343889148779873, 0.40000026882147105]
    inferred_params=[0.0026892085518520903, 434.4452655954981, 7056.309660109621, 0.00012080446316527372, 0.0024666925378970194, 0.0011465681186305678, 2.595397583499278e-10, 9.01594116289882, 1.600581233850582, 4.7914694562583975, 0.4000025625960726]
    inferred_params=[-0.0831455230466103, 421.75682491012714, 7078.2779711954745, 0.00012080477835014745, 0.004110694113544712, 0.0011465683948502243, 2.596730147883652e-10, 9.015930319756032, 2.1181508092892587, 5.346599955446149, 0.5999999924210264]
    inferred_params=[-0.06202878859625431, 56.47425198391122, 2659.9466646191718, 0.00012080450895231545, 0.0041108063511217634, 0.0006879436598670587, 9.999999857331693e-11, 9.015929295024732, 2.84613056627866, 5.215571873921595, 0.4019491213369458]
    inferred_params=[-0.019106005884705374, 53.25042268640264, 2655.3091133089174, 0.00012080441347347774, 0.0024664868033714405, 0.0006879469519832639, 9.999996948023375e-11, 9.0159352779017, 2.8375701928376857, 5.213915050992704, 0.5999987942107046]
    inferred_params=[0.002707266157404936, 431.5317292838629, 7057.763756614384, 0.00012080649627857818, 0.0024668636255099218, 0.0011465616336472683, 2.596058209356642e-10, 9.015934259922634, 1.780821768948755, 4.97400312309354, 0.400001156051716]
    inferred_params=[0.0015529742952713477, 437.9572645980276, 7015.08062205601, 0.00012081581944559124, 0.0024665180267551346, 0.0011465650711908403, 2.580921906494901e-11, 9.015938141061776, 6.235329585923734, 3.1425953996543674, 0.4000002714850346]
    inferred_params=[-0.021495031150668878,  17.570527719697008, 1949.3033882011057, 0.00020134013002972008, 0.00411080805846069, 0.0006879411270873684, 2.3340827906876003e-10, 9.014898777848584, 2.5763763013646175, 4.359308770046119, 0.4000000157539228]
    inferred_params=[-0.06999997557877347,1e-3, 1.245269093253469, 701.0390205847876, 0.00012081970706517015, 0.0024664965342477527, 0.0011465681607091556, 1.7085728483891467e-09, 9.014958257811347, 6.014524412936475, 5.676241758957758, 0.4683583542196279]
    inferred_params=[-0.06999999976247695, 0.054653156208964916, 0.423123012427712, 361.21853877194775, 0.0005089406894416405, -0.055068050329032774, 0.0022654775318873364, 7.263427801734963e-11, 9.01488754940552, 1.5710075181470322, 5.2472698323265075, 0.4837659989252462]
    #inferred_params=[-0.06999997869349413, 0.07228496915886849, 13.380910552686727, 136.7193405893562, 0.0019999989494070857, -0.017387868180937807, 0.0020622497400090457, 9.999999928198332e-11, 9.012923768523683, 1.5707963282578978, 4.706986793173553, 0.4394551917936057]
    #inferred_params=[-0.04360455454950035, 0.00010270662292309546, 37.359772287060075, 134.9762212627008, 0.00199970388607249, 0.004777479982016303, 0.001299865559919934, 3.0601694353969734e-10, 9.015450285831738, 3.1912456506402505, 3.4376582058977876, 0.5730279057738448]
    #inferred_params=[-0.09999992323276853, 0.09999984026530274, 2476.4846992762937, 212.64576475763195, 0.00199987543575968, -0.02761477871367725, 0.00022959276066804486, 4.494113408952504e-09, 9.015859859943566, 5.380508213684635, 5.339712400929006, 0.5901009973509725]
    inferred_params=[-0.029684587659467895, 0.015423677503228435, 184.54350534229397, 7209.054457183826, 1.4814777324317683e-05, -0.09933550996421193, 0.004192576352992021, 2.4832179807042775e-10, 9.016466705940926, 6.283184749628701, 4.941242800185401, 0.599999882604709]

    #inferred_params=[-0.04261899345311761, 22.381555395265497, 1288.6182661878825, 0.00012080655547508927, 0.004110810366477168, 0.0006879429762117952, 1.1386685166132384e-09, 9.01490124897687, 4.761022898261431, 4.125701872157008, 0.5999999960651834]
    #inferred_params=[-0.052381022493849856, 38.83009178900811, 630.2720365182151, 0.0002013405699652845, 0.0041108108134498035, 0.0006879411561647431, 1.0684972937740034e-09, 9.015279033844829, 3.4070976356125953, 5.432687270921383, 0.5742869813297893]


    true_data=current_results
    fourier_arg=cyt.top_hat_filter(true_data)


    cmaes_test=cyt.test_vals(inferred_params, "timeseries")
    #plt.plot(cmaes_test)
    #plt.show()
    h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, data_time_series=cmaes_test,  xaxis=voltage_results, alpha_increment=0.3)

    plt.plot(fourier_arg)
    plt.plot(cyt.top_hat_filter(cmaes_test))
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
    num_runs=10
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
        #plt.plot(cmaes_time)
        #plt.plot(fourier_arg)
        #plt.show()
        #plt.subplot(1,2,1)
        #plt.plot(voltage_results, cmaes_time)
        #plt.plot(voltage_results, true_data, alpha=0.5)
        #plt.show()
        #h_class.plot_harmonics(times=time_results, experimental_time_series=current_results, data_time_series=cmaes_time,  xaxis=voltage_results)
        #plt.subplot(1,2,2)
        #plt.plot(time_results, cyt.define_voltages()[cyt.time_idx:])
        #plt.plot(time_results, voltage_results)
