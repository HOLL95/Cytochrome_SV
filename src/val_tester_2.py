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
    current_results=current_data[0::dec_amount,1]
    time_results=current_data[0::dec_amount,0]
except:
    raise ValueError("No current file of that scan and frequency found")
try:
    voltage_results=voltage_data[0::dec_amount,1]
except:
    raise ValueError("No voltage file of that scan and frequency found")

param_list={
    "E_0":0.2,
    'E_start':  min(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]), #(starting dc voltage - V)
    'E_reverse':max(voltage_results[len(voltage_results)//4:3*len(voltage_results)//4]),
    'omega':8.94,#8.88480830076,  #    (frequency Hz)
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
    'sampling_freq' : (1.0/200),
    'phase' : 3*(math.pi/2),
    "time_end": None,
    'num_peaks': 30
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
    "experiment_time": time_results,
    "experiment_current": current_results,
    "experiment_voltage":voltage_results,
    "bounds_val":20,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[0.2, 0.3],
    "E0_std": [1e-5,  0.1],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
    "noise":[0, 100]
}
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
print(cyt.nd_param.c_I0)
def hann_window(series):
    hann=np.hanning(len(series))
    return np.multiply(hann, series)
cyt.define_boundaries(param_bounds)
time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"]*cyt.nd_param.c_T0, 0.5)


cyt.dim_dict["noise"]=0
cyt.dim_dict["phase"]=3*math.pi/2
print(len(current_results))
#cyt.def_optim_list(["E_0","k0_shape", "k0_scale","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
cyt.simulation_options["dispersion_bins"]=[5]
cyt.simulation_options["GH_quadrature"]=True
cyt.def_optim_list(["E_0","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"])
reduced_list=["E_0","k_0","Ru","gamma","omega","cap_phase","phase", "alpha"]
reduced_vals=[-0.15336602503285124, 34.94661847742798, 862.8515851277263, 8.172998783135129e-11, 8.941111613936217, 2.185756798482757, 3.658426955099939, 0.5999999999999948]
real_reduced_vals=[-0.2298233557733859, 54.02734878486922, 56.98720340934313, 1.0568500401803649e-11, 8.939393208568873, 5.756812883487079, 4.944735887956814, 0.40000000000000213]
real_reduced_vals=[-0.22143480816315367, 9.554852103917577, 110.62804301180374, 1.1700021020182037e-11, 8.938519564358028, 5.815856354422392, 5.288179056973903, 0.462783815464132]
abs_reduced_vals=[0.11632868285877823, 253.1340866293384, 410.3675393800388, 2.2700606724717796e-11, 8.941015495770522, 6.267334286127515, 6.028389988317695, 0.5999999999999979]
abs_reduced_vals=[0.12524648512800418, 45.81189242167458, 248.23178090620584, 1.934991492656277e-11, 8.941052845573875, 4.888998266359435, 3.2603293133761335, 0.40000000000003355]
imag_reduced_vals=[-0.36797649999999843, 74.67748059185197, 312.827878694299, 9.390339181258608e-11, 8.94049803991402, 1.5707963267948966, 4.414787723470457, 0.4]
vals=[-0.01655918225883679, 28.37267320998559, 999.9999999999768, 0.0001226316755359932, 0.010515263489958092, -0.0014258396718951558, 9.037611264615161e-11, 8.940736406988611,4.607754120894463, 6.1218970386536, 0.40000000000002656]
abs_vals=[0.1415555782071663, 538.1069767143139, 639.2518440388463, 4.503671120719274e-05, -0.023095932541434994, 0.0032692266939467977, 1.7958574261781276e-11, 8.941040633214785, 1.646172891288431, 5.800777668826027, 0.42049644198301717]
real_vals=[-0.32814304875062755, 156.2132295274366, 672.7743490371313, 0.0007206393126657284, 0.12389661943676065, 0.00877327740842467, 5.121409345799231e-11, 8.938770401530908, 5.890285773219234, 3.205828836698493, 0.5800926400246942]
imag_vals=[-0.02355425718380222, 29.006812943893074, 790.4035761965491, 0.0001733560705809961, 0.0643049888801522, 0.00999329777977639, 9.956885412841783e-11, 8.941147763567404, 5.098595570269756, 4.987013756372768, 0.54958625705231]
all_vals=[0.02033533819663358, 28.347813937220756, 999.9999999806909, 0.0003013806013473721, -0.01892301369255068, -0.004948533055054176, 7.840053796231529e-11, 8.940988000369643, 2.9142983739839714, 5.459297204676048, 0.4000000000000008]
imag_e0=[0.25154525912497694, 0.04776397494695416, 28.045021208386437, 310.2861947806952, 0.0007418424428828407, 0.011576584391785626, 0.005837777563835097, 7.745776863774142e-11, 8.940567899623963, 5.798645555881929, 4.409584917666756, 0.4163338871105171]
real_e0=[0.27962188042218383, 0.025673544382111127, 182.00012539796222, 999.995507576053, 0.00034998499951815093, -0.03609368808580215, -0.0037276229790483336, 3.362205150254791e-11, 8.940444592733577, 6.273105850163433, 3.412448697161386, 0.4370739814229856]
abs_e0=[0.279803167104502, 0.0021528213104500134, 928.7717059555468, 155.73367646127838, 1.9864684046884364e-05, 0.026630409339232358, -0.006006758875122577, 9.799748258522194e-11, 8.940733077600111, 5.810040827846592, 4.077470828989698, 0.5247845922838983]
#all_vals=[-0.19896709588060044, 0.0117447572739623, 663.0070333127449, 15.60742873191647, 0.0004775254125236546, -0.012948911979412857, 0.009191060972021893, 1.5331044249093517e-11, 8.605649285898744, 2.4548253689680077, 5.709340060264028, 0.5485117501096282]
comp_vals=[-0.3055372331950631, 991.0626312468332, 89.21401193705475, 0.0007634486126395742, 0.0980481274632901, -0.005811384646150023, 5.3223663919983905e-11, 8.940733077600111, 5.771825413883179, 0.5901757781891152]



titles=["Real", "Imaginary"]
funcs=[np.real, np.imag]
results=[comp_vals, comp_vals]
"""for i in range(0, len(reduced_vals)):
    vals[cyt.optim_list.index(reduced_list[i])]=imag_reduced_vals[i]
print(vals)"""
for j in range(0, len(results)):
    vals=abs_vals
    true_signal=cyt.test_vals(results[j], "timeseries")
    test_data=cyt.add_noise(true_signal, 0*0.05*max(true_signal))
    #cyt.simulation_options["alpha_dispersion"]="uniform"
    #cyt.def_optim_list(["Ru","Cdl","CdlE1", "CdlE2",'omega',"phase","cap_phase"])
    #cyt.dim_dict["gamma"]=0
    true_data=current_results


    f=np.fft.fftfreq(len(time_results), time_results[1]-time_results[0])

    where_idx=tuple(np.where((f>-other_values["harmonic_range"][-1]-0.5)&(f<other_values["harmonic_range"][-1]+0.5)))
    y=np.fft.fft(hann_window(true_data))
    y2=np.fft.fft(hann_window(test_data))
    func=funcs[j]
    covariance=np.cov(np.array([np.real(y), np.imag(y)*1j],dtype="complex"))
    print(covariance)
    inv_cov=np.linalg.inv(covariance)
    print(-np.log(np.linalg.det(covariance)))
    error=y-y2
    real_error=np.mean(np.real(error))
    imag_error=np.mean(np.imag(error))*1j
    mean=np.array([real_error, imag_error])
    hermite_error=np.transpose(np.conj(mean))
    print(hermite_error)
    term2=np.dot(hermite_error, inv_cov)
    term2=np.dot(term2, mean)
    print("complex?",term2)


    plt.plot(f[where_idx], func(y2[where_idx]), label="Simulation")
    plt.plot(f[where_idx], func(y[where_idx]), label="Data", alpha=0.7)
    plt.xlabel("Nondim Frequency")
    plt.ylabel(titles[j]+" Amplitude")
    plt.title(titles[j]+" Fit")
    plt.legend()
    plt.show()
    fourier_arg=cyt.top_hat_filter(true_data)
    cyt.secret_data_fourier=fourier_arg
    test_fourier=cyt.test_vals(results[j], "fourier", test=False)

    data_harmonics=harms.generate_harmonics(time_results,(current_results), funcs[j])
    syn_harmonics=harms.generate_harmonics(time_results, (test_data), funcs[j])
    fig, ax=plt.subplots(len(data_harmonics), 1)
    for i in range(0, len(data_harmonics)):


        ax[i].plot(voltage_results, (syn_harmonics[i,:]), label="Simulation")
        ax[i].plot(voltage_results, (data_harmonics[i,:]),  label="Data")
        #ax[i].plot(voltage_results, np.subtract(data_harmonics[i,:],syn_harmonics[i,:]), alpha=0.7, label="Residual")
        ax2=ax[i].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(other_values["harmonic_range"][i], rotation=0)
        if i==0:
            ax[i].set_title(titles[j]+" Fit")
            ax[i].legend(loc="upper right")
        if i==len(data_harmonics)-1:
            ax[i].set_xlabel("Nondim voltage")
        if i==3:
            ax[i].set_ylabel("Nondim current")

    plt.show()
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
score = pints.SumOfSquaresError(cmaes_problem)#[4.56725844e-01, 4.44532637e-05, 2.98665132e-01, 2.96752050e-01, 3.03459391e-01]#
CMAES_boundaries=pints.RectangularBoundaries(list([np.zeros(len(cyt.optim_list))]), list([np.ones(len(cyt.optim_list))]))
cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=False
#cyt.simulation_options["test"]=True
num_runs=5
param_mat=np.zeros((num_runs,len(cyt.optim_list)))
score_vec=np.ones(num_runs)*1e6

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
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)
    #plt.subplot(1,2,1)
    plt.plot(voltage_results, cmaes_time)
    plt.plot(voltage_results, true_signal)
    plt.plot(voltage_results, test_data, alpha=0.5)
    #plt.subplot(1,2,2)
    #plt.plot(time_results, cyt.define_voltages()[cyt.time_idx:])
    #plt.plot(time_results, voltage_results)
    plt.show()
    #cmaes_fourier=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
    param_mat[i,:]=cmaes_results
    score_vec[i]=found_value
    print("Finish?")
    #i, o, e = select.select( [sys.stdin], [], [], 5)
    #if len(i) != 0:
    #    break
