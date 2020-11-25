import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from mpl_toolkits.mplot3d import Axes3D
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import scipy.integrate as integ
import scipy.optimize as opt
import math
import copy
import pints
from single_e_class_unified import single_electron
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import itertools
from matplotlib import cm
import seaborn

directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/SV"
files=os.listdir(data_loc)
scan="3"
freq="_9_"
dec_amount=8
harm_range=list(range(3,8,2))
num_harms=len(harm_range)+1


def flatten(something):
    if isinstance(something, (list, tuple, set, range)):
        for sub in something:
            yield from flatten(sub)
    elif isinstance(something, dict):
        for key in something.keys():
            yield from flatten(something[key])
    else:
        yield something

unit_dict={
    "E_0": "V",
    'E_start': "V", #(starting dc voltage - V)
    'E_reverse': "V",
    'omega':"Hz",#8.88480830076,  #    (frequency Hz)
    'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
    'v': '$s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm^{-2}$',
    'k_0': '$s^{-1}$', #(reaction rate s-1)
    'alpha': "",
    'E0_skew':"",
    "E0_mean":"V",
    "E0_std": "V",
    "k0_shape":"",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"rads",
    'phase' : "rads",
    "alpha_mean": "",
    "alpha_std": "",
    "":"",
    "noise":"",
    "error":"$\\mu A$",
}
fancy_names={
    "E_0": '$E^0$',
    'E_start': '$E_{start}$', #(starting dc voltage - V)
    'E_reverse': '$E_{reverse}$',
    'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
    'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "v",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Ru",  #     (uncompensated resistance ohms)
    'Cdl': "$C_{dl}$", #(capacitance parameters)
    'CdlE1': "$C_{dlE1}$",#0.000653657774506,
    'CdlE2': "$C_{dlE2}$",#0.000245772700637,
    'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
    'gamma': '$\\Gamma',
    'E0_skew':"$E^0 \\kappa$",
    'k_0': '$k_0$', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "cap_phase":"C$_{dl}$ phase",
    "k0_shape":"$\\log (k^0) \\sigma$",
    "k0_scale":"$\\log (k^0) \\mu$",
    "alpha_mean": "$\\alpha\\mu$",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "Phase",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"RMSE",
}

param_list={
    "E_0":-0.25,
    'E_start':  -0.5, #(starting dc voltage - V)
    'E_reverse':0.1,
    'omega':8.940960632790196,#8.88480830076,  #    (frequency Hz)
    "original_omega":8.940960632790196,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 10.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 1e-5*0,#0.000653657774506,
    'CdlE2': 1e-5*0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1e4, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":-0.25,
    "E0_std": 0.05,
    "E0_skew":0,
    "k0_shape":0.25,
    "k0_scale":1000,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 5,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3, 8)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.1,1e5],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1000],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-1], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
    'k_0': [0.1, 1e6], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[param_list['E_start'],param_list['E_reverse']],
    "E0_std": [1e-4,  0.2],
    "E0_skew":[-10, 10],
    "d_E":[0.05, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [0, 2*math.pi],
}
table_params=["E_start", "E_reverse", "E_0","k_0","Ru","Cdl","gamma", "alpha", "v", "omega", "phase","d_E","sampling_freq", "area"]
def RMSE(series1, series2):
    return np.sqrt(np.mean(np.power(np.subtract(series1, series2),2)))
def approx_error(series1, series2):
     return np.mean(np.divide(abs(np.subtract(series1, series2)),series1))*100


class PSV_FIM:
    def __init__(self, existing_class, val_dict, sim_params, ED_parameters=[], delta=1e-2):
        self.ED_parameters=ED_parameters
        self.sim_params=sim_params
        self.delta=delta
        self.existing_class=existing_class
        self.core_times=self.existing_class.time_vec[self.existing_class.time_idx]
        self.val_dict=val_dict
        self.init_class(val_dict)
    def init_class(self, val_dict):
        dim_dict=self.existing_class.dim_dict
        sim_options=copy.deepcopy(self.existing_class.simulation_options)
        param_list=list(val_dict.keys())
        for param in param_list:
            dim_dict[param]=val_dict[param]
            if param=="omega":
                sim_options["no_transient"]=1/val_dict[param]
                dim_dict["original_omega"]=val_dict[param]
            if param=="d_E":
                dim_dict["E_start"]=dim_dict["E_0"]-val_dict[param]
                dim_dict["E_reverse"]=dim_dict["E_0"]+val_dict[param]
        self.optim_vals=[dim_dict[x] for x in self.sim_params]
        self.new_class=single_electron(None, dim_dict, sim_options, copy.deepcopy(self.existing_class.other_values), self.existing_class.param_bounds)
        self.new_class.def_optim_list(self.sim_params)
    def sensitivity(self, param_of_interest, delta=None):
        if delta==None:
            delta=self.delta
        interest_position=self.new_class.optim_list.index(param_of_interest)
        orig_val=self.new_class.dim_dict[param_of_interest]
        change_val=delta*self.new_class.dim_dict[param_of_interest]
        params=copy.deepcopy(self.optim_vals)
        no_change_param=(self.new_class.test_vals(params, "timeseries"))
        #plt.subplot(1,2,1)
        #plt.plot(self.new_class.e_nondim(self.new_class.define_voltages()))
        #plt.subplot(1,2,2)
        #plt.title(self.new_class.dim_dict["d_E"])
        #plt.plot(no_change_param)
        #plt.show()
        params[interest_position]=self.new_class.dim_dict[param_of_interest]+change_val
        upper_param=(self.new_class.test_vals(params, "timeseries"))
        params[interest_position]=self.new_class.dim_dict[param_of_interest]-change_val
        normal_param=(self.new_class.test_vals(params, "timeseries"))
        #plt.plot(upper_param)
        #plt.plot(normal_param)
        #plt.show()
        derivative=np.divide(abs(np.subtract(upper_param, normal_param)), 2*change_val)
        normed_derivative=np.divide(derivative, no_change_param)
        return np.multiply(derivative, orig_val), self.new_class.time_vec[self.new_class.time_idx], max(no_change_param)
    def FIM(self, val_dict, delta=None):
        self.init_class(val_dict)
        sensitivity_matrix=np.zeros((len(self.core_times), len(self.new_class.optim_list)))

        for i in range(0, len(self.new_class.optim_list)):
            sensitivity, times, max_current=self.sensitivity(self.new_class.optim_list[i], delta)
            #plt.plot(sensitivity)
            #plt.title(self.new_class.optim_list[i])
            #plt.show()
            sensitivity_matrix[:,i]=np.interp(self.core_times, times, sensitivity)
        cov=np.identity(len(self.core_times))*(1/(0.05*max_current))
        FIM_1=np.matmul(sensitivity_matrix.transpose(), cov)
        FIM=np.matmul(FIM_1, sensitivity_matrix)

        #print(FIM)
        return FIM
    def simulate(self, parameters):
        for i in range(0, len(self.ED_parameters)):
            self.val_dict[self.ED_parameters[i]]=self.existing_class.un_normalise(parameters[i], self.existing_class.param_bounds[self.ED_parameters[i]])
        FIM=self.FIM(self.val_dict)
        inv_FIM=np.linalg.inv(FIM)
        return_arg=np.linalg.det(inv_FIM)
        return (np.log10(return_arg))


def information_calc(val_dict, param_of_interest, current_class,other_values, delta):
    disp_params=["E0_mean","E0_std","k0_scale", "k0_shape", "alpha_mean", "alpha_std"]
    dim_dict=current_class.dim_dict
    sim_options=copy.deepcopy(current_class.simulation_options)
    for param in val_dict.keys():
        dim_dict[param]=val_dict[param]
        if param=="omega":
            sim_options["no_transient"]=1/val_dict[param]
            dim_dict["original_omega"]=val_dict[param]
    optim_list=[x for x in val_dict.keys() if x in disp_params]
    if param_of_interest not in optim_list:
        optim_list+=[param_of_interest]

    new_class=single_electron(None, dim_dict, sim_options, other_values, current_class.param_bounds)
    new_class.def_optim_list(optim_list)
    params=[val_dict[key] for key in new_class.optim_list]

    interest_position=new_class.optim_list.index(param_of_interest)
    change_val=delta*dim_dict[param_of_interest]
    params[interest_position]=dim_dict[param_of_interest]+change_val

    #volts=new_class.define_voltages()[new_class.time_idx]

    upper_param=(new_class.test_vals(params, "timeseries"))
    params[interest_position]=dim_dict[param_of_interest]-change_val
    normal_param=(new_class.test_vals(params, "timeseries"))
    derivative=np.divide(abs(np.subtract(upper_param, normal_param)), 2*change_val)
    #plt.plot(upper_param)
    #plt.plot(normal_param)
    #plt.plot(volts, np.subtract(upper_param, normal_param))
    #plt.show()
    #total_information=integ.trapz(np.power(derivative,2))
    return derivative, new_class.time_vec[new_class.time_idx], max(normal_param)

def k0_diff_calc(val_dict,  current_class,other_values):
    disp_params=["E0_mean","E0_std","k0_scale", "k0_shape", "alpha_mean", "alpha_std"]
    dim_dict=current_class.dim_dict
    sim_options=copy.deepcopy(current_class.simulation_options)
    for param in val_dict.keys():
        dim_dict[param]=val_dict[param]
        if param=="omega":
            sim_options["no_transient"]=1/val_dict[param]
            dim_dict["original_omega"]=val_dict[param]

    new_class=single_electron(None, dim_dict, sim_options, other_values, current_class.param_bounds)
    new_class.def_optim_list(["k_0"])
    no_disp=(new_class.test_vals([dim_dict["k0_scale"]], "timeseries"))
    print(new_class.dim_dict["k_0"])
    new_class.def_optim_list(["k0_shape", "k0_scale"])
    volts=new_class.define_voltages()[new_class.time_idx]
    disp=(new_class.test_vals([dim_dict["k0_shape"], dim_dict["k0_scale"]], "timeseries"))
    volts2=new_class.define_voltages()[new_class.time_idx]
    print(new_class.dim_dict["k0_shape"], new_class.dim_dict["k0_scale"])


    return (RMSE(disp, no_disp))
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
cyt.def_optim_list(["Cdl"])
cyt.simulation_options["dispersion_bins"]=[16]

cyt.def_optim_list(["k0_shape", "k0_scale"])
voltage=cyt.define_voltages()[cyt.time_idx]
times=cyt.time_vec[cyt.time_idx]
frequencies=np.arange(10, 100, 1)
amplitudes=np.arange(0.3, 0.31)
k0_exps=np.linspace(-1, 4, 10)
k0_vals=[10**x for x in k0_exps]#3z\13ave
ru_vals=copy.deepcopy(k0_vals)
#k0_vals=np.linspace(0.1, 0.8, 5)
copied_vals=copy.deepcopy(other_values)
CDLS=[10**x for x in range(-5, -1)]
parameters=["E_0","k_0", "Ru", "Cdl", "alpha"]
exp_params=["omega","phase"]
val_dict=dict(zip(parameters, [param_list[x] for x in parameters]))
for q in range(0, 0):
    param_list["Cdl"]=CDLS[q]


    ED_1=PSV_FIM(cyt, val_dict, parameters, exp_params, 1e-2)


    #print(ED_1.simulate(c_params))
    for i in range(0, len(exp_params)):#
        c_params[i]=cyt.un_normalise(c_params[i], cyt.param_bounds[exp_params[i]])
    #print(c_params)


    from scipy.optimize import Bounds, minimize
    from pints import plot

    import cma
    es = cma.CMAEvolutionStrategy([1e-3, 0.5], 0.5, {"bounds":[[0 for x in range(0, len(exp_params))], [1 for x in range(0, len(exp_params))]], "tolfun":1e-4, "tolx":1e-4})
    es.optimize(ED_1.simulate)

    es.result_pretty()
c_params_5=[0.0029994507624747048, 0.10981321424416143, 0.26678155063651277]
c_params_4=[0.0017019464396750257, 0.0649385293382252, 0.45602341239476873]
c_params_3=[0.00033283345562488135, 0.10909027070019929, 0.06802690966390106]
c_params_2=[0.00010643572041858844, 0.1317199828970463, 0.14197851244040813]
c_params=[0.015015683247741541, 0.999963926012231, 0.0008135651067916193]
c_params=[0.06127637762376781, 0.11063427063068046]
test_vals=[c_params]
for i in range(0,len(test_vals)):
    for j in range(0, len(test_vals[i])):
        test_vals[i][j]=cyt.un_normalise(test_vals[i][j], cyt.param_bounds[exp_params[j]])
sens_vals=[[10000,0.3, 3*math.pi/2]]*4
param_sets=[test_vals, sens_vals]
print(test_vals)
table_params=["E_start", "E_reverse", "E_0","k_0","Ru","Cdl","gamma", "alpha", "omega", "phase","d_E", "area"]
print([val_dict[x] if x in val_dict else cyt.dim_dict[x] for x in table_params ])
filenames=["ED_MCMC_d_E", "Sens_MCMC_d_E"]
for i in range(0, len(param_sets)):
    for j in range(0, len(param_sets[0])):
        for q in range(0, len(exp_params)):
            val_dict[exp_params[q]]=param_sets[i][j][q]
        val_dict["Cdl"]=CDLS[j]
        print(val_dict)
        ED_1=PSV_FIM(cyt, val_dict, parameters, exp_params, 1e-2)
        syn_data=ED_1.new_class.test_vals(ED_1.optim_vals, "timeseries")
        error= 0.05*max(syn_data)
        print("VALS", error)
        noisy_syn_data=ED_1.new_class.add_noise(syn_data,error)
        mcmc_problem=pints.SingleOutputProblem(ED_1.new_class, ED_1.new_class.time_vec[ED_1.new_class.time_idx], noisy_syn_data)
        updated_lb=[param_bounds[x][0] for x in parameters]+[0.1*error]
        updated_ub=[param_bounds[x][1] for x in parameters]+[10*error]
        updated_b=[updated_lb, updated_ub]
        updated_b=np.sort(updated_b, axis=0)
        log_liklihood=pints.GaussianLogLikelihood(mcmc_problem)
        log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
        log_posterior=pints.LogPosterior(log_liklihood, log_prior)
        mcmc_parameters=ED_1.optim_vals

        mcmc_parameters=np.append(mcmc_parameters, error)
        xs=[mcmc_parameters,
            mcmc_parameters,
            mcmc_parameters
            ]
        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)
        mcmc.set_parallel(True)
        mcmc.set_max_iterations(10000)
        chains=mcmc.run()
        #plot.trace(chains)
        #plt.show()
        f=open("MCMC_results"+"/"+filenames[i]+"_"+str(-5+j), "wb")
        np.save(f, chains)
        f.close()
