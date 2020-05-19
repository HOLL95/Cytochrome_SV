import numpy as np
import matplotlib.pyplot as plt
import sys
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from scipy.integrate import odeint
from single_e_class_unified import single_electron
from pybamm_solve import pybamm_solver
import time
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/DCV/22_Aug/"
files=os.listdir(data_loc)
number="2"
blank_nos=[str(x) for x in range(1, 7)]
for letter in ["A"]:
    for file in files:
        if ".ids" not in file:
            if "183"+letter in file and number in file:
                dcv_file=np.loadtxt(data_loc+file, skiprows=1)
                plt.plot(dcv_file[:,1], dcv_file[:, 2])
    plt.show()
current_results1=dcv_file[:, 2]
voltage_results1=dcv_file[:, 1]
time_results1=dcv_file[:, 0]
for letter in ["blank"]:
    for number in blank_nos:
        for file in files:
            if ".ids" not in file:
                if letter in file and number in file:
                    blank_file=np.loadtxt(data_loc+file, skiprows=1)
                    blank_current=blank_file[:, 2]
                    blank_voltage=blank_file[:, 1]
                    blank_time=blank_file[:, 0]
                    plt.plot(blank_voltage, current_results1-blank_current, label=number)
plt.plot(voltage_results1, blank_current)
plt.legend()
plt.show()


#current_results1=current_results1-blank_current
param_list={
    "E_0":0.2,
    'E_start':  min(voltage_results1), #(starting dc voltage - V)
    'E_reverse':max(voltage_results1),
    'omega':8.94, #8.88480830076,  #    (frequency Hz)
    "v":0.03,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
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
    'sampling_freq' : (1.0/2000),
    'phase' :0.1,
    "time_end": -1,
    'num_peaks': 30,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":16,
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
    "harmonic_range":list(range(3,9,1)),
    "experiment_time": time_results1,
    "experiment_current": current_results1,
    "experiment_voltage":voltage_results1,
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.3,-0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e5],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
    'k_0': [0, 1e4], #(reaction rate s-1)
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
}
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)

time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
plt.plot(voltage_results, current_results)
plt.show()
volts=cyt.define_voltages()
plt.plot(volts)
plt.plot(voltage_results)
plt.show()
cyt.simulation_options["dispersion_bins"]=[16]
cyt.simulation_options["GH_quadrature"]=True
cyt.def_optim_list([ "E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma", "alpha"])
norm_vals=[ "E_0", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma", "alpha"]
curr_best=[-0.12138673699417057, 5653.711865172047, 6.251920538521568e-07, 0.0005113312954312908, 0.032688180215297846, -0.0016959746570033296, -4.363761285119949e-05, 5.118104508670231, -0.2856860853954224, -0.012643633142239707, -0.0001955648458022985, 6.441193562150172e-12, 0.5250534731453991]
blank_sub=[0.09306177584874675, 1.9344951154832872, 7.4238908166570345, 0.0001868507457614954, 0.03418077288539134, -0.003700461869470928, -8.222260002632709e-05, 1.8219980821049462, -0.4487186289315357, -0.007048236265024821, 0.00020170449845746674, 1.4283135211180592e-12, 0.5569545398118998]
with_params=[-0.25819776867643385,  0.06004627245093131, 83951.6420075274, 6.251920538521568e-07, 0.0005113312954312908, 0.032688180215297846, -0.0016959746570033296, -4.363761285119949e-05, 5.118104508670231, -0.2856860853954224, -0.012643633142239707, -0.0001955648458022985, 2.0464993756449906e-10, 0.5250534731453991]
with_params_sub=[-0.25819776867643385,  0.06004627245093131, 83951.6420075274, 7.4238908166570345, 0.0001868507457614954, 0.03418077288539134, -0.003700461869470928, -8.222260002632709e-05, 1.8219980821049462, -0.4487186289315357, -0.007048236265024821, 0.00020170449845746674, 2.0464993756449906e-10, 0.5569545398118998]

plot_results=[current_results1,current_results1, current_results1-blank_current, current_results1-blank_current]
vals=[curr_best,with_params, blank_sub,  with_params_sub]
vals=[-0.22724866541126082, 0.06775741946664324, 100, 10.48267882705497, 0.0003891656168817575, 0.11518075211542111, 0.006116081631796888, 0.00015255858946039424, 3.15, -0.11518075211542111, -0.006116081631796888, -0.00015255858946039424, 5.501037311619498e-10, 0.468582417253183]
vals=[-0.22724866541126082,  100, 5.48267882705497, 0.0003891656168817575, 0.21518075211542111, 0.0006116081631796888, 0.000015255858946039424,9.01037311619498e-10, 0.468582417253183]
cyt.nd_param.nd_param_dict["time_end"]=time_results[-1]

#cyt.times()
print(len(cyt.time_vec))
volts=cyt.define_voltages()
cyt.simulation_options["numerical_method"]="pybamm"
cyt.def_optim_list(norm_vals)
k_idx=cyt.optim_list.index("Ru")
orig_k=vals[k_idx]
k_str=str(round(orig_k, 3))
print("HELLO")
"""rs=np.flip([1, 10, 100,200, 600, 800,  1000, 10000, 100000])
error10=[0.04416823387145996,
        0.05545496940612793,
        0.06423497200012207,
        0.06735754013061523,
        0.07774925231933594,
        0.24072909355163574,
        5.253161430358887,
        7.299848794937134,
        8.343536853790283]

plt.loglog(rs, error10, label="Rtol=1e-10, Atol=1e-8")
plt.xlabel("Ru")
plt.ylabel("simulation time(secs)")
plt.legend()
plt.show()"""
rs=[800,  1000, 10000, 100000]
for i in range(0, len(rs)):
    plt.subplot(2, len(rs)//2, i+1)
    vals[k_idx]=rs[i]
    current_range=cyt.test_vals(vals, "timeseries")

    plt.title("$R_u=10^"+str(int(np.log10(rs[i])))+"\Omega$")


    #plt.show()
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = time_results[-1]
    numpoints = len(current_range)

    w0 = [current_range[0],0, voltage_results[0]]

    # Call the ODE solver.
    start=time.time()
    wsol = odeint(cyt.current_ode_sys, w0, time_results,
                  atol=abserr, rtol=relerr)
    print("python", time.time()-start)
    adaptive_current=wsol[:,0]
    adaptive_potential=wsol[:,2]
    adaptive_theta=wsol[:, 1]#cyt.calc_theta(wsol[:,0])
    gradients=np.zeros((len(time_results),len(wsol[0])))
    jacobian_eig=np.zeros((2,len(time_results)))
    #for j in range(0, len(time_results)):
        #gradients[j, :]=cyt.current_ode_sys([adaptive_current[j], adaptive_theta[j], adaptive_potential[j]], time_results[j])
        #multiply_vec=[cyt.Cdlp*cyt.nd_param.nd_param_dict["Ru"]*-1, cyt.nd_param.nd_param_dict["gamma"], cyt.Cdlp]
        #gradients[j, :]=np.multiply(gradients[j, :], multiply_vec)
        #jacobian=cyt.system_jacobian([adaptive_current[j], adaptive_theta[j], adaptive_potential[j]], time_results[j])
        #eigens, vectors=np.linalg.eig(jacobian)
        #jacobian_eig[:, j]=[min(eigens), max(eigens)]
    ax=plt.gca()
    p1,=ax.plot(time_results, adaptive_current, label="I", alpha=0.7, color="blue")
    p2, =ax.plot(time_results, current_range, label="I pybamm", alpha=0.7, color="red")
    #p1, =ax.plot(time_results, jacobian_eig[0, :], label="min(eig)")
    #ax2=ax.twinx()
    #p2, =ax2.plot(time_results, jacobian_eig[1, :], color="red", label="max(eig)")
    #ax.set_xlabel("Nondim time")
    #ax.set_ylabel("min(eig)")
    #ax2.set_ylabel("max(eig)")
    #scaled_dI=gradients[:, 0]
    #scaled_dtheta=gradients[:, 1]
    #scaled_E=gradients[:, 2]
    #grad_1=np.add(scaled_dtheta, scaled_E)
    #grad_2=np.add(grad_1, scaled_dI)
    #p5, =ax.plot(time_results, grad_2, color="yellow", linestyle="--", label="dI+d$\\theta$+dE")
    #p2,=ax.plot(time_results, gradients[:, 0], label="dI", color="red", alpha=0.7)
    #p3,=ax.plot(time_results, gradients[:, 1], label="d$\\theta$", color="green", alpha=0.7)
    #p4,=ax.plot(time_results, gradients[:, 2], label="dE", color="purple", alpha=0.7, linestyle="--")
    plt.legend(handles=[p1, p2], loc="lower left")

#plt.tight_layout()
plt.show()
colours=["red", "blue", "green"]
rs=[2e-8, 2e-5, 0.02, 0.2, 2, 20, 200, 2000, 20000]
sfs=[1/x for x in [50, 200, 500, 2000]]
#fig, ax=plt.subplots(len(sfs), len(rs))
plot="current"
"""rs=[100, 1000, 10000, 100000]
exps=np.flip(range(0, 9))
r_exps=range(2, 6)
cdls=[1/(10**x) for x in exps]
for i in range(0, len(cdls)):
    for j in range(0, len(rs)):
        vals[r_idx]=rs[j]
        vals[cdl_idx]=orig_cdl*cdls[i]
        current_range=cyt.test_vals(vals, "timeseries")
        plt.subplot(3, 3, i+1)
        plt.plot(voltage_results, current_range, label='$R_u=10^'+str(int(np.log10(rs[j])))+" \\Omega$")
        plt.xlabel("Nondim Voltage")
        plt.ylabel("Nondim Current")
    plt.legend(loc="upper left")
    plt.title("Cdl="+str(orig_cdl*cdls[i])+"F")
plt.show()"""
"""for z in range(0, len(sfs)):
    param_list["sampling_freq"]=sfs[z]
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    cyt.def_optim_list(norm_vals)
    cyt.nd_param.nd_param_dict["time_end"]=time_results[-1]
    cyt.times()
    volts=cyt.define_voltages()
    if plot=="domain":
        times=[x*5*cyt.nd_param.nd_param_dict["sampling_freq"] for x in range(1, 10)]
        times=np.append(0.1*cyt.nd_param.nd_param_dict["sampling_freq"], times)
    elif plot=="current":
        times=[x*cyt.nd_param.nd_param_dict["sampling_freq"] for x in range(1, 2)]
    for i in range(0, len(rs)):
        axis=ax[z][i]
        cyt.bounds_val=rs[i]
        for j in range(0, len(times)):
            #print("DTTTT", cyt.time_vec[1]-cyt.time_vec[0], cyt.nd_param.nd_param_dict["sampling_freq"])
            vals[k_idx]=orig_k*100
            if plot=="domain":
                cyt.simulation_options["numerical_debugging"]=times[j]
                current_range, residual=cyt.test_vals(vals, "timeseries")
                axis.plot(current_range, residual, label=str(round(times[j], 3)))
            elif plot=="current":
                current_range=cyt.test_vals(vals, "timeseries")
                axis.plot(volts, current_range)
            #plt.plot(cyt.e_nondim(voltage_results), cyt.i_nondim(current_results), lw=2, label="data")
        axis.set_title("Bv="+str(cyt.bounds_val)+" Sr="+str(1/sfs[z]))
        if i==len(rs)-1:
            axis.legend(loc="upper right")
        if z==len(sfs)-1:
            if plot=="domain":
                axis.set_xlabel("Function domain(A)")
            elif plot=="current":
                axis.set_xlabel("Nondim voltage")
        if i==0:
            if plot=="domain":
                axis.set_ylabel("Residual value")
            elif plot=="current":
                axis.set_xlabel("Nondim current")
plt.show()"""
"""for i in range(0, len(vals)):
    if len(vals[i])>len(norm_vals):
        cyt.def_optim_list([ "E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv","gamma", "alpha"])
    else:
        cyt.def_optim_list(norm_vals)
    test=cyt.test_vals(vals[i], "timeseries")
    plt.subplot(2,2,i+1)
    if i==1:
        plt.ylim([-2, 2])
    if i>1:
        plt.ylim([-1, 1])
    plt.plot(voltage_results1, cyt.i_nondim(test)*1e6)
    plt.plot(voltage_results1, plot_results[i]*1e6)
    plt.xlabel("voltage(V)")
    plt.ylabel("Current($\\mu$A)")
plt.show()"""
cdl_vals=[0.0005113312954312908, 0.032688180215297846, -0.0016959746570033296, -4.363761285119949e-05, 5.118104508670231, -0.2856860853954224, -0.012643633142239707, -0.0001955648458022985]
cdl_params=["Cdl","CdlE1", "CdlE2", "CdlE3","Cdlinv","CdlE1inv", "CdlE2inv", "CdlE3inv"]
for i in range(0, len(cdl_vals)):
    cyt.dim_dict[cdl_params[i]]=cdl_vals[i]
cyt.def_optim_list([ "E_0", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma", "alpha"])
cyt.simulation_options["test"]=False
cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["adaptive_ru"]=True
cmaes_problem=pints.SingleOutputProblem(cyt, time_results, current_results)
score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
#x0=cyt.change_norm_group(ifft_vals, "norm")
cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
cmaes_fitting.set_parallel(False)
found_parameters, found_value=cmaes_fitting.run()
print(found_parameters)
cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
print(list(cmaes_results))
cmaes_time=cyt.test_vals(cmaes_results, likelihood="timeseries", test=False)

plt.plot(voltage_results, cmaes_time)
plt.plot(voltage_results, current_results, alpha=0.5)

plt.show()
