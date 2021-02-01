from single_e_anal import analytical_electron
import sys
sys.path.append('..')
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


k0_vals=[10**x for x in range(-1, 5)]

num_oscillations=[10, 50, 100, 200, 300]
time_ends=[10.0, 5.0,0.7, 0.07, 0.015, 0.015]
num_freqs=6
T=(273+25)
F=96485.3328959
R=8.314459848
plt.figure(num=None, figsize=(9,10), dpi=120, facecolor='w', edgecolor='k')
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))
def pc_error(y, y_pred):
    no_0_idx=np.where(y!=0)
    return np.mean(abs(np.subtract(y_pred[no_0_idx], y[no_0_idx]))/(abs(y[no_0_idx])))*100
def approx_error(y, y_true):
    subtractor=abs(y-y_true)
    denom=abs(y_true)
    zero_loc=np.where(denom!=0)
    divisor=np.divide(subtractor[zero_loc], denom[zero_loc])
    return np.mean(divisor)
d_E_vals=[3*((R*T)/F), 5*((R*T)/F)]
desired_harms=list(range(2, 5))
error=np.zeros(num_freqs)
k_val=1
for lcv_0 in [1]:

    log_k=np.log10(k_val)
    print(log_k)
    freq_range=[(10**x) for x in np.linspace(log_k+5, log_k+6, num_freqs)]

    for lcv_1 in range(0, num_freqs):
        peak_heights=np.zeros((2, len(desired_harms)))
        estart=-0.25
        erev=0.25
        param_list={
            'E_start':  estart, #(starting dc voltage - V)
            'E_reverse':  erev ,   #  (reverse dc voltage - V)
            'omega':freq_range[lcv_1],#8.88480830076,  #    (frequency Hz)
            'd_E': 0.25,   #(ac voltage amplitude - V) freq_range[j],#
            'original_omega':freq_range[lcv_1],
              #       (scan rate s^-1)
            'area': 0.03, #(electrode surface area cm^2)
            'Ru': 1.0,  #     (uncompensated resistance ohms)
            'Cdl': 1e-5, #(capacitance parameters)
            'CdlE1': 0,#0.000653657774506,
            'CdlE2': 0,#0.000245772700637,
            'CdlE3': 0,#1.10053945995e-06,
            'gamma': 1e-10,
            'original_gamma':1e-10,      # (surface coverage per unit area)
            'E_0': 0,      #       (reversible potential V)
            'k_0': k_val, #(reaction rate s-1)
            'alpha': 0.5,
            'sampling_freq' : (1.0/200),
            'phase' : 0,
            'time_end':-1,
            "num_peaks":20
        }
        simulation_options={
            "no_transient":False,
            "numerical_debugging": False,
            "experimental_fitting":False,
            "dispersion":False,
            "dispersion_bins":None,
            "test": False,
            "method": "sinusoidal",
            "phase_only":True,
            "likelihood":"timeseries",
            "numerical_method": "Brent minimisation",
            "label": "MCMC",
            "optim_list":[]
        }
        other_values={
            "filter_val": 0.5,
            "harmonic_range":list(range(3, 8)),
            "bounds_val":20000,
        }
        anal=analytical_electron(param_list, 0.0)
        time_range=np.linspace(0,param_list["num_peaks"]/param_list["omega"], int(param_list["num_peaks"]*(1/param_list["sampling_freq"])))
        non_dim_time_range=time_range/anal.nd_param.c_T0
        time_len=len(time_range)
        i_val=np.zeros(time_len)
        e_val=np.zeros(time_len)
        for j in range(0, time_len):
            i_val[j]=anal.i(non_dim_time_range[j])
            e_val[j]=anal.e(non_dim_time_range[j])
        non_dim_time_range=time_range
        wsol = odeint(anal.diff_eq_gamma, 0, non_dim_time_range)

        scipy_current=np.zeros(len(wsol))
        for i in range(0, len(wsol)):
            scipy_current[i]=anal.diff_eq_gamma(wsol[i], non_dim_time_range[i])
        #plt.subplot(2,3,lcv_1+1)
        dim_scipy=np.multiply(scipy_current, anal.nd_param.c_I0)
        dim_i=np.multiply(i_val, anal.nd_param.c_I0)
        nondim_i=anal.nd_param.c_I0
        harmonic_range=np.arange(1,10,1)
        numerical=single_electron(None, param_list, simulation_options, other_values, {})

        synthetic_data=numerical.test_vals([], "timeseries")
        print(len(synthetic_data))
        scipy_sol= odeint(numerical.current_ode_sys, [scipy_current[0], 0, e_val[0]],numerical.time_vec)

        sci_current=numerical.i_nondim(scipy_sol[:,0])
        numerical_time=numerical.t_nondim(numerical.time_vec)
        interped_anal=np.interp(numerical_time, time_range, dim_i)
        interped_numeric=np.interp(time_range, numerical_time, synthetic_data)
        plt.subplot(2,3,lcv_1+1)
        plt.plot(numerical_time, synthetic_data, label="Numerical (Brent method)")

        #plt.plot(numerical_time, interped_anal, label="Analytical")
        plt.plot(numerical_time,scipy_sol[:,0] , label="Numerical (adaptive method)", color="red", linestyle="--")

        #print(anal.nd_param.nd_omega)
        title=pc_error(dim_scipy, synthetic_data)#rmse(dim_i, interped_numeric)/np.mean(abs(synthetic_data))
        plt.title(title)
        plt.xlabel("Time(s)")
        plt.ylabel("Current(A)")
        if lcv_0==4:
            plt.legend()

#ax-plt.gca()
plt.show()
