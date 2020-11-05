from single_e_anal import analytical_electron
import sys
sys.path.append('..')
from single_e_class_unified import single_electron
from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt
freq_range=[5*(10**x) for x in np.arange(0,7)]
time_ends=[10.0, 5.0,0.7, 0.07, 0.015, 0.015]
num_freqs=len(freq_range)
plt.figure(num=None, figsize=(9,10), dpi=120, facecolor='w', edgecolor='k')
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))
for lcv_1 in range(0, num_freqs):
    param_list={
        'E_start': -0.25, #(starting dc voltage - V)
        'E_reverse': 0.25,    #  (reverse dc voltage - V)
        'omega':freq_range[lcv_1],#8.88480830076,  #    (frequency Hz)
        'd_E': 0.25,   #(ac voltage amplitude - V) freq_range[j],#
        'original_omega':freq_range[lcv_1],
          #       (scan rate s^-1)
        'area': 0.03, #(electrode surface area cm^2)
        'Ru': 0.0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-6*0, #(capacitance parameters)
        'CdlE1': 0,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        'CdlE3': 0,#1.10053945995e-06,
        'gamma': 9.5e-12,
        'original_gamma':9.5e-12,      # (surface coverage per unit area)
        'E_0': 0.1,      #       (reversible potential V)
        'k_0': 1000, #(reaction rate s-1)
        'alpha': 0.55,
        'sampling_freq' : (1.0/400),
        'phase' : 0,
        'time_end':-1,
        "num_peaks":30
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
    anal=analytical_electron(param_list, 1.0)
    time_range=np.linspace(0,30/param_list["omega"], int(30*(1/param_list["sampling_freq"])))
    non_dim_time_range=time_range/anal.nd_param.c_T0
    #non_dim_time_range=np.multiply(time_range, param_list['k_0'])
    time_len=len(time_range)
    i_val=np.zeros(time_len)
    e_val=np.zeros(time_len)
    for j in range(0, time_len):
        i_val[j]=anal.i(non_dim_time_range[j])
        e_val[j]=anal.e(non_dim_time_range[j])

    non_dim_time_range=time_range

    dim_i=np.multiply(i_val, anal.nd_param.c_I0)
    #plt.plot(non_dim_time_range, i_val)
    #plt.show()
    nondim_i=anal.nd_param.c_I0
    #plt.subplot(2, 3, i+1)

    f=np.fft.fftfreq(time_len, anal.nd_param.sampling_freq)
    #Y=np.fft.fft(i_val)
    harmonic_range=np.arange(1,10,1)
    numerical=single_electron(None, param_list, simulation_options, other_values)
    synthetic_data=numerical.test_vals([], "timeseries")
    abserr = 1.0e-8
    relerr = 1.0e-6
    numpoints = len(numerical.time_vec)
    w0 = [synthetic_data[0],1, anal.e(0)]
    """wsol = odeint(numerical.current_ode_sys, w0, numerical.time_vec,
                  atol=abserr, rtol=relerr)"""

    synthetic_data=numerical.i_nondim(synthetic_data)
    plt.plot(numerical.t_nondim(numerical.time_vec), synthetic_data)
    #plt.plot(time_range, i_val)
    #plt.plot(numerical.define_voltages(), synthetic_data)
    plt.plot(time_range, dim_i)
    interped_anal=np.interp(numerical.t_nondim(numerical.time_vec), time_range, dim_i)
    plt.title(rmse(synthetic_data, interped_anal)/np.mean([np.mean(abs(interped_anal)), np.mean(abs(synthetic_data))]))
    plt.show()
    #nd_dict=vars(param_list)
    #keys=nd_dict.keys()
    #for i in range(0, len(keys)):
    #    print keys[i], param_list[keys[i]]
        #print nd_dict[keys[i]]
    #plt.axhline(max(i_val[time_len/2:]), color="black", linestyle="--")
    #plt.axhline(max(synthetic_data[time_len/2:]), color="black", linestyle="--")
"""    plt.subplot(2, num_freqs/2, lcv_1+1)
    plt.plot(time_range, i_val*1000, label="analytical")#1.1245022593473897
    plt.plot(time_range, synthetic_data*1000, label="numerical")
    percent_diff=abs(peak_ratio-peak_ratio_inv)*100
    #plt.plot(np.subtract(i_val, synthetic_data))
    plt.title(str(freq_range[lcv_1])+ "    "+"$\Delta$=" + str(round(percent_diff,3))+ "%")
    plt.legend()
    plt.xlabel('Time(s)')
    plt.ylabel('Current(mA)')
plt.subplots_adjust(left=0.05,right=0.98, bottom=0.05, top=0.95, wspace=0.27)
plt.show()"""
