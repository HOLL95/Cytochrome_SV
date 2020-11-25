from single_e_anal import analytical_electron
import sys
sys.path.append('..')
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt



k0_vals=[10**x for x in range(-1, 5)]

num_oscillations=[10, 50, 100, 200, 300]
time_ends=[10.0, 5.0,0.7, 0.07, 0.015, 0.015]
num_freqs=5
T=(273+25)
F=96485.3328959
R=8.314459848
plt.figure(num=None, figsize=(9,10), dpi=120, facecolor='w', edgecolor='k')
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))
def approx_error(y, y_true):
    subtractor=abs(np.subtract(y, y_true))
    denom=np.max([np.abs(y), np.abs(y_true)], axis=0)
    divisor=np.divide(subtractor, denom)
    divisor[np.where(denom==0)]=0
    #plt.plot(y)
    #plt.plot(y_true)
#    plt.plot(np.subtract(y, y_true))
    #plt.show()
    return np.mean(divisor)
d_E_vals=[3*((R*T)/F), 5*((R*T)/F)]
desired_harms=list(range(2, 5))

error=np.zeros(num_freqs)
for lcv_0 in range(0, len(k0_vals)):
    log_k=np.log10(k0_vals[lcv_0])
    freq_range=[5*(10**x) for x in np.linspace(log_k-1, log_k+3, num_freqs)]
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
            'Ru': 0.0,  #     (uncompensated resistance ohms)
            'Cdl': 1e-6*0, #(capacitance parameters)
            'CdlE1': 0,#0.000653657774506,
            'CdlE2': 0,#0.000245772700637,
            'CdlE3': 0,#1.10053945995e-06,
            'gamma': 9.5e-12,
            'original_gamma':9.5e-12,      # (surface coverage per unit area)
            'E_0': 0.1,      #       (reversible potential V)
            'k_0': k0_vals[lcv_0], #(reaction rate s-1)
            'alpha': 0.55,
            'sampling_freq' : (1.0/2000),
            'phase' : 0,
            'time_end':-1,
            "num_peaks":50
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
        dim_i=np.multiply(i_val, anal.nd_param.c_I0)
        nondim_i=anal.nd_param.c_I0
        harmonic_range=np.arange(1,10,1)
        numerical=single_electron(None, param_list, simulation_options, other_values)
        synthetic_data=numerical.i_nondim(numerical.test_vals([], "timeseries"))
        numerical_time=numerical.t_nondim(numerical.time_vec)
        interped_anal=np.interp(numerical_time, time_range, dim_i)
        plt.plot(interped_anal)
        plt.plot(synthetic_data)
        plt.show()
        error[lcv_1]=approx_error(interped_anal, synthetic_data)#(rmse(interped_anal, synthetic_data)/rmse(synthetic_data, np.mean(synthetic_data)))*100#(max([max(interped_anal), max(synthetic_data)])))*100

        """plt.subplot(1, len(freq_range), lcv_1+1)
        plt.title(error[lcv_1])
        plt.plot(numerical_time, interped_anal, label=param_list["k_0"])
        plt.plot(numerical_time, synthetic_data)
        plt.legend()
        plt.show()"""
    print(error)
    plt.loglog(freq_range, error, label="$k^0=$"+str(k0_vals[lcv_0]))
plt.legend()

#ax-plt.gca()
plt.xlabel("Frequency(Hz)")
#ax.get_xticks()

plt.ylabel("Percentage error")
fig=plt.gcf()
fig.set_size_inches((3.25, 4.5))
plt.show()
fig.savefig("Bell_error", dpi=500)
