from single_e_anal import analytical_electron
import sys
sys.path.append('..')
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt
freq_range=[5*(10**x) for x in np.arange(1,2)]
freq_range=[50]
noises=np.linspace(-3, -1, 20)
noises=[x/100 for x in [0.5, 1, 2, 3, 4, 5]]

num_oscillations=[10, 50, 100, 200, 300]
time_ends=[10.0, 5.0,0.7, 0.07, 0.015, 0.015]
num_freqs=len(freq_range)
T=(273+25)
F=96485.3328959
R=8.314459848
plt.figure(num=None, figsize=(9,10), dpi=120, facecolor='w', edgecolor='k')
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))
d_E_vals=[3*((R*T)/F), 5*((R*T)/F)]
desired_harms=list(range(2, 5))
estimate_array=np.zeros(len(noises))
estimate_stds=np.zeros(len(noises))
num_noise_repeats=1000
noise_repeat_array=np.zeros(num_noise_repeats)
big_array=[]
for lcv_0 in range(0, len(noises)):
    noise_repeat_array=np.zeros(num_noise_repeats)
    for noise_repeats in range(0, num_noise_repeats):
        for lcv_1 in range(0, num_freqs):
            peak_heights=np.zeros((2, len(desired_harms)))
            E_ins=[0,0]
            for lcv_2 in range(0, len(d_E_vals)):
                estart=E_ins[lcv_2]-d_E_vals[lcv_2]
                erev=E_ins[lcv_2]+d_E_vals[lcv_2]
                param_list={
                    'E_start':  estart, #(starting dc voltage - V)
                    'E_reverse':  erev ,   #  (reverse dc voltage - V)
                    'omega':freq_range[lcv_1],#8.88480830076,  #    (frequency Hz)
                    'd_E': d_E_vals[lcv_2],   #(ac voltage amplitude - V) freq_range[j],#
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
                    'E_0': 1*((R*T)/F),      #       (reversible potential V)
                    'k_0': 1, #(reaction rate s-1)
                    'alpha': 0.55,
                    'sampling_freq' : (1.0/400),
                    'phase' : 0,
                    'time_end':-1,
                    "num_peaks":300
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
                E_ins[lcv_2]=(estart+erev)/2
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
                #synthetic_data=numerical.i_nondim(numerical.test_vals([], "timeseries"))
                numerical_time=numerical.t_nondim(numerical.time_vec)
                #interped_anal=np.interp(numerical_time, time_range, dim_i)
                harms=harmonics(desired_harms, param_list["omega"], 0.1)
                noisy_i=numerical.add_noise(i_val, noises[lcv_0]*max(i_val))
                print("NOISES", noise_repeats)
                anal_harms=harms.generate_harmonics(time_range, noisy_i, hanning=True)
                for i in range(0, len(desired_harms)):
                    peak_heights[lcv_2, i]=max(abs(anal_harms[i,:]))
            e0_estimates=np.zeros(len(desired_harms))
            for i in range(0, len(desired_harms)):
                e0_estimates[i]=anal.nd_param_estimator(desired_harms[i], peak_heights[0,i], peak_heights[1, i], E_in_1=0, Delta1=3, eta1=-1, alpha=0.55, E_in_2=0, Delta2=5, eta2=-1)

        noise_repeat_array[noise_repeats]=np.mean(e0_estimates[~np.isnan(e0_estimates)])
        print(np.mean(e0_estimates[~np.isnan(e0_estimates)]))
    print(noise_repeat_array)
    big_array.append({str(noises[lcv_0]):noise_repeat_array})
print(big_array)
np.save("Parameter_inference_changing_noise", big_array)
