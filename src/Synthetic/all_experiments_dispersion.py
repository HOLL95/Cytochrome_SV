import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
plot=True
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import math
import copy
import time
import pints
from single_e_class_unified import single_electron
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Ramped"
files=os.listdir(data_loc)
harm_range=list(range(3,8,2))
num_harms=len(harm_range)
scan="_1_"
freq="_9Hz"
dec_amount=1
def flatten(something):
    if isinstance(something, (list, tuple, set, range)):
        for sub in something:
            yield from flatten(sub)
    elif isinstance(something, dict):
        for key in something.keys():
            yield from flatten(something[key])
    else:
        yield something




regime="reversible"

figure=multiplot(3,3, **{"harmonic_position":0, "num_harmonics":num_harms, "orientation":"landscape",  "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":2, "plot_height":1, "harmonic_spacing":0})
inset_plot_row="row3"
extra_plot_list=[]
for i in range(1, 3):
    extra_plot_list.append( inset_axes(figure.axes_dict[inset_plot_row][i], width="35%", height="35%", loc=3, borderpad=1.35))
    for label in (extra_plot_list[i-1].get_xticklabels() + extra_plot_list[i-1].get_yticklabels()):
        label.set_fontsize(6)
    extra_plot_list[i-1].tick_params(direction='in', length=1, width=1)


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
    'k_0': 's^{-1}$', #(reaction rate s-1)
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
    'E0_skew':"$E^0 \\alpha$",
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

RV_param_list={
    "E_0":-0.25,
    'E_start':  -500e-3, #(starting dc voltage - V)
    'E_reverse':100e-3,
    'omega':10,  #    (frequency Hz)
    "v":    22.35174e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 0.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "k0_scale":10,
    "k0_shape":0.6,
    "E0_mean":-0.25,
    "E0_std": 0.05,
    "E0_skew":0,
    "cap_phase":4.712388,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :4.712388,
}
table_params=RV_param_list.keys()
disped_params=["E0_mean", "E0_std", "E0_skew","k0_scale", "k0_shape",
                    "alpha_mean", "alpha_std"]

param_bounds={
    "E_start":[-2, 2],
    "E_reverse":[0, 4],
    "area":[1e-5, 0.1],
    "sampling_freq":[1/1000.0, 1/10.0],
    'E_0':[-2, 4],
    'omega':[1, 1e5],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 5e5],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-3], #(capacitance parameters)
    'CdlE1': [-0.01,0.01],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-4*RV_param_list["original_gamma"],1e4*RV_param_list["original_gamma"]],
    'k_0': [0.1, 1e6], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    'phase' : [0, 2*math.pi],
    "cap_phase":[0, 2*math.pi],
    "E0_mean":[-2,4],
    "E0_std": [1e-4,  1],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],

}
time_start=2/(RV_param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "ramped",
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[]
}

dcv_simulation_options={
    "no_transient":0.00002,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "dcv",
    "phase_only":False,
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[]
}
table_dict={key:RV_param_list[key] for key in param_bounds.keys()}
orig_table_dict=copy.deepcopy(table_dict)
#for param in forbidden_params:
#    del table_dict[param]
table_data={key:table_dict[key] for key in table_dict.keys()}
table_names=list(table_dict.keys())
SV_simulation_options=copy.deepcopy(simulation_options)
DCV_simulation_options=copy.deepcopy(simulation_options)
DCV_simulation_options["method"]="dcv"

SV_param_list=copy.deepcopy(RV_param_list)
DCV_param_list=copy.deepcopy(RV_param_list)
DCV_param_list["v"]=1
changed_SV_params=["d_E", "phase", "cap_phase", "num_peaks", "original_omega", "sampling_freq"]
changed_sv_vals=[300e-3, 3*math.pi/2,  3*math.pi/2, 25, RV_param_list["omega"], 1/200.0]
for key, value in zip(changed_SV_params, changed_sv_vals):
    SV_param_list[key]=value
disp_bin_val=20
SV_simulation_options["method"]="sinusoidal"
other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "bounds_val":20000,
}
get_colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
RV=single_electron(None, RV_param_list, simulation_options, other_values, param_bounds)
SV=single_electron(None, SV_param_list, SV_simulation_options, other_values, param_bounds)
DCV=single_electron(None, DCV_param_list, dcv_simulation_options, other_values, param_bounds)
dispersion_groups={"E_0":["E0_mean", "E0_std", "E0_skew"], "k_0":["k0_shape", "k0_scale"], "alpha":["alpha_mean", "alpha_std"]}
parameters=[["E_0"], ["k_0"], ["E_0", "k_0"]]
labels=["$E^0$", "$k_0$", "$E^0, k_0$"]
exp_keys=["ramped", "sinusoidal", "dcv"]
exp_class_dict=dict(zip(exp_keys, [RV, SV, DCV]))
DCV_scan_rate=np.power(10,np.linspace(-2.5, 0, 30))
def trumpet_plots(old_class, DCV_scan_rate, sim_params):
    peak_pos=np.zeros((2, len(DCV_scan_rate)))
    for q in range(0, len(DCV_scan_rate)):
        new_param_list=copy.deepcopy(old_class.dim_dict)
        new_param_list["v"]=DCV_scan_rate[q]
        new_sim_options=copy.deepcopy(old_class.simulation_options)
        new_sim_options["no_transient"]=False
        new_param_list["sampling_freq"]=new_param_list["sampling_freq"]/DCV_scan_rate[q]
        DCV_new=single_electron(None, new_param_list, new_sim_options, old_class.other_values, old_class.param_bounds)
        DCV_new.def_optim_list(old_class.optim_list)
        syn=DCV_new.test_vals(sim_params, "timeseries")
        syn_time=DCV_new.i_nondim(syn)*1e3
        volts=DCV_new.e_nondim(DCV_new.define_voltages()[DCV_new.time_idx])
        peak_pos[0][q]=DCV_new.e_nondim(volts[np.where(syn_time==max(syn_time))])
        peak_pos[1][q]=DCV_new.e_nondim(volts[np.where(syn_time==min(syn_time))])
    return peak_pos
for i in range(1, 2):

    experiment_type=exp_keys[i]
    current_class=exp_class_dict[experiment_type]
    longs="~"*50
    ts=current_class.test_vals([], "timeseries")
    non_disped_time=current_class.i_nondim(ts)
    if experiment_type=="dcv":
        non_disped_trumpet=trumpet_plots(current_class, DCV_scan_rate, [])
        non_disped_ts_plot=current_class.i_nondim(ts)
        non_disped_volt_dcv=current_class.e_nondim(current_class.define_voltages()[current_class.time_idx])
    if experiment_type=="ramped":
        times=current_class.t_nondim(current_class.time_vec[current_class.time_idx])
        harms=harmonics(harm_range, current_class.dim_dict["omega"], 0.05)
        non_disped_harmonics=harms.generate_harmonics(times, non_disped_time, hanning=True)
    for j in range(0, len(parameters)):
        optim_list=list(flatten([dispersion_groups[key] for key in parameters[j]]))
        current_class.simulation_options["dispersion_bins"]=[disp_bin_val]*len(parameters[j])
        current_class.def_optim_list(optim_list)
        params=[RV_param_list[key] for key in optim_list]
        start=time.time()
        syn_time=current_class.i_nondim(current_class.test_vals(params, "timeseries"))
        print(time.time()-start)
        print(experiment_type, longs)
        if experiment_type=="dcv":
            disped_trumpet=trumpet_plots(current_class, DCV_scan_rate, params)
            disped_volt_dcv=current_class.e_nondim(current_class.define_voltages()[current_class.time_idx])
        if parameters[j]==["E_0"]:
            e0_plot=syn_time
            if experiment_type=="dcv":
                e0_trumpet=disped_trumpet
        if experiment_type=="ramped":
            times=current_class.t_nondim(current_class.time_vec[current_class.time_idx])
            harms=harmonics(harm_range, current_class.dim_dict["omega"], 0.05)
            syn_harmonics=harms.generate_harmonics(times, syn_time, hanning=True)
            if parameters[j]==["E_0"]:
                e0_harms=syn_harmonics
            for q in range(0, len(syn_harmonics)):
                current_ax=figure.axes_dict["row1"][q+(j*len(harm_range))]
                current_ax.plot(times, abs(non_disped_harmonics[q, :]*1e6))
                ticks= current_ax.get_yticks()
                current_ax.set_yticks([ticks[1], ticks[-2]])
                current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                current_ax.text(0.9, 0.75,harm_range[q],
                                                                    horizontalalignment='center',
                                                                    verticalalignment='center',
                                                                    transform = current_ax.transAxes,
                                                                    fontsize=10)
                if q==len(syn_harmonics)//2 and j==0:
                    current_ax.set_ylabel("Current($\\mu$ A)")

                if q==len(syn_harmonics)-1:
                        current_ax.set_xlabel("Time(s)")
                if parameters[j]==["E_0", "k_0"]:
                    current_ax.plot(times, abs(e0_harms[q, :]*1e6))
                if parameters[j]==["k_0"]:
                    current_ax.plot(times, abs(syn_harmonics[q, :]*1e6), color="red")
                else:
                    current_ax.plot(times, abs(syn_harmonics[q, :]*1e6))
        elif experiment_type=="dcv" and parameters[j]!=["E_0"]:
            current_ax=figure.axes_dict["row"+str(i+1)][j]
            extra_plot_list[j-1].plot(non_disped_volt_dcv, non_disped_ts_plot*1e6)
            extra_plot_list[j-1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            if parameters[j]==["k_0"]:
                extra_plot_list[j-1].plot(disped_volt_dcv, syn_time*1e6, color="red")
            plot_vs=np.log10(DCV_scan_rate)
            current_ax.scatter(plot_vs, non_disped_trumpet[0,:]*1e3, color=get_colours[0], marker="D", s=5, label="Oxidation V",facecolors='none')
            current_ax.scatter(plot_vs, non_disped_trumpet[1,:]*1e3,color=get_colours[0], s=5, label="Reduction V")
            current_ax.set_ylim([min(non_disped_trumpet[1,:])*1.05*1e3, max(non_disped_trumpet[0,:])*0.95*1e3])
            leg=current_ax.legend(loc="upper left",fontsize=8, frameon=False)
            if parameters[j]==["E_0", "k_0"]:
                extra_plot_list[j-1].plot(disped_volt_dcv,  e0_plot*1e6)
                extra_plot_list[j-1].plot(disped_volt_dcv, syn_time*1e6)
                current_ax.scatter(plot_vs, e0_trumpet[0,:]*1e3, color=get_colours[1], marker="D", s=5,facecolors='none')
                current_ax.scatter(plot_vs, e0_trumpet[1,:]*1e3,color=get_colours[1], s=5)
                current_ax.scatter(plot_vs, disped_trumpet[0,:]*1e3,color=get_colours[2], marker="D", s=5,facecolors='none')
                current_ax.scatter(plot_vs, disped_trumpet[1,:]*1e3,color=get_colours[2], s=5)
            else:
                current_ax.scatter(plot_vs, disped_trumpet[0,:]*1e3,color="red", marker="D", s=5,facecolors='none')
                current_ax.scatter(plot_vs, disped_trumpet[1,:]*1e3,color="red", s=5)#

            current_ax.set_xlabel("Log(v)")
            current_ax.set_ylabel("Peak position(mV)")
            current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        else:
            volts=current_class.e_nondim(current_class.define_voltages()[current_class.time_idx])
            current_ax=figure.axes_dict["row"+str(i+1)][j]
            if experiment_type=="dcv":
                factor=1e6
            else:
                factor=1e6
            current_ax.plot(volts, non_disped_time*factor, label="None")
            ticks= current_ax.get_yticks()
            current_ax.set_yticks([ticks[1], ticks[len(ticks)//2],ticks[-2]])
            current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            current_ax.set_xlabel("Potential(V)")
            if j==0:
                current_ax.set_ylabel("Current($\\mu$A)")
            if parameters[j]==["E_0", "k_0"]:
                current_ax.plot(volts, e0_plot*factor, label="$E^0$")
            if parameters[j]==["k_0"]:
                current_ax.plot(volts, syn_time*factor,color="red", label=labels[j])
            else:
                current_ax.plot(volts, syn_time*factor, label=labels[j])
            if experiment_type=="sinusoidal":
                xmin, xmax =current_ax.get_xlim()
                current_ax.set_xlim([xmin, xmax+((xmax-xmin)*0.5)])
                current_ax.legend(loc="upper right",handlelength=0.65, frameon=False, bbox_to_anchor=(1.05, 1.0))
plt.subplots_adjust(top=0.985,
                    bottom=0.085,
                    left=0.11,
                    right=0.985,
                    hspace=0.1,
                    wspace=0.15)
fig=plt.gcf()
fig.set_size_inches((7, 7))
save_path="all_experiments.png"
plt.show()
fig.savefig(save_path, dpi=500)






#ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
