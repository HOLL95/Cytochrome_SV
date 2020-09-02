import numpy as np
import matplotlib.pyplot as plt
plot=True
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
from matplotlib.ticker import FormatStrFormatter
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/Ramped"
files=os.listdir(data_loc)
harm_range=list(range(1,9,1))
num_harms=len(harm_range)+1
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
regime="reversible"
regime="irreversible"
for regime in ["reversible", "irreversible"]:
    figure=multiplot(3, 2, **{"distribution_position":list(range(0,3)), "num_harmonics":3, "orientation":"landscape",  "plot_width":5, "row_spacing":1,"col_spacing":2, "plot_height":1})
    for j in range(2, 4):
        figure.axes_dict["row3"][j].set_axis_off()
    if regime=="reversible":
        scale_scan_vals=np.flip([50, 100, 500])
        k0_val=100
    elif regime=="irreversible":
        scale_scan_vals=np.flip([0.5, 1, 2])
        k0_val=1
    params={"E_0":{"E0_mean":[-0.2, -0.25, -0.3],
                    "E0_std":[0.03,0.05 , 0.07],
                    "E0_skew":[-5, 0, 5]},
            "k_0":{"k0_shape":([0.25, 0.5, 0.65]),
                    "k0_scale":scale_scan_vals}}
    flattened_list=list(flatten(params))
    total_length=len(flattened_list)
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
        'k_0': '$k_0', #(reaction rate s-1)
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
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
        'omega':8.881077023434541,#8.88480830076,  #    (frequency Hz)
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
        'k_0': k0_val, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":-0.25,
        "E0_std": 0.05,
        "E0_skew":0,
        "k0_shape":0.25,
        "k0_scale":k0_val,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :0.0,
        "time_end": None,
        'num_peaks': 30,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=1/(param_list["omega"])
    simulation_options={
        "no_transient":time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":16,
        "test": False,
        "method": "ramped",
        "phase_only":False,
        "likelihood":likelihood_options[1],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }
    other_values={
        "filter_val": 0.5,
        "harmonic_range":harm_range,
        "experiment_time": time_results1,
        "experiment_current": current_results1,
        "experiment_voltage":voltage_results1,
        "bounds_val":20000,
    }
    param_bounds={
        'E_0':[param_list['E_start'],param_list['E_reverse']],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1000],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.05,0.15],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],100*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[param_list['E_start'],param_list['E_reverse']],
        "E0_std": [1e-4,  0.2],
        "E0_skew":[-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        "k0_range":[1e2, 1e4],
        'phase' : [0, 2*math.pi],
    }
    table_params=["E_start", "E_reverse", "E_0","k_0","Ru","Cdl","gamma", "alpha", "v", "omega", "phase","d_E","sampling_freq", "area", "E0_mean", "E0_std", "E0_skew", "k0_shape", "k0_scale"]

    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    print([cyt.dim_dict[x] if x in cyt.dim_dict else 0 for x in table_params])
    original_params=list(params.keys())
    table_params=["E_0","k0_scale", "k0_shape","Ru","Cdl","CdlE1","gamma","phase", "alpha"]
    print([param_list[x] for x in table_params])
    cyt.def_optim_list(original_params)
    no_disp=cyt.test_vals([0.2, 100], "timeseries")
    harms=harmonics(cyt.other_values["harmonic_range"], cyt.dim_dict["omega"], 0.5)
    time_results=cyt.t_nondim(cyt.other_values["experiment_time"])
    current_results=cyt.i_nondim(cyt.other_values["experiment_current"])
    voltage_results=cyt.e_nondim(cyt.other_values["experiment_voltage"])
    no_disp_harmonics=harms.generate_harmonics(time_results, cyt.i_nondim(no_disp))
    cyt.simulation_options["dispersion_bins"]=[16]
    cyt.simulation_options["GH_quadrature"]=False
    def e0_re_dim(arg):
        return np.multiply(arg, cyt.nd_param.c_E0)
    def k0_re_dim(arg):
        return np.divide(arg, cyt.nd_param.c_T0)
    plot_locs={"E_0":{"E0_mean":{"row":"row1", "col":0, "col_end":False, "func":e0_re_dim},
                    "E0_std":{"row":"row2", "col":0, "col_end":False, "func":e0_re_dim},
                    "E0_skew":{"row":"row3", "col":0, "col_end":True, "func":e0_re_dim}},
            "k_0":{"k0_shape":{"row":"row2", "col":1, "col_end":True, "func":k0_re_dim},
                    "k0_scale":{"row":"row1", "col":1, "col_end":False, "func":k0_re_dim}}}
    for parameter in original_params:
        disped_params=list(params[parameter].keys())
        updated_optim_list=copy.deepcopy(original_params)
        param_idx=original_params.index(parameter)
        updated_optim_list.pop(param_idx)
        updated_optim_list.extend(disped_params)
        if "E0_mean" not in updated_optim_list:
            cyt.simulation_options["GH_quadrature"]=False
        else:
            cyt.simulation_options["GH_quadrature"]=False
        cyt.def_optim_list(updated_optim_list)
        for disp_param in disped_params:
            j+=1
            sim_params=[param_list[key] for key in updated_optim_list]
            param_loc=updated_optim_list.index(disp_param)
            for i in range(0, len(params[parameter][disp_param])):
                sim_params[param_loc]=params[parameter][disp_param][i]
                syn_time=cyt.i_nondim(cyt.test_vals(sim_params, "timeseries"))*1e3
                func=plot_locs[parameter][disp_param]["func"]
                values, weights=cyt.return_distributions(1000)
                values=func(values)
                disp_ax=figure.axes_dict[plot_locs[parameter][disp_param]["row"]][(2*plot_locs[parameter][disp_param]["col"])]
                if parameter=="k_0":
                    disp_ax.text(0.1,1.9, fancy_names[disp_param], transform=disp_ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')
                else:
                    disp_ax.text(-0.03,1.9, fancy_names[disp_param], transform=disp_ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')
                disp_ax.plot(values, weights, label=str(round(params[parameter][disp_param][i], 3))+" "+str(unit_dict[disp_param]))
                disp_ax.xaxis.tick_top()
                disp_ax.set_xlabel(fancy_names[parameter]+"("+unit_dict[parameter]+")")
                disp_ax.xaxis.set_label_position('top')
                disp_ax.set_ylabel("PD")
                axis_position=1+(2*plot_locs[parameter][disp_param]["col"])
                axis_row=figure.axes_dict[plot_locs[parameter][disp_param]["row"]]
                current_ax=axis_row[axis_position]
                times=cyt.t_nondim(cyt.time_vec[cyt.time_idx])
                syn_harmonics=harms.generate_harmonics(times, cyt.i_nondim(syn_time*1e6), hanning=True)
                y_plot=[max(abs(syn_harmonics[i,:])) for i in range(0, len(syn_harmonics))]
                current_ax.semilogy(harm_range, y_plot)# alpha=1-(0.1*i)
                current_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                current_ax.set_ylabel("Current($mA$)")
                #current_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))



                if plot_locs[parameter][disp_param]["col_end"]==True:
                    current_ax.set_xlabel("Voltage(V)")
                else:
                    current_ax.set_xticklabels([])
            disp_xlim=disp_ax.get_xlim()
            disp_ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            disp_ax.set_xlim(disp_xlim[0], disp_xlim[1]+(abs(disp_xlim[1]-disp_xlim[0])*0.5))
            disp_ax.legend(loc="center right", fontsize=8, frameon=False)


    fig=plt.gcf()
    plt.subplots_adjust(top=0.94,
                        bottom=0.06,
                        left=0.12,
                        right=0.975,
                        hspace=0.115,
                        wspace=0.6)
    fig.set_size_inches((7, 9))
    save_path="SV_parameter_scans_"+regime+".png"
    plt.show()
    fig.savefig(save_path, dpi=500)
    plt.clf()






    #ax.plot(time_results, abs(syn_harmonics[i,:]), label="Sim")
