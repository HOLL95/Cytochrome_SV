import sys
sys.path.append("..")
import numpy as np
from multiplotter import multiplot
import matplotlib.pyplot as plt
import sys
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from matplotlib.ticker import FormatStrFormatter
import time
from single_e_class_unified import single_electron
from scipy.integrate import odeint
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/FTACV"
files=os.listdir(data_loc)
experimental_dict={}
harm_range=list(range(4,8,1))
num_harms=len(harm_range)
param_file=open(data_loc+"/FTACV_params", "r")
useful_params=dict(zip(["max", "start", "Amp[0]", "freq", "rate"], ["E_reverse", "E_start", "d_E", "omega", "v"]))
figure=multiplot(4,2, **{"harmonic_position":[0,1,2,3], "num_harmonics":num_harms, "orientation":"landscape",  "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":1, "plot_height":1, "harmonic_spacing":1})
print(figure.axes_dict)
dec_amount=64
for line in param_file:
    split_line=line.split()
    print(split_line)
    if split_line[0] in useful_params.keys():
        experimental_dict[useful_params[split_line[0]]]=float(split_line[1])
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]

for i in range(3, 4):
    file_name="FTACV_Cyt_{0}_cv_".format(i)
    current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  -0.33898177567074794, #(starting dc voltage - V)
        'E_reverse':0.26049326614698887,
        'omega':8.959320552342025, #8.88480830076,  #    (frequency Hz)
        "v":0.022316752195354346,
        'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        'phase' :6.283185307179562,
        "time_end": -1,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["omega"])
    simulation_options={
        "no_transient":False,#time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[15],
        "GH_quadrature":True,
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
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":200,
    }
    param_bounds={
        'E_0':[-0.1, 0.0],
        "E_start":[0.9*param_list["E_start"], 1.1*param_list["E_start"]],
        "E_reverse":[0.9*param_list["E_reverse"], 1.1*param_list["E_reverse"]],
        "v":[0.9*param_list["v"], 1.1*param_list["v"]],
        'omega':[0.8*param_list['omega'],1.2*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.05,0.05],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.1, 0.0],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [0, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.other_values["experiment_current"]
    print(current_results[0], current_results[-1])
    voltage_results=cyt.other_values["experiment_voltage"]
    h_class=harmonics(other_values["harmonic_range"], param_list["omega"]*cyt.nd_param.c_T0, 0.05)

    volts=cyt.define_voltages()



    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","omega","phase", "alpha"])
    ramped_inferred_0=[-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333, -0.004699728058449013,0, 2.1898117688472174e-11, 8.959294458587753, 0.9281447797610709, 0.5592126258301378]
    ramped_inferred_1=[-0.08250348574002123, 0.037628433600570534, 4.635568923834899, 153.10805911446752, 9.792321199260773e-05, -0.09249415046996813, 8.59651543872117e-05, 8.294496746195923e-05, 2.7796118957386284e-11,8.959294458587753, 0, 0.5999999844213646]
    ramped_inferred_2=[-0.08064031224498379, 0.020906827217859487, 63.01964378454537, 112.80693965100555, 0.0007989280702348236, -0.008716352699352406, 0.0012650098345725197, -2.60067208995296e-05, 2.9678863151432805e-11, 8.959294458587753, 0, 0.5999999698409333]
    ramped_inferred_3=[-0.05897077320642114, 0.012624609618582739, 569.8770986714504, 137.37161854510956, 0.0008889805390117373, 0.008340803598526805, 0.0010715780154989737, 4.964566158449646e-05, 5.2754243128435144e-11, 8.959294458587753, 0, 0.5819447228966595]
    ramped_fiddled_0=[-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333, -0.004699728058449013,0, 2.1898117688472174e-11, 8.959294458587753, 0.9281447797610709, 0.5592126258301378]
    ramped_fiddled_1=[-0.08250348574002123, 0.037628433600570534*0.3, 4.635568923834899, 153.10805911446752, 9.792321199260773e-05, -0.09249415046996813, 8.59651543872117e-05, 8.294496746195923e-05, 3*2.7796118957386284e-11, 8.959294458587753,0, 0.5999999844213646]
    ramped_fiddled_2=[-0.065, 0.020906827217859487, 63.01964378454537, 112.80693965100555, 0.0007989280702348236, -0.008716352699352406, 0.0012650098345725197, -2.60067208995296e-05, 2.9678863151432805e-11*1.3, 8.959294458587753, 0, 0.5999999698409333]
    ramped_fiddled_3=[-0.05897077320642114, 0.012624609618582739*1.9, 569.8770986714504, 137.37161854510956, 0.0008889805390117373, 0.008340803598526805, 0.0010715780154989737, 4.964566158449646e-05, 5.2754243128435144e-11*0.65, 8.959294458587753, 0, 0.5819447228966595]
    SV_params=[[ramped_inferred_0,ramped_inferred_1, ramped_inferred_2, ramped_inferred_3], [ramped_fiddled_0,ramped_fiddled_1, ramped_fiddled_2, ramped_fiddled_3]]
    SV_params=[[ramped_inferred_0,ramped_fiddled_0], [ramped_inferred_1,ramped_fiddled_1],[ramped_inferred_2,ramped_fiddled_2],[ramped_inferred_3,ramped_fiddled_3]]
    print(SV_params[1])
    #ramped_inferred_4=[-0.05744624908677006, 0.02788517403038323, 164.40170561479968, 268.7042102662674, 0.00019999999918567714, -0.01781717221653123, -0.0010262335333251996, 2.4414183646021532e-11, 8.959320552342025, 6.283184732454485, 0.40000000000073965]
    #ramped_inferred=[-0.07963988256472493, 0.043023349442016245, 20.550878790990733, 581.5147052074157, 8.806259570340898e-05, -0.045583168350011485, -0.00011159862236990309, 2.9947102043021914e-11, 8.959320552342025, 0, 0.5999994350046962]
    #
    titles=["Ramped inferred", "SV set 1", "SV set 2", "SV set 3"]
    cyt.simulation_options["method"]="dcv"
    dcv_volt=cyt.e_nondim(cyt.define_voltages()[cyt.time_idx])
    cyt.simulation_options["method"]="ramped"
    experimental_harmonics=h_class.generate_harmonics(time_results, current_results,hanning=True)
    figure_keys=list(figure.axes_dict.keys())
    title_counter=0
    for j in range(0, len(figure_keys)):
        row_position=0
        for q in range(0, len(SV_params[j])):
            cmaes_test=cyt.test_vals(SV_params[j][q], "timeseries")
            synthetic_harmonics=h_class.generate_harmonics(time_results, cmaes_test, hanning=True)
            for k in range(0, num_harms):
                print(row_position)

                ax=figure.axes_dict[figure_keys[j]][row_position]

                if k%num_harms==0:
                    if "Ramped" not in titles[j]:
                        if row_position!=0:
                            ax.set_title(titles[j]+ "(Adjusted)")
                        else:
                            ax.set_title(titles[j])
                    else:
                        ax.set_title(titles[j])

                if j==len(figure_keys)-1:
                    if k==num_harms-1:
                        ax.set_xlabel("Nondimensional time")
                if q==0:
                    if k==num_harms//2:
                        ax.set_ylabel("Nondimensional current")

                ax2=ax.twinx()
                ax2.set_ylabel(h_class.harmonics[k], rotation=0)
                ax2.set_yticks([])
                ax.plot(time_results, abs(experimental_harmonics[k,:]), label="Experiment")
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.plot(time_results, abs(synthetic_harmonics[k,:]), label="Simulation")
                if j==0:
                    if row_position==num_harms:
                        ax.legend(loc="upper right", bbox_to_anchor=[0.1, 2.8], frameon=False)
                row_position+=1
plt.subplots_adjust(top=0.95,
bottom=0.05,
left=0.115,
right=0.985,
hspace=0.265,
wspace=0.6)
plt.show()
