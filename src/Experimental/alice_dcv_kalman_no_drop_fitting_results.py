import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
sys.path.append("..")
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from scipy.integrate import odeint
from single_e_class_unified import single_electron
from scipy.optimize import curve_fit
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-2])+"/Experiment_data/Alice_2_11_20/DCV"
files=os.listdir(data_loc)
experimental_dict={}
file_name="/dcV_Cjx-183D_WT_pH_7_1_3"
dcv_file=np.loadtxt(data_loc+file_name, skiprows=2)
dcv_file_time=dcv_file[:,0]
dcv_file_voltage=dcv_file[:,1]
dcv_file_current=dcv_file[:,2]
param_list={
    "E_0":-0.05,
    'E_start':  -0.39, #(starting dc voltage - V)
    'E_reverse':0.3,
    'omega':8.94, #8.88480830076,  #    (frequency Hz)
    "v":30*1e-3,
    'd_E': 0,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 0, #(capacitance parameters)
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
    'sampling_freq' : (1.0/200),
    'phase' :0,
    "Q":1,
    "time_end": -1,
    'num_peaks': 30,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[5],
    "GH_quadrature":True,
    "test": False,
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "cmaes",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":range(0, 1),
    "experiment_time": dcv_file_time,
    "experiment_current": dcv_file_current,
    "experiment_voltage":dcv_file_voltage,
    "bounds_val":200,
}
param_bounds={
    'E_0':[-0.1,0.0],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 300],  #     (uncompensated resistance ohms)
    "Q":[1e-5, 1],
    'Cdl': [1e-6,1e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'Cdlinv': [1e-5, 1e-3], #(capacitance parameters)
    'CdlE1inv': [-0.1,0.1],#0.000653657774506,
    'CdlE2inv': [-0.1,0.1],#0.000245772700637,
    'CdlE3inv': [-0.1,0.1],#1.10053945995e-06,
    'gamma': [0.5*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [1e-3, 1e4], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[-0.1,0.1],
    "E0_std": [1e-5,  0.5],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
}
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
def poly_2(x, a, b, c):
    return (a*x**2)+b*x+c
def poly_3(x, a, b, c, d):
    return (a*x**3)+(b*x**2)+c*x+d
def poly_4(x, a, b, c, d, e):
    return (a*x**4)+(b*x**3)+(c*x**2)+d*x+e
time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
middle_idx=list(voltage_results).index(max(voltage_results))
first_idx=10#len(voltage_results)//15
current=[]
idx_1=[first_idx, middle_idx+20]
idx_2=[middle_idx, -first_idx]
func=poly_3
interesting_section=[[-0.15, 0.12], [-0.15, 0.08]]
subtract_current=np.zeros(len(current_results))
fitted_curves=np.zeros(len(current_results))
nondim_v=cyt.e_nondim(voltage_results)
#plt.plot(nondim_v, current_results)
for i in range(0, 2):

    current_half=current_results[idx_1[i]:idx_2[i]]
    time_half=time_results[idx_1[i]:idx_2[i]]
    volt_half=cyt.e_nondim(voltage_results[idx_1[i]:idx_2[i]])
    noise_idx=np.where((volt_half<interesting_section[i][0]) | (volt_half>interesting_section[i][1]))
    noise_voltages=volt_half[noise_idx]
    noise_current=current_half[noise_idx]
    noise_times=time_half[noise_idx]
    popt, pcov = curve_fit(func, noise_times, noise_current)
    fitted_curve=[func(t, *popt) for t in time_half]
    subtract_current[idx_1[i]:idx_2[i]]=np.subtract(current_half, fitted_curve)
    fitted_curves[idx_1[i]:idx_2[i]]=fitted_curve
    #plt.plot(volt_half, fitted_curve, color="red")
    #plt.plot(noise_voltages, noise_current)

cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","gamma", "alpha"])
PSV_optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"]
ramped_optim_list=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"]
PSV_params=[-0.021495031150668878, 17.570527719697008, 1949.3033882011057, 0.00020134013002972008, 0.00411080805846069, 0.0006879411270873684, 2.3340827906876003e-10, 9.014898777848584, 2.5763763013646175, 4.359308770046119, 0.4000000157539228]
PSV_params=[-0.06999997557877347, 1.245269093253469, 701.0390205847876, 0.00012081970706517015, 0.0024664965342477527, 0.0011465681607091556, 1.7085728483891467e-09, 9.014958257811347, 6.014524412936475, 5.676241758957758, 0.4683583542196279]
PSV_params=[-0.06999999976247695, 0.054653156208964916, 0.423123012427712, 361.21853877194775, 0.0005089406894416405, -0.055068050329032774, 0.0022654775318873364, 7.263427801734963e-11, 9.01488754940552, 1.5710075181470322, 5.2472698323265075, 0.4837659989252462]
ramped_params=[-0.04485376873500503, 293.2567587982391, 146.0113118472105, 0.0001576519851347672, 0.006105674536299788, 0.0012649370988525588, 2.2215281961212185e-11, 8.959294996508683, 6.147649245979944,0, 0.5372803774088237]
ramped_params=[-0.06158660103168602, 0.02767157845943783, 86.53789947453802, 40.56259593918775, 0.0009310670277647225, 0.030236335577448786, -0.0002525820042452911, 2.2619461371744093e-11, 8.959288930500506, 5.644994973337578, 0.5170561841197072]
cyt.def_optim_list(["k_0", "Ru", "Cdl","CdlE1", "CdlE2","CdlE3"])

volts=cyt.define_voltages()
#plt.plot(noisy_current)
tr=cyt.nd_param.nd_param_dict["E_reverse"]-cyt.nd_param.nd_param_dict["E_start"]
u=np.ones(len(cyt.time_vec))
u[np.where(cyt.time_vec>tr)]*=-1
#plt.plot(u)
#plt.show()


"""counter=0
for resistances in [10, 100, 1000, 10000]:
    counter+=1
    DCV_inferred=[1000, resistances, 5e-5, 1e-2, 1e-3, 1e-5]
    current=cyt.test_vals(DCV_inferred, "timeseries")
    #print(cyt.dim_dict["k_0"])
    #plt.plot(current)
    #plt.show()
    error=0.01*max(current)
    noisy_current=cyt.add_noise(current, error )
    cyt.secret_data_time_series=noisy_current
    for z in range(0, 2):
        plt.subplot(2, 4, counter+(4*z))

        #plt.scatter(cyt.time_vec[time_start:time_idx], noisy_current[time_start:time_idx],  label="Data", s=2, color="black")
        for ru_scale in np.flip([error, 1e-2,5e-2,  0.5]):
                #DCV_inferred[1]=true_ru*ru_scale

                q=ru_scale
                pred_cap, kalman_plot= cyt.kalman_pure_capacitance(cyt.secret_data_time_series, q)
                if z==1:
                    kalman_plot=kalman_plot*cyt.kalman_u
                psi_s="$\\Phi_s$= "
                if ru_scale==error:
                    plt.plot(cyt.time_vec, kalman_plot, linestyle="--", label=psi_s+"True error")
                else:
                    plt.plot(cyt.time_vec,kalman_plot,  label=psi_s+"{:.1e}".format(ru_scale))
                plt.ylabel("Nondim current")
                plt.xlabel("Nondim time")
                plt.title("R="+str(resistances))
                #cyt.test_vals(DCV_inferred, "timeseries")
        if z==0:
            plt.plot(cyt.time_vec,noisy_current, alpha=0.7, label="I")
            plt.legend(loc="upper right", frameon=False)
        else:
            plt.plot(cyt.time_vec,pred_cap, alpha=0.7, label="I/${\\frac{dE}{dt}}$")
            plt.legend(loc="lower left", frameon=False)
plt.show()"""

k_vals=[1e-2, 0.1, 10, 1000]
r_vals=[10000, 100000]
fit_params=["E_0", "k_0","Ru","Cdl","gamma", "alpha", "Q"]
results=np.zeros((len(k_vals), len(r_vals), len(fit_params)))
results[0, 0, :]=[1]*len(fit_params)
set2=[
[-0.044600567007731066, 0.01026626742798857, 199.21710983086552, 4.792295932697819e-05, 8.898738485345558e-12, 0.48038850859339416, 0.0841948417719602],
[-0.050408767370443024, 0.09497513747631915, 299.99200900853094, 4.728801367347065e-05, 1.0398640363481766e-11, 0.4992071888510464, 0.0841946716462203],
[-0.049588625722034846, 6.103735749139773, 192.64048456759815, 4.8073483046125545e-05, 1.0366379901695914e-11, 0.5999999464840962, 0.08419467172612506],
[-0.048253631595097685, 5.511533432688336, 140.3635154281575, 4.793657949747877e-05, 9.863673924834324e-12, 0.4000000047394874, 0.08419466835136118],
[-0.05325091470873754, 5817.362378434118, 293.14917582794953, 3.243162428309858e-05, 5.000000036742205e-12, 0.41423389888838263, 0.0827068580059358],
[-5.414609871579046e-10, 6927.369657192836, 66.7618881518237, 3.166843361959333e-05, 5.000000043777827e-12, 0.44256509663522103, 0.08206490877019161],
[-0.04867890494247045, 0.0010094838321794417, 178.09984477733062, 3.385149190321681e-05, 7.306577873461583e-12, 0.40000018901706585, 0.0820210519736999],
[-0.049845293435488484, 0.001000004805802035, 5.209226467393232, 3.1814499689707456e-05, 7.592244875496075e-12, 0.4955705563440733, 0.08206237243857595]]


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
counter=1
mode="integrated"
error_params=[r"$E^0\epsilon =$", r"$k^0\epsilon =$", r"$\Gamma\epsilon=$"]
import copy
for i in range(0, len(r_vals)):
    for j in range(0, len(k_vals)):
        cyt.simulation_options["numerical_method"]="Brent minimisation"
        cyt.def_optim_list(["k_0", "Ru", "Cdl","CdlE1", "CdlE2","CdlE3", "gamma", "E_0", "alpha"])
        DCV_inferred=[k_vals[j], r_vals[i], 5e-5, 1e-2, 1e-3, 1e-5, 1e-11, -0.05, 0.5]
        True_values=[-0.05, k_vals[j], DCV_inferred[6]]
        pure_farad=copy.deepcopy(DCV_inferred)
        pure_farad[2]*=0
        current=cyt.test_vals(DCV_inferred, "timeseries")
        pure_farad_current=cyt.test_vals(pure_farad, "timeseries")
        pure_cap=copy.deepcopy(DCV_inferred)
        pure_cap[6]*=0
        pure_cap_current=cyt.test_vals(pure_cap, "timeseries")
        print(cyt.dim_dict["gamma"], "GAMMA")
        error=0.01*max(current)
        noisy_current=cyt.add_noise(current, error)
        test=False
        cyt.param_bounds["Q"][1]=5*error
        cyt.secret_data_time_series=noisy_current
        cyt.dim_dict["Q"]=error
        cyt.simulation_options["numerical_method"]="Kalman_simulate"
        #cyt.test_vals(DCV_inferred, "timeseries", test=False)
        cyt.def_optim_list(fit_params)
        cmaes_problem=pints.SingleOutputProblem(cyt, cyt.time_vec, noisy_current)
        score = pints.SumOfSquaresError(cmaes_problem)
        CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
        val=100
        volts=cyt.define_voltages()

        x0=abs(np.random.rand(cyt.n_parameters()))
        cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
        cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
        cmaes_fitting.set_parallel(True)
        #plt.plot(current)
        #plt.show()

        #found_parameters, found_value=cmaes_fitting.run()
        #print(found_parameters)
        #cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
        #print(list(cmaes_results))

        #cyt.simulation_options["Kalman_capacitance"]=True
        #results[i, j, :]=list(saved_results)
        #print(counter)
        plt.subplot(2,4, counter)
        pred_vals=[set2[counter-1][0], set2[counter-1][1], set2[counter-1][4]]
        percent_error=abs(np.divide(np.subtract(pred_vals, True_values), True_values))
        axes_text=["" for x in range(0, len(percent_error))]
        print(pred_vals, True_values)
        print(percent_error*100)
        for q in range(0, len(percent_error)):
            axes_text[q]=error_params[q]+str(round(percent_error[q]*100,2))+r"%"

        if mode=="integrated":
            cmaes_time=cyt.test_vals(set2[counter-1], likelihood="timeseries", test=False)
            counter+=1
            textstr = '\n'.join(axes_text)
            plt.plot(volts, cmaes_time, label="Fitted")
            plt.plot(volts, noisy_current, alpha=0.7, label="Data")
            ax=plt.gca()
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.15, 0.75, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
            #for q in range(0, len(axes_text)):
            #    plt.text(0+q, 0.5, axes_text[q])
        else:
            cyt.simulation_options["Kalman_capacitance"]=True
            cmaes_time=cyt.test_vals(set2[counter-1], likelihood="timeseries", test=False)
            counter+=1
            plt.plot(volts, cyt.farad_current, label="Fitted $I_f$")
            plt.plot(volts, pure_farad_current, label="True $I_f$")
            plt.plot(volts, pure_cap_current, label="True $I_c$")
            plt.plot(volts, cyt.pred_cap, label="Fitted $I_c$")
        plt.title("$k^0="+str(k_vals[j])+",R_u="+str(r_vals[i])+"$")
        if i==len(r_vals)-1:
            plt.xlabel("Nondim potential")
        if j==0:
            plt.ylabel("Nondim current")
            plt.legend(ncol=2, loc="lower center", bbox_to_anchor=[0.5, 0.2])


plt.subplots_adjust(top=0.955,
bottom=0.065,
left=0.06,
right=0.96,
hspace=0.2,
wspace=0.2)
plt.show()

        #cyt.simulation_options["Kalman_capacitance"]=False

print(results)
save_dict={"results":results, "resistances":r_vals, "kinetics":k_vals, "parameters":fit_params}
np.save("Kalman_fits.npy", save_dict)
val1=[-0.05155097415338214, 7876.303924421097, 6896475.263383517, 4.7308994764134254e-05, 7.034705070960326e-12, 0.4158400921399645, 0.08419466805543942]
val2=[-0.09999999458465042, 0.007427226004530573, 1999999.9983219688, 3.1928273403555495e-05, 9.847535660797748e-11, 0.4000005174983166, 0.0820649087741374]
val3=[-7.647758634710122e-09, 8030.8314641560955, 1999998.181058591, 2.270442857648828e-06, 5.000000060959137e-12, 0.4308152874838279, 0.07170256361958534]
val4=[-0.061546406871540706, 0.0838295994074412, 112611.97731142382, 1.2051154721580916e-06, 5.000866261450389e-12, 0.43203151616759217, 0.07170252099298625]
#set_high=
