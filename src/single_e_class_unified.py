import isolver_martin_brent
#import isolver_martin_NR
from scipy.stats import norm, lognorm
import math
import numpy as np
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
from params_class import params
from dispersion_class import dispersion
from decimal import Decimal
import copy
import time
import pickle
import warnings
class single_electron:
    def __init__(self,file_name=None, dim_parameter_dictionary={}, simulation_options={}, other_values={}, param_bounds={}, results_flag=True):
        if type(file_name) is dict:
            raise TypeError("Need to define a filename - this is currently a dictionary!")

        if len(dim_parameter_dictionary)==0 and len(simulation_options)==0 and len(other_values)==0:
            self.file_init=True
            file=open(file_name, "rb")
            save_dict=pickle.load(file, encoding="latin1")
            dim_parameter_dictionary=save_dict["param_dict"]
            simulation_options=save_dict["simulation_opts"]
            other_values=save_dict["other_vals"]
            param_bounds=save_dict["bounds"]
            self.save_dict=save_dict
        else:
            self.file_init=False

        if simulation_options["method"]=="ramped":
            dim_parameter_dictionary["v_nondim"]=True
        self.nd_param=params(dim_parameter_dictionary)
        self.nd_param_dict=self.nd_param.nd_param_dict
        self.dim_dict=copy.deepcopy(dim_parameter_dictionary)
        self.simulation_options=simulation_options
        self.other_values=other_values
        self.check_inputs()
        self.optim_list=self.simulation_options["optim_list"]
        self.harmonic_range=self.other_values["harmonic_range"]
        self.num_harmonics=len(self.harmonic_range)
        self.filter_val=self.other_values["filter_val"]
        self.bounds_val=self.other_values["bounds_val"]
        self.time_array=[]
        if self.simulation_options["experimental_fitting"]==True:
            if simulation_options["method"]=="sinusoidal":
                time_end=(self.other_values["num_peaks"]/self.nd_param.nd_param_dict["omega"])
            elif simulation_options["method"]=="ramped":
                time_end=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])
            elif simulation_options["method"]=="dcv":
                time_end=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])
            if simulation_options["no_transient"]!=False:
                if simulation_options["no_transient"]>time_end:
                    warnings.warn("Previous transient removal method detected")
                    time_idx=tuple(np.where(self.other_values["experiment_time"]<time_end))
                    desired_idx=tuple((range(simulation_options["no_transient"],time_idx[0][-1])))
                    self.time_idx=desired_idx[0]
                else:
                    desired_idx=tuple(np.where((self.other_values["experiment_time"]<time_end) & (self.other_values["experiment_time"]>simulation_options["no_transient"])))
                    time_idx=tuple(np.where(self.other_values["experiment_time"]<time_end))
                    self.time_idx=desired_idx[0][0]
            else:
                desired_idx=tuple(np.where(self.other_values["experiment_time"]<time_end))
                time_idx=desired_idx
                self.time_idx=0
            if self.file_init==False or results_flag==True:
                self.time_vec=self.other_values["experiment_time"][time_idx]/self.nd_param.c_T0
                self.other_values["experiment_time"]=self.other_values["experiment_time"][desired_idx]/self.nd_param.c_T0
                self.other_values["experiment_current"]=self.other_values["experiment_current"][desired_idx]/self.nd_param.c_I0
                self.other_values["experiment_voltage"]=self.other_values["experiment_voltage"][desired_idx]/self.nd_param.c_E0
            else:
                if simulation_options["method"]=="sinusoidal":
                    self.other_values["time_end"]=(self.other_values["num_peaks"])#/self.nd_param.nd_param_dict["omega"])
                else:
                    self.other_values["time_end"]=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])
                self.times()
        else:
            if simulation_options["method"]=="sinusoidal":
                self.other_values["time_end"]=(self.other_values["num_peaks"])#/self.nd_param.nd_param_dict["omega"])
            else:
                self.other_values["time_end"]=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])
            self.times()
            if simulation_options["no_transient"]!=False:
                    transient_time=self.t_nondim(self.time_vec)
                    start_idx=np.where(transient_time>simulation_options["no_transient"])
                    self.time_idx=start_idx[0][0]
            else:
                    self.time_idx=0
        self.def_optim_list(self.simulation_options["optim_list"])
        frequencies=np.fft.fftfreq(len(self.time_vec), self.time_vec[1]-self.time_vec[0])
        self.frequencies=frequencies[np.where(frequencies>0)]
        last_point= (self.harmonic_range[-1]*self.nd_param.nd_param_dict["omega"])+(self.nd_param.nd_param_dict["omega"]*self.filter_val)
        self.test_frequencies=frequencies[np.where(self.frequencies<last_point)]
        self.param_bounds=param_bounds
        if self.simulation_options["experimental_fitting"]==True:
            self.secret_data_fourier=self.top_hat_filter(self.other_values["experiment_current"])
            self.secret_data_time_series=self.other_values["experiment_current"]
    def GH_setup(self):
        """
        We assume here that for n>1 normally dispersed parameters then the order of the integral
        will be the same for both. The function defines the nodes, weights and weights required for the calculation of a normal expectation
        using Gaussian-hermite quadrature.
        """
        try:
            disp_idx=self.simulation_options["dispersion_distributions"].index("normal")
        except:
            print(dispersion_distributions)
            raise KeyError("GH quadrature is only appropriate for a normal distribution")
        nodes=self.simulation_options["dispersion_bins"][disp_idx]
        labels=["nodes", "weights", "normal_weights"]
        nodes, weights=np.polynomial.hermite.hermgauss(nodes)
        normal_weights=np.multiply(1/math.sqrt(math.pi), weights)
        self.other_values["GH_dict"]=dict(zip(labels, [nodes, weights, normal_weights]))
    def def_optim_list(self, optim_list):
        """
        Accepts a list of parameters to simulate.
        Checks if the parameters are already defined for the model
        Defines the boundaries variable for normalisation using the param_bounds using the param bounds dictionary
        Checks if any of the parameters are dispersed parameters
        """
        keys=list(self.dim_dict.keys())
        for i in range(0, len(optim_list)):
            if optim_list[i] in keys:
                continue
            else:
                raise KeyError("Parameter " + optim_list[i]+" not found in model")
        self.optim_list=optim_list
        param_boundaries=np.zeros((2, self.n_parameters()))
        check_for_bounds=vars(self)
        if "param_bounds" in list(check_for_bounds.keys()):
            for i in range(0, self.n_parameters()):
                    param_boundaries[0][i]=self.param_bounds[self.optim_list[i]][0]
                    param_boundaries[1][i]=self.param_bounds[self.optim_list[i]][1]

            self.boundaries=param_boundaries
        disp_flags=["mean", "scale", "upper"]
        disp_check=[[y in x for y in disp_flags] for x in self.optim_list]
        if True in [True in x for x in disp_check]:
            self.simulation_options["dispersion"]=True
            distribution_flags=["normal", "lognormal", "uniform"]
            self.simulation_options["dispersion_parameters"]=[]
            self.simulation_options["dispersion_distributions"]=[]
            for i in range(0, len(self.optim_list)):
                count=0
                for j in range(0, len(disp_flags)):
                    if count>1:
                        raise ValueError("Multiple dispersion flags in "+self.optim_list[i])
                    if disp_flags[j] in self.optim_list[i]:
                        index=self.optim_list[i].find("_"+disp_flags[j])
                        self.simulation_options["dispersion_parameters"].append(self.optim_list[i][:index])
                        self.simulation_options["dispersion_distributions"].append(distribution_flags[j])
            if type(self.simulation_options["dispersion_bins"])is int:
                if len(self.simulation_options["dispersion_distributions"])>1:
                    warnings.warn("Only one set of bins defined for multiple distributions. Assuming all distributions discretised using the same number of bins")
                num_dists=len(self.simulation_options["dispersion_distributions"])
                self.simulation_options["dispersion_bins"]=[self.simulation_options["dispersion_bins"]]*num_dists
            if "GH_quadrature" in self.simulation_options:
                if self.simulation_options["GH_quadrature"]==True:
                    self.GH_setup()
            self.disp_class=dispersion(self.simulation_options, optim_list)
        else:
            self.simulation_options["dispersion"]=False
        if "phase" in optim_list and "cap_phase" not in optim_list:
            self.simulation_options["phase_only"]=True
        else:
            self.simulation_options["phase_only"]=False
    def normalise(self, norm, boundaries):
        """
        Normalises the value (norm) provided to a value between 0 and 1 by its position relative to the value of the boundaries
        """
        return  (norm-boundaries[0])/(boundaries[1]-boundaries[0])
    def un_normalise(self, norm, boundaries):
        """
        Norm is a value between 0 and 1, which the function normalises to the appropriate value between the provided boundaries
        """
        return (norm*(boundaries[1]-boundaries[0]))+boundaries[0]
    def i_nondim(self, current):
        return np.multiply(current, self.nd_param.c_I0)
    def e_nondim(self, potential):
        return np.multiply(potential, self.nd_param.c_E0)
    def t_nondim(self, time):
        return np.multiply(time, self.nd_param.c_T0)
    def n_outputs(self):
        return 1
    def n_parameters(self):
        return len(self.optim_list)
    def define_voltages(self, transient=True):
        """
        Returns the potential input as defined by the experimental paramters in the parameter dictionary.
        If transient is equal to False then the voltage is also truncated appropriately
        """
        voltages=np.zeros(len(self.time_vec))
        if self.simulation_options["method"]=="sinusoidal":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], (self.time_vec[i]))
        elif self.simulation_options["method"]=="ramped":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],(self.time_vec[i]))
        elif self.simulation_options["method"]=="dcv":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.dcv_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) , 1,(self.time_vec[i]))
        if transient==False:
            voltages=voltages[self.time_idx:]
        return voltages
    def top_hat_filter(self, time_series):
        frequencies=self.frequencies
        L=len(time_series)
        window=np.hanning(L)
        time_series=np.multiply(time_series, window)
        f=np.fft.fftfreq(len(time_series), self.time_vec[1]-self.time_vec[0])
        Y=np.fft.fft(time_series)
        #Y_pow=np.power(copy.deepcopy(Y[0:len(frequencies)]),2)
        top_hat=copy.deepcopy(Y[0:len(frequencies)])
        scale_flag=False
        true_harm=self.nd_param.nd_param_dict["omega"]*self.nd_param.c_T0
        if "fourier_scaling" in self.simulation_options:
            if self.simulation_options["fourier_scaling"]!=None:
                scale_flag=True
        if sum(np.diff(self.harmonic_range))!=len(self.harmonic_range)-1 or scale_flag==True or self.other_values["filter_val"]!=0.5:
            results=np.zeros(len(top_hat), dtype=complex)
            for i in range(0, self.num_harmonics):
                true_harm_n=true_harm*self.harmonic_range[i]
                index=tuple(np.where((frequencies<(true_harm_n+(self.nd_param.nd_param_dict["omega"]*self.filter_val))) & (frequencies>true_harm_n-(self.nd_param.nd_param_dict["omega"]*self.filter_val))))
                if scale_flag==True:
                    filter_bit=abs(top_hat[index])
                    min_f=min(filter_bit)
                    max_f=max(filter_bit)
                    filter_bit=[self.normalise(x, [min_f, max_f]) for x in filter_bit]
                else:
                    filter_bit=top_hat[index]
                results[index]=filter_bit
        else:
            first_harm=(self.harmonic_range[0]*true_harm)-(self.nd_param.nd_param_dict["omega"]*self.filter_val)
            last_harm=(self.harmonic_range[-1]*true_harm)+(self.nd_param.nd_param_dict["omega"]*self.filter_val)
            likelihood=top_hat[np.where((frequencies>first_harm) & (frequencies<last_harm))]
            results=np.zeros(len(top_hat), dtype=complex)
            results[np.where((frequencies>first_harm) & (frequencies<last_harm))]=likelihood
        comp_results=np.append((np.real(results)), np.imag(results))
        return abs(results)
    def abs_transform(self, data):
        window=np.hanning(len(data))
        hanning_transform=np.multiply(window, data)
        f_trans=abs(np.fft.fft(hanning_transform[len(data)/2+1:]))
        return f_trans
    def saved_param_simulate(self, params):
        if self.file_init==False:
            raise ValueError('No file provided')
        else:
            self.def_optim_list(self.save_dict["optim_list"])
            type=self.simulation_options["likelihood"]
            return self.test_vals(params,type, test=False)
    def save_state(self, results, filepath, filename, params):
        other_vals_save=self.other_values
        other_vals_save["experiment_time"]=results["experiment_time"]
        other_vals_save["experiment_current"]=results["experiment_current"]
        other_vals_save["experiment_voltage"]=results["experiment_voltage"]
        file=open(filepath+"/"+filename, "wb")
        save_dict={"simulation_opts":self.simulation_options, \
                    "other_vals":other_vals_save, \
                    "bounds":self.param_bounds, \
                    "param_dict":self.dim_dict ,\
                    "params":params, "optim_list":self.optim_list}
        pickle.dump(save_dict, file, pickle.HIGHEST_PROTOCOL)
        file.close()
    def calc_theta(self, current):
        voltages=self.define_voltages()
        if self.simulation_options["no_transient"]!=True:
            voltages=voltages[self.time_idx:]
        theta=np.zeros(len(current))
        theta[0]=0
        dt=self.nd_param.nd_param_dict["sampling_freq"]
        for i in range(1, len(current)):
            Er=voltages[i]-self.nd_param.nd_param_dict["Ru"]*current[i]
            expval1=Er-self.nd_param.nd_param_dict["E_0"]
            exp11=np.exp((1-self.nd_param.nd_param_dict["alpha"])*expval1)
            exp12=np.exp((-self.nd_param.nd_param_dict["alpha"])*expval1)
            u1n1_top=dt*self.nd_param.nd_param_dict["k_0"]*exp11 + theta[i-1]
            denom = ((dt*self.nd_param.nd_param_dict["k_0"]*exp11) +(dt*self.nd_param.nd_param_dict["k_0"]*exp12) + 1)
            theta[i]=u1n1_top/denom
        return theta
    def times(self):
        self.time_vec=np.arange(0, self.other_values["time_end"], self.nd_param.nd_param_dict["sampling_freq"])
        #self.time_vec=np.linspace(0, self.nd_param.nd_param_dict["time_end"], num_points)
    def change_norm_group(self, param_list, method):
        normed_params=copy.deepcopy(param_list)
        if method=="un_norm":
            for i in range(0,len(param_list)):
                normed_params[i]=self.un_normalise(normed_params[i], [self.boundaries[0][i],self.boundaries[1][i]])
        elif method=="norm":
            for i in range(0,len(param_list)):
                normed_params[i]=self.normalise(normed_params[i], [self.boundaries[0][i],self.boundaries[1][i]])
        return normed_params
    def variable_returner(self):
        variables=self.nd_param.nd_param_dict
        for key in list(variables.keys()):
            if type(variables[key])==int or type(variables[key])==float or type(variables[key])==np.float64:
                print(key, variables[key])
    def test_vals(self, parameters, likelihood, test=False):
        orig_likelihood=self.simulation_options["likelihood"]
        orig_label=self.simulation_options["label"]
        orig_test=self.simulation_options["test"]
        self.simulation_options["likelihood"]=likelihood
        self.simulation_options["label"]="MCMC"
        self.simulation_options["test"]=test
        results=self.simulate(parameters, self.frequencies)
        if sum(results)==0:
            raise ValueError("Not simulated, check options")
        self.simulation_options["likelihood"]=orig_likelihood
        self.simulation_options["label"]=orig_label
        self.simulation_options["test"]=orig_test
        return results
    def paralell_disperse(self, solver):
        time_series=np.zeros(len(self.time_vec))
        if "GH_quadrature" in self.simulation_options:
            if self.simulation_options["GH_quadrature"]==True:
                sim_params, values, weights=self.disp_class.generic_dispersion((self.nd_param.nd_param_dict), self.other_values["GH_dict"])
        else:
                sim_params, values, weights=self.disp_class.generic_dispersion((self.nd_param.nd_param_dict))

        for i in range(0, len(weights)):
            for j in range(0, len(sim_params)):
                self.nd_param_dict[sim_params[j]]=values[i][j]
            time_series_current=solver(self.nd_param_dict, self.time_vec,self.simulation_options["method"], -1, self.bounds_val)
            time_series=np.add(time_series, np.multiply(time_series_current, np.prod(weights[i])))
        return time_series
    def numerical_plots(self, solver):
        self.debug_time=self.simulation_options["numerical_debugging"]
        time_series=solver(self.nd_param_dict, self.time_vec,self.simulation_options["method"], self.debug_time, self.bounds_val)
        residual=time_series[1]
        residual_gradient=time_series[2]
        #plt.subplot(1,2,1)
        #plt.semilogy(current, np.abs(residual))
        Nt=self.time_vec[-1]/self.nd_param.nd_param_dict["sampling_freq"]
        bounds_val=max(10*self.nd_param.nd_param_dict["Cdl"]*self.nd_param.nd_param_dict["d_E"]*self.nd_param.nd_param_dict["nd_omega"]/Nt,1.0)
        middle_index=(len(time_series[0])-1)/2 + 1
        I0=residual[middle_index]
        if  self.simulation_options["numerical_method"]=="Newton-Raphson":
            plt.subplot(1,2,1)
            plt.title("Residual, t="+str(self.debug_time))
            plt.plot(current, residual)
            plt.axvline(time_series[3][1], color="red",linestyle="--")
            plt.axvline(time_series[3][0]+time_series[3][2], color="black", linestyle="--")
            plt.axvline(time_series[3][0]-time_series[3][2], color="black",linestyle="--")
            plt.subplot(1,2,2)
            plt.title("Residual gradient")
            plt.plot(current, ((residual_gradient)))
            plt.show()
        else:
            plt.plot(current, residual, label=round(self.debug_time, 4))
            plt.show()
    def simulate(self,parameters, frequencies=None):
        if len(parameters)!= len(self.optim_list):
            print(self.optim_list)
            print(parameters)
            raise ValueError('Wrong number of parameters')
        if self.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(parameters, "un_norm")
        else:
            normed_params=copy.deepcopy(parameters)
        for i in range(0, len(self.optim_list)):
            self.dim_dict[self.optim_list[i]]=normed_params[i]
        if "phase_only" in self.simulation_options and self.simulation_options["phase_only"]==True:
            self.dim_dict["cap_phase"]=self.dim_dict["phase"]
        self.nd_param=params(self.dim_dict)
        self.nd_param_dict=self.nd_param.nd_param_dict
        if self.simulation_options["numerical_method"]=="Brent minimisation":
            solver=isolver_martin_brent.brent_current_solver
        elif self.simulation_options["numerical_method"]=="Newton-Raphson":
            solver=isolver_martin_NR.NR_current_solver
            if self.simulation_options["method"]=="dcv":
                raise ValueError("Newton-Raphson dcv simulation not implemented")
        else:
            raise ValueError('Numerical method not defined')
        start=time.time()
        if self.simulation_options["numerical_debugging"]!=False:
            self.numerical_plots(solver)
        else:
            if self.simulation_options["dispersion"]==True:
                time_series=self.paralell_disperse(solver)
            else:
                time_series=solver(self.nd_param_dict, self.time_vec, self.simulation_options["method"],-1, self.bounds_val)
        if self.simulation_options["no_transient"]!=False:
            time_series=time_series[self.time_idx:]
        time_series=np.array(time_series)
        if self.simulation_options["likelihood"]=='fourier':
            filtered=self.top_hat_filter(time_series)
            if (self.simulation_options["test"]==True):
                print(list(normed_params))
                self.variable_returner()
                plt.plot(self.secret_data_fourier, label="data")
                plt.plot(filtered , alpha=0.7, label="numerical")
                plt.legend()
                plt.show()

            return filtered
        elif self.simulation_options["likelihood"]=='timeseries':
            if self.simulation_options["test"]==True:
                print(list(normed_params))
                self.variable_returner()
                if self.simulation_options["experimental_fitting"]==True:
                    plt.subplot(1,2,1)
                    plt.plot(self.other_values["experiment_voltage"],time_series)
                    plt.subplot(1,2,2)
                    plt.plot(self.other_values["experiment_time"],time_series)
                    plt.plot(self.other_values["experiment_time"],self.secret_data_time_series, alpha=0.7)
                    plt.show()
                else:
                    plt.plot(self.time_vec[self.time_idx:], time_series)
                    plt.plot(self.time_vec[self.time_idx:], self.secret_data_time_series)
                    plt.show()
            return (time_series)
    def check_inputs(self):
        required_params=set(["E_0", "k_0", "alpha", "gamma", "Ru", "Cdl", "CdlE1","CdlE2","CdlE3", "E_start", \
                        "original_gamma", "area","E_reverse", "omega", "phase", "d_E", "sampling_freq"])
        user_params=set(self.dim_dict.keys())
        req_union=required_params.intersection(user_params)
        if len(req_union)!=len(required_params):
            missing_params=required_params-req_union
            raise KeyError("Essential parameter(s) mising:",missing_params)
        if "experimental_fitting" not in self.simulation_options:
            raise KeyError("Please specify whether to fit to experimental data or not")
        if "no_transient" not in self.simulation_options:
            self.simulation_options["no_transient"]=False
        elif self.simulation_options["no_transient"]!=False:
            try:
                int(self.simulation_options["no_transient"])
            except:
                raise ValueError("Please specify the time before which you want to remove the signal")
        if "numerical_debugging" not in self.simulation_options:
            self.simulation_options["numerical_debugging"]=False
        if "dispersion" not in self.simulation_options:
            self.simulation_options["dispersion"]=False
        if "test" not in self.simulation_options:
            self.simulation_options["test"]=False
        if "method" not in self.simulation_options:
            raise KeyError("Please specify method (sinusoidal or ramped)")
        if "likelihood" not in  self.simulation_options:
            raise KeyError("Please specify either frequency or time-domain likelihood (options-fourier or timeseries)")
        if "numerical_method" not in self.simulation_options:
            self.simulation_options["numerical_method"]="Brent minimisation"
        if "label" not in self.simulation_options:
            self.simulation_options["label"]="MCMC"
        if "optim_list" not in self.simulation_options:
            raise KeyError("Please specify parameters with which to simulate")
        if self.simulation_options["likelihood"]=="fourier":
            if "filter_val" not in self.other_values:
                warnings.warn("Assuming width of filter is 0.5*omega")
                self.other_values["filter_val"]=0.5
            if "harmonic_range" not in self.other_values:
                raise KeyError("Please specify which harmonics to use for the fitting process")
        if self.simulation_options["experimental_fitting"]==True:
            if "experiment_time" not in self.other_values:
                raise KeyError("Please an experimental time")
            if "experiment_voltage" not in self.other_values:
                raise KeyError("Please an experimental voltage")
            if "experiment_current" not in self.other_values:
                raise KeyError("Please an experimental current")
        if self.simulation_options["method"]=="sinusoidal":
            if "num_peaks" not in self.other_values:
                if "num_peaks" in self.dim_dict:
                    self.other_values["num_peaks"]=self.dim_dict["num_peaks"]
                else:
                    raise KeyError("Specify the number of peaks desired for the experiment")
        if "bounds_val" not in self.other_values:
            #Brent minimisation algorithm maximum change in one time interval dt
            self.other_values["bounds_val"]=20
        if self.simulation_options["method"]=="ramped":
            if "original_omega" in self.dim_dict:
                raise KeyError("Need to delete original_omega from the list of parameters for the ramped method")
            if "v" not in self.dim_dict:
                raise KeyError("The ramped method needs a linear potential scan rate v for non-dimensionalisation")
        if "phase only" in self.simulation_options and "cap_phase" not in user_params:
            raise KeyError("Specify either phase only or a capacitance phase")
class paralell_class:
    def __init__(self, params, times, method, bounds, solver):
        self.params=params
        self.times=times
        self.method=method
        self.bounds=bounds
        self.solver=solver
    def paralell_simulate(self, weight_val_entry):
        start=time.time()
        self.sim_params=copy.deepcopy(self.params)
        for i in range(len(weight_val_entry[0])):
            self.sim_params[weight_val_entry[0][i]]=weight_val_entry[1][i]
        time_series=self.solver(self.sim_params, self.times, self.method,-1, self.bounds)
        time_series=np.multiply(time_series, weight_val_entry[2])
        return (time_series)
    def paralell_dispersion(self, weight_list):
        p = mp.Pool(4)
        start1=time.time()
        sc = p.map_async(self,  [weight for weight in weight_list])
        start=time.time()
        results=sc.get()
        p.close()
        disped_time=np.sum(results, axis=0)
        start2=time.time()
        return disped_time
    def __call__(self, x):
        return self.paralell_simulate(x)
