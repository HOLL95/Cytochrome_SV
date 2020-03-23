from scipy.stats import norm, lognorm
import numpy as np
import itertools
import copy
import math
class dispersion:
    def __init__(self, simulation_options, optim_list):
        self.simulation_options=simulation_options
        if "dispersion_parameters" not in self.simulation_options:
            raise ValueError("Dispersion parameters not defined")
        if len(self.simulation_options["dispersion_bins"])!=len(self.simulation_options["dispersion_parameters"]):
            print(self.simulation_options["dispersion_bins"],self.simulation_options["dispersion_parameters"])
            raise ValueError("Need to define number of bins for each parameter")
        if len(self.simulation_options["dispersion_distributions"])!=len(self.simulation_options["dispersion_parameters"]):
            print(self.simulation_options["dispersion_distributions"],self.simulation_options["dispersion_parameters"])
            raise ValueError("Need to define distributions for each parameter")
        for i in range(0, len(self.simulation_options["dispersion_parameters"])):
            if self.simulation_options["dispersion_distributions"][i]=="uniform":
                if (self.simulation_options["dispersion_parameters"][i]+"_lower" not in optim_list) or (self.simulation_options["dispersion_parameters"][i]+"_upper" not in optim_list):
                    raise ValueError("Uniform distribution requires "+self.simulation_options["dispersion_parameters"][i]+"_lower and " + self.simulation_options["dispersion_parameters"][i]+"_upper")
            elif self.simulation_options["dispersion_distributions"][i]=="normal":
                if (self.simulation_options["dispersion_parameters"][i]+"_mean" not in optim_list) or (self.simulation_options["dispersion_parameters"][i]+"_std" not in optim_list):
                    raise ValueError("Normal distribution requires "+self.simulation_options["dispersion_parameters"][i]+"_mean and " + self.simulation_options["dispersion_parameters"][i]+"_std")
            elif self.simulation_options["dispersion_distributions"][i]=="lognormal":
                if (self.simulation_options["dispersion_parameters"][i]+"_shape" not in optim_list)  or (self.simulation_options["dispersion_parameters"][i]+"_scale" not in optim_list):
                    raise ValueError("Lognormal distribution requires "+self.simulation_options["dispersion_parameters"][i]+"_shape and " + self.simulation_options["dispersion_parameters"][i]+"_loc and "  + self.simulation_options["dispersion_parameters"][i]+"_scale")
            else:
                raise KeyError(self.simulation_options["dispersion_distributions"][i]+" distribution not implemented")
    def generic_dispersion(self, nd_dict, GH_dict=None):
        weight_arrays=[]
        value_arrays=[]
        for i in range(0, len(self.simulation_options["dispersion_parameters"])):
            if self.simulation_options["dispersion_distributions"][i]=="uniform":
                    value_arrays.append(np.linspace(self.simulation_options["dispersion_parameters"][i]+"_lower", self.simulation_options["dispersion_parameters"][i]+"_upper", self.simulation_options["dispersion_bins"][i]))
                    weight_arrays.append([1/self.simulation_options["dispersion_bins"][i]]*self.simulation_options["dispersion_bins"][i])
            elif self.simulation_options["dispersion_distributions"][i]=="normal":
                    param_mean=nd_dict[self.simulation_options["dispersion_parameters"][i]+"_mean"]
                    param_std=nd_dict[self.simulation_options["dispersion_parameters"][i]+"_std"]
                    if type(GH_dict) is dict:
                        param_vals=[(param_std*math.sqrt(2)*node)+param_mean for node in GH_dict["nodes"]]
                        param_weights=GH_dict["normal_weights"]
                    else:
                        min_val=norm.ppf(1e-4, loc=param_mean, scale=param_std)
                        max_val=norm.ppf(1-1e-4, loc=param_mean, scale=param_std)
                        param_vals=np.linspace(min_val, max_val, self.simulation_options["dispersion_bins"][i])
                        param_weights=np.zeros(self.simulation_options["dispersion_bins"][i])
                        param_weights[0]=norm.cdf(param_vals[0],loc=param_mean, scale=param_std)
                        param_midpoints=np.zeros(self.simulation_options["dispersion_bins"][i])
                        param_midpoints[0]=norm.ppf((1e-4/2), loc=param_mean, scale=param_std)
                        for j in range(1, self.simulation_options["dispersion_bins"][i]):
                            param_weights[j]=norm.cdf(param_vals[j],loc=param_mean, scale=param_std)-norm.cdf(param_vals[j-1],loc=param_mean, scale=param_std)
                            param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
                        param_midpoints=param_vals
                    value_arrays.append(param_vals)
                    weight_arrays.append(param_weights)
            elif self.simulation_options["dispersion_distributions"][i]=="lognormal":
                    param_loc=0
                    param_shape=nd_dict[self.simulation_options["dispersion_parameters"][i]+"_shape"]
                    param_scale=nd_dict[self.simulation_options["dispersion_parameters"][i]+"_scale"]
                    min_val=lognorm.ppf(1e-4, param_shape, loc=param_loc, scale=param_scale)
                    max_val=lognorm.ppf(1-1e-4, param_shape, loc=param_loc, scale=param_scale)
                    param_vals=np.linspace(min_val, max_val, self.simulation_options["dispersion_bins"][i])
                    param_weights=np.zeros(self.simulation_options["dispersion_bins"][i])
                    param_weights[0]=lognorm.cdf(param_vals[0],param_shape, loc=param_loc, scale=param_scale)
                    param_midpoints=np.zeros(self.simulation_options["dispersion_bins"][i])
                    param_midpoints[0]=norm.ppf((1e-4/2), loc=param_mean, scale=param_std)
                    for j in range(1, self.simulation_options["dispersion_bins"][i]):
                        param_weights[j]=norm.cdf(param_vals[j],param_shape, loc=param_loc, scale=param_scale)-norm.cdf(param_vals[j-1],param_shape, loc=param_loc, scale=param_scale)
                        param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
                    value_arrays.append(param_midpoints)
                    weight_arrays.append(param_weights)
        total_len=np.prod(self.simulation_options["dispersion_bins"])
        weight_combinations=list(itertools.product(*weight_arrays))
        value_combinations=list(itertools.product(*value_arrays))
        sim_params=copy.deepcopy(self.simulation_options["dispersion_parameters"])
        for i in range(0, len(sim_params)):
            if sim_params[i]=="E0":
                sim_params[i]="E_0"
            if sim_params[i]=="k0":
                sim_params[i]="k_0"
        return sim_params, value_combinations, weight_combinations
