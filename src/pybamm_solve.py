import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time
class pybamm_solver:
    def __init__(self, e_class):
        self.dim_dict=e_class.dim_dict
        self.nd_dict=e_class.nd_param.nd_param_dict
        self.model=pybamm.BaseModel()
        self.parameter_dict={}
        self.pybam_val_dict={}
        self.simulation_options=e_class.simulation_options
        self.dim_keys=self.dim_dict.keys()
        self.time=e_class.time_vec
        for key in self.dim_keys:
            self.parameter_dict[key]=pybamm.InputParameter(key)
            self.pybam_val_dict[key]=None
        current=pybamm.Variable("current")
        theta=pybamm.Variable("theta")
        Edc_forward = pybamm.t
        Edc_backwards = -(pybamm.t - 2*self.parameter_dict["tr"])
        E_t = self.parameter_dict["E_start"]+ \
        (pybamm.t <= self.parameter_dict["tr"]) * Edc_forward + \
        (pybamm.t > self.parameter_dict["tr"]) * Edc_backwards
        Er=E_t-(self.parameter_dict["Ru"]*current)
        ErE0=Er-self.parameter_dict["E_0"]
        alpha=self.parameter_dict["alpha"]
        if self.simulation_options["method"]=="dcv":
            Cdlp=self.parameter_dict["Cdl"]*(1+self.parameter_dict["CdlE1"]*Er+self.parameter_dict["CdlE2"]*(Er**2)+self.parameter_dict["CdlE3"]*(Er**3))
        else:
            Cdlp=(pybamm.t <= self.parameter_dict["tr"]) *(self.parameter_dict["Cdl"]*(1+self.parameter_dict["CdlE1"]*Er+self.parameter_dict["CdlE2"]*(Er**2)+self.parameter_dict["CdlE3"]*(Er**3)))+\
            (pybamm.t > self.parameter_dict["tr"]) *(self.parameter_dict["Cdlinv"]*(1+self.parameter_dict["CdlE1inv"]*Er+self.parameter_dict["CdlE2inv"]*(Er**2)+self.parameter_dict["CdlE3inv"]*(Er**3)))

        self.model.variables={"current":current, "theta":theta}
        d_thetadt=((1-theta)*self.parameter_dict["k_0"]*pybamm.exp((1-alpha)*ErE0))-(theta*self.parameter_dict["k_0"]*pybamm.exp((-alpha)*ErE0))
        dIdt=(E_t.diff(pybamm.t)-(current/Cdlp)+self.parameter_dict["gamma"]*d_thetadt*(1/Cdlp))/self.parameter_dict["Ru"]
        self.model.rhs={current:dIdt, theta:d_thetadt}
        self.model.initial_conditions={theta:pybamm.Scalar(0), current:pybamm.Scalar(0)}
    def simulate(self, nd_param_dict,time_vec ,*args):
        for key in self.dim_keys:
            self.pybam_val_dict[key]=nd_param_dict[key]
        #param=pybamm.ParameterValues(self.pybam_val_dict)
        #param.process_model(self.model)
        disc=pybamm.Discretisation()
        disc.process_model(self.model)
        solver=pybamm.ScipySolver()
        solution=solver.solve(self.model, time_vec, inputs=self.pybam_val_dict)

        return solution["current"].entries
    def potential_fun(self):
        if self.simulation_options["method"]=="sinusoidal":
                Et=isolver_martin_brent.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
                dEdt=isolver_martin_brent.dEdt(self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
        elif self.simulation_options["method"]=="ramped":
                tr=self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]
                Et=isolver_martin_brent.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
                dEdt=isolver_martin_brent.c_dEdt(tr ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
        elif self.simulation_options["method"]=="dcv":
                Edc_forward = pybamm.t
                Edc_backwards = -(pybamm.t - 2*self.parameter_dict["tr"])
                Eapp = 27.2+ \
                (pybamm.t <= self.parameter_dict["tr"]) * Edc_forward + \
                (pybamm.t > self.parameter_dict["tr"]) * Edc_backwards
        return Eapp
