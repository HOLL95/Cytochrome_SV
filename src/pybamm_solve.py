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
        self.current=pybamm.Variable("current")
        self.theta=pybamm.Variable("theta")
        if self.simulation_options["method"]=="dcv":
            Edc_forward = pybamm.t
            Edc_backwards = -(pybamm.t - 2*self.parameter_dict["tr"])
            E_t = self.parameter_dict["E_start"]+ \
            (pybamm.t <= self.parameter_dict["tr"]) * Edc_forward + \
            (pybamm.t > self.parameter_dict["tr"]) * Edc_backwards
        elif self.simualtion_options["method"]=="sinusoidal":
            E_t=self.parameter_dict["E_start"]+self.parameter_dict["d_E"]+(self.parameter_dict["d_E"]*pybamm.sin((self.parameter_dict["nd_omega"]*pybamm.t)+self.parameter_dict["phase"]))
        elif self.simualtion_options["method"]=="ramped":
            Edc_forward = pybamm.t
            Edc_backwards = -(pybamm.t - 2*self.parameter_dict["tr"])
            E_t = self.parameter_dict["E_start"]+ \
            (pybamm.t <= self.parameter_dict["tr"]) * Edc_forward + \
            (pybamm.t > self.parameter_dict["tr"]) * Edc_backwards+\
            (self.parameter_dict["d_E"]*pybamm.sin((self.parameter_dict["nd_omega"]*pybamm.t)+self.parameter_dict["phase"]))
        Er=E_t-(self.parameter_dict["Ru"]*self.current)
        ErE0=Er-self.parameter_dict["E_0"]
        alpha=self.parameter_dict["alpha"]
        Cdlp=self.parameter_dict["Cdl"]*(1+self.parameter_dict["CdlE1"]*Er+self.parameter_dict["CdlE2"]*(Er**2)+self.parameter_dict["CdlE3"]*(Er**3))
        if "Cdlinv" not in e_class.optim_list:
            Cdlp=self.parameter_dict["Cdl"]*(1+self.parameter_dict["CdlE1"]*Er+self.parameter_dict["CdlE2"]*(Er**2)+self.parameter_dict["CdlE3"]*(Er**3))
        else:
            Cdlp=(pybamm.t <= self.parameter_dict["tr"]) *(self.parameter_dict["Cdl"]*(1+self.parameter_dict["CdlE1"]*Er+self.parameter_dict["CdlE2"]*(Er**2)+self.parameter_dict["CdlE3"]*(Er**3)))+\
            (pybamm.t > self.parameter_dict["tr"]) *(self.parameter_dict["Cdlinv"]*(1+self.parameter_dict["CdlE1inv"]*Er+self.parameter_dict["CdlE2inv"]*(Er**2)+self.parameter_dict["CdlE3inv"]*(Er**3)))
        self.model.variables={"current":self.current, "theta":self.theta}
        d_thetadt=((1-self.theta)*self.parameter_dict["k_0"]*pybamm.exp((1-alpha)*ErE0))-(self.theta*self.parameter_dict["k_0"]*pybamm.exp((-alpha)*ErE0))
        dIdt=(E_t.diff(pybamm.t)-(self.current/Cdlp)+self.parameter_dict["gamma"]*d_thetadt*(1/Cdlp))/self.parameter_dict["Ru"]
        self.model.rhs={self.current:dIdt, self.theta:d_thetadt}

    def simulate(self, nd_param_dict,time_vec ,*args):
        for key in self.dim_keys:
            self.pybam_val_dict[key]=nd_param_dict[key]
        self.model.initial_conditions={self.theta:pybamm.Scalar(0), self.current:pybamm.Scalar(0)}

        disc=pybamm.Discretisation()
        disc.process_model(self.model)
        if self.simulation_options["method"]!="dcv":
            solver=pybamm.CasadiSolver(mode="fast")
        else:
            solver=pybamm.ScipySolver()
        try:
            solution=solver.solve(self.model, time_vec, inputs=self.pybam_val_dict)
            sol=solution["current"].entries
        except:
            sol=np.zeros(len(time_vec))
        return sol
