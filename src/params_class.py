#!/usr/bin/env python
import math
import warnings
import copy
class params:
    def __init__(self,param_dict):
        self.param_dict=copy.deepcopy(param_dict)

        self.T=(273+25)
        self.F=96485.3328959
        self.R=8.314459848
        self.c_E0=(self.R*self.T)/self.F
        self.c_Gamma=self.param_dict["original_gamma"]
        if "v" not in self.param_dict or "original_omega" in self.param_dict:
            self.param_dict["v"]=self.c_E0*self.param_dict["original_omega"]
        self.c_T0=abs(self.c_E0/self.param_dict["v"])
        self.c_I0=(self.F*self.param_dict["area"]*self.c_Gamma)/self.c_T0

        self.method_switch={
                            'e_0':self.e0,
                            'k_0':self.k0,
                            'cdl':self.cdl,
                            'e_start' :self.estart,
                            'e_reverse': self.erev,
                            'omega':self.omega_d,
                            'd_e' :self.de,
                            'ru':self.ru,
                            'gamma':self.Gamma,
                            'sampling_freq':self.sf
                            }
        keys=sorted(param_dict.keys())
        for i in range(0, len(keys)):
            if keys[i].lower() in self.method_switch:
                self.non_dimensionalise(keys[i], param_dict[keys[i]])
        self.nd_param_dict=self.param_dict


    def non_dimensionalise(self, name,name_value):
        function = self.method_switch[name.lower()]
        function(name_value, 'non_dim')
    def re_dimensionalise(self,name, name_value):
        if name.lower() in self.method_switch:
                function = self.method_switch[name.lower()]
                function(name_value, 're_dim')
        else:
            raise ValueError(name + " not in param list!")
    def e0(self, value, flag):
        if flag=='re_dim':
            self.param_dict["E_0"]=value*self.c_E0
            if "E0_std" in self.param_dict:
                self.param_dict["E0_std"]=self.param_dict["E0_std"]*self.c_E0
                self.param_dict["E0_mean"]=self.param_dict["E0_mean"]*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["E_0"]=value/self.c_E0
            if "E0_std" in self.param_dict:
                self.param_dict["E0_std"]=self.param_dict["E0_std"]/self.c_E0
                self.param_dict["E0_mean"]=self.param_dict["E0_mean"]/self.c_E0
    def k0(self, value, flag):
        if flag=='re_dim':
            self.param_dict["k_0"]=value/self.c_T0
            if "k0_scale" in self.param_dict:
                self.param_dict["k0_scale"]=self.param_dict["k0_scale"]/self.c_T0
        elif flag == 'non_dim':
            self.param_dict["k_0"]=value*self.c_T0
            if "k0_scale" in self.param_dict:
                self.param_dict["k0_scale"]=self.param_dict["k0_scale"]*self.c_T0
    def cdl(self, value, flag):
        if flag=='re_dim':
            self.param_dict["Cdl"]=value*self.c_I0*self.c_T0/(self.param_dict["area"]*self.c_E0)
        elif flag == 'non_dim':
            self.param_dict["Cdl"]=value/self.c_I0/self.c_T0*(self.param_dict["area"]*self.c_E0)
    def estart(self, value, flag):
        if flag=='re_dim':
            self.param_dict["E_start"]=value*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["E_start"]=value/self.c_E0
    def erev(self, value, flag):
        if flag=='re_dim':
            self.param_dict["E_reverse"]=value*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["E_reverse"]=value/self.c_E0
    def omega_d(self, value, flag):

        if flag=='re_dim':
            self.param_dict["omega"]=value/(2*math.pi*self.c_T0)
        elif flag == 'non_dim':
            self.param_dict["nd_omega"]=value*(2*math.pi*self.c_T0)
    def de(self, value, flag):
        if flag=='re_dim':
            self.param_dict["d_E"]=value*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["d_E"]=value/self.c_E0
    def ru(self, value, flag):
        if flag=='re_dim':
            self.param_dict["Ru"]=value*self.c_E0/self.c_I0
        elif flag == 'non_dim':
            self.param_dict["Ru"]=value/self.c_E0*self.c_I0
    def Gamma(self, value, flag):
        if flag=='re_dim':
            self.param_dict["gamma"]=value*self.c_Gamma
        elif flag == 'non_dim':
            self.param_dict["gamma"]=value/self.c_Gamma
    def sf(self, value, flag):

        if flag=='re_dim':
            self.param_dict["sampling_freq"]=value/((2*math.pi)/self.param_dict["nd_omega"])
        elif flag == 'non_dim':
            self.param_dict["sampling_freq"]=value*((2*math.pi)/self.param_dict["nd_omega"])
