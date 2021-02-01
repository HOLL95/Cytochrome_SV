import math
import numpy as np
import matplotlib.pyplot as plt
from anal_params_class import params
import copy
import time
from scipy.special import iv
class analytical_electron:
    def __init__(self, dim_paramater_dictionary, gamma_0):
        key_list=list(dim_paramater_dictionary.keys())
        self.dim_dict=dim_paramater_dictionary
        self.nd_param=params(dim_paramater_dictionary)
        for i in range(0, len(key_list)):
            self.nd_param.non_dimensionalise(key_list[i], dim_paramater_dictionary[key_list[i]])
        nd_dict=self.nd_param.__dict__
        self.nu=((self.nd_param.E_reverse+self.nd_param.E_start)/2)-self.nd_param.E_0
        I_0_1_alpha=iv(0, (1-self.nd_param.alpha)*self.nd_param.d_E)
        I_0_alpha=iv(0, self.nd_param.alpha*self.nd_param.d_E)
        self.gamma_inf=I_0_1_alpha/(I_0_1_alpha+(np.exp(-self.nu)*I_0_alpha))
        self.sigma=(np.exp((1-self.nd_param.alpha)*self.nu)*I_0_1_alpha)+np.exp(-self.nd_param.alpha*self.nu)*I_0_alpha
        self.gamma_0=gamma_0
        self.I_0_1_alpha=I_0_1_alpha
        self.I_0_alpha=I_0_alpha
        self.a_coeff=2*(np.exp(-self.nd_param.alpha*self.nu)/(self.I_0_1_alpha+(np.exp(-self.nu)*self.I_0_alpha)))
    def h(self, t):
        h=np.exp((1-self.nd_param.alpha)*self.nu)*np.exp((1-self.nd_param.alpha)*self.nd_param.d_E*np.sin(self.nd_param.nd_omega*t+self.nd_param.phase))
        return h
    def e(self, t):
        return self.nd_param.d_E*np.sin(self.nd_param.nd_omega*t+self.nd_param.phase)
    def update_Ein(self, times):
        volts=[self.e(t) for t in times]
        self.nu=np.mean(volts)-self.nd_param.E_0
    def g(self, t):
        g=self.h(t)+np.exp(-self.nd_param.alpha*self.nu)*np.exp((-self.nd_param.alpha)*self.nd_param.d_E*np.sin(self.nd_param.nd_omega*t+self.nd_param.phase))
        return g
    def i(self,t):
        i=self.h(t)-(self.gamma_inf+(self.gamma_0-self.gamma_inf)*np.exp(-self.sigma*t))*self.g(t)
        return i
    def gamma_t(self, t):
        return self.gamma_inf+(self.gamma_0-self.gamma_inf)*np.exp(-self.sigma*t)
    def diff_eq_gamma(self, gamma, t):
        return self.h(t)-self.g(t)*gamma
    def p_k(self, m, alpha, Delta, eta):
        if m%2==0:
            p_k=(iv(0,(alpha*Delta))*iv(m,(1-alpha)*Delta))-(iv(0,(1-alpha)*Delta)*iv(m,(alpha)*Delta))
        elif m%2==1:
            p_k=(iv(0,(alpha*Delta))*iv(m,(1-alpha)*Delta))+(iv(0,(1-alpha)*Delta)*iv(m,(alpha)*Delta))
        return p_k
    def M(self, t1, p):
        M0=(self.nd_param.nd_omega/(2*p*math.pi*self.sigma))*(np.exp((1-self.nd_param.alpha)*self.nu)*self.I_0_1_alpha-(self.gamma_0*self.sigma))*(1-np.exp(-(2*p*math.pi*self.sigma)/self.nd_param.nd_omega))
        return M0*np.exp(-self.sigma*t1)
    def bell_amplitude(self, eta=-1, alpha=0.55, Delta=3):
        print("Bell params", eta, alpha, Delta)
        A2act=(2*np.exp((1-alpha)*eta)/(np.exp(eta)*iv(0,(1-alpha)*Delta)+iv(0,-alpha*Delta))*abs(iv(0,alpha*Delta)*iv(2,(1-alpha)*Delta)-iv(0,(1-alpha)*Delta)*iv(2,alpha*Delta)))
        A3act=(2*np.exp((1-alpha)*eta)/(np.exp(eta)*iv(0,(1-alpha)*Delta)+iv(0,-alpha*Delta))*abs(iv(0,alpha*Delta)*iv(3,(1-alpha)*Delta)+iv(0,(1-alpha)*Delta)*iv(3,alpha*Delta)))
        A4act=(2*np.exp((1-alpha)*eta)/(np.exp(eta)*iv(0,(1-alpha)*Delta)+iv(0,-alpha*Delta))*abs(iv(0,alpha*Delta)*iv(4,(1-alpha)*Delta)-iv(0,(1-alpha)*Delta)*iv(4,alpha*Delta)))
        return [A2act, A3act, A4act]
    def nd_param_estimator(self, m, amp_1, amp_2, **kwargs):
        #print(amp_1, amp_2, "amp_1, amp2")
        alpha=kwargs["alpha"]
        dE1=kwargs["Delta1"]
        eta1=kwargs["eta1"]
        dE2=kwargs["Delta2"]
        eta2=kwargs["eta2"]
        E_in_1=kwargs["E_in_1"]
        E_in_2=kwargs["E_in_2"]
        p_k_de1=abs(self.p_k(m, alpha, dE1, eta1))
        p_k_de2=abs(self.p_k(m, alpha, dE2, eta2))
        arg1=np.exp((1-alpha)*E_in_1)*p_k_de1*iv(0, -alpha*dE2)*amp_2
        arg2=np.exp((1-alpha)*E_in_2)*p_k_de2*iv(0, -alpha*dE1)*amp_1
        arg3=np.exp((1-alpha)*(E_in_1+E_in_2))*p_k_de2*iv(0, (1-alpha)*dE1)*amp_1
        arg4=np.exp((1-alpha)*(E_in_1+E_in_2))*p_k_de1*iv(0, (1-alpha)*dE2)*amp_2
        RHS=(arg1-arg2)/(arg3-arg4)
        if RHS>0:
            E0_estimate=-np.log(RHS)
        else:
            E0_estimate=None
        return E0_estimate
    def nd_harmonic_amp(self, m, **kwargs):
        alpha=kwargs["alpha"]
        Delta=kwargs["Delta"]
        eta=kwargs["eta"]
        numerator=2*np.exp((1-alpha)*eta)
        denominator=(np.exp(eta)*iv(0,(1-alpha)*Delta)+iv(0,-alpha*Delta))
        return (numerator/denominator)*abs(self.p_k(m, alpha, Delta, eta))
    def analytical_e0_estimate(self, harm_range, set1, set2):
        if type(harm_range) is not list:
            harm_range=[harm_range]
        E0_estimates=np.zeros(len(harm_range))
        for i in range(0, len(harm_range)):
            m=harm_range[i]
            amp_1=self.nd_harmonic_amp(m, alpha=set1["alpha"], Delta=set1["Delta"], eta=set1["eta"])
            amp_2=self.nd_harmonic_amp(m, alpha=set2["alpha"], Delta=set2["Delta"], eta=set2["eta"])
            E0_estimates[i]=self.nd_param_estimator(m, amp_1, amp_2,
                                                    alpha=set1["alpha"], Delta1=set1["Delta"], eta1=set1["eta"], E_in_1=set1["E_in"],
                                                    Delta2=set2["Delta"], eta2=set2["eta"], E_in_2=set2["E_in"])
        return E0_estimates
