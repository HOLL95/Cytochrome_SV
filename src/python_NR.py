import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
class NR_test:
    def __init__(self, In0, u1n0,E, dE,dt, param_dict):
        self.param_dict=param_dict
        self.In0=In0
        self.u1n0=u1n0
        self.dE=dE
        self.E=E
        self.dt=dt
    def residual(self, In1):
        self.update_temporaries(In1)
        return self.Cdlp*(self.dt*self.dE-self.param_dict["Ru"]*(In1-self.In0)) - self.dt*In1 + self.param_dict["gamma"]*(self.u1n1-self.u1n0)


    def residual_gradient(self, In1):
        return -self.Cdlp*self.param_dict["Ru"] - self.dt + self.param_dict["gamma"]*self.du1n1


    def update_temporaries(self, In1):
        Ereduced = self.E - self.param_dict["Ru"]*In1
        cap_E=self.E
        cER=cap_E-self.param_dict["Ru"]*In1
        Ereduced2 = np.power(cER,2)
        Ereduced3 = cER*Ereduced2
        expval1 = Ereduced - self.param_dict["E_0"]
        exp11 = np.exp((1.0-self.param_dict["alpha"])*expval1)
        exp12 = np.exp(-self.param_dict["alpha"]*expval1)

        dexp11 = -self.param_dict["Ru"]*(1.0-self.param_dict["alpha"])*exp11
        dexp12 = self.param_dict["Ru"]*self.param_dict["alpha"]*exp11

        u1n1_top = self.dt*self.param_dict["k_0"]*exp11 + self.u1n0
        du1n1_top = self.dt*self.param_dict["k_0"]*dexp11
        denom = (self.dt*self.param_dict["k_0"]*exp11 +self.dt*self.param_dict["k_0"]*exp12 + 1)
        ddenom = self.dt*self.param_dict["k_0"]*(dexp11 + dexp12)
        tmp = 1.0/denom
        tmp2 = np.power(tmp,2)
        self.u1n1 = u1n1_top*tmp
        self.du1n1 = -(u1n1_top*ddenom + du1n1_top*denom)*tmp2
        self.Cdlp = self.param_dict["Cdl"]*(1.0 + self.param_dict["CdlE1"]*cER + self.param_dict["CdlE2"]*Ereduced2 + self.param_dict["CdlE3"]*Ereduced3)
class python_NR_simulation:
    def __init__(self, times, param_dict, potential_func, dt=None):
        if dt==None:
            dt=times[1]-times[0]
        E, dE=potential_func(0)
        I1=param_dict["Cdl"]*dE
        I0=I1
        theta_0=0
        t1=0
        numerical_current=np.zeros(len(times))
        numerical_current[0]=I1
        desired_error=3e-2
        r=1.01*desired_error
        h_scaling=0.9
        modified_dt=dt
        solutions=[0, 0]
        dt_scaling=[1, 0.5]
        for i in range(1, len(times)):
            r=1.01*desired_error
            while r>desired_error:
                if modified_dt!=dt:
                    modified_dt=modified_dt*0.9*(desired_error/r)
                h=modified_dt
                for  j in range(0, len(dt_scaling)):
                    modified_dt*=dt_scaling[j]
                    I0=numerical_current[i-1]
                    #print(I0, "prev current")
                    t1=times[i-1]
                    current_test=[]
                    time_test=[]
                    while t1<times[i]:
                        I0=I1
                        E,dE=potential_func(t1)
                        solver_class=NR_test(I0, theta_0, E, dE, modified_dt, param_dict)
                        I1=newton(solver_class.residual, I0, solver_class.residual_gradient)
                        #current_test.append(I1)
                        #print(I1, modified_dt, j)
                        #time_test.append(t1)
                        #print("time=", t1," current= ", I1, j)

                        theta_0=solver_class.u1n1
                        t1+=modified_dt

                    print("time=", t1, "expected time", times[i], j)
                    solutions[j]=I1
                    #plt.plot(time_test, current_test, alpha=0.7)
                    #plt.axvline(times[i])
                r=abs(solutions[1]-solutions[0])/h
                print(r, i)
                print("solution 2=", solutions[1], "solution 1=", solutions[0], r)
                #plt.show()
            numerical_current[i]=solutions[1]#((I1-I0)*(times[i]-t1+dt)/dt)+I0
            #plt.plot(numerical_current[np.where(numerical_current!=0)])
            #plt.show()
        self.numerical_current=numerical_current
