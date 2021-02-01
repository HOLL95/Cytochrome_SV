import numpy as np
import matplotlib.pyplot as plt
import pints
from pints import plot
def chain_appender(chains, burn=5000):
    first_chain=chains[0, burn:]
    for i in range(1, len(chains[:, 0])):
        first_chain=np.append(first_chain, chains[i, burn:])
    return first_chain
true_vals=[-0.25, 10000.0, 10.0, 1e-05*1e6, 0.5]
params=["$E^0(V)$", "$k_0(s^{-1})$", "$R_u(\\Omega)$", "$C_{dl}(\\mu F)$", "$\\alpha$","$\\sigma$"]
files=["Sens_MCMC_high_f","ED_MCMC"]
d_E_errors=[2.1973630798514976, 0.5772175913790867]
just_f_errors=[0.19127081488301423,0.14939889969993458]
sd_ratios=np.zeros(len(params))
labels=["$\it{de}$ $\it{novo}$ parameters", "FIM parameters"]
for z in range(0, len(files)):
    chains=np.load("MCMC_results/"+ files[z])
    for i in range(0, len(chains[0,0,:])):
        plt.subplot(2,3, i+1)
        #for q in range(0, len(chains[:,0,0])):


        plt.xlabel(params[i])
        ax=plt.gca()
        if "C_" in params[i]:
            chain=chain_appender(chains[:, :, i], 5000)*1e6
            plt.hist(chain, alpha=0.5)
        else:
            chain=chain_appender(chains[:, :, i], 5000)
            plt.hist(chain, alpha=0.5, label=labels[z])
        if z==0:
            sd_ratios[i]=np.std(chain)
        else:
            sd_ratios[i]=sd_ratios[i]/np.std(chain)
        if "\\alpha" in params[i]:
            ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.4], ncol=2, frameon=False)
        plt.ylabel("Frequency")
        if i<len(chains[0,0,:])-1:
            plt.axvline(true_vals[i], color="black", linestyle="--")
            if z==1:
                ax.text(0.85, 0.9,round(sd_ratios[i],2),
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,
                fontsize=12)
            #ax.set_title(round(sd_ratios[i],2))
plt.subplots_adjust(top=0.99,
                    bottom=0.13,
                    left=0.11,
                    right=0.99,
                    hspace=0.285,
                    wspace=0.46)
fig=plt.gcf()
plt.show()
fig.savefig("SD_ratios.png", dpi=500)
