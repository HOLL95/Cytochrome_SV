import numpy as np
import matplotlib.pyplot as plt
import pints
from pints import plot
def chain_appender(chains, burn=5000):
    first_chain=chains[0, burn:]
    for i in range(1, len(chains[:, 0])):
        first_chain=np.append(first_chain, chains[i, burn:])
    return first_chain
true_vals=[-0.25, 10000.0, 10.0, 1e-05, 0.5]
files=["ED_MCMC_d_E", "Sens_MCMC_d_E"]
d_E_errors=[2.1973630798514976, 0.5772175913790867]
just_f_errors=[0.19127081488301423,0.14939889969993458]
for q in range(-5, -1):
    for z in range(0, len(files)):
        chains=np.load("MCMC_results/"+files[z]+"_"+str(q))
        for i in range(0, len(chains[0,0,:])):
            plt.subplot(2,3, i+1)
            #for q in range(0, len(chains[:,0,0])):
            plt.hist(chain_appender(chains[:, :, i], 5000), alpha=0.5)
            if i<len(chains[0,0,:])-1:
                if i==3:
                    plt.axvline(10**q, color="black", linestyle="--")
                else:
                    plt.axvline(true_vals[i], color="black", linestyle="--")


    plt.show()
