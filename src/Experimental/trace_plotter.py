import numpy as np
import matplotlib.pyplot as plt
from pints import plot
chains=np.load("alice_cyt_1_MCMC_log_transformed")

#plot.pairwise(np.vstack(chains[:, 10000:]), parameter_names=["E0_mean","E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","omega","cap_phase","phase"])
plot.trace(chains[:, :, :],  parameter_names=["E0_mean","E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","omega","cap_phase","phase"])
plt.show()
