import os
import matplotlib.pyplot as plt
import numpy as np
from pints import plot
chains=np.load("alice_cyt_2_MCMC")
plot.trace(chains)
plt.show()
