import pints
from pints.plot import trace
import numpy as np
import matplotlib.pyplot as plt
chains=np.load("alice_cyt_1_MCMC")
trace(chains[:, :, :])
plt.show()
