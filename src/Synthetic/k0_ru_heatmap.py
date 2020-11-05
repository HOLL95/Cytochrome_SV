import seaborn
import numpy as np
import copy
import matplotlib.pyplot as plt
k0_vals=[10**x for x in range(-2, 4)]#3z\13ave
ru_vals=copy.deepcopy(k0_vals)
kinetic_information=np.load("Scan_results.txt.npy")
ax=seaborn.heatmap(np.log10(kinetic_information), xticklabels=ru_vals, yticklabels=k0_vals, annot=True, cbar=True,cbar_kws={'label': 'log10(Input frequency)'})
ax.invert_yaxis()
ax.set_xlabel("Ru")
ax.set_ylabel("k0")
plt.show()
