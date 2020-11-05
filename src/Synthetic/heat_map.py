import numpy as np
import seaborn
import matplotlib.pyplot as plt
results_mat=np.load("Scan_results.txt.npy")
k0_exps=np.linspace(-1, 5, 20)
k0_vals=[10**x for x in k0_exps]
shape_linspace=np.linspace(0.2, 0.8, 20)
ax=seaborn.heatmap(np.log10(results_mat), xticklabels=shape_linspace, yticklabels=k0_exps, cbar=True)
ax.invert_yaxis()
ax.set_xlabel("k0_shape")
ax.set_ylabel("k0_scale")
plt.show()
