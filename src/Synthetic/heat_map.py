import numpy as np
import seaborn
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.colors import LogNorm
k0_exps=np.linspace(-1, 4, 50)
k0_vals=[10**x for x in k0_exps]#3z\13ave
ru_vals=copy.deepcopy(k0_vals)
def truncate(n, round_n):
    denom=10**round_n
    return int(n*denom)/denom
fig, ax=plt.subplots()
k0_powers=range(-1, 5)
"""long_k0_exps=np.linspace(-1, 4, 500)
long_k0=[10**x for x in long_k0_exps]
xx,yy=np.meshgrid(long_k0, long_k0)
results_mat=np.load("FIM_K0.npy")
z=results_mat
points=np
grid_z0 = griddata(points, values, (xx, yy), method='nearest')"""
k0_xtick=[10**x for x in k0_powers]

heat_map_results=["FIM_K0.npy", "FIM_K0_ru.npy"]
xx,yy=np.meshgrid(k0_vals, ru_vals)
results_mat=np.load("FIM_K0.npy")
z=results_mat
f = interpolate.interp2d(k0_vals, ru_vals, z, kind='linear')
long_k0_exps=np.linspace(-1, 4, 1000)
long_k0=[10**x for x in long_k0_exps]
fnew=f(long_k0, long_k0)


interpolated_postions=np.interp(k0_xtick, long_k0, range(0, len(long_k0)))
k0_xtick_labels=["10$^{"+str(x)+"}$" for x in k0_powers]
results_mat=fnew
log_norm = LogNorm(vmin=results_mat.min().min(), vmax=results_mat.max().max())
ax=seaborn.heatmap(fnew, norm=log_norm, cmap="viridis",cbar_kws={"label":"$\\omega(Hz)$"})
seaborn.heatmap(results_mat,norm=log_norm, cbar=True, ax=ax,cbar_kws={"ticks":None})
ax2=plt.contour(np.log10(fnew))
ax.set_xticks(interpolated_postions)
ax.set_xticklabels([str(x) for x in k0_xtick_labels], rotation=0)
ax.set_yticks(interpolated_postions)
ax.set_yticklabels([str(x) for x in k0_xtick_labels])
ax.invert_yaxis()
#cbar = ax.collections[0].colorbar
#cbar.set_ticklabels(["10$^{"+str(x)+"}$" for x in range(-1, 4)])
ax.set_xlabel("$R_u(\\Omega)$")
ax.set_ylabel("$k_0(s^{-1})$")
plt.subplots_adjust(top=0.975,
                    bottom=0.115,
                    left=0.12,
                    right=0.98,
                    hspace=0.2,
                    wspace=0.2)
fig.set_size_inches((6, 4.5))
plt.show()
fig.savefig("KI_best_freq.png", dpi=500)
