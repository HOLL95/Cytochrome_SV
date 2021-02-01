from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
curves=np.load("Likelihood_curves.npy", allow_pickle=True)
surfaces=np.load("2D_likelihoods.npy", allow_pickle=True)
curve_dict=curves.item()
likelihood_dict=surfaces.item()
params=["E0_mean","E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","cap_phase","phase", "alpha"]
"""for desired_key in likelihood_dict.keys():
    if "omega" in desired_key:
        X = likelihood_dict[desired_key]["X"]
        Y = likelihood_dict[desired_key]["Y"]

        Z = likelihood_dict[desired_key]["Z"]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(X, Y, np.array(Z), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.

        ax.set_title(desired_key)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()"""
fig=plt.figure()
counter=0
for i in range(0, len(params)):
    for j in range(0, len(params)):

            #axes=fig.add_subplot(len(params), len(params), (i*len(params))+1)
            #print((i*len(params))+1)
            #axes.plot(curve_dict[params[i]]["X"], curve_dict[params[i]]["Y"])
        if i>j:
            ax=fig.add_subplot(len(params), len(params), (i*len(params))+j+1)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            #axes.set_title((i*len(params))+j+1)
            """desired_key=params[i]+"_"+params[j]
            X = likelihood_dict[desired_key]["X"]
            Y = likelihood_dict[desired_key]["Y"]
            Z = likelihood_dict[desired_key]["Z"]"""
        elif i==j:
            #
            ax=fig.add_subplot(len(params), len(params), (i*len(params))+1+j)

plt.show()
