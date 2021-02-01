import os
import matplotlib.pyplot as plt
import numpy as np
from pints import plot
dir=os.getcwd()
location=dir+"/MCMC/DCV/"
files=os.listdir(location)
def plot_kde_2d(x, y, ax):
    # Get minimum and maximum values
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(x, y, s=0.5, alpha=0.5)#, cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax])
    ax.locator_params(nbins=2)

burn=5000
optim_list=["E0_mean", "E0_std", "k_0", "Ru", "Cdl", "gamma", "alpha", "noise"]

unit_dict={
    "E_0": "V",
    'E_start': "V", #(starting dc voltage - V)
    'E_reverse': "V",
    'omega':"Hz",#8.88480830076,  #    (frequency Hz)
    'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
    'v': '$s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm^{-2}$',
    'k_0': 's^{-1}$', #(reaction rate s-1)
    'alpha': "",
    "E0_mean":"V",
    "E0_std": "V",
    "k0_shape":"",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"",
    'phase' : "",
    "alpha_mean": "",
    "alpha_std": "",
    "":"",
    "noise":"",
}
fancy_names={
    "E_0": '$E^0$',
    'E_start': '$E_{start}$', #(starting dc voltage - V)
    'E_reverse': '$E_{reverse}$',
    'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
    'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "v",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Ru",  #     (uncompensated resistance ohms)
    'Cdl': "$C_{dl}$", #(capacitance parameters)
    'CdlE1': "$C_{dlE1}$",#0.000653657774506,
    'CdlE2': "$C_{dlE2}$",#0.000245772700637,
    'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
    'gamma': '$\\Gamma',
    'k_0': '$k_0', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "cap_phase":"$C_{dl}$ phase",
    "alpha_mean": "$\\alpha\\mu$",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "Phase",
    "":"Experiment",
    "noise":"$\sigma$",
}
titles=[fancy_names[x]+"("+unit_dict[x]+")" if (unit_dict[x]!="") else fancy_names[x] for x in optim_list]
for filename in files:
    if "MCMC_2" not in filename:
        chain=np.load(location+filename)
        #plot.trace(chain)
        #plt.show()
        if len(chain[0,:,0])>1000:

            print(chain)
            print(filename)
            n_params=len(chain[0,0,:])
            fig, ax=plt.subplots(n_params, n_params)

            for i in range(0, n_params):
                for j in range(0, n_params):
                    if i==j:
                        for q in range(0, 3):
                            ax[i,j].hist(chain[q, burn:, i])
                    elif i<j:
                        ax[i,j].axis('off')
                    else:
                        for q in range(0, 3):
                            plot_kde_2d(chain[q, burn:, i], chain[q, burn:, j], ax[i,j])
                    if i!=0:
                        ax[i, 0].set_ylabel(titles[i])
                    if i<n_params-1:
                        ax[i,j].set_xticklabels([])#
                    if j>0 and i!=j:
                        ax[i,j].set_yticklabels([])
                    if j!=n_params:
                        ax[-1, i].set_xlabel(titles[i])
                        plt.setp( ax[-1, i].xaxis.get_majorticklabels(), rotation=15 )
            plt.show()
            fig.clf()
