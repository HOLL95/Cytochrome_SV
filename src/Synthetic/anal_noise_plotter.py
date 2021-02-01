import numpy as np
import matplotlib.pyplot as plt
import pickle
noise_array=np.load("Parameter_inference_changing_noise.npy", allow_pickle=True)
MCMC_array=np.load("Numerical_anal_comp_MCMC", allow_pickle=True)
#print(noise_dict)
#noises=[x/100 for x in np.flip([0.5, 1, 2, 3, 4, 5])]
def approx_error(x, x_true):
    return (abs(x-x_true)/abs(x_true))*100
T=(273+25)
F=96485.3328959
R=8.314459848
RTF=R*T/F
bins=20
for i in np.flip(range(1, len(noise_array))):
    noise_dict=noise_array[i]
    MCMC_dict=MCMC_array[i]
    for key in noise_dict.keys():
        label=str(float(key)*100)+"%"
        plt.subplot(1,2,1)
        plt.hist(np.multiply(noise_dict[key], RTF), alpha=0.7, label=label, bins=bins)
        plt.xlabel("$\~E^0$(V)")
        plt.ylabel("Frequency")
        plt.title("Analytical inference")
        plt.axvline(1*RTF, color="black", linestyle="--")
        plt.subplot(1,2,2)
        MC_key=list(MCMC_dict.keys())[0]
        E0_chains=MCMC_dict[MC_key][0::4]
        E0_plot_chains=[]
        for j in range(1, 4):
            E0_plot_chains=np.append(E0_plot_chains, E0_chains[j*9000:j*10000])
        plt.hist(E0_plot_chains, alpha=0.7, label=label, bins=bins)
        plt.legend()
        plt.axvline(1*RTF, color="black", linestyle="--")
        no_nan=np.isnan(noise_dict[key])
        result_list=[label, str(RTF*np.mean(noise_dict[key][~no_nan])),str(RTF*np.std(noise_dict[key][~no_nan])), str(np.mean(E0_plot_chains)),str(np.std(E0_plot_chains))]

        print((" & ").join(result_list)+"\\\\")
        plt.xlabel("$\~E^0$(V)")
        plt.ylabel("Frequency")
        plt.title("Numerical inference")

plt.legend()
plt.subplots_adjust(top=0.947,
bottom=0.091,
left=0.062,
right=0.987,
hspace=0.2,
wspace=0.144)
fig=plt.gcf()

plt.show()
fig.savefig("Anal_vs_MCMC.png", dpi=500)
