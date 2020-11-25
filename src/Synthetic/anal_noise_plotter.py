import numpy as np
import matplotlib.pyplot as plt
noise_array=np.load("Parameter_inference_changing_noise.npy")
noises=np.linspace(-3, -1, 20)
noises=[0]+[10**x for x in noises]
def approx_error(x, x_true):
    return (abs(x-x_true)/abs(x_true))*100
T=(273+25)
F=96485.3328959
R=8.314459848
RTF=R*T/F
noise_array_error=[approx_error(x, 1) for x in noise_array[0]]
plt.errorbar(np.log10(noises), noise_array[0]*RTF, yerr=noise_array[1]*RTF)
ax=plt.gca()
xticks=ax.get_xticks()
ax.set_xticklabels(["0%"]+[str(round(x, 1))+"%" for x in np.power(10, xticks[1:])*100])
plt.xlabel("Noise standard deviation")
plt.ylabel("Estimated $E_0(V)$")
plt.axhline(1*RTF, color="black", linestyle="--")
plt.show()
