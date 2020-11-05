import numpy as np
import matplotlib.pyplot as plt
noise_array=np.load("Parameter_inference_changing_noise.npy")
noises=np.linspace(-3, -1, 20)
noises=[0]+[10**x for x in noises]
plt.errorbar(np.log10(noises), noise_array[0], yerr=noise_array[1])
ax=plt.gca()
xticks=ax.get_xticks()
ax.set_xticklabels(["0%"]+[str(round(x, 1))+"%" for x in np.power(10, xticks[1:])*100])
plt.xlabel("Noise standard deviation")
plt.ylabel("Estimated $E_0$")
plt.axhline(1, color="black", linestyle="--")
plt.show()
