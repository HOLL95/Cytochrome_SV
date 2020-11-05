import numpy as np
import matplotlib.pyplot as plt
freq_array=np.load("Parameter_inference_changing_freqs.npy")
freq_range=np.linspace(1, 200, 50)
num_oscillations=[10, 50, 100, 200, 300]
print(len(freq_array))
plt.plot(freq_range, (freq_array[0]))
plt.xlabel("Input frequency")
plt.ylabel("Estimated $E_0$")
plt.axhline(1, color="black", linestyle="--")
plt.show()
