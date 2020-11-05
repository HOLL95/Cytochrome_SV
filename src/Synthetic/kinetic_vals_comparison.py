import numpy as np
import matplotlib.pyplot as plt
kinetic_val_range=np.linspace(0.0002, 0.22, 1000)
mM_concs=[1, 2.5, 5]
concs=[x*1e-6 for x in mM_concs]
v_sol=[np.multiply(kinetic_val_range, x) for x in concs]
for i in range(0, len(v_sol)):
    plt.plot(kinetic_val_range, np.divide(v_sol[i], 1e-10), label=mM_concs[i])
plt.legend()
plt.show()
