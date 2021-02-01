import numpy as np
import sys
print("hey")
parameters=["E0_mean","E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","omega","cap_phase","phase", "alpha"]
novel_dictionary={}
for q in parameters:
    filename="Likelihood_surfaces_"+q+".npy"
    file=np.load(filename, allow_pickle=True)
    dictionary=file.item()
    for key in dictionary:
        novel_dictionary[key]=dictionary[key]
np.save("2D_likelihoods_high_gamma.npy", novel_dictionary)
