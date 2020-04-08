import numpy as np
"""error=y-y2
real_error=np.mean(np.real(error))
imag_error=np.mean(np.imag(error))*1j
mean=np.array([real_error, imag_error])
hermite_error=np.transpose(np.conj(mean))
print(hermite_error)
term2=np.dot(hermite_error, inv_cov)
term2=np.dot(term2, mean)
print("complex?",term2)"""
