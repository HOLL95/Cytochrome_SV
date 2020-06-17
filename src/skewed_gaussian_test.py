import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
length=100
location=-0.2
scale=0.1
for a in np.arange(-3, 3):
    err=1e-5
    distribution_vals=np.zeros(length)
    distribution_weights=np.zeros(length)
    start=skewnorm.ppf(err, a, loc=location, scale=scale)
    end=skewnorm.ppf(1-err, a, loc=location, scale=scale)
    vals=np.linspace(start, end, length)
    distribution_vals[0]=skewnorm.ppf(err/2, a, loc=location, scale=scale)
    distribution_weights[0]=skewnorm.cdf(start,a, loc=location, scale=scale)
    for i in range(1, len(vals)):
        distribution_weights[i]=skewnorm.cdf(vals[i],a, loc=location, scale=scale)-skewnorm.cdf(vals[i-1], a, loc=location, scale=scale)
        distribution_vals[i]=(vals[i]+vals[i-1])/2
    plt.plot(distribution_vals, distribution_weights, label=a)
    print(np.sum(distribution_weights))
plt.legend()
plt.show()
