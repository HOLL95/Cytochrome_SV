class oscillator:
    def __init__(self, times):
        self.times=times
    def damped_osc(self, A,lamb,frequency,phase):
        cosine=np.cos(np.add(np.multiply(self.times, frequency), phase))
        exponent=np.exp(np.multiply(-lamb, self.times))
        return A*exponent*cosine
