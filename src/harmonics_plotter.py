import  matplotlib.pyplot as plt
import pints.plot
import numpy as np
import copy
class harmonics:
    def __init__(self, harmonics, input_frequency, filter_val):
        self.harmonics=harmonics
        self.num_harmonics=len(harmonics)
        self.input_frequency=input_frequency
        self.filter_val=filter_val
        print("initialised!")
    def reorder(list, order):
        return [list[i] for i in order]
    def generate_harmonics(self, times, data):
        L=len(data)
        window=np.hanning(L)
        time_series=data
        f=np.fft.fftfreq(len(time_series), times[1]-times[0])
        Y=np.fft.fft(time_series)
        last_harm=(self.harmonics[-1]*self.input_frequency)
        frequencies=f[np.where((f>0) & (f<(last_harm+(0.5*self.input_frequency))))]
        top_hat=(copy.deepcopy(Y[0:len(frequencies)]))
        harmonics=np.zeros((self.num_harmonics, len(time_series)), dtype="complex")
        for i in range(0, self.num_harmonics):
            true_harm=self.harmonics[i]*self.input_frequency
            freq_idx=np.where((frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (frequencies>true_harm-(self.input_frequency*self.filter_val)))
            filter_bit=(top_hat[freq_idx])
            harmonics[i,np.where((frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (frequencies>true_harm-(self.input_frequency*self.filter_val)))]=filter_bit
            harmonics[i,:]=((np.fft.ifft(harmonics[i,:])))
        return harmonics
    def empty(self, arg):
        return arg
    def inv_objective_fun(self, func, time_series):
        obj_func=np.append(func(time_series),0)
        time_domain=(np.fft.ifft(np.append(obj_func,np.flip(obj_func))))
        return time_domain
    def harmonic_selecter(self, ax, time_series, times, box=True, arg=np.real, line_label=None, alpha=1.0, extend=False):
        f=np.fft.fftfreq(len(time_series), times[1]-times[0])
        hann=np.hanning(len(time_series))
        time_series=np.multiply(time_series, hann)
        Y=np.fft.fft(time_series)
        last_harm=(5+self.harmonics[-1])*self.input_frequency
        first_harm=self.harmonics[0]*self.input_frequency
        frequencies=f[np.where((f>=0 )& (f<(last_harm+(0.5*self.input_frequency))))]
        fft_plot=Y[np.where((f>=0 )& (f<(last_harm+(0.5*self.input_frequency))))]
        ax.semilogy(frequencies, abs(arg(fft_plot)), label=line_label, alpha=alpha)
        if box==True:
            len_freq=np.linspace(0, 100, len(frequencies))
            longer_len_freq=np.linspace(0, 100, 10000)
            extended_frequencies=np.interp(longer_len_freq, len_freq, frequencies)
            box_area=np.zeros(len(extended_frequencies))
            for i in range(0, self.num_harmonics):
                true_harm=self.harmonics[i]*self.input_frequency
                peak_idx=np.where((frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (frequencies>true_harm-(self.input_frequency*self.filter_val)))
                extended_peak_idx=np.where((extended_frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (extended_frequencies>true_harm-(self.input_frequency*self.filter_val)))
                box_area[extended_peak_idx]=max(fft_plot[peak_idx])
            ax.plot(extended_frequencies, box_area, color="r", linestyle="--")

    def plot_harmonics(self, times, method, **kwargs):
        if method=="abs":
            a=abs
        else:
            a=self.empty
        harmonics_list=[]
        harmonics_labels=[]
        for key, value in list(kwargs.items()):
            harmonics_list.append(value)
            harmonics_labels.append(key)
        fig, ax=plt.subplots(self.num_harmonics,1)
        for i in range(0, self.num_harmonics):
            for j in range(0, len(harmonics_labels)):
                    ax[i].plot(times, a(harmonics_list[j][i,:]), label=harmonics_labels[j])
        ax[i].yaxis.set_label_position("right")
        ax[i].set_ylabel(str(self.harmonics[i]), rotation=0)
        plt.legend()
        plt.show()
    def harmonics_plus(self, title, method, times, **kwargs):
        plt.rcParams.update({'font.size': 12})
        large_plot_xaxis=times
        fig=plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
        if method=="abs":
            a=abs
        else:
            a=self.empty
        time_list=[]
        titles=[]
        for key, value in list(kwargs.items()):
            if key=="voltage":
                large_plot_xaxis=voltages
                continue
            time_list.append(value)
            titles.append(key)

        title_lower=[x.lower() for x in titles]
        exp_idx=title_lower.index("experimental")
        new_order=list(range(0, len(titles)))
        new_order[0]=exp_idx
        new_order[exp_idx]=0
        titles=[titles[x] for x in new_order]
        time_list=[time_list[x] for x in new_order]
        harmonics_list=[]
        print("~"*50)
        for i in range(0, len(time_list)):
            harms=self.generate_harmonics(times, time_list[i])
            harmonics_list.append(harms)
        harm_axes=[]
        harm_len=2
        fig.text(0.03, 0.5, 'Current($\mu$A)', ha='center', va='center', rotation='vertical')

        for i in range(0,self.num_harmonics):
            harm_axes.append(plt.subplot2grid((self.num_harmonics,harm_len*2), (i,0), colspan=harm_len))
            for q in range(0, len(titles)):
                harm_axes[i].plot(times, np.multiply(a(harmonics_list[q][i,:]), 1e6), label=titles[q])
            harm_axes[i].yaxis.set_label_position("right")
            harm_axes[i].set_ylabel(str(self.harmonics[i]), rotation=0)
        harm_axes[i].legend()
        harm_axes[i].set_xlabel("Time(s)")

        time_ax=plt.subplot2grid((self.num_harmonics,harm_len*2), (0,harm_len), rowspan=self.num_harmonics, colspan=harm_len)
        for p in range(0, len(titles)):
            if titles[p].lower()=="experimental":
                time_ax.plot(large_plot_xaxis, np.multiply(time_list[p], 1e3), label=titles[p], alpha=1.0)
            else:
                time_ax.plot(large_plot_xaxis, np.multiply(time_list[p], 1e3), label=titles[p], alpha=0.5)
        time_ax.set_ylabel("Current(mA)")
        time_ax.set_xlabel("Time(s)")
        plt.legend()
        plt.suptitle(title)
        plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.92, wspace=0.23)
        plt.show()
