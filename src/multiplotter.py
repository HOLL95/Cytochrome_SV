import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
class multiplot:
    def __init__(self, num_rows, num_cols, **kwargs):
        if (num_rows%1)!=0 or (num_cols%1)!=0:
            raise ValueError("require integer row and column numbers")
        else:
            num_rows=int(num_rows)
            num_cols=int(num_cols)
        #mpl.rcParams['xtick.labelsize'] = 12
        #mpl.rcParams['ytick.labelsize'] = 12
        #mpl.rcParams['axes.labelsize'] = 12

        if "num_harmonics" not in kwargs:
            kwargs["num_harmonics"]=4
        if "orientation" not in kwargs:
            kwargs["orientation"]="portrait"
        if "row_spacing" not in kwargs:
            kwargs["row_spacing"]=1
        if "font_size" not in kwargs:
            mpl.rcParams["font.size"]=10
        else:
            mpl.rcParams["font.size"]=kwargs["font_size"]
        if "col_spacing" not in kwargs:
            kwargs["col_spacing"]=1
        if "plot_width" not in kwargs:
            kwargs["plot_width"]=2
        if "plot_height" not in kwargs:
            kwargs["plot_height"]=1
        if "fourier_position" in kwargs:
            kwargs["fourier_plot"]=True
        if "fourier_plot" not in kwargs:
            kwargs["fourier_plot"]=False
        if "fourier_position" not in kwargs and kwargs["fourier_plot"]!=False:
            kwargs["fourier_position"]=[1]
        if "fourier_position" in kwargs and type(kwargs["fourier_position"]) is not list:
            kwargs["fourier_position"]=[kwargs["fourier_position"]]
        if "harmonic_position" not in kwargs:
            print("harmonic position set to 0")
            kwargs["harmonic_position"]=[0]
        if type(kwargs["harmonic_position"]) is not list:
            kwargs["harmonic_position"]=[kwargs["harmonic_position"]]
        for pos in kwargs["harmonic_position"]:
            if kwargs["orientation"] =="landscape":
                if pos>=num_rows:
                    raise ValueError(str(pos)+" is greater than largest row number "+str(num_rows))
            elif kwargs["orientation"] =="portrait":
                if pos>=num_cols:
                    raise ValueError(str(pos)+" is greater than largest column number "+str(num_cols))
        y_dim=(kwargs["num_harmonics"]*num_rows*kwargs["plot_height"])+(kwargs["row_spacing"]*(num_rows-1))
        x_dim=(kwargs["plot_width"]*num_cols)+(kwargs["col_spacing"]*(num_cols-1))
        print(y_dim, x_dim)
        if kwargs["orientation"] =="landscape":
            total_axes=[]
            axes=["row" +str(x) for x in range(1, num_rows+1)]
            print(kwargs)
            for i in range(0, num_rows):
                row_axes=[]
                if i in kwargs["harmonic_position"]:

                    for q in range(0, kwargs["num_harmonics"]*num_cols):
                        ax=plt.subplot2grid((y_dim, x_dim), (i*(kwargs["num_harmonics"]+kwargs["row_spacing"])+(q%kwargs["num_harmonics"]),int(np.floor(q/kwargs["num_harmonics"]))*(kwargs["plot_width"]+kwargs["col_spacing"])), rowspan=kwargs["plot_height"], colspan=kwargs["plot_width"])
                        row_axes.append(ax)

                elif kwargs["fourier_plot"]!=False and i in kwargs["fourier_position"]:

                    if kwargs["num_harmonics"]%2==1:
                        f_val=int(np.floor(kwargs["num_harmonics"]/2)+1)
                    else:
                        f_val=int(kwargs["num_harmonics"]/2)
                    for q in range(0, 2*num_cols):
                        ax=plt.subplot2grid((y_dim, x_dim), (i*(kwargs["num_harmonics"]+kwargs["row_spacing"])+((q%2)*(f_val)),int(np.floor(q/2))*(kwargs["plot_width"]+kwargs["col_spacing"])), rowspan=int(np.floor(kwargs["num_harmonics"]/2)), colspan=kwargs["plot_width"])
                        row_axes.append(ax)
                else:
                    for j in range(0, num_cols):
                        ax=plt.subplot2grid((y_dim, x_dim), (i*(kwargs["num_harmonics"]+kwargs["row_spacing"]),j*(kwargs["plot_width"]+kwargs["col_spacing"])), rowspan=(kwargs["num_harmonics"]), colspan=kwargs["plot_width"])
                        row_axes.append(ax)
                total_axes.append(row_axes)
            self.total_axes=total_axes
            self.axes_dict=(dict(zip(axes, total_axes)))
        elif kwargs["orientation"] =="portrait":
            axes=["col" +str(x) for x in range(1, num_cols+1)]
            total_axes=[]
            for i in range(0, num_cols):
                row_axes=[]
                if i in kwargs["harmonic_position"]:
                    for j in range(0, num_rows):
                        for q in range(0, kwargs["num_harmonics"]):
                            ax=plt.subplot2grid((y_dim, x_dim), (j*(kwargs["num_harmonics"]+kwargs["row_spacing"])+q,i*(kwargs["plot_width"]+kwargs["col_spacing"])), rowspan=kwargs["plot_height"], colspan=kwargs["plot_width"])
                            row_axes.append(ax)

                elif kwargs["fourier_plot"]!=False and i in kwargs["fourier_position"]:
                    if kwargs["num_harmonics"]%2==1:
                        f_val=int(np.floor(kwargs["num_harmonics"]/2)+1)
                    else:
                        f_val=int(kwargs["num_harmonics"]/2)
                    for j in range(0, num_rows):
                        for q in range(0, 2):
                            ax=plt.subplot2grid((y_dim, x_dim), ((j*(kwargs["num_harmonics"]+kwargs["row_spacing"])+(q*f_val)),i*(kwargs["plot_width"]+kwargs["col_spacing"])), rowspan=int(np.floor(kwargs["num_harmonics"]/2)), colspan=kwargs["plot_width"])
                            row_axes.append(ax)
                else:
                    for j in range(0, num_rows):
                        ax=plt.subplot2grid((y_dim, x_dim), (j*(kwargs["num_harmonics"]+kwargs["row_spacing"]),i*(kwargs["plot_width"]+kwargs["col_spacing"])), rowspan=(kwargs["num_harmonics"]), colspan=kwargs["plot_width"])
                        row_axes.append(ax)
                total_axes.append(row_axes)
            self.axes_dict=(dict(zip(axes, total_axes)))
