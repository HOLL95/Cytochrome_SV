import numpy as np
import matplotlib.pyplot as plt
import sys
plot=True
from harmonics_plotter import harmonics
import os
import math
import copy
import pints
from single_e_class_unified import single_electron
directory=os.getcwd()
dir_list=directory.split("/")
data_loc=("/").join(dir_list[:-1])+"/Experiment_data/DCV"
files=os.listdir(data_loc)
file="X183A 25.ids"
dcv_file=open(data_loc+"/"+file, "rb")

results_dict={}
result_key="Exp_"
print_yes=False
exp_count=1
for line in dcv_file:
    string=str(line.decode("latin1"))
    if print_yes==True and b"\x00\x00\x00\x00" in line:
        exp_count+=1
        print_yes=False
    if b"primary_data" in line:
        count=3
        print_yes=True
        current_key=result_key+str(exp_count)
        results_dict[current_key]=np.array([])
    if print_yes==True:
        if count<0:
            ascii_line=line.decode("ascii")

            seperate_line=ascii_line.strip().split(" ")
            try:
                num_list=list([np.float(x) for x in seperate_line if x!=""])
                results_dict[current_key]=np.append(results_dict[current_key],seperate_line)
            except:
                print(seperate_line)
            #print(num_list, ",")

        count-=1
