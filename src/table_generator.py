import os
from decimal import Decimal
import pickle
import numpy as np
unit_dict={
    "E_0": "V",
    'E_start': "V", #(starting dc voltage - V)
    'E_reverse': "V",
    'omega':"Hz",#8.88480830076,  #    (frequency Hz)
    'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
    'v': '$V s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm^{-2}$',
    'k_0': 's^{-1}$', #(reaction rate s-1)
    'alpha': "",
    'E0_skew':"",
    "E0_mean":"V",
    "E0_std": "V",
    "k0_shape":"",
    "sampling_freq":"$s^{-1}$",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"rads",
    'phase' : "rads",
    "alpha_mean": "",
    "alpha_std": "",
    "":"",
    "noise":"",
    "error":"$\\mu A$",
}
fancy_names={
    "E_0": '$E^0$',
    'E_start': '$E_{start}$', #(starting dc voltage - V)
    'E_reverse': '$E_{reverse}$',
    'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
    'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "v",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Ru",  #     (uncompensated resistance ohms)
    'Cdl': "$C_{dl}$", #(capacitance parameters)
    'CdlE1': "$C_{dlE1}$",#0.000653657774506,
    'CdlE2': "$C_{dlE2}$",#0.000245772700637,
    'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
    'gamma': '$\\Gamma',
    'E0_skew':"$E^0$ skew",
    'k_0': '$k_0', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "cap_phase":"C$_{dl}$ phase",
    "k0_shape":"$\\log(k^0) \\sigma$",
    "k0_scale":"$\\log(k^0) \\mu$",
    "alpha_mean": "$\\alpha\\mu$",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "Phase",
    "sampling_freq":"Sampling rate",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"RMSE",
}

optim_list=["","E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"]
values=[[-0.051002951188454194, 1.0000168463482572e-05, 0.5871952270775241, 3.4006592554802374e-07,0,0,0,0, 3.107639506959287e-11, 0,0,0,0.4521221354084581],
        [-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333, -0.004699728058449013,0, 2.1898117688472174e-11, 8.959294458587753,0, 0,0.5592126258301378],
        [-0.049697063448006555, 0.0037230268328498823, 0.3527926751864493, 4.773196671248675e-10, 0,0,0,0,1.196896571851427e-11, 0,0,0,0.5513930594926286],
        [-0.07963988256472493, 0.043023349442016245, 20.550878790990733, 581.5147052074157, 8.806259570340898e-05, -0.045583168350011485, -0.00011159862236990309, 0.00018619134662841048, 2.9947102043021914e-11, 9.014976375142606, 5.699844468024501, 5.18463541959069, 0.5999994350046962]]


name_list=[fancy_names[x] for x in optim_list]
parameter_orientation="row"
param_num=len(name_list)+1
names=["DCV\\ 1", "ramped", "DCV\\ 2", "SV"]

title_num=len(names)+1
table_file=open("image_tex_edited.tex", "w")


#names=my_list[0]
if parameter_orientation=="row":
    f =open("image_tex_test.tex", "r")
    table_title="\\multicolumn{"+str(param_num-1)+"}{|c|}{Parameter values}\\\\ \n"
    table_control1="\\begin{tabular}{|"+(("c|"*param_num))+"}\n"
    table_control2="\\begin{tabular}{|"+(("p{3cm}|"*param_num))+"}\n"
    value_rows=[]
    row_1=""
    for i in range(0, len(name_list)):
        if unit_dict[optim_list[i]]!="":
            unit_dict[optim_list[i]]=" ("+unit_dict[optim_list[i]]+")"
        if i ==len(values[0]):
            row_1=row_1+name_list[i]+unit_dict[optim_list[i]] +"\\\\\n"
        else:
            row_1=row_1+(name_list[i]+unit_dict[optim_list[i]]+" & ")
    value_rows.append(row_1)
    for i in range(0, len(names)):
        row_n=""
        row_n=row_n+(names[i]+ " & ")
        for j in range(0, len(values[0])):
            if j ==len(values[0])-1:
                end_str="\\\\\n"
            else:
                end_str=" & "
            print(names[i])
            if abs(values[i][j])>1e-2 or values[i][j]==0:

                row_n=row_n+(str(round(values[i][j],3))+ end_str)
            else:
                row_n=row_n+("{:.3E}".format(Decimal(str(values[i][j])))+ end_str)
        value_rows.append(row_n)

    control_num=0

    for line in f:
        if line[0]=="\\":
            if line.strip()=="\hline":
                table_file.write(line)
            try:
                line.index("{")
                command=line[line.index("{")+1:line.index("}")]
                if (command=="tabular")and (control_num==0):
                    line=table_control1
                    control_num+=1
                elif (command=="tabular") and (control_num==2):
                    line=table_control2
                    control_num+=1
                elif command=="4":
                    line=table_title
                table_file.write(line)
            except:
                continue
        elif line[0]=="e":
            for q in range(0, len(names)+1):
                line=value_rows[q]
                table_file.write(line)
                table_file.write("\hline\n")
elif parameter_orientation =="column":
    f =open("image_tex_test_param_col.tex", "r")
    table_control_1="\\begin{tabular}{|"+(("c|"*title_num))+"}\n"
    titles=["& {0}".format(x) for x in names]
    titles="Parameter "+(" ").join(titles)+"\\\\\n"

    row_headings=[name_list[i]+" ("+unit_dict[optim_list[i]]+") " if unit_dict[optim_list[i]]!="" else name_list[i] for i in range(1, len(optim_list))]
    numerical_rows=[]
    for j in range(0, len(values[0])):
        int_row=""
        for q in range(0, len(names)):
            if values[q][j]>1e-2 or values[q][j]==0:
                int_row=int_row+"& "+(str(round(values[q][j],3)))+" "
            else:
                int_row=int_row+"& "+"{:.3E}".format(Decimal(str(values[q][j])))+" "

        numerical_rows.append(int_row+"\\\\\n\hline\n")
    for i in range(0, len(numerical_rows)):
        numerical_rows[i]=row_headings[i]+numerical_rows[i]
    for line in f:
        if line[0]=="\\":
            if "begin{tabular}" in line:
                table_file.write(table_control_1)
            else:
                table_file.write(line)
        elif "Parameter_line" in line:
                table_file.write(titles)
        elif "Data_line" in line:
            for q in range(0, len(numerical_rows)):
                table_file.write(numerical_rows[q])



f.close()

table_file.close()
filename=""
filename="alice_multi_exp_"
filename=filename+"table.png"
os.system("pdflatex image_tex_edited.tex")
os.system("convert -density 300 -trim image_tex_edited.pdf -quality 100 " + filename)
os.system("mv " +filename+" ~/Documents/Oxford/Cytochrome_SV/Results/Param_tables")
