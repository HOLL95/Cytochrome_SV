import re
set1=["E0_mean", "E0_std","k0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","cap_phase", "cap_phase","phase", "alpha"]
set2=["mean", "std", "skew", "shape", "scale", "upper", "lower"]
disp_param_dict={}
for i in range(0, len(set1)):
    for j in range(0, len(set2)):
        if set2[j] in set1[i]:
            m=re.search('.+?(?=_'+set2[j]+')', set1[i])
            param=m.group(0)
            if param in disp_param_dict:
                disp_param_dict[param].append(set2[j])
            else:
                disp_param_dict[param]=[set2[j]]
disp_flags=[["mean", "std"], ["shape","scale"],["lower","upper"], ["mean","std" ,"skew"]]
distribution_names=["normal", "lognormal", "uniform", "skewed_normal"]
distribution_dict=dict(zip(distribution_names, disp_flags))
print(disp_param_dict)
param_names=list(disp_param_dict.keys())
distribution_list=[]
i=0
for param in param_names:
    param_set=set(disp_param_dict[param])
    i+=1
    for key in distribution_dict.keys():
        if set(distribution_dict[key])==param_set:
            distribution_list.append(key)
print(param_names, distribution_list)
