import re
k_classes=[]
for i in range(0, 25):
    k_string="k_"+str(i)
    if i!=0:
        if i%3==0:
            k_string+="_scale"
        if i%4==0:
            k_string+="_shape"
    k_classes.append(k_string)
print(k_classes)

for i in range(0, len(k_classes)):
    m=p.search(k_classes[i])
    if m!=None:
        print(k_classes[i])
