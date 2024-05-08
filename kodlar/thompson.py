
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('Ads_CTR_Optimisation.csv')



#Thompson
import random

N=10000 #10.000 tıklama var
d=10 # 10 tane ilan var

#Ni(n)

toplam = 0 # toplam ödül = 0
secilenler=[]
birler = [0]*d
sifirlar =[0]*d 
for n in range(0,N):
    ad=0 #secilen ilan
    max_th=0
    for i in range(0,d):
        rasbeta= random.betavariate(birler[i]+1,sifirlar[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
        
    secilenler.append(ad)
    odul= veriler.values[n,ad] # verilerdeki n. satır =1 ise ödül =1
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1
    toplam = toplam + odul 

print('Toplam Ödül')
print(toplam)
    
plt.hist(secilenler)
plt.show() 