
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('Ads_CTR_Optimisation.csv')


#Random Selection
"""
import random

N=10000
d=10
toplam=0
secilenler= []
for n in range(0,N):
    ad = random.randrange(10)
    secilenler.append(ad)
    odul= veriler.values[n,ad] # verilerdeki n. satır =1 ise ödül =1
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()

"""

#UCB
import math

N=10000 #10.000 tıklama var
d=10 # 10 tane ilan var
#Ri(n)
oduller = [0]*d   # 10 tane 0 elemanlı # ilk başta bütün ilanlaron ödülü 0
#Ni(n)
tiklamalar = [0]*d
toplam = 0 # toplam ödül = 0
secilenler=[]
for n in range(0,N):
    ad=0 #secilen ilan
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama= oduller[i]/tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb= ortalama+delta
        else:
            ucb=N*10
        if max_ucb < ucb: # max'tan büyük bir ucb çıktı
            max_ucb = ucb
            ad=i
    secilenler.append(ad)
    tiklamalar[ad]= tiklamalar[ad]+1
    odul= veriler.values[n,ad] # verilerdeki n. satır =1 ise ödül =1
    oduller[ad]=oduller[ad]+odul
    toplam = toplam + odul 

print('Toplam Ödül')
print(toplam)
    
plt.hist(secilenler)
plt.show()   
    
    
    
    
    
    
    
    
    
    
    