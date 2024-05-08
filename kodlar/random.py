
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

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