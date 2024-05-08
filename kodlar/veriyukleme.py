#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kodlar

#veri yükleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

print(veriler)
boy=veriler[['boy']]
print(boy)
boykilo=veriler[['boy','kilo']]
print(boykilo)


class insan:
    boy =180
    def kosmak(self,b):
        return b+10


ali= insan()
print(ali.boy)
print(ali.kosmak(90))

#veri ön işleme
