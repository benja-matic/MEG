import numpy as np
import matplotlib.pyplot as plt
Nc = 272
fs = 1200
ttot = 2.5*fs

A=np.load('epochs.baselinenorm.npy', encoding='latin1')
A1_=A.item().get(‘g30’)
A1=zeros((len(A1_),Nc,ttot))
y1=zeros((len(A1_),1))
k=-1
for i in A1_:
   k+=1
   print k
   y1[k,0]=i[0]
   A1[k,:,:]=i[1]

del(A1_)


#
A2_=A.item().get(‘g120’)
A2=zeros((len(A2_),Nc,ttot))
y2=zeros((len(A2_),1))
k=-1
for i in A2_:
   k+=1
   print k
   y2[k,0]=-i[0]
   A2[k,:,:]=i[1]

del(A2_)
