from numpy import *
from matplotlib.pyplot import *


Nc=272
fs=1200
ttot=int(2.5*fs)

A=load('epochs.baselinenorm.npy', encoding = 'latin1')

A1_=A.item().get('g30')
#A*_ has each sample as a list..where [0]:=contrast [1]:=meg data
A1=zeros((len(A1_),Nc,ttot))
y1=zeros((len(A1_),1))
k=-1
for i in A1_:
    k+=1
    print(k)
    y1[k,0]=i[0]
    A1[k,:,:]=i[1]

del(A1_)


#
A2_=A.item().get('g120')
A2=zeros((len(A2_),Nc,ttot))
y2=zeros((len(A2_),1))
k=-1
for i in A2_:
    k+=1
    print(k)
    y2[k,0]=-i[0]
    A2[k,:,:]=i[1]

del(A2_)

t0=int(.5*fs)
t1=int(1.5*fs)

#splint into training and validation epochs
ntr=200
A1tr=A1[:ntr,:,t0:t1]
A2tr=A2[:ntr,:,t0:t1]
y1=zeros((len(A1tr),2))
y1[:,0]=1
y2=zeros((len(A2tr),2))
y2[:,1]=1

Xtr=vstack((A1tr,A2tr))
labelstr=vstack((y1,y2))
del(A1tr,A2tr,y1,y2)
ind=range(len(Xtr))
random.shuffle(ind)
Xtr=Xtr[ind,:,:]
ytr=ytr[ind,:]

A1v=A1[ntr:,:,t0:t1]
A2v=A2[ntr:,:,t0:t1]
y1=zeros((len(A1v),2))
y1[:,0]=1
y2=zeros((len(A2v),2))
y2[:,1]=1

Xv=vstack((A1v,A2v))
labelsv=vstack((y1,y2))
del(A1,A2,A1v,A2v,y1,y2)
ind=range(len(Xv))
random.shuffle(ind)
Xv=Xv[ind,:,:]
yv=yv[ind,:]

save('Xtr',Xtr)
save('ytr',ytr)
save('Xv',Xv)
save('yv',yv)
