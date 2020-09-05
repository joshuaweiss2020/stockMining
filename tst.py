import matplotlib.pyplot as plt
import numpy as np

a= np.array([1,3.43,4,5],dtype='int')

print(a)

b=np.array([range(i,i+3) for i in [2,4,6]])

print(b)

c=np.ones((4,5))
print(c)

d=np.full((4,5),10)
print(d)

e=np.arange(0,30,3)
print(e)

f=np.random.random()
print(f)

g=np.array(np.random.normal(0,5,(10)),dtype='int')
print(g)

print(d[2,3])

print(g[0:9:2])

h = np.array([range(i,i+3) for i in [0,1,2]])
print(h)
print(h[2,:])
print(h[:,2])

i = np.array([range(i,i+5) for i in [0,1,2]])

print(i)
i1,i2=np.vsplit(i,[2])

print("i1:",i1)
print("i2:",i2)

i3,i4=np.hsplit(i,[2])

print("i3:",i3)
print("i4:",i4)

print(i4.reshape(1,9))

print("concatentate:")
print(np.concatenate([i1,i2]))
print(np.concatenate([i3,i4],axis=1))

print(sum(i4[1,:]))
print(i4+1)

print("min:",i4.min(axis=0))

np.random.seed(0)
k=np.random.randint(1,100,size=(10))

print("k:",k)
print("k 75",np.percentile(k,75))
print("k 50",np.median(k))
print("k var",np.var(k))
print("k min",np.min(k))

v=np.array([0,1,3,5])
print(k[v])

print(k[k>50])

print(k>50)

print(np.sum(k>50))

print(np.any(k<0))

print(np.all(k>0))

print(k[(k>50) & (k<80)])