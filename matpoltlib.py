import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-100,100,100)

# fig,ax  = plt.subplots(3,1,sharex=False,sharey=False)
# plt.subplot(2,1,1)
# plt.xlim(-10,10)
# plt.ylim(-10,10)
#
#
# ax[0].plot(x,x*x,"--r")
# ax[0].plot(x,x+3,":g")
# ax[0].set(xlim=(-100,100),ylim=(-100,100),title='ax0')
#
# x= np.linspace(-10,10,100)
# ax[1].plot(x,np.sin(x),"-y")
# ax[1].set(xlim=(0,10),ylim=(-1,1),title='ax1')

ax2= plt.axes()
rng = np.random.RandomState(0)
for m in ['o','.',',','x','+','v','^','<','>','s','d']:
    ax2.plot(rng.rand(5),rng.rand(5),m+"-87jk",label="maker={}".format(m))
ax2.legend()
ax2.set(xlim=(0,1),ylim=(0,1),title='ax2')
# plt.plot(x,x*x,"--r")
# plt.plot(x,x+3,":g")
# plt.plot(x,np.sin(x)+50,"-y")
#控制plt 当前对象
# plt.subplot(2,1,2)
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.axis('tight')
# plt.plot(x,x*x*x,"--b")
# plt.xlim(-100,100)
# plt.ylim(-100,100)
plt.show()