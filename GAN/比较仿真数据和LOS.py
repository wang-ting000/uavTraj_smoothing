import numpy as np
import matplotlib.pyplot as plt

# los = np.load('../CKM/los_gain.npy')
# ch = np.load('../CKM/channel_gains.npy')
# plt.plot(ch,'k')
# plt.plot(los,'r')
#
# plt.show()


qs = np.load('../CKM/qs.npy')

gs = np.load('../CKM/channel_gains.npy')

print(np.hstack((qs, gs))[0])