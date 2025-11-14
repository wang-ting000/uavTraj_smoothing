import os
import numpy as np

v = 10
with open('tx.txrx', 'r') as file:
    with open('tx_new_1.txrx', 'w') as f:
        for i in range(57):
            l0 = file.readline()
            f.write(l0)
            if i == 25:
                f.write("nVertices "+str(v) +'\n')
                x = 0
                y = 0
                z = 250
                l = str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
                f.write(l)
                for j in range(1,v):
                    x = np.random.uniform(0,1000)
                    y = np.random.uniform(0,1000)
                    z = np.random.uniform(250,750)
                    l1 = str(x) +' ' + str(y) + ' ' + str(z) + '\n'
                    f.write(l1)
