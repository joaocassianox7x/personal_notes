import matplotlib.pyplot as plt
import numpy as np
from time import time


tempos = []
pis = []
Ns = range(10,100000,1000)
for N in Ns:
    soma = 0
    t0 = time()
    for _ in range(N):
        x_r = np.random.rand()
        y_r = np.random.rand()
        soma+=1*((x_r**2 + y_r**2) < 1)

    tempo_py = round(time() - t0,5)
    #print(4 * soma/N, )


    t0 = time()
    soma = 4*np.sum(1*((np.random.rand(N)**2 + np.random.rand(N)**2) < 1)) / N
    tempo_np = round(time() - t0,5)
    pi_np = soma

    tempos.append([tempo_py,tempo_np])
    pis.append(pi_np)


tempos = np.array(tempos)
pis = np.array(pis)

plt.figure()
plt.subplot(211)
plt.plot(Ns, tempos[:,0], label='Python')
plt.plot(Ns, tempos[:,1], label='Numpy')
plt.legend()
plt.tight_layout()

plt.subplot(212)
plt.plot(Ns, np.abs(pis - np.pi), label='Convergencia')
plt.legend()
plt.tight_layout()

plt.savefig('teste.png')