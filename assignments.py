import numpy as np
import matplotlib.pyplot as plt

Sx = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
Sy = 1/np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
D = 2.87 # GHz
y = 28 # GHz/T
Bmag = y*np.linspace(0, 50e-3, 100) # T
theta = np.pi/2
eigenfreq1 = np.zeros(Bmag.shape)
eigenfreq2 = np.zeros(Bmag.shape)
eigenfreq3 = np.zeros(Bmag.shape)

for i in range(0, Bmag.shape[0]):
    Bx = 0
    By = Bmag[i] * np.sin(theta)
    Bz = Bmag[i] * np.cos(theta)
    H = D*np.matmul(Sz, Sz) + Bx*Sx + By*Sy + Bz*Sz
    w, v = np.linalg.eig(H)
    sortedw = np.sort(np.real(w))
    eigenfreq1[i] = sortedw[0]
    eigenfreq2[i] = sortedw[1]
    eigenfreq3[i] = sortedw[2]

plt.figure(figsize=(4,3))
# plt.plot(Bmag/y*1e3, np.ones(Bmag.shape) * D, 'k--')
plt.plot(Bmag/y*1e3, eigenfreq1, color='k', label=r'$m_s = 0$')
plt.plot(Bmag/y*1e3, eigenfreq2, color='b', label=r'$m_s = -1$')
plt.plot(Bmag/y*1e3, eigenfreq3, color='r', label=r'$m_s = 1$')
plt.legend(loc='right')
plt.xlabel(r'$B$ [mT]'), plt.ylabel(r'Eigenenergy [GHz]')
plt.xlim([Bmag[0]/y*1e3, Bmag[-1]/y*1e3])
plt.subplots_adjust(top=0.95,
    bottom=0.202,
    left=0.173,
    right=0.963,
    wspace=0.2,
    hspace=0.2)

# Make the ESR plot that we're actually looking for
plt.figure(figsize=(4,3))
ESR1 = np.abs(eigenfreq2 - eigenfreq1) # ms=0 <-> ms=1 transition
ESR2 = np.abs(eigenfreq3 - eigenfreq1) # ms=0 <-> ms=-1 transition
plt.plot(Bmag/y*1e3, ESR1, color='b', label=r'$m_s=0$ to $m_s = -1$')
plt.plot(Bmag/y*1e3, ESR2, color='r', label=r'$m_s=0$ to $m_s = 1$' )
plt.legend()
plt.xlabel(r'$B$ [mT]'), plt.ylabel(r'$f_{ESR}$ [GHz]')
plt.xlim([Bmag[0]/y*1e3, Bmag[-1]/y*1e3])
plt.subplots_adjust(top=0.95,
    bottom=0.202,
    left=0.173,
    right=0.963,
    wspace=0.2,
    hspace=0.2)
# plt.tight_layout()
plt.show()
