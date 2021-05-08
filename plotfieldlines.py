import numpy as np
import matplotlib.pyplot as plt

# Global parameters
mu0 = 1.25663706212e-6 # vacuum permeability, T*m/A

# Calculate theoretical B-field lines
z = np.linspace(-5e-2, 5e-2, 20) # -10 to 0 cm
y = np.linspace(-4e-2, 4e-2, 20) # -5 to 5 cm
Z, Y = np.meshgrid(z,y) # coordinates meshgrid
R = np.sqrt(Z**2 + Y**2) # distance to origin (meshgrid)
rz = Z/R # unit vector x-component (meshgrid)
ry = Y/R # unit vector y-component (meshgrid)
mz = 1.500 # magnetic moment points in negative x-direction (mx is scalar)
my = 0 # so y-component is zero (my is scalar)

Bz_th = mu0/(4*np.pi) * (3*rz*(mz*rz + my*ry) - mz)/R**3
By_th = mu0/(4*np.pi) * (3*ry*(mz*rz + my*ry) - my)/R**3

plt.figure(figsize=(6,4))
plt.streamplot(Z*1e2, Y*1e2, Bz_th*1e3, By_th*1e3, density=0.75, minlength=0.3, color='k', zorder=0)
plt.xlabel(r'$z$ [cm]'), plt.ylabel(r'$y$ [cm]')
plt.xlim([-5, 5]), plt.ylim([-4, 4])
plt.tight_layout()
plt.show()
