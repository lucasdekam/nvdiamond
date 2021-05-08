import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

# Global parameters
D = 2.87 # zero-field splitting frequency, GHz
gamma = 28 # electron gyromagnetic ratio, GHz/T
mu0 = 1.25663706212e-6 # vacuum permeability, T*m/A
m_fitted = 1.511
R0 = -0.001897 # correction found from angle_analysis.py
th0 = -0.057407 # correction found from angle_analysis.py
R = 3.2e-2 - R0 # measured distance to magnet center, include correction R0

def ESR_freqs(par, measured_ESR_freqs):
    """ This function calculates the 8 NV ESR freqs for a given
    magnetic field B.
    The field is expressed in the frame of the diamond
    The diamond has a [001] top face

    From T. van der Sar, translated into Python by Lucas"""

    # Input:
    B_mag = par[0] # Absolute value of magnetic field B
    B_theta = par[1] # Polar angle of B
    B_phi = par[2] # Azimuthal angle of B

    Sx = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Sy = 1/np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

    NV_theta = np.arccos(1/np.sqrt(3))

    # Matrix with orientation vectors of the NV center
    nNV = 1/np.sqrt(3) * np.array([[1, 1, 1,-1],
                                   [1, 1,-1, 1],
                                   [1,-1, 1, 1]]).T

    # B-field unit vector small bhat, here nB:
    nB = np.array([np.sin(B_theta)*np.cos(B_phi),
                   np.sin(B_theta)*np.sin(B_phi),
                   np.cos(B_theta)])

    nBpar = np.matmul(nNV, nB) # parallel component for the 4 families
    nBperp = np.sqrt(1 - nBpar**2) # perpendicular component for the 4 families

    ESR_freqs = np.zeros((4,2))

    for i in range(0,4):
        # Calculate ground state ESR frequencies for all NV orientations
        H = D*np.matmul(Sz, Sz) + gamma*B_mag*(nBpar[i]*Sz + nBperp[i]*Sy)
        w, v = np.linalg.eig(H)
        sortedw = np.sort(np.real(w))
        eigenfreq1 = sortedw[0]
        eigenfreq2 = sortedw[1]
        eigenfreq3 = sortedw[2]
        ESR_freqs[i,0] = np.abs(eigenfreq3 - eigenfreq1) # ms=0 <-> ms=-1 transition
        ESR_freqs[i,1] = np.abs(eigenfreq2 - eigenfreq1) # ms=0 <-> ms=1 transition

    return np.sort(ESR_freqs.reshape(-1)) - measured_ESR_freqs

def Bmagvstheta_shift(theta, R0, th0):
  m = m_fitted
  return mu0*m/(4*np.pi*(R-R0)**3) * np.sqrt(3*np.cos(theta-th0)**2 + 1)

# Load data and split into arrays
data = np.loadtxt('angledata_adapted.txt')
theta_pts = data[:,0] # angle of points at which we measured, deg
theta_pts = np.array([x*np.pi/180 for x in theta_pts]) - th0 # convert to rad, include correction theta0
peak_freqs = data[:,1:9] # GHz
peak_sigma = data[:,9:] # GHz

# Initialize arrays
Bmag_extracted = np.zeros(theta_pts.shape) # for storing averaged B's in
Btheta_extracted = np.zeros(theta_pts.shape)
Bphi_extracted = np.zeros(theta_pts.shape)

# Guesses for fitting
guesses_theta = np.linspace(0, np.pi/2, theta_pts.shape[0])
guesses_mag = Bmagvstheta_shift(theta_pts, R0, th0)

# Extract B-field using least squares with ESR frequencies
nrows = peak_freqs.shape[0]

for i in range(0, nrows):
    result = least_squares(ESR_freqs,
                           x0=np.array([guesses_mag[i], guesses_theta[i], np.pi/2]),
                           bounds=([0,0,0], [np.inf, np.pi, 2*np.pi]),
                           args=(peak_freqs[i,:],),
                           loss='linear',
                           max_nfev=1e6)
    Bvec_extracted = result.x
    Bmag_extracted[i] = Bvec_extracted[0]
    Btheta_extracted[i] = Bvec_extracted[1]
    Bphi_extracted[i] = Bvec_extracted[2]

# Extracted B-field in the frame of the diamond
Bzprime_extr = Bmag_extracted*np.cos(Btheta_extracted)
Byprime_extr = Bmag_extracted*np.sin(Btheta_extracted)*np.sin(Bphi_extracted)
Bxprime_extr = Bmag_extracted*np.sin(Btheta_extracted)*np.cos(Bphi_extracted)

# Apply rotation matrix to get B-field in the frame of the magnet
Bz_extr = Bzprime_extr*np.cos(theta_pts) - Byprime_extr*np.sin(theta_pts)
By_extr = Bzprime_extr*np.sin(theta_pts) + Byprime_extr*np.cos(theta_pts)
Bx_extr = Bxprime_extr # no rotation
print(Btheta_extracted*180/np.pi)
print(Bphi_extracted*180/np.pi)

# Locations of the diamond in the frame of the magnet
zpts = R*np.cos(theta_pts)
ypts = R*np.sin(theta_pts)

# Calculate theoretical B-field lines
z = np.linspace(-2e-2, 5e-2, 20) # -10 to 0 cm
y = np.linspace(-2e-2, 4e-2, 20) # -5 to 5 cm
Z, Y = np.meshgrid(z,y) # coordinates meshgrid
Rmag = np.sqrt(Z**2 + Y**2) # distance to origin (meshgrid)
rz = Z/Rmag # unit vector x-component (meshgrid)
ry = Y/Rmag # unit vector y-component (meshgrid)
mz = m_fitted # magnetic moment points in negative x-direction (mx is scalar)
my = 0 # so y-component is zero (my is scalar)

Bz_th = mu0/(4*np.pi) * (3*rz*(mz*rz + my*ry) - mz)/R**3
By_th = mu0/(4*np.pi) * (3*ry*(mz*rz + my*ry) - my)/R**3

plt.figure(figsize=(6,4))
plt.streamplot(Z*1e2, Y*1e2, Bz_th*1e3, By_th*1e3, density=0.75, minlength=0.3, color=(0.8, 0.8, 0.8), zorder=0)
plt.quiver(zpts*1e2, ypts*1e2, Bz_extr*1e3, By_extr*1e3, width=0.005, color='k', zorder=5, label='measured field')
plt.xlabel(r'$z$ [cm]'), plt.ylabel(r'$y$ [cm]')
plt.xlim([-1, 5]), plt.ylim([0, 4])
plt.legend()
plt.tight_layout()

# plt.figure(figsize=(4,3))
# plt.streamplot(Z*1e2, Y*1e2, Bz_th*1e3, By_th*1e3, density=0.75, minlength=0.3, color=(0.8, 0.8, 0.8), zorder=0)
# plt.quiver(zpts*1e2, ypts*1e2, Bz_extr*1e3, By_extr*1e3, width=0.005, color='k', zorder=5, label='measured field')
# plt.xlabel(r'$z$ [cm]'), plt.ylabel(r'$y$ [cm]')
# plt.xlim([-1, 5]), plt.ylim([0, 4])
# # plt.legend()
# plt.subplots_adjust(top=0.93,
#     bottom=0.2,
#     left=0.215,
#     right=0.95,
#     hspace=0.2,
#     wspace=0.2)

plt.figure(figsize=(4,3))
plt.plot(theta_pts*180/np.pi, Bx_extr*1e3, 'k.')
plt.xlabel(r'$\theta$ [deg]'), plt.ylabel(r'$B_x$ [mT]')
# plt.ylim([0, 20])
plt.subplots_adjust(top=0.93,
    bottom=0.2,
    left=0.215,
    right=0.95,
    hspace=0.2,
    wspace=0.2)
plt.show()
