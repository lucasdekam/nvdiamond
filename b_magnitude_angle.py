import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Global parameters
D = 2.87 # zero-field splitting frequency, GHz
gamma = 28 # electron gyromagnetic ratio, GHz/T
mu0 = 1.25663706212e-6 # vacuum permeability, T*m/A
m_fitted = 1.511
R = 3.2e-2 # measured distance to magnet center

# Functions
def extractBmag(fu, fl):
    """ Extract the B-field magnitude from two peak frequencies fu and fl given
    in units of GHz. fu: upper, fu: lower """
    return 1/np.sqrt(3) * np.sqrt(fu**2 + fl**2 - fu*fl - D**2) / gamma

def varianceBmag(fu, fl, su, sl):
    """ Compute the variance of a B-value computed using extractBmag(),
    using partial derivatives """
    dB_dfu = 1/(2*np.sqrt(3)) * (2*fu-fl)/np.sqrt(fu**2 + fl**2 - fu*fl - D**2) / gamma
    dB_dfl = 1/(2*np.sqrt(3)) * (2*fl-fu)/np.sqrt(fu**2 + fl**2 - fu*fl - D**2) / gamma
    return dB_dfu**2 * su**2 + dB_dfl**2 * sl**2

def Bmagvstheta(theta, R0):
  m = m_fitted
  return mu0*m/(4*np.pi*(R-R0)**3) * np.sqrt(3*np.cos(theta)**2 + 1)

def Bmagvstheta_shift(theta, R0, th0):
  m = m_fitted
  return mu0*m/(4*np.pi*(R-R0)**3) * np.sqrt(3*np.cos(theta-th0)**2 + 1)

# Load data and split into arrays
data = np.loadtxt('angledata.txt')
theta_pts = data[:,0] # angle of points at which we measured, deg
theta_pts = np.array([x*np.pi/180 for x in theta_pts]) # convert to rad
peak_freqs = data[:,1:9] # GHz
peak_sigma = data[:,9:] # GHz

# Initialize arrays
B_extracted = np.zeros(theta_pts.shape) # for storing averaged B's in
B_sigma = np.zeros(theta_pts.shape)

# Compute B and the uncertainties for each value of B. Each row of peak_freqs
# gives one value of B
# First, define the for loops
nrows = peak_freqs.shape[0]
ncols = peak_freqs.shape[1]

for i in range(0, nrows):
    row_B = [] # list for B's extracted from current row
    row_var = [] # variances for extracted B's in current row
    for j in range(0, 4): # always 8 columns, iterate over half so 4
        if peak_freqs[i,j] != 0:
            fu = peak_freqs[i,7-j]
            fl = peak_freqs[i,j]
            su = peak_sigma[i,7-j]
            sl = peak_sigma[i,j]
            row_B.append(extractBmag(fu, fl))
            row_var.append(varianceBmag(fu, fl, su, sl))
        else:
            print("Element {0},{1} skipped because zero was found.".format(i,j))
    B_extracted[i] = np.mean(row_B) # average over found B's
    # uncertainty is smaller when there are multiple values of B found
    total_variance_B = np.sum(row_var)/len(row_B)**2
    B_sigma[i] = np.sqrt(total_variance_B)

# Fit data
fit_params, fit_cov = curve_fit(Bmagvstheta, theta_pts, B_extracted, p0=0)
th0fit_params, th0fit_cov = curve_fit(Bmagvstheta_shift, theta_pts, B_extracted, p0=(0, 0))
# The distance to magnet center was also a fitting parameter. Print the error
# as compared to the distance that we measured:
print('normal, R0 = {0:.6f}'.format(fit_params[0]))
print('shift, R0 = {0:.6f}'.format(th0fit_params[0]))
print('shift = {0:.6f}'.format(th0fit_params[1]))

# Plot
theta_axis = np.linspace(-0.1, np.pi/2+0.1, 100) # theta from 0 to 90 degrees, but in radians
B_fit = Bmagvstheta(theta_axis, fit_params)
B_fit_shift = Bmagvstheta_shift(theta_axis, *th0fit_params)

plt.figure(figsize=(6,4))
plt.plot(theta_axis*180/np.pi, B_fit*1e3, 'r-', label=r'$\theta_0 = 0^\circ$')
plt.plot(theta_axis*180/np.pi, B_fit_shift*1e3, 'k--', label=r'$\theta_0 = {0:.2f}^\circ$'.format(th0fit_params[1]*180/np.pi))
plt.errorbar(theta_pts*180/np.pi, B_extracted*1e3, yerr=3*B_sigma*1e3, fmt='k.', ecolor='k', capsize=4, label='data')
plt.xlabel(r'$\theta$ [deg]'), plt.ylabel(r'$B$ [mT]')
plt.xlim([-5, 95])
plt.xticks(np.arange(0, 10) * 10, np.arange(0, 10) * 10)
plt.legend()
plt.tight_layout()
plt.show()
