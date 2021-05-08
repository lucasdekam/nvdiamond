import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Global parameters
D = 2.87 # zero-field splitting frequency, GHz
gamma = 28 # electron gyromagnetic ratio, GHz/T
mu0 = 1.25663706212e-6 # vacuum permeability, T*m/A

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

def inversecube(z, c1, z0):
    """ B = c1/(z-z0)^p with p=3 """
    return c1/(z-z0)**3

def inversepow(z, c1, p, z0):
    """ B = c1/(z-z0)^p """
    return c1/(z-z0)**p

# Load data and split into arrays
data = np.loadtxt('distancedata.txt')
d_pts = data[:,0]*1e-2 # distance of points at which we measured, m
peak_freqs = data[:,1:9] # GHz
peak_sigma = data[:,9:] # GHz

# Initialize arrays
B_extracted = np.zeros(d_pts.shape) # for storing averaged B's in
B_sigma = np.zeros(d_pts.shape)

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

# Fitting
d_pts_fit = d_pts
B_to_fit = B_extracted
# First, fit with an inverse cubic polynomial
cubefit_params, cubefit_cov = curve_fit(inversecube, d_pts, B_extracted, p0=(mu0/(4*np.pi), 0))
# Then a fit with the power of the polynomial also a fitting parameter
pfit_params, pfit_cov = curve_fit(inversepow, d_pts_fit, B_to_fit, p0=(mu0/(4*np.pi), 3, 0))
# Determine the magnetic moment of the magnet from the cubic fit
m_cubefit = 2*np.pi*cubefit_params[0]/mu0
# Print the magnetic moment that was found (compare to theoretical value of 1.5)
print('m_cube = {0:.3f}'.format(m_cubefit))
# The horizontal shift in distance was also a fitting parameter. This corresponds
# to the error in our measurement of the initial distance of magnet center to diamond.
print('z0 = {0:.6f} (cubefit)'.format(cubefit_params[1]))
print('z0 = {0:.6f} (pfit)'.format(pfit_params[2]))

# Define a smooth axis
d_axis = np.linspace(2.4e-2, 5.1e-2, 100) # smooth axis

# Plot stuff
plt.figure(figsize=(6,4))
plt.plot(d_axis*1e2, inversecube(d_axis, *cubefit_params)*1e3, color='r', label=r'$p=3$')
plt.plot(d_axis*1e2, inversepow(d_axis, *pfit_params)*1e3, 'k--', label=r'$p={0:.2f}$'.format(pfit_params[1]))
# For the error bars, make them 3 times as large (so total height of error bar
# is 6sigma, centered around the mean) to make it a bit more visible.
# For normally distributed data, deviation of 3sigma corresponds to certainty of 99.7%
plt.errorbar(d_pts*1e2, B_extracted*1e3, yerr=3*B_sigma*1e3, fmt='k.', ecolor='k', capsize=4, label='data')
plt.xlabel(r'$z$ [cm]'), plt.ylabel(r'$B$ [mT]')
plt.xlim([2.4, 5.1]), plt.ylim([0, 35])
plt.legend()
plt.tight_layout()
plt.show()
