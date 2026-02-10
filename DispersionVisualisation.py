import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import astropy
from math import pi

filename=os.path.join("example_data/20260203_165003_B0329+54.npz")
obsdata = np.load(filename)
print(obsdata['header'])
print("")


data = obsdata['data']
print("Data array shape:",data.shape)

# Here we infer the number of channels and numnber of phase bins based on the last entry in the file.
nsub, nchan,nphase = data.shape

# Print this out for verification
print("Nsub = {} Nchan = {} Nphase= {}".format(nsub, nchan,nphase))


# Here is where we reshape the 1-d array into a 2-d data structure
iphase=np.arange(nphase)
ichan=np.arange(nchan)
isub = np.arange(nsub)

# We can integrate over all frequency channels by using the `sum` routine from numpy...
fully_averaged=np.mean(data,axis=(0,1))
time_averaged = np.mean(data,axis=0)
freq_averaged = np.mean(data,axis=1)

# Plot the 2-d data:
plt.figure(figsize=(16,6))
plt.subplot(131)
plt.imshow(time_averaged,aspect='auto',origin='lower')
plt.xlabel("Phase (iphase)")
plt.ylabel("Channels (ichan)")

plt.subplot(132)
plt.imshow(freq_averaged,aspect='auto',origin='lower')
plt.xlabel("Phase (iphase)")
plt.ylabel("Sub-integrations (isub)")

# Plot the integrated profile:
plt.subplot(133)
plt.plot(fully_averaged)
plt.xlabel("Phase (iphase)")
plt.ylabel("Intensity (Arbitrary)")
plt.title("Integrated profile")
plt.show()

##
#  This function will shift each row of a 2-d (3-d) array by the the number of columns specified in the "shifts" array.
#  data_in - the 2-d (3-d) array to work on
#  shifts - the shifts to apply
#  Returns: The shifted array
##
def shift_rows(data_in, shifts):
    shifted = np.zeros_like(data_in)
    if data_in.ndim == 3:
        for sub in range(nsub):
            shifted[sub] = shift_rows(data_in[sub],shifts)
    else:
        for chan in range(nchan):
            shifted[chan] = np.roll(data_in[chan],int(shifts[chan]))
    return shifted

# This example scaling is wrong - you need to determine the right values to scale!
#freq = 50 + ichan * 5
# right scaling
f_c = 611        # MHz (central frequency)
bw = 10          # MHz total bandwidth

P = 0.714        # seconds, pulsar period
DM = 26.7        # pc cm^-3, trial dispersion measure

nchan = time_averaged.shape[0]

freq = np.linspace(
    f_c - bw/2,
    f_c + bw/2,
    nchan
)

# Time resolution per phase bin
dt = P / nphase   # seconds per bin


###############################################################################
# Compute dispersion delay
###############################################################################
# Dispersion constant
K = 4.148808e3  # ms MHz^2 pc^-1 cm^3

# Reference frequency (highest frequency channel)
nu_ref = freq.max()

# Delay relative to reference frequency
delay_s = K * DM * (freq**-2 - nu_ref**-2)

# Convert delay to phase-bin shifts (negative = de-disperse)
bindelay = -np.round(delay_s / dt).astype(int)

# This array is going to contain the phase shifts in "bins".
# For this demo, we just shift by the frequency, which is clearly not correct!
# bindelay=freq

# Here we call our row-shifting function
# Remeber that the shift is in phase bins, not time units!
dedispersed = shift_rows(time_averaged,bindelay)
# Again, sum along axis zero to integrate over frequency.
integrated=np.sum(dedispersed,axis=0)

# plot the data again... If you have de-dispersed correctly, you the S/N should be maximised.
plt.figure(figsize=(12,12))
plt.subplot(211)
plt.imshow(
    dedispersed,
    aspect='auto',
    origin='lower',
    extent=(0, P, freq[0], freq[-1])
)
plt.imshow(dedispersed,aspect='auto',origin='lower',extent=(0,1,freq[0],freq[-1]))
plt.xlabel("Time (s)") # phase
plt.ylabel("Frequency (MHz)") # frequency
plt.title(f"De-dispersed data (DM = {DM:.1f} pc cm$^{{-3}}$)")

# Integrated pulse profile
plt.subplot(212)
plt.plot(np.linspace(0, P, nphase), integrated)
plt.xlabel("Time (s)")
plt.ylabel("Intensity (arbitrary)")
plt.title("Integrated profile")

plt.tight_layout()
plt.show()

