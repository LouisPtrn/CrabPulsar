import os
import numpy as np
from  matplotlib import pyplot as plt
from math import pi

# Program to calculate the period of a pulsar from the integrated profile using Fourier transform
#################################################################################################

#filename=os.path.join("mydata/20260203_165003_B0329+54.npz")
#filename=os.path.join("mydata/20260217_095751_B1933+16.npz")
#filename=os.path.join("mydata/20260217_121849_B2020+28.npz")

data_directory=os.path.join('LovellTimeSeries/')

print("Files avaliable:",os.listdir(data_directory))

# this is the file we will work on. Change this to select the observation you want to process.

# This is 'unformatted' raw binary data. The data are arranged in sequence with 1 byte per sample.
# Each sample is a single 8-bit signed integer, i.e. taking a value between -128 and +127.
# The samples are spaced uniformly in time.
datfile = "psr5.dat"
hdrfile= "psr5.hdr"

dt = None
with open(os.path.join(data_directory,hdrfile)) as f:
    for line in f:
        line=line.strip()
        print(line)
        if line.startswith("Tsamp"):
            dt = float(line.split()[1])
            dt = dt*10**(-3) # ms to s

# Now we read in the data from the file. We use the `fromfile` routine from numpy to read in the data as a 1-d array of 8-bit integers.
data = np.fromfile(os.path.join(data_directory,datfile), dtype=np.int8)
#for item in data:
    #print(item)

t = np.arange(len(data))*dt

# We can plot the data.
print("Total observation time is {} seconds".format(len(data)*dt))
plt.figure(figsize=(10,3))
plt.plot(t,data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("data that wasn't simulated")
plt.show()


################################################################
#Calc fourier transform and get period

ft = np.fft.rfft(data)
ft[0] = 0 # set the zero frequency component to zero, as this is just the mean of the data and doesn't contain any information about the period.
power = np.real(ft)*np.real(ft) + np.imag(ft)*np.imag(ft)

# sample frequency is the inverse of the sample interval
fs = 1/dt

freqs = np.fft.rfftfreq(len(data),d=dt)
plt.figure(figsize=(10,10))
plt.plot(freqs,power)
plt.xlabel("Frequency (Hz)") # change this line to add your x-axis label
plt.ylabel("Power")
plt.title("Power spectrum of real data")
plt.show()

# calculate period
peak_frequency = freqs[np.argmax(power)]
print("Peak frequency is {} Hz".format(peak_frequency))
print("This corresponds to a period of {} seconds".format(1/peak_frequency))











