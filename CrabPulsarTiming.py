import os
import numpy as np

from scipy import interpolate

from astropy import coordinates as coord
from astropy import units as u
from astropy import constants as const
from astropy import time as astrotime

from matplotlib import pyplot as plt
from math import pi

# Load files
filename=os.path.join("mydata/20260217_143556_B0531+21.npz")
obsdata = np.load(filename)
print(obsdata["header"])
period_guess = obsdata['approx_period']

toafile  = os.path.join("mydata/20260217_143556_B0531+21.npz.toas.txt")

baryfile = os.path.join("ssb_files/ssb_2026.txt") # will work for all of 2026

year, month, day, xpos, ypos, zpos = np.loadtxt(baryfile,unpack=True)
toa_list, toa_errs = np.loadtxt(toafile,unpack=True)

def get_interp():
    # Interpolation of Earth-Barycenter position
    bary_positions = np.array([xpos,ypos,zpos]).T

    # Create array of dates from the barycenter file
    year_arr = year.astype(int)
    month_arr = month.astype(int)
    day_arr = day.astype(int)

    # Convert to astropy Time objects for easier date comparison
    # Create ISO format date strings for each day
    date_strings = [f"{int(y)}-{int(m):02d}-{int(d):02d}" for y, m, d in zip(year_arr, month_arr, day_arr)]
    times = astrotime.Time(date_strings, format='iso', scale='utc')

    # Target date for interpolation
    target_year = 2026
    target_month = 2
    target_day = 17
    target_time = astrotime.Time(f"{target_year}-{target_month:02d}-{target_day:02d}", format='iso', scale='utc')

    # Find the index closest to the target date
    target_jd = target_time.jd
    time_jd = times.jd
    distances = np.abs(time_jd - target_jd)

    # Get the index of the closest point
    center_idx = np.argmin(distances)

    # Determine the range for interpolation (using a symmetric window around the target date)
    # Typical Lagrange interpolation uses 4-6 points for good accuracy
    interp_number = 5  # Number of points to use for interpolation
    half_window = interp_number // 2

    start_point = max(0, center_idx - half_window)
    end_point = min(len(bary_positions), center_idx + half_window + 1)

    # Adjust start_point if end_point is at the boundary
    if end_point - start_point < interp_number:
        if end_point == len(bary_positions):
            start_point = max(0, end_point - interp_number)
        else:
            end_point = min(len(bary_positions), start_point + interp_number)

    print(f"Target date: {target_year}-{target_month:02d}-{target_day:02d}")
    print(f"Interpolation range: indices {start_point} to {end_point-1}")
    print(f"Dates for interpolation:")
    for i in range(start_point, end_point):
        print(f"  Index {i}: {year_arr[i]}-{month_arr[i]:02d}-{day_arr[i]:02d}")

    # Convert times to Julian Date for interpolation (numeric values needed for lagrange)
    times_jd = times.jd
    interp_function = interpolate.lagrange(times_jd[start_point:end_point],\
                                         bary_positions[start_point:end_point])
    return interp_function


# Nasty function treat as black box. Gets time of arrivals
########################################################################
def get_toas(ddfreq_averaged, obsdata, plots=False):
    times = obsdata['times']  # The time of phase zero for each subint
    approx_period = obsdata['approx_period']  # The approximate period of the pulsar
    toas = []
    toa_errs = []
    tempo2_toas = []

    # Equation A7 in Taylor 1992
    def get_dchi(tau, N, nbin):
        dphi = np.angle(xspec)[1:N]

        k = np.arange(1, N)

        dchi = np.sum(k * np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.sin(dphi + 2 * np.pi * k * tau / nbin))
        return dchi

    # Equation A9 in Taylor 1992
    def get_b(tau, N, nbin):
        dphi = np.angle(xspec)[1:N]
        k = np.arange(1, N)
        scale = np.sum(np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.cos(dphi + 2 * np.pi * k * tau / nbin))
        scale /= np.sum(np.abs(f_template[1:N]) ** 2)
        return scale

    # Equation A10 in Taylor 1992
    def get_sigma_tau(tau, N, nbin, b):
        dphi = np.angle(xspec)[1:N]
        k = np.arange(1, N)
        chi2 = np.sum(np.abs(f_prof[1:N]) ** 2 + b ** 2 * np.abs(f_template[1:N])) - 2 * b * np.sum(
            np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.cos(dphi + 2 * np.pi * k * tau / nbin))
        sigma2 = chi2 / (N - 1)
        de = np.sum(
            (k ** 2) * np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.cos(dphi + 2 * np.pi * k * tau / nbin))
        fac = nbin / (2 * np.pi)
        return np.sqrt(sigma2 / (2 * b * de)) * fac

    # Just for plotting, rotates an array by a fractional phase shift using Fourier transform
    def rotate_phs(ff, phase_shift):
        fr = ff * np.exp(-1.0j * 2 * np.pi * np.arange(len(ff)) * phase_shift)
        return np.fft.irfft(fr)

    # Loop over every sub integration
    for ip in range(len(ddfreq_averaged)):
        try:
            prof = ddfreq_averaged[ip]
            nbin = len(prof)

            # We are going to do a cross correlation by means of the Fourier transform and the Wiener-Kinchin theorem
            f_template = np.fft.rfft(template)
            f_prof = np.fft.rfft(prof)

            # The cross correlation of a and b is the inverse transform of FT(a) times the conjugate of FT(b)
            xspec = f_template.conj() * f_prof  # "cross spectrum"
            xcor = np.fft.irfft(xspec)  # Cross correlation

            ishift = np.argmax(np.abs(xcor))  # estimate of the shift directly from the peak cross-correlation

            # We need to define some bounds to search. (Actually this might not be optimal)
            lo = ishift - 1
            hi = ishift + 1
            nh = len(xspec)
            # We minimise the chisquare parameter by findng the root of it's derivatiive following Taylor 1992
            # This root_scalar method uses the 'Brent 1973' algorithm for root finding.
            ret = opt.root_scalar(get_dchi, bracket=(lo, hi), x0=ishift, args=(nh, nbin), method='brentq')

            # tau is the bin shift between data and template, which will become our ToA
            tau = ret.root
            # Again folow the math of Taylor 1992 to get the scale factor, which it calls 'b'
            scale = get_b(tau, nh, nbin)
            # And finally given the shift and scale we can find the uncertainty on the shift.
            sigma_tau = get_sigma_tau(tau, nh, nbin, scale)

            # Phase shift is bin shift divided by nbins
            phase_shift = tau / nbin

            # ToA is the phase shift converted to a time shift and added to the time of phase zero.
            toa = times[ip] + approx_period * tau / nbin / 86400.0
            toa_err = approx_period * sigma_tau / nbin
            tempo2_toa = " test 611 {:.16f} {} jb42\n".format(toa, toa_err * 1e6)

            toas.append(toa)
            toa_errs.append(toa_err)
            tempo2_toas.append(tempo2_toa)

            phase = np.linspace(0, 1, nbin)

            rotate_and_scaled_template = scale * rotate_phs(f_template, phase_shift)
            diff = prof - rotate_and_scaled_template

            d = np.amax(prof) - np.amin(prof)
            if plots:
                # And do some plotting...
                plt.figure(figsize=(12, 8))
                plt.xlabel("Pulse Phase")
                plt.ylabel("Flux Density (arbitrary)")

                plt.title(r"{} Profile {:d}:  $\delta t =$ {:.3f} $\pm$ {:.3f} ms".format(obsdata['source_name'], ip,
                                                                                          1e3 * approx_period * phase_shift,
                                                                                          1e3 * toa_err))
                plt.step(phase, prof, color='black', linewidth=1.0, label="Data")
                plt.plot(phase, rotate_and_scaled_template, color='red', label="Template")

                plt.step(phase, diff - d, color='green', linewidth=1.0, alpha=0.8, label=r"Data$-$template")
                plt.axhline(-d, ls=":", color='k')

                plt.xlim(0, 1)
                plt.ylim(np.amin(prof) - 1.2 * d, np.amax(prof) + 0.1 * d)

                plt.legend(loc="lower left", ncol=3)

                plt.show()
                plt.close()
        except ValueError as e:
            print(e)
    return np.array(toas), np.array(toa_errs), tempo2_toas


######################################################################

def get_earth_delay(toas):
    # POSITION OF CRAB PULSAR ra = 5.575 hours, dec = 22.0145 degrees
    pulsarpos = coord.SkyCoord(ra=5.575 * u.hourangle, dec=22.0145 * u.deg, frame='icrs')
    # Position of lovell telescope
    lovellpos = coord.EarthLocation(lat=53.236 * u.deg, lon=-2.305 * u.deg, height=25 * u.m)

    # convert to times using the astropy time module
    # time1 = astrotime.Time("2022-09-07T23:34:01.000", scale='utc')
    # time2 = astrotime.Time("2025-12-07T23:34:01.500", scale='utc')

    # toas is in MJD, so we can convert it to astropy Time objects
    times = astrotime.Time(toas, format='mjd', scale='utc')

    # To compute the angle between the pulsar and the earth we can use astropy to tell us the elevation angle to the pulsar.
    # First Transform the coordinate system to an Alt-Az system. This needs the location of the telescope and the times
    # of the observation.
    altaz = pulsarpos.transform_to(coord.AltAz(obstime=times, location=lovellpos))

    earth_delay = (lovellpos.x * np.cos(altaz.az) * np.cos(altaz.alt) + \
                   lovellpos.y * np.sin(altaz.az) * np.cos(altaz.alt) + \
                   lovellpos.z * np.sin(altaz.alt)) / const.c

    print(earth_delay)
    #print(np.mean(earth_delay))

get_earth_delay(toa_list)