import os
import numpy as np

from scipy import interpolate
import scipy.optimize as opt

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
ra = 5.575
dec = 22.0145
time_start = "2026-02-17T14:35:56.000"  # Start time in iso format
time_end = "2026-02-17T16:35:56.000"    # End time in iso format

# barycentre file has the x y z values of the unity vector from the earth to the barycentre at each time step
year, month, day, xpos, ypos, zpos = np.loadtxt(baryfile,unpack=True)
toa_list, toa_errs = np.loadtxt(toafile,unpack=True) # list of modified julian dates of arrival and errors


def get_interp(target_year, target_month, target_day):
    # Interpolation of Earth-Barycenter position
    bary_positions = np.array([xpos, ypos, zpos]).T # coords of unit vector from earth to barycenter at each time step
    # Create array of dates from the barycenter file
    year_arr = year.astype(int)
    month_arr = month.astype(int)
    day_arr = day.astype(int)

    # Convert to astropy Time objects for easier date comparison
    # Create ISO format date strings for each day
    date_strings = [f"{int(y)}-{int(m):02d}-{int(d):02d}" for y, m, d in zip(year_arr, month_arr, day_arr)]
    times = astrotime.Time(date_strings, format='iso', scale='utc')

    # Target date for interpolation
    target_time = astrotime.Time(f"{target_year}-{target_month:02d}-{target_day:02d}", format='iso', scale='utc')

    # Find the index closest to the target date
    target_mjd = target_time.mjd
    time_mjd = times.mjd

    distances = np.abs(time_mjd - target_mjd)

    # Get the index of the closest point
    center_idx = np.argmin(distances)

    interp_number = 5  # PUTTING THIS HIGHER THAN 3 BREAKS EVERYTHING. computer can't handle ~61000^4 or higher it loses accuracy
    # Fixed by offsetting MJD values by ~ -60000
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
    print(f"Interpolation range: indices {start_point} to {end_point - 1}")
    print(f"Dates for interpolation:")
    for i in range(start_point, end_point):
        print(f"  Index {i}: {year_arr[i]}-{month_arr[i]:02d}-{day_arr[i]:02d}")

    # Convert times to Julian Date for interpolation (numeric values needed for lagrange)
    # Perform Lagrange interpolation for each coordinate (x, y, z)
    times_mjd = times.mjd -61000  # Shift MJD to reduce numerical issues in interpolation

    interp_funcs = []
    for i in range(3):  # For x, y, z
        interp_func = interpolate.lagrange(times_mjd[start_point:end_point], bary_positions[start_point:end_point, i])
        print(times_mjd[start_point:end_point])
        print(bary_positions[start_point:end_point, i])
        interp_funcs.append(interp_func)
    return interp_funcs


def get_earth_delay(toas):
    # POSITION OF CRAB PULSAR ra = 5.575 hours, dec = 22.0145 degrees
    pulsarpos = coord.SkyCoord(ra=5.575 * u.hourangle, dec=22.0145 * u.deg, frame='icrs')
    # Position of lovell telescope
    lovellpos = coord.EarthLocation(lat=53.236 * u.deg, lon=-2.305 * u.deg, height=78 * u.m)
    #lovellpos = coord.EarthLocation(lat=180, lon=270, height=25) # for testing

    # convert to times using the astropy time module
    # time1 = astrotime.Time("2022-09-07T23:34:01.000", scale='utc')
    # time2 = astrotime.Time("2025-12-07T23:34:01.500", scale='utc')

    # toas is in MJD, so we can convert it to astropy Time objects
    times = astrotime.Time(toas, format='mjd', scale='utc')

    # To compute the angle between the pulsar and the earth we can use astropy to tell us the elevation angle to the pulsar.
    # First Transform the coordinate system to an Alt-Az system. This needs the location of the telescope and the times
    # of the observation.
    altaz = pulsarpos.transform_to(coord.AltAz(obstime=times, location=lovellpos))

    earth_delay = (lovellpos.x * np.cos(altaz.az) * np.cos(altaz.alt) +
                   lovellpos.y * np.sin(altaz.az) * np.cos(altaz.alt) +
                   lovellpos.z * np.sin(altaz.alt)) / const.c.to(u.m/u.s)
    #print("Earth delays:")
    #print(earth_delay)
    return np.mean(earth_delay)


def calculate_roemer_delay(time, ra, dec):
    # Convert the time to an astropy Time object
    time = astrotime.Time(time, scale='utc')

    # Convert the RA and Dec to an astropy SkyCoord object.
    # Get position of crab pulsar
    pulsarpos = coord.SkyCoord(ra=ra*u.hourangle, dec=dec*u.deg, frame='icrs')
    n_hat = pulsarpos.cartesian.xyz


    # Get the position of the Earth at the given time
    earth_pos = coord.get_body_barycentric('earth', time) # inaccurate?
    r_vector = earth_pos.xyz.to(u.m)

    # Calculate the Roemer delay
    c = const.c.to(u.m/u.s)
    roemer_delay = (r_vector.dot(n_hat)) / c

    return roemer_delay.to(u.s).value

def calc_roem(time, interp_funcs):
    # Convert the time to an astropy Time object
    time = astrotime.Time(time, scale='utc')

    # Get VECTOR for earth to barycenter at given time (IN UNITS OF AU, but we will convert to meters)
    pos_x = interp_funcs[0](time.mjd - 61000) * u.AU
    pos_y = interp_funcs[1](time.mjd - 61000) * u.AU
    pos_z = interp_funcs[2](time.mjd - 61000) * u.AU

    # Get position of crab pulsar
    pulsarpos = coord.SkyCoord(ra=ra*u.hourangle, dec=dec*u.deg, frame='icrs')
    n_hat = pulsarpos.cartesian.xyz

    # Calculate the Roemer delay
    c = const.c.to(u.m/u.s)
    roemer_delay = (pos_x * n_hat[0] + pos_y * n_hat[1] + pos_z * n_hat[2]) / c
    return roemer_delay.to(u.s).value






def test_interp():
    # Test the interpolation function by comparing it to the actual position at a known time
    test_time = "2026-02-18T14:35:56.000"
    interp_funcs = get_interp(2026,2,18)
    interp_x = interp_funcs[0](astrotime.Time(test_time, scale='utc').mjd-61000)
    interp_y = interp_funcs[1](astrotime.Time(test_time, scale='utc').mjd-61000)
    interp_z = interp_funcs[2](astrotime.Time(test_time, scale='utc').mjd-61000)
    print(interp_x, interp_y, interp_z)


if __name__ == "__main__":
    interp_funcs = get_interp(2026,2,17) # polynomial function
    test_interp()

    earth_delay = get_earth_delay(toa_list).to(u.s).value
    # Calculated roamer delay for the start time of the observation
    r_delay_1 = calculate_roemer_delay(time_start, ra, dec)
    r_delay_2 = calc_roem(time_start, interp_funcs)
    print(f"Roemer delay (astropy): {r_delay_1} seconds")
    print(f"Roemer delay (real method): {r_delay_2} seconds")
    print(f"Earth delay: {earth_delay} seconds")

    total_delay = r_delay_2 - earth_delay
    print(f"Total delay: {total_delay} seconds")

    # now we can apply this delay to every TOA in the list to make the times of arrival
    # as if they were observed at the barycentre of the solar system rather than at the earth
    bary_toas = toa_list - total_delay/86400 # convert delay from seconds to days (MJD is in days)
    print(toa_list)
    print("First 10 barycentered TOAs:")
    print(bary_toas[:10])




