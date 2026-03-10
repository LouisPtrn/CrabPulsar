from CrabPulsarTiming import *







if __name__ == "__main__":
    filename=os.path.join("mydata/20260217_143556_B0531+21.npz")
    obsdata = np.load(filename)
    period_guess = obsdata['approx_period']

    toafile  = os.path.join("mydata/20260217_143556_B0531+21.npz.toas.txt")
    baryfile = os.path.join("ssb_files/ssb_2026.txt")
    ra = 5.575
    dec = 22.0145

    year, month, day, xpos, ypos, zpos = np.loadtxt(baryfile,unpack=True)
    toa_list, toa_errs = np.loadtxt(toafile,unpack=True) # list of modified julian dates of arrival and errors

