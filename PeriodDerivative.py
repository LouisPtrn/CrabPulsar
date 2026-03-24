import numpy as np

def get_age(m):
    tau = 0


if __name__ == "__main__":
    # format (mjd period error)
    # read period of last data point
    with open("Calculated_periods.txt","r") as f:
        lines = f.readlines()
        last_line = lines[-1]
        P = float(last_line.split()[1])
        f.close()

    print(P)
    m = 3.725 * 10 ** -8  # pulsar period (s) / time (mjd)
    # convert to second/second
    m = m/(24*3600)
