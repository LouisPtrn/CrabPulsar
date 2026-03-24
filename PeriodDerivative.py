import numpy as np


# def get_true_period(toa_list, period_guess):
#     residuals = []
#     for i in range(len(toa_list)-1):
#         diff = (bary_toas[i] - bary_toas[0])*24*3600
#         n = diff/period_guess
#         n_int = round(n)
#         residual = (n-n_int)
#         residuals.append(residual)
#
#     # get true period from gradient of residuals
#     residuals = np.array(residuals)
#     toa_list = np.array(toa_list)
#     grad = (np.gradient(residuals))
#     print(grad)
#     if grad > 0:
#         period = period_guess + grad*(diff/n_int)
#     else:
#         period = period_guess - grad*(diff/n_int)
#     return period


def get_age(P, Pdot):
    age = P/(2*Pdot) # age in seconds
    age = age/(365.25*24*3600) # convert to years
    return age


def get_b_field(P, Pdot):
    B = 3.2 * 10**19 * np.sqrt(P*Pdot)
    return B


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
    print("Characteristic age: " + str(round(get_age(P,m),3)) + " years")
    print("B-field: " + str(round(get_b_field(P,m),3)))


