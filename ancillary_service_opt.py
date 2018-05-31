
# %% Imports
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import interpolate


# %% Read in Data
path_load = "~/Documents/Homework/CEE272R/Project/Data/load_profiles_actual.csv"

foldername_AS = "~/Documents/Homework/CEE272R/Project/Data/" + \
                "2_Day_AS_Disclosure_2017/ext.00013057.0000000000000000." + \
                "20170101.031244298.48_Hour_AS_Disclosure/"
filename_regup = "48h_Cleared_DAM_AS_REGUP-01-JAN-17.csv"
filename_regdn = "48h_Cleared_DAM_AS_REGDN-01-JAN-17.csv"
filename_regup_price = "48h_Agg_AS_Offers_REGUP-01-JAN-17.csv"
filename_regdn_price = "48h_Agg_AS_Offers_REGDN-01-JAN-17.csv"

loads = pd.read_csv(path_load)

regup_pwr = pd.read_csv(foldername_AS + filename_regup)
regdn_pwr = pd.read_csv(foldername_AS + filename_regdn)
regup_price = pd.read_csv(foldername_AS + filename_regup_price)
regdn_price = pd.read_csv(foldername_AS + filename_regdn_price)

regup_price['Delivery Date'] = regup_price['Delivery Date'].astype('datetime64[ns]')

regup_price[regup_price['Hour Ending'] == 1]

# %% Functions to get prices for one day

def get_prices_dn(reg_price, reg_pwr):
    """ Get prices from files given by ERCOT. Depends on not changing names from
    ERCOT names though, so be careful with that!!.

    inputs:
        reg_price = file read from 48h_Agg_AS_Offers_REGDN-01-JAN-17.csv
        reg_pwr = file read from 48h_Cleared_DAM_AS_REGDN-01-JAN-17.csv
    outputs:
        dataframe with columns 'regdn' (power) and 'price' (associated price of that power)
    """
    prices = np.zeros(24);
    for hour in range(1,24+1):
        reg_hr = reg_price[reg_price['Hour Ending'] == hour]
        f_up = interpolate.interp1d(reg_hr['MW Offered'], reg_hr['REGDN Offer Price'], kind = 'next')
        prices[hour - 1] = f_up(reg_pwr['Total Cleared AS - RegDown'][hour - 1])
    df = pd.DataFrame({'regdn' : reg_pwr['Total Cleared AS - RegDown'], 'price' : prices})
    return df

def get_prices_up(reg_price, reg_pwr):
    """ Get prices from files given by ERCOT. Depends on not changing names from
    ERCOT names though, so be careful with that!!.

    inputs:
        reg_price = file read from 48h_Agg_AS_Offers_REGUP-01-JAN-17.csv
        reg_pwr = file read from 48h_Cleared_DAM_AS_REGUP-01-JAN-17.csv
    outputs:
        dataframe with columns 'regup' (power) and 'price' (associated price of that power)
    """
    prices = np.zeros(24);
    for hour in range(1,24+1):
        reg_hr = reg_price[reg_price['Hour Ending'] == hour]
        f_up = interpolate.interp1d(reg_hr['MW Offered'], reg_hr['REGUP Offer Price'], kind = 'next')
        prices[hour - 1] = f_up(reg_pwr['Total Cleared AS - RegUp'][hour - 1])
    df = pd.DataFrame({'regup' : reg_pwr['Total Cleared AS - RegUp'], 'price' : prices})
    return df

# %% Get parameters

T = 24
regup = get_prices_up(regup_price, regup_pwr)
regdn = get_prices_dn(regdn_price, regdn_pwr)

capacity = 4000
regup['price'] * regup['regup']
len(regup)

# %% Construct problem

# Variables
charge =   cp.Variable(T)
discharge = cp.Variable(T)
battery_energy = cp.Variable(T+1)

# Objective!
objective = cp.Minimize(-cp.sum(discharge * regup['price'] - charge * regdn['price']))

# Constraints
constraints = [ battery_energy <= capacity, \
                battery_energy >= 0, \
                battery_energy[0] == 0, \
                charge <= regdn['regdn'], \
                charge <= 5, \
                charge >= 0, \
                discharge <= regup['regup'], \
                discharge >= 0]
for t in range(T):
    constraints.append(battery_energy[t+1] == battery_energy[t] + charge[t] - discharge[t])

prblm = cp.Problem(objective, constraints)

# %% Solve Problem
result = prblm.solve()

print("status:", prblm.status)
print("optimal value", -prblm.value)

print(charge.value)
print(discharge.value)
print(battery_energy.value)
