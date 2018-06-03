# Doing Ancillary Services over multiple time frames

# %% imports

import os
import zipfile as zip
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy import interpolate
import matplotlib.pyplot as plt


# %% unzip all files in a folder and rename in a certain way
# FUCK YEAH THIS WORKS NOW!!


def unzip_to_folders(relative_path_folder, new_folder_suffix):
    ''' Aggregates data from a bunch of folders that are named by previous script.
    inputs:
        relative_path_folder: relative path to the folder you want
        new_folder_suffix: suffix of the folders you want to create (plus date)
    returns: nothing
    '''
    for zip_file in [f for f in os.listdir(relative_path_folder) if f.endswith('.zip')]:
        path_new_zip = os.path.join(relative_path_folder, zip_file)
        new_zip = zip.ZipFile(path_new_zip)
        #print(new_zip)
        new_name = zip_file[30:38] + "_" +  new_folder_suffix # Modify this if needed
        path_new_name = os.path.join(relative_path_folder, new_name)
        new_zip.extractall(path_new_name)

# %% Get one file out of each folder and read it into a large dataframe

relative_path_folder = "../../Data/2_Day_AS_Disclosure_2017/"
#relative_path_folder = "../../test_zips"
def aggregate_data_from_folders(relative_path_folder, new_folder_suffix, possible_file_prefixes, sort_values, csv_name):
    ''' Aggregates data from a bunch of folders that are named by previous script.
    inputs:
        relative_path_folder: relative path to the folder you want
        new_folder_suffix: suffix of the folders you're trying to read from
        possible_file_prefixes: possible prefixes of the file you're trying to read. Must be a tuple!!
        sort_values: The values you want to sort by (in a list of strings)
    returns:
        new_csv_name, the relative path to the new csv
    '''
    new_csv = pd.DataFrame()

    # Iterate over the folders
    for new_foldername in \
        [f for f in os.listdir(relative_path_folder) if f.endswith(new_folder_suffix)]:
        path_single_folder = os.path.join(relative_path_folder, new_foldername)
        # Find the file in this folder that you want
        for new_filename in \
            [f for f in os.listdir(path_single_folder) if f.startswith(possible_file_prefixes)]:
            path_new_file = os.path.join(path_single_folder, new_filename)
            new_csv = new_csv.append(pd.read_csv(path_new_file))

    # Sort files so in chronological order
    #new_csv['Delivery Date'] = new_csv['Delivery Date'].astype('datetime64[ns]')
    #new_csv['Interval Time'] = new_csv['Interval Time'].astype('datetime64[ns]')
    new_csv['SCED Time Stamp'] = new_csv['SCED Time Stamp'].astype('datetime64[ns]')

    new_csv = new_csv.sort_values(by = sort_values)

    # Write to csv
    path_csv_name = os.path.join(relative_path_folder, csv_name)
    new_csv.to_csv(path_csv_name)
    return path_csv_name


# %% Format and read in data

# Unzip!

relative_path_folder = "../../Data/2_Day_AS_Disclosure_2017/"
#relative_path_folder = "../../test_zips"
new_folder_suffix = "2_day_AS_Disclosure"


unzip_to_folders(relative_path_folder, new_folder_suffix)

# %% Aggregate Data
price_prefixes = ('48h_Agg_AS_Offers_', '2d_Agg_AS_Offers_')
regup_price_prefixes = tuple([str + "REGUP" for str in price_prefixes])
regdn_price_prefixes = tuple([str + "REGDN" for str in price_prefixes])
pwr_prefixes = ('48h_Cleared_DAM_AS_', '2d_Cleared_DAM_AS_')
regup_pwr_prefixes = tuple([str + "REGUP" for str in pwr_prefixes])
regdn_pwr_prefixes = tuple([str + "REGDN" for str in pwr_prefixes])

regdn_price_csv_name = 'aggregated_data_regdn_price.csv'
regdn_pwr_csv_name = 'aggregated_data_regdn_pwr.csv'
regup_price_csv_name = 'aggregated_data_regup_price.csv'
regup_pwr_csv_name = 'aggregated_data_regup_pwr.csv'

pwr_sort_values = ['Delivery Date', 'Hour Ending']
price_sort_values = ['Delivery Date', 'Hour Ending', 'MW Offered']



path_regdn_price = aggregate_data_from_folders(relative_path_folder, new_folder_suffix, \
    regdn_price_prefixes, price_sort_values, regdn_price_csv_name)
regdn_price = pd.read_csv(path_regdn_price)
path_regup_price = aggregate_data_from_folders(relative_path_folder, new_folder_suffix, \
    regup_price_prefixes, price_sort_values, regup_price_csv_name)
regup_price = pd.read_csv(path_regup_price)



path_regup_pwr = aggregate_data_from_folders(relative_path_folder, new_folder_suffix, \
    regup_pwr_prefixes, pwr_sort_values, regup_pwr_csv_name)
regup_pwr = pd.read_csv(path_regup_pwr)

path_regdn_pwr = aggregate_data_from_folders(relative_path_folder, new_folder_suffix, \
    regdn_pwr_prefixes, pwr_sort_values, regdn_pwr_csv_name)
regdn_pwr = pd.read_csv(path_regdn_pwr)

# %% Functions to get prices for one day

def get_prices_dn(reg_price, reg_pwr):
    """ Get prices from files given by ERCOT. Depends on not changing names from
    ERCOT names though, so be careful with that!!.

    inputs:
        reg_price = file read from 48h_Agg_AS_Offers_REGDN-01-JAN-17.csv
        reg_pwr = file read from 48h_Cleared_DAM_AS_REGDN-01-JAN-17.csv
    outputs:
        dataframe with columns 'date', 'hour', 'regdn' (power) and 'price' (associated price of that power)
    """
    hours = reg_pwr['Hour Ending'].unique()
    prices = np.zeros(len(hours));
    for hour in range(len(hours)):
        price_hr = reg_price[reg_price['Hour Ending'] == hours[hour]]
        pwr_hr = reg_pwr[reg_pwr['Hour Ending'] == hours[hour]]

        interp_pwr = price_hr['MW Offered'].values
        interp_price = price_hr['REGDN Offer Price'].values

        interp_pwr = np.insert(interp_pwr, 0, 0)
        interp_price = np.insert(interp_price, 0, 0)

        f_up = interpolate.interp1d(interp_pwr, interp_price, kind = 'next')
        prices[hour] = f_up(pwr_hr['Total Cleared AS - RegDown'])
    df = pd.DataFrame({'date' : reg_pwr['Delivery Date'], 'hour'  : reg_pwr['Hour Ending'], \
        'regdn' : reg_pwr['Total Cleared AS - RegDown'], 'price' : prices})
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
    hours = reg_pwr['Hour Ending'].unique()
    prices = np.zeros(len(hours));
    for hour in range(len(hours)):
        price_hr = reg_price[reg_price['Hour Ending'] == hours[hour]]
        pwr_hr = reg_pwr[reg_pwr['Hour Ending'] == hours[hour]]

        interp_pwr = price_hr['MW Offered'].values
        interp_price = price_hr['REGUP Offer Price'].values

        interp_pwr = np.insert(interp_pwr, 0, 0)
        interp_price = np.insert(interp_price, 0, 0)

        f_up = interpolate.interp1d(interp_pwr, interp_price, kind = 'next')
        prices[hour] = f_up(pwr_hr['Total Cleared AS - RegUp'])
    df = pd.DataFrame({'date' : reg_pwr['Delivery Date'], 'hour'  : reg_pwr['Hour Ending'], \
        'regup' : reg_pwr['Total Cleared AS - RegUp'], 'price' : prices})
    return df



# %% Make the big files


regup = pd.DataFrame()
regdn = pd.DataFrame()

for day in regup_price['Delivery Date'].unique():
    print(day)
    regup_price_day = regup_price[regup_price['Delivery Date'] == day]
    regdn_price_day = regdn_price[regdn_price['Delivery Date'] == day]

    regup_pwr_day = regup_pwr[regup_pwr['Delivery Date'] == day]
    regdn_pwr_day = regdn_pwr[regdn_pwr['Delivery Date'] == day]

    regup = regup.append(get_prices_up(regup_price_day, regup_pwr_day))
    regdn = regdn.append(get_prices_dn(regdn_price_day, regdn_pwr_day))

regup_csv_name = os.path.join(relative_path_folder, 'regup_yearlong.csv')
regdn_csv_name = os.path.join(relative_path_folder, 'regdn_yearlong.csv')

regup.to_csv(regup_csv_name)
regdn.to_csv(regdn_csv_name)


# %%

# Read in the yearlong csv's:

regup = pd.read_csv(regup_csv_name)
regdn = pd.read_csv(regdn_csv_name)
T = len(regup)

capacity = 1000 # MWh

max_charge_rate = capacity / 10
max_discharge_rate = capacity / 10

# %% Construct problem

# Variables
charge =   cp.Variable(T)
discharge = cp.Variable(T)
battery_energy = cp.Variable(T+1)

# Objective!
objective = cp.Minimize(-cp.sum(discharge * regup['price'] + charge * regdn['price']))

# Constraints
constraints = [ battery_energy <= capacity, \
                battery_energy >= 0, \
                battery_energy[0] == 0, \
                charge <= regdn['regdn'], \
                charge <= max_charge_rate, \
                charge >= 0, \
                discharge <= regup['regup'], \
                discharge <= max_discharge_rate, \
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

plt.plot(discharge.value)
plt.plot(charge.value)
