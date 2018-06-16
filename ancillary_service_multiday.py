# Doing Ancillary Services over multiple time frames

# %% imports

import os
import zipfile as zip
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter

# %% plot names

AS_plot_name = 'ancillary_services_plot.png'
rev_vs_cap_name = 'rev_vs_cap.png'
profit_vs_cost_name = 'profit_vs_cost.png'

def millions(x, pos):
    intermediate = round(float(x) / 10**6)
    return "$" + str(intermediate) + "M"

def test_func(x, pos):
    return '400'

mil_format = FuncFormatter(millions)
test_format = FuncFormatter(test_func)


# %% Define variables and parameters

relative_path_folder = "../../Data/2_Day_AS_Disclosure_2017/"
regup_csv_name = os.path.join(relative_path_folder, 'regup_yearlong.csv')
regdn_csv_name = os.path.join(relative_path_folder, 'regdn_yearlong.csv')


# Read in the yearlong csv's:

regup = pd.read_csv(regup_csv_name)
regdn = pd.read_csv(regdn_csv_name)
T = len(regup)

capacity = 100 # MWh
cost_kWh = 209 # $/KWh
battery_lifetime = 10 # years


eff_charge = 0.90
eff_discharge = 0.90

max_charge_rate = capacity / 4
max_discharge_rate = capacity / 4

price_arb = 30 # $/MWh
capacity_range = range(100, 2000, 100)

# %% Define function for optimization

def optimize_AS(regup, regdn, capacity, eff_in, eff_out, max_charge, max_discharge, price_arb):

    # Variables
    charge =   cp.Variable(T)
    discharge = cp.Variable(T)
    discharge_arbitrage = cp.Variable(T)
    battery_energy = cp.Variable(T+1)

    # Objective! (negative because it works and thought might be necessary for convexity)
    objective = cp.Maximize(cp.sum(eff_out * (discharge) * regup['price']) + cp.sum(charge * regdn['price']))

    # Constraints
    constraints = [ battery_energy <= capacity, \
                    battery_energy >= 0, \
                    battery_energy[0] == capacity / 2, \
                    charge <= regdn['regdn'], \
                    charge <= max_charge, \
                    charge >= 0, \
                    discharge <= regup['regup'], \
                    discharge  + discharge_arbitrage <= max_discharge, \
                    discharge >= 0, \
                    discharge_arbitrage >= 0, \
                    discharge_arbitrage <= capacity / 4]#cp.maximum(battery_energy[0:-1] - capacity / 4, np.zeros(battery_energy[0:-1].shape))]
    for t in range(T):
        constraints.append(battery_energy[t+1] == battery_energy[t] +
            eff_in * charge[t] - discharge[t] - discharge_arbitrage[t])

    prblm = cp.Problem(objective, constraints)

    # Solve Problem
    result = prblm.solve()
    print("Total Revenue: ", objective.value)
    print('Discharge: ', sum(eff_out * discharge.value * regup['price']), "\nCharge: ", sum(charge.value * regdn['price']), "\nArbitrage: ", sum(eff_out * discharge_arbitrage.value * price_arb))
    print("Capacity: ", capacity, "\nStatus: ", prblm.status, "\n")
    return prblm, charge, discharge, discharge_arbitrage, battery_energy



# %% Run for loop to find cost curve
 
revenue = []
charge = []
discharge  = []
discharge_arb = []
battery_energy = []

for capacity in capacity_range:
    max_charge_rate = capacity / 4
    max_discharge_rate = capacity / 4
    prblm_new, charge_new, discharge_new, dis_arb_new, battery_energy_new = optimize_AS(regup, \
        regdn, capacity, eff_charge, eff_discharge, max_charge_rate, max_discharge_rate, price_arb)

    if prblm_new.status in ['optimal', 'optimal_accurate', 'optimal_inaccurate']:
        print('Appending new lines')
        new_rev = prblm_new.value + sum(eff_discharge * discharge_new.value * price_arb)
        revenue.append(new_rev)
        charge.append(charge_new.value)
        discharge.append(discharge_new.value)
        discharge_arb.append(dis_arb_new.value)
        battery_energy.append(battery_energy_new.value)



#prblm, charge, discharge, battery_energy = optimize_AS(regup, regdn, capacity, \
#    eff_charge, eff_discharge, max_charge_rate, max_discharge_rate)
# %% Play around with battery energy

np.mean(battery_energy[0])

adjusted_mean = []
for n in range(0, len(capacity)):
    energy_mean = np.mean(battery_energy[n])
    adjusted_mean.append(energy_mean / (capacity[n] / 4))
print(adjusted_mean)
plt.plot(adjusted_mean)

#amean_df = pd.DataFrame({'capacity': capacity, 'adjusted_mean_energy': adjusted_mean})
#amean_df.to_csv('adjusted_mean_energy.csv')



# %% Print curve results
#x = range(50,2050,50)
capacity = np.array(capacity_range)
# Write CSV with results so we don't lose them lol
#rev_csv = pd.DataFrame({'Capacity': capacity, 'Revenue': revenue})
#rev_csv.to_csv('arb_ancillary_revenue.csv')
rev_csv = pd.read_csv('arb_ancillary_revenue.csv')
revenue = rev_csv['Revenue']

rev_array = np.array(revenue)
cost_array = np.linspace(10, 300, len(capacity))
'''
cap_mat = np.transpose(np.tile(capacity, (len(capacity),1))) # Rows are different capacities
rev_mat = np.transpose(np.tile(rev_array, (len(rev_array),1)))

cost_array = np.linspace(10, 300, len(capacity))
cost_mat = np.tile(cost_array, (len(capacity),1))


profit_mat = rev_mat * 10 - cost_mat * cap_mat * 1000

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(cap_mat, cost_mat, profit_mat)
plt.xlabel('Capacity (MWh)')
plt.ylabel('Cost of bateries (kWh)')
plt.show()
'''

plt.close()

plt.figure()
plt.plot(capacity, revenue)
plt.title('Revenue per year')
plt.xlabel('Capacity (MWh)')
plt.ylabel('Revenue ($)')
ax.yaxis.set_major_formatter(tick)
plt.show()
# %%
plt.savefig(rev_vs_cap_name, dpi = 300)
# %%

'''
plt.figure()
#plt.subplot(2,1,2)
plt.plot(capacity, cost_kWh * 1000 * capacity)
plt.xlabel('Capacity (MWh)')
plt.ylabel('Cost ($)')
plt.title('Capital Cost Using ' + str(cost_kWh) + ' $/kWh')
'''
cap_num = 0
profit = revenue[cap_num] * 10 - cost_array * 1000 * capacity[cap_num]
# Find point with 209
f_profit = interpolate.interp1d(cost_array, profit)
profit_point = f_profit(cost_kWh)

plt.figure()
plt.plot(cost_array, profit, cost_kWh, profit_point, 'ro')
plt.title('Breakeven Cost for just ancillary services is ~$50/kWh')
plt.xlabel('Capital Cost of Battery ($/kWh)')
plt.ylabel('Profit ($)')
plt.show()
# %%
plt.savefig(profit_vs_cost_name, dpi = 300)

# %% Plot Profit!!
profit = rev_array * 10 - capacity * 1000 * cost_kWh

plt.figure()
plt.plot(capacity, profit, capacity[np.argmax(profit)], np.max(profit), 'ro')
plt.title('Maximum Profit is $350 Million')
plt.xlabel('Total Battery Capacity (MWh)')
plt.ylabel('Profit ($)')
plt.show()
plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))

# %%
plt.savefig("arb_profit_capacity.png", dpi = 300)


# %% Plot Different types of ancillary services

relative_path_folder
specific_day_path = '20170516_2_day_AS_Disclosure'
relative_path_day = os.path.join(relative_path_folder, specific_day_path)

filenames = [f for f in os.listdir(relative_path_day) if f.startswith("48h_Cleared")]

AS_day = np.zeros([24, len(filenames)])
col_names = []
for n in range(0, len(filenames)):
    next_file_path = os.path.join(relative_path_day, filenames[n])
    next_file = pd.read_csv(next_file_path)

    AS_day[:,n] = next_file.iloc[:,2]
    col_names.append(next_file.columns[2])

    print(next_file.columns[2])

col_names = [str.split(' - ')[1] for str in col_names]
plt.figure()
plt.plot(AS_day)
plt.legend(col_names)
plt.title('Different Ancillary Services on May 14th, 2017')
plt.xlabel('Hour of Day')
plt.ylabel('Power Requirement (MW)')


# %%Use this once you've made the plot larger:
plt.savefig(AS_plot_name, dpi = 300)

# %% Print single results

print("Revenue: ", prblm.value)
print("Profit: ",  prblm.value * battery_lifetime - cost_kWh * 1000 * capacity)

print(charge.value)
print(discharge.value)
print(battery_energy.value)

plt.plot(discharge.value)
plt.plot(charge.value)
plt.plot(battery_energy.value)



# %% Problem outside of for loop if needed

# Variables
charge =   cp.Variable(T)
discharge = cp.Variable(T)
battery_energy = cp.Variable(T+1)

# Objective! (negative because it works and thought might be necessary for convexity)
objective = cp.Maximize(cp.sum(discharge * regup['price'] + charge * regdn['price']))

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
    constraints.append(battery_energy[t+1] == battery_energy[t] +
        eff_charge * charge[t] - eff_discharge * discharge[t])

prblm = cp.Problem(objective, constraints)

# Solve Problem
result = prblm.solve()
print("status:", prblm.status)
