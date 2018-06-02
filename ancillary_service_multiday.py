# Doing Ancillary Services over multiple time frames

# %% imports

import os
import zipfile as zip
import pandas as pd

## General plan:
# 1. List all zip files in path
# 2. for zip file in list of zip files:
#   2a. unzip
#   2b. read the four files we need to read
#   2c. Append those files to the list we want to append them to
# THis should give you four big files with each day and hour on it.
# 3. Then, we slice into each file as needed.



# %% List files in directory

foldername_AS = "~/Documents/Homework/CEE272R/Project/Data/" + \
                "2_Day_AS_Disclosure_2017"

relative_path_folder = "../../Data/2_Day_AS_Disclosure_2017/"
foldername_AS
zip_1 = os.listdir("../../Data/2_Day_AS_Disclosure_2017/")[0]
new_name = zip_1[30:38] + "_" + zip_1[49:70]
#'ext.00013057.0000000000000000.20171201.032008581.48_Hour_AS_Disclosure.zip'
os.path.join(relative_path_folder, zip_1)


# %% Test folder rename

os.listdir()
 # THis is how to rename!! rename takes in a relative path that you rename the file to.
for file in [f for f in os.listdir() if f.endswith('.ipynb')]:
    print(file)
    file_path = os.path.join("..", file)
    os.rename(file, '../test_for_rename.ipynb')

# %% Figure out how to unzip files
path_zip_1 = os.path.join(relative_path_folder, zip_1)
path_zip_1
zip1 = zip.ZipFile(path_zip_1)
foldername_current = os.path.join(relative_path_folder, new_name)
zip1.extractall(foldername_current)

os.listdir(foldername_current)
for file in [f for f in os.listdir(foldername_current) if f.startswith('48h_Agg_AS_Offers_REGDN')]:
    print(file)


# %% unzip all files in a folder and rename in a certain way
# FUCK YEAH THIS WORKS NOW!!

#relative_path_folder = "../../Data/2_Day_AS_Disclosure_2017/"
relative_path_folder = "../../test_zips"

for zip_file in [f for f in os.listdir(relative_path_folder) if f.endswith('.zip')]:
    path_new_zip = os.path.join(relative_path_folder, zip_file)
    new_zip = zip.ZipFile(path_new_zip)
    #print(new_zip)
    new_name = zip_file[30:38] + "_" + zip_file[49:70] # Modify this if needed
    path_new_name = os.path.join(relative_path_folder, new_name)
    new_zip.extractall(path_new_name)

# %% Get one file out of each folder and read it into a large dataframe

relative_path_folder = "../../test_zips"
new_csv = pd.DataFrame()
for new_foldername in \
    [f for f in os.listdir(relative_path_folder) if f.endswith('_48_Hour_AS_Disclosure')]:
    path_single_folder = os.path.join(relative_path_folder, new_foldername)

    for new_filename in \
        [f for f in os.listdir(path_single_folder) if f.startswith('48h_Agg_AS_Offers_REGDN')]:
        path_new_file = os.path.join(path_single_folder, new_filename)

    new_csv = new_csv.append(pd.read_csv(path_new_file))

# Sort files so in chronological order
new_csv['Delivery Date'] = new_csv['Delivery Date'].astype('datetime64[ns]')
new_csv = new_csv.sort_values(by = ['Delivery Date', 'Hour Ending', 'MW Offered'])

# Write to csv
new_csv_name = os.path.join(relative_path_folder, 'aggregated_data.csv')
new_csv.to_csv(new_csv_name)

# %%
new_foldername = [f for f in os.listdir(relative_path_folder) if f.endswith('_48_Hour_AS_Disclosure')][1]
path_single_folder = os.path.join(relative_path_folder, new_foldername)

new_filename = [f for f in os.listdir(path_single_folder) if f.startswith('48h_Agg_AS_Offers_REGDN')][0]
path_new_file = os.path.join(path_single_folder, new_filename)

new_csv = new_csv.append(pd.read_csv(path_new_file))


new_csv
