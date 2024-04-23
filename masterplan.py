'''
Preparatory file for setting up the simulation

Variable Definitions
    activity: A list to store all activity codes.
    PROJECT: A list to store project numbers.

Loop for Data Extraction 
    Iterates over each row in the input data, extracting and organizing the information into the defined data structures.

Process
    DataframeSource generates new parts based on the simulation schedule, introducing them into the system at defined times.
    Parts are routed through various Process instances, undergo specified processing times.
    The Sink collects finished parts.
'''

import time
import pandas as pd
import numpy as np

# Loads information about different activities or operations within the simulation environment
input_data = pd.read_excel('./data/MCM_ACTIVITY.xls') 

# data 
data_num = len(input_data)

pd.to_datetime(input_data['PLANSTARTDATE'], unit='s')  
STARTDATE = []  

# Setting Simulation start date 
for i in range(data_num):
    if input_data['PLANSTARTDATE'][i].year >= 2018:
        STARTDATE.append(input_data['PLANSTARTDATE'][i])

initial_date = np.min(STARTDATE)  # 2018-12-10

'''
Data Storage & Processing
Data storage format:
{
    Project Number:
    {
        Block Code (Location Code):
        {
            Activity Code: [Start Date (interval from initial date), Duration]
        }
    }
}
'''
# 변수
raw_data = {}  # Data before preprocessing, used for load calculations
preproc_data = {}  # Data after preprocessing, used for Simcomponents execution
activity = []  # List to store all activity codes
PROJECT = []  # List to store project numbers


# Iterates over each row in the input data, extracting and organizing the information into the defined data structures.

# Retrieve data
for i in range(data_num):
    temp = input_data.loc[i]  # Read one row at a time
    proj_name = temp.PROJECTNO  # Project number
    proj_block = temp.LOCATIONCODE  # Block code
    activity_code = temp.ACTIVITYCODE[5:]  # Activity code

    # Discard data from 2005 and data with activity codes starting with '#'
    if (temp.PLANSTARTDATE.year >= 2018) and (proj_block != 'OOO'):
        # Create a dictionary with project number as key -> value is a dictionary with block code as key
        if proj_name not in PROJECT:
            PROJECT.append(proj_name)
            raw_data[proj_name] = {}
            preproc_data[proj_name] = {}
        if proj_block not in raw_data[proj_name].keys():
            # Create a dictionary with block code as key -> value is a list with activity code, start date, and duration
            # location code: {ACTIVITYCODE: start date, duration}
            raw_data[proj_name][proj_block] = {}
            preproc_data[proj_name][proj_block] = {}

        # Set start date to 0 to calculate time difference in days
        interval_t = temp.PLANSTARTDATE - initial_date  # + datetime.timedelta(days=1)

        if activity_code not in activity:
            activity.append(activity_code)

        # [Start time interval, Total process time]
        raw_data[proj_name][proj_block][activity_code] = [interval_t.days, temp.PLANDURATION]
        preproc_data[proj_name][proj_block][activity_code] = [interval_t.days, temp.PLANDURATION]


# Sort by date order
for name in PROJECT:
    for location_code in raw_data[name].keys():
        # Sort based on start date
        sorted_raw = sorted(raw_data[name][location_code].items(), key=lambda x: x[1][0])
        sorted_preproc = sorted(preproc_data[name][location_code].items(), key=lambda x: x[1][0])
        raw_data[name][location_code] = sorted_raw
        preproc_data[name][location_code] = sorted_preproc

# Sort by the first process start time within each block
for name in PROJECT:
    block_list_raw = list(raw_data[name].items())
    block_sorted_raw = sorted(block_list_raw, key=lambda x: x[1][0][1][0])
    raw_data[name] = block_sorted_raw

    block_list_preproc = list(preproc_data[name].items())
    block_sorted_preproc = sorted(block_list_preproc, key=lambda x: x[1][0][1][0])
    preproc_data[name] = block_sorted_preproc

# Calculate start time intervals between blocks (compare the first process start times)
# Initial = 0 / ith: ith - (i-1)

IAT = {}
for name in PROJECT:
    IAT[name] = []
    for i in range(len(raw_data[name])):
        if i == 0:
            IAT[name].append(0)
        else:
            interval_AT = raw_data[name][i][1][0][1][0] - raw_data[name][i-1][1][0][1][0]
            IAT[name].append(interval_AT)
    dict_block = dict(preproc_data[name])
    preproc_data[name] = dict_block


# Handle overlapping or included parts (for SimComponents)
for name in PROJECT:
    for location_code in preproc_data[name].keys():
        for i in range(0, len(preproc_data[name][location_code])-1):
            # End time of the preceding process
            date1 = preproc_data[name][location_code][i][1][0] + preproc_data[name][location_code][i][1][1] - 1
            # Start time of the trailing process
            date2 = preproc_data[name][location_code][i+1][1][0]
            # End time of the subsequent process
            date3 = preproc_data[name][location_code][i+1][1][0] + preproc_data[name][location_code][i+1][1][1] - 1

            if date1 > date2:  # When the preceding process finishes later than the succeeding process
                if date1 < date3:  # When the preceding process overlaps with the subsequent process
                    preproc_data[name][location_code][i+1][1][0] = date1
                else:  # When included
                    preproc_data[name][location_code][i+1][1][0] = date1
                    preproc_data[name][location_code][i+1][1][1] = 1
                    preproc_data[name][location_code][i+1][1].append("##")  # Mark fully included

for name in PROJECT:
    for location_code in preproc_data[name].keys():
        temp_list = []
        for i in range(0, len(preproc_data[name][location_code])):
            if len(preproc_data[name][location_code][i][1]) < 3:
                temp_list.append(preproc_data[name][location_code][i])
        preproc_data[name][location_code] = dict(temp_list)

import simpy
import random
from collections import OrderedDict
from SimComponents_for_masterplan import DataframeSource, Sink, Process
import matplotlib.pyplot as plt

# Generator objects that create IAT data and block data one by one in order
def gen_schedule(inter_arrival_time_data):
    project_list = list(inter_arrival_time_data.keys())
    print(project_list)
    for project in project_list:
        for inter_arrival_time in inter_arrival_time_data[project]:
            yield inter_arrival_time

def gen_block_data(block_data):
    project_list = list(block_data.keys())
    for project in project_list:
        block_list = list(block_data[project].keys())
        for location_code in block_list:
            activity = OrderedDict(block_data[project][location_code])
            yield [project, location_code, activity]

print(IAT)  # Printing IAT data

IAT_gen = gen_schedule(IAT)
preproc_data_gen = gen_block_data(preproc_data)

# Simulation starts
random.seed(42)

RUN_TIME = 45000

env = simpy.Environment()

process_dict = {}
Source = DataframeSource(env, "Source", IAT_gen, preproc_data_gen, process_dict)
Sink = Sink(env, 'Sink', debug=False, rec_arrivals=True)

process = []
for i in range(len(activity)):
    process.append(Process(env, activity[i], 10, process_dict, 10))

for i in range(len(activity)):
    process_dict[activity[i]] = process[i]

process_dict['Sink'] = Sink

# Run it
start = time.time()  # Save start time
env.run(until=RUN_TIME)
print("simulation time :", time.time() - start)

print('#'*80)
print("Results of simulation")
print('#'*80)

# Example of planned data
print(preproc_data['U611']['A11C'])
# Example of simulation results data generated
print(Sink.block_project_sim['U611']['A11C'])

#### WIP calculation

process_time = Sink.last_arrival
WIP = [0 for i in range(process_time)]

# for name in block_name:
for location_code in Sink.block_project_sim['U611'].keys():
    p = dict(Sink.block_project_sim['U611'][location_code])
    q = list(p.items())
    Sink.block_project_sim['U611'][location_code] = q
    for i in range(0, len(Sink.block_project_sim['U611'][location_code])-1):
        ### End time of preceding process
        date1 = Sink.block_project_sim['U611'][location_code][i][1][0] + Sink.block_project_sim['U611'][location_code][i][1][1] -1
        ### Start time of subsequent process
        date2 = Sink.block_project_sim['U611'][location_code][i+1][1][0]
        lag = date2-date1
        if lag > 3:
            for j in range(date1, date2):
                WIP[j] += 1

plt.plot(WIP)
plt.xlabel('time')
plt.ylabel('WIP')
plt.title('WIP')
plt.show()

fig, axis = plt.subplots()
axis.hist(Sink.waits, bins=100, density=True)
axis.set_title("Histogram for waiting times")
axis.set_xlabel("time")
axis.set_ylabel("normalized frequency of occurrence")
plt.show()

fig, axis = plt.subplots()
axis.hist(Sink.arrivals, bins=100, density=True)
axis.set_title("Histogram for Sink Interarrival times")
axis.set_xlabel("time")
axis.set_ylabel("normalized frequency of occurrence")
plt.show()