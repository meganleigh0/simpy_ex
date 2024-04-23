import os
import random
import urllib.request
import xlrd
import csv
import pandas as pd
import functools
import simpy
from SimComponents_for_supply_chain import DataframeSource, Sink, Process, Monitor
import matplotlib.pyplot as plt
import time

""" Data loading
    1. request raw excel file from github of jonghunwoo
    2. then, change the format from excel to csv
    3. Data frame object of product data is generated from csv file
"""

# ./data 폴더에 해당 파일이 없으면 실행
if not os.path.isfile('./data/spool_data_for_simulation.csv'):
    url = "https://raw.githubusercontent.com/jonghunwoo/public/master/spool_data_for_simulation.xlsx"
    filename = "./data/spool_data_for_simulation.xlsx"
    urllib.request.urlretrieve(url, filename)

    def csv_from_excel(excel_name, file_name, sheet_name):
        workbook = xlrd.open_workbook(excel_name)
        worksheet = workbook.sheet_by_name(sheet_name)
        csv_file = open(file_name, 'w')
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        for row_num in range(worksheet.nrows):
            writer.writerow(worksheet.row_values(row_num))

        csv_file.close()

    csv_from_excel('./data/spool_data_for_simulation.xlsx', './data/spool_data_for_simulation.csv', 'Sheet1')

# csv 파일 pandas 객체 생성
data = pd.read_csv('./data/spool_data_for_simulation.csv')
df = data[["NO_SPOOL", "DIA", "Length", "Weight", "MemberCount", "JointCount", "Material", "제작협력사", "도장협력사", "Plan_makingLT", "Actual_makingLT", "Predicted_makingLT", "Plan_paintingLT", "Actual_paintingLT", "Predicted_paintingLT"]]

df.rename(columns={'NO_SPOOL': 'part_no', "제작협력사": 'proc1', '도장협력사': 'proc2', 'Plan_makingLT': 'ct1', 'Actual_makingLT': 'ct3', 'Predicted_makingLT': 'ct5', 'Plan_paintingLT': 'ct2', 'Actual_paintingLT': 'ct4', 'Predicted_paintingLT': 'ct6'}, inplace=True)
print(df.shape[0])

""" Simulation
    Dataframe with product data is passed to Source.
    Then, source create product with interval time of defined adist function.
    For this purpose, DataframeSource is defined based on original Source.
"""

random.seed(42)
adist = functools.partial(random.randrange,1,10) # Inter-arrival time
samp_dist = functools.partial(random.expovariate, 1) # need to be checked
proc_time = functools.partial(random.normalvariate,5,1) # sample process working time

RUN_TIME = 45000

env = simpy.Environment()

Source = DataframeSource(env, "Source", adist, df)
Sink = Sink(env, 'Sink', debug=False, rec_arrivals=True)

proc1_name_list = list(df.drop_duplicates(['proc1'])['proc1'])
proc2_name_list = list(df.drop_duplicates(['proc2'])['proc2'])

proc1_list = []
proc2_list = []
monitor1_list = []
monitor2_list = []

#print(proc1_name_list)

proc1_qlimit = [10, 10, 10, 10, 10, 10, 10]
proc2_qlimit = [10, 10, 10, 10, 10, 10, 10]
proc1_subprocess = [5, 5, 5, 5, 5, 5, 5]
proc2_subprocess = [5, 5, 5, 5, 5, 5, 5]

#proc1_qlimit = [1, 1, 1, 1, 1, 1, 1]
#proc2_qlimit = [1, 1, 1, 1, 1, 1, 1]
#proc1_subprocess = [1, 1, 1, 1, 1, 1, 1]
#proc2_subprocess = [1, 1, 1, 1, 1, 1, 1]

for i in range(len(proc1_name_list)):
    proc1_list.append(Process(env, "proc1", "{}".format(proc1_name_list[i]), proc_time, proc1_subprocess[i], qlimit=proc1_qlimit[i], limit_bytes=False))
    #monitor1_list.append(Monitor(env, proc1_list[i], samp_dist))
    Source.outs['{}'.format(proc1_list[i].name)] = proc1_list[i]

for i in range(len(proc2_name_list)):
    proc2_list.append(Process(env, "proc2", "{}".format(proc2_name_list[i]), proc_time, proc2_subprocess[i], qlimit=proc2_qlimit[i], limit_bytes=False))
    #monitor2_list.append(Monitor(env, proc2_list[i], samp_dist))
    proc2_list[i].outs['Sink'] = Sink

for i in range(len(proc1_list)):
    for j in range(len(proc2_list)):
        proc1_list[i].outs['{}'.format(proc2_list[j].name)] = proc2_list[j]

start = time.time()  # 시작 시간 저장
# Run it
env.run(until=RUN_TIME)
print("time :", time.time() - start)

print("Total Lead Time : ", Sink.last_arrival)

# 공정별 가동률
for i in range(len(proc1_list)):
    print("utilization of {0}: {1:2.2f}".format(proc1_list[i].name ,proc1_list[i].working_time / Sink.last_arrival))
for i in range(len(proc2_list)):
    print("utilization of {0}: {1:2.2f}".format(proc2_list[i].name ,proc2_list[i].working_time / Sink.last_arrival))

#for i in range(len(proc1_list)):
    #print("average system occupancy of {0}: {1:.3f}".format(proc1_list[i].name, float(sum(monitor1_list[i].sizes)) / len(monitor1_list[i].sizes)))
#for i in range(len(proc2_list)):
    #print("average system occupancy of {0}: {1:.3f}".format(proc2_list[i].name, float(sum(monitor2_list[i].sizes)) / len(monitor2_list[i].sizes)))

# 공정별 대기시간의 합

names = ["(주)성광테크", "건일산업(주)", "부흥", "삼성중공업(주)거제", "성일", "성일SIM함안공장", "해승케이피피", "삼녹", "성도", "하이에어"]
waiting_time = {}
for name in names:
    t = 0
    for i in range(len(Sink.waiting_list)):
        if len(Sink.waiting_list[i]) == 2:
            if name + ' ' + "waiting start" == list(Sink.waiting_list[i].keys())[0]:
                t += Sink.waiting_list[i][name + " waiting finish"] - Sink.waiting_list[i][name + " waiting start"]
                #print("2 ", Sink.waiting_list[i][name + " waiting finish"], " - ", Sink.waiting_list[i][name + " waiting start"])

        elif len(Sink.waiting_list[i]) == 4:
            #print(list(Sink.waiting_list[i].keys())[0])
            #print(list(Sink.waiting_list[i].keys())[1])
            #print(list(Sink.waiting_list[i].keys())[2])
            #print(list(Sink.waiting_list[i].keys())[3])
            if name + ' ' + "waiting start" == list(Sink.waiting_list[i].keys())[0]:
                t += Sink.waiting_list[i][name + " waiting finish"] - Sink.waiting_list[i][name + " waiting start"]
                #print("4 ", Sink.waiting_list[i][name + " waiting finish"], " - ", Sink.waiting_list[i][name + " waiting start"])
            if name + ' ' + "waiting start" == list(Sink.waiting_list[i].keys())[2]:
                t += Sink.waiting_list[i][name + " waiting finish"] - Sink.waiting_list[i][name + " waiting start"]
                #print("4 ", Sink.waiting_list[i][name + " waiting finish"], " - ", Sink.waiting_list[i][name + " waiting start"])
        else:
            print(Sink.waiting_list[i])

    waiting_time[name] = t

print("total waiting time of (주)성광테크 : ",waiting_time["(주)성광테크"])
print("total waiting time of 건일사업(주) : ",waiting_time["건일산업(주)"])
print("total waiting time of 부흥 : ",waiting_time["부흥"])
print("total waiting time of 삼성중공업(주)거제 : ",waiting_time["삼성중공업(주)거제"])
print("total waiting time of 성일 : ",waiting_time["성일"])
print("total waiting time of 성일SIM함안공장 : ",waiting_time["성일SIM함안공장"])
print("total waiting time of 해승케이피피 : ",waiting_time["해승케이피피"])
print("total waiting time of 삼녹 : ",waiting_time["삼녹"])
print("total waiting time of 성도 : ",waiting_time["성도"])
print("total waiting time of 하이에어 : ",waiting_time["하이에어"])

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

fig, axis = plt.subplots()
axis.hist(monitor1_list[0].sizes, bins=10, density=True)
axis.set_title("Histogram for Process1 WIP")
axis.set_xlabel("time")
axis.set_ylabel("normalized frequency of occurrence")
plt.show()