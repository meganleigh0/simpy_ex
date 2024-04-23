"""
Example use of SimComponents to simulate a certain assembly line model
Copyright 2019 Dr. Jonathan Woo
"""
import random
import functools
import simpy
from SimComponents_assembly_line import Source, Sink, Process, Monitor
import time

if __name__ == '__main__':

    random.seed(42)

    adist = functools.partial(random.randrange,3,7)
    samp_dist = functools.partial(random.expovariate, 1)

    proc_time1 = functools.partial(random.normalvariate,5,1)
    proc_time2 = functools.partial(random.normalvariate,5,1)
    proc_time3 = functools.partial(random.normalvariate,5,1)
    proc_time4 = functools.partial(random.normalvariate,5,1)
    proc_time5 = functools.partial(random.normalvariate,5,1)

    RUN_TIME = 500

    # Create the SimPy environment
    env = simpy.Environment()

    # Create the packet generators and sink
    Sink = Sink(env, 'Sink', debug=False, rec_arrivals=True)
    Source = Source(env, "Source", adist)
    Process1 = Process(env, 'Process1', proc_time1, qlimit=5, limit_bytes=False)
    Process2 = Process(env, 'Process2', proc_time2, qlimit=5, limit_bytes=False)
    Process3 = Process(env, 'Process3', proc_time3, qlimit=5, limit_bytes=False)
    Process4 = Process(env, 'Process4', proc_time4, qlimit=5, limit_bytes=False)
    Process5 = Process(env, 'Process5', proc_time5, qlimit=5, limit_bytes=False)

    # Using a PortMonitor to track each status over time
    Monitor1 = Monitor(env, Process1, samp_dist)
    Monitor2 = Monitor(env, Process2, samp_dist)
    Monitor3 = Monitor(env, Process3, samp_dist)
    Monitor4 = Monitor(env, Process4, samp_dist)
    Monitor5 = Monitor(env, Process5, samp_dist)

    # Connection
    Source.out = Process1
    Process1.out = Process2
    Process2.out = Process3
    Process3.out = Process4
    Process4.out = Process5
    Process5.out = Sink

    # Run it
    start = time.time()  # 시작 시간 저장
    # Run it
    env.run(until=RUN_TIME)
    print("simulation time :", time.time() - start)

    print('#'*80)
    print("Results of simulation")
    print('#'*80)
    print("Lead time of Last 10 Parts: " + ", ".join(["{:.3f}".format(x) for x in Sink.waits[-10:]]))

    print("Process1: Last 10 queue sizes: {}".format(Monitor1.sizes[-10:]))
    print("Process2: Last 10 queue sizes: {}".format(Monitor2.sizes[-10:]))
    print("Process3: Last 10 queue sizes: {}".format(Monitor3.sizes[-10:]))
    print("Process4: Last 10 queue sizes: {}".format(Monitor4.sizes[-10:]))
    print("Process5: Last 10 queue sizes: {}".format(Monitor5.sizes[-10:]))


    print("Sink: Last 10 arrival times: " + ", ".join(["{:.3f}".format(x) for x in Sink.arrivals[-10:]])) # 모든 공정을 거친 assembly가 최종 노드에 도착하는 시간 간격 - TH 계산 가능
    print("Sink: average lead time = {:.3f}".format(sum(Sink.waits)/len(Sink.waits))) # 모든 parts들의 리드타임의 평균

    print("sent {}".format(Source.parts_sent))
    print("received: {}, dropped {} of {}".format(Process1.parts_rec, Process1.parts_drop, Process1.name))
    print("received: {}, dropped {} of {}".format(Process2.parts_rec, Process2.parts_drop, Process2.name))
    print("received: {}, dropped {} of {}".format(Process3.parts_rec, Process3.parts_drop, Process3.name))
    print("received: {}, dropped {} of {}".format(Process4.parts_rec, Process4.parts_drop, Process4.name))
    print("received: {}, dropped {} of {}".format(Process5.parts_rec, Process5.parts_drop, Process5.name))

    print("average system occupancy of Process1: {:.3f}".format(float(sum(Monitor1.sizes))/len(Monitor1.sizes)))
    print("average system occupancy of Process2: {:.3f}".format(float(sum(Monitor2.sizes))/len(Monitor2.sizes)))
    print("average system occupancy of Process3: {:.3f}".format(float(sum(Monitor3.sizes))/len(Monitor3.sizes)))
    print("average system occupancy of Process4: {:.3f}".format(float(sum(Monitor4.sizes)) / len(Monitor4.sizes)))
    print("average system occupancy of Process5: {:.3f}".format(float(sum(Monitor5.sizes)) / len(Monitor5.sizes)))

    print("utilization of Process1: {:2.2f}".format(Process1.working_time/RUN_TIME))
    print("utilization of Process2: {:2.2f}".format(Process2.working_time/RUN_TIME))
    print("utilization of Process3: {:2.2f}".format(Process3.working_time/RUN_TIME))
    print("utilization of Process4: {:2.2f}".format(Process4.working_time / RUN_TIME))
    print("utilization of Process5: {:2.2f}".format(Process5.working_time / RUN_TIME))
