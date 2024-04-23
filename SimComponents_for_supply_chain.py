import simpy
from collections import deque
from datetime import datetime, timedelta
import plotly
import plotly.figure_factory as ff
'''
Includes detailed conditional logic and additional attributes for handling specific simulation scenarios such as queuing and process delays.
Additional checks and operations in the Process and Source classes. Uses waiting events (self.wait[0])
'''

today = datetime(datetime.now().year, datetime.now().month, datetime.now().day, hour=0, minute=0, second=0)

class Part(object):

    def __init__(self, time, id, src="a", dst="z", flow_id=0):
        self.time = time
        self.id = id
        self.src = src
        self.dst = dst
        self.flow_id = flow_id
        self.data = []

    def __repr__(self):
        return "id: {}, src: {}, time: {}".format(self.id, self.src, self.time)


class DataframePart(object):

    def __init__(self, time, df, id, src="a", dst="z", flow_id=0):
        self.time = time
        self.df = df
        self.id = id
        self.src = src
        self.dst = dst
        self.flow_id = flow_id
        self.data = []
        self.waiting = {}

    def __repr__(self):
        return "id: {}, src: {}, time: {}, df: {}".format(self.id, self.src, self.time, self.df)


class Source(object):

    def __init__(self, env, id,  adist, initial_delay=0, finish=float("inf"), flow_id=0):
        self.id = id
        self.env = env
        self.adist = adist
        self.initial_delay = initial_delay
        self.finish = finish
        self.out = None
        self.parts_sent = 0
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.flow_id = flow_id
        self.wait = [self.env.event()]

    def run(self):
        self.out.wait_pre = self.wait
        yield self.env.timeout(self.initial_delay)
        while self.env.now < self.finish:
            yield self.env.timeout(self.adist())
            self.parts_sent += 1
            p = Part(self.env.now, self.parts_sent, src=self.id, flow_id=self.flow_id)

            if (self.out.__class__.__name__ == 'Process'):
                if len(self.out.store.items) >= self.out.qlimit - 1:
                    self.out.stop = True
                    yield self.wait[0]

            #print("part{0} left source at {1}".format(p.id, self.env.now))
            self.out.put(p)


class DataframeSource(object):

    def __init__(self, env, id, adist, df, initial_delay=0, finish=float("inf"), flow_id=0):
        self.id = id
        self.env = env
        self.adist = adist
        self.df = df
        self.initial_delay = initial_delay
        self.delay_time = 0
        self.finish = finish
        self.outs = {}
        self.parts_sent = 0
        self.action = env.process(self.run())
        self.flow_id = flow_id

    def run(self):
        yield self.env.timeout(self.initial_delay)

        while self.env.now < self.finish:
            yield self.env.timeout(self.adist()/10)

            p = DataframePart(self.env.now, self.df.iloc[self.parts_sent], self.parts_sent, src=self.id, flow_id=self.flow_id)

            #print("Part {part_name} sents to {next_process}/{no} at {time}".format(part_name=p.df["part_no"], next_process=p.df["proc1"], no=self.routing_num, time=self.env.now ))

            if 'Sink' not in list(self.outs.keys()):
                if self.outs[p.df["proc1"]].inventory + self.outs[p.df["proc1"]].busy >= self.outs[p.df["proc1"]].qlimit:
                    stop = self.env.event()
                    self.outs[p.df["proc1"]].wait1.append(stop)
                    yield stop

            #print("part{0} sent at {1}".format(p.id, self.env.now))
            self.parts_sent += 1

            self.outs[p.df["proc1"]].put(p)

            if len(self.df) == self.parts_sent + 1:
                #print("All parts are sent at {time}".format(time=self.env.now))
                break


class Sink(object):

    def __init__(self, env, name, rec_arrivals=True, absolute_arrivals=False, rec_waits=True, debug=True, selector=None):
        self.name = name
        self.store = simpy.Store(env)
        self.env = env
        self.rec_waits = rec_waits
        self.rec_arrivals = rec_arrivals
        self.absolute_arrivals = absolute_arrivals
        self.waits = []
        self.arrivals = []
        self.debug = debug
        self.parts_rec = 0
        self.selector = selector
        self.last_arrival = 0.0
        self.data = []
        self.waiting_list = []  # part 별로 대기시간 기록한 dictionary 모아주는 list

    def put(self, part):
        if not self.selector or self.selector(part):
            now = self.env.now
            if self.rec_waits:
                self.waits.append(self.env.now - part.time)
            if self.rec_arrivals:
                if self.absolute_arrivals:
                    self.arrivals.append(now)
                else:
                    self.arrivals.append(now - self.last_arrival)
                self.last_arrival = now
            self.parts_rec += 1
        self.waiting_list.append(part.waiting)  # part의 waiting dictionary list에 추가 해줌

        if self.debug:
            print(part)
        for i in range(0, len(part.data), 3):
            self.data.append(dict(Task='Part{0}'.format(part.id), Start=part.data[i+1], Finish=part.data[i+2], Resource='{0}'.format(part.data[i])))

    def show_chart(self):
        fig = ff.create_gantt(self.data, index_col='Resource', title='Chart of each part', group_tasks=True, show_colorbar=True)
        plotly.offline.plot(
            fig, filename='chart.html'
        )


class Process(object):

    def __init__(self, env, process_kind, name, rate, subprocess_num, qlimit=None, limit_bytes=True, debug=False):
        self.process_kind = process_kind
        self.name = name
        self.store = simpy.Store(env)
        self.rate = rate
        self.proc_time = 0
        self.env = env
        self.subprocess_num = subprocess_num
        self.wait1 = deque([])
        self.wait2 = self.env.event()
        self.parts_rec = 0
        self.parts_drop = 0
        self.qlimit = qlimit
        self.limit_bytes = limit_bytes
        self.debug = debug
        self.inventory = 0
        self.busy = 0
        self.action = env.process(self.run())
        self.working_time = 0
        self.outs = {}

    def run(self):
        while True:
            if self.busy < self.subprocess_num:
                msg = (yield self.store.get())
                self.inventory -= 1
                self.busy += 1
                self.env.process(self.subrun(msg))
            else:
                yield self.wait2

    def subrun(self, msg):

        self.proc_time = msg.df["ct{}".format(self.process_kind[-1])]

        if self.proc_time == 0:
            self.proc_time = 0.1

        #print("Process {name} cycle time is {time}".format(name=self.name, time=proc_time))

        self.start_time = self.env.now
        yield self.env.timeout(self.proc_time)
        self.working_time += self.env.now - self.start_time

        msg.waiting[self.name + " waiting start"] = self.env.now  # 대기 시작

        #print("Part {part_name} sents to {next_process}/{no} at {time}".format(part_name=msg.df["part_no"], next_process=msg.df["proc2"], no=self.routing_num, time=self.env.now))

        if 'Sink' not in list(self.outs.keys()):
            next_process_kind = "proc{}".format(int(self.process_kind[-1]) + 1)
            if self.outs[msg.df[next_process_kind]].inventory + self.outs[msg.df[next_process_kind]].busy >= self.outs[msg.df[next_process_kind]].qlimit:
                stop = self.env.event()
                self.outs[msg.df[next_process_kind]].wait1.append(stop)
                yield stop
            msg.data.append((today + timedelta(minutes=self.env.now)).strftime("%Y-%m-%d %H:%M:%S"))
            self.outs[msg.df[next_process_kind]].put(msg)
        else:
            msg.data.append((today + timedelta(minutes=self.env.now)).strftime("%Y-%m-%d %H:%M:%S"))
            self.outs['Sink'].put(msg)

        #print("Current process is {curr_process} and routing number is {routing_num}".format(curr_process=self.process_kind+self.name, routing_num=self.routing_num))
        msg.waiting[self.name + " waiting finish"] = self.env.now  # 대기 종료

        self.busy -= 1
        self.wait2.succeed()
        self.wait2 = self.env.event()

        if self.debug:
            print(msg)

        if self.inventory + self.busy < self.qlimit and len(self.wait1) > 0:
            temp = self.wait1.popleft()
            temp.succeed()

    def put(self, part):
        self.inventory += 1
        self.parts_rec += 1
        part.data.append("{0}".format(self.name))
        part.data.append((today + timedelta(minutes=self.env.now)).strftime("%Y-%m-%d %H:%M:%S"))
        if self.qlimit is None:
            return self.store.put(part)
        elif len(self.store.items) >= self.qlimit:
            self.parts_drop += 1
        else:
            return self.store.put(part)


class Monitor(object):

    def __init__(self, env, port, dist):
        self.port = port
        self.env = env
        self.dist = dist
        self.sizes = []
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            total = self.port.inventory + self.port.busy
            self.sizes.append(total)