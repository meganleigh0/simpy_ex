import simpy
import random
from datetime import datetime, timedelta
import plotly
import plotly.figure_factory as ff
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
                if ((len(self.out.store.items) >= self.out.qlimit+1 and self.out.qlimit == 0)
                    and (len(self.out.store.items) >= self.out.qlimit and self.out.qlimit > 0)):
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
        self.finish = finish
        self.out = None
        self.parts_sent = 0
        self.action = env.process(self.run())
        self.flow_id = flow_id
        self.wait = [self.env.event()]

    def run(self):
        self.out.wait_pre = self.wait
        yield self.env.timeout(self.initial_delay)
        while self.env.now < self.finish:

            if self.out.__class__.__name__ == 'Process':
                if self.out.inventory + self.out.busy >= self.out.qlimit:
                    self.out.stop = True
                    yield self.wait[0]

            print('source leaves at {0}'.format(self.env.now))
            self.parts_sent += 1
            p = DataframePart(self.env.now, self.df.iloc[self.parts_sent], self.parts_sent, src=self.id,
                              flow_id=self.flow_id)
            print(p.df[0])
            self.out.put(p)

            print('self.df.count', len(self.df))
            print('self.parts_sent', self.parts_sent)
            if len(self.df) == self.parts_sent + 1:
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

    def __init__(self, env, name, rate, qlimit=None, limit_bytes=True, debug=False):
        self.name = name
        self.store = simpy.Store(env)
        self.rate = rate
        self.env = env
        self.out = None
        self.wait_pre = None
        self.stop = False
        self.wait = [self.env.event()]
        self.parts_rec = 0
        self.parts_drop = 0
        self.qlimit = qlimit
        self.limit_bytes = limit_bytes
        self.debug = debug
        self.busy = 0
        self.inventory = 0
        self.action = env.process(self.run())
        self.working_time = 0

    def run(self):
        self.out.wait_pre = self.wait
        while True:
            msg = (yield self.store.get())

            self.inventory -= 1
            self.busy += 1

            self.start_time = self.env.now
            #print('processing at {0} at {1}'.format(self.name, self.env.now))
            yield self.env.timeout(self.rate())
            self.working_time += self.env.now - self.start_time

            if self.out.__class__.__name__ == 'Process':
                if self.out.inventory + self.out.busy >= self.out.qlimit:
                    self.out.stop = True
                    yield self.wait[0]

            msg.data.append((today + timedelta(minutes=self.env.now)).strftime("%Y-%m-%d %H:%M:%S"))
            self.out.put(msg)
            self.busy -= 1
            if self.debug:
                print(msg)

            if self.inventory + self.busy < self.qlimit and self.stop:
                self.stop = False
                self.wait_pre[0].succeed()
                self.wait_pre[0] = self.env.event()

    def put(self, part):
        self.parts_rec += 1
        self.inventory += 1
        part.data.append("{0}".format(self.name))
        part.data.append((today + timedelta(minutes=self.env.now)).strftime("%Y-%m-%d %H:%M:%S"))
        if self.qlimit is None:
            # print(self.name, ': infinite qlimit')
            return self.store.put(part)
        elif len(self.store.items) >= self.qlimit:
            # print('dropped at', self.name)
            self.parts_drop += 1
        else:
            # print(self.name, ': accept transfer')
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
            total = len(self.port.store.items) + self.port.busy
            self.sizes.append(total)


class RandomBrancher(object):
    """ A demultiplexing element that chooses the output port at random.

        Contains a list of output ports of the same length as the probability list
        in the constructor.  Use these to connect to other network elements.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        probs : List
            list of probabilities for the corresponding output ports
    """
    def __init__(self, env, probs):
        self.env = env
        self.probs = probs
        self.ranges = [sum(probs[0:n+1]) for n in range(len(probs))]  # Partial sums of probs
        if self.ranges[-1] - 1.0 > 1.0e-6:
            raise Exception("Probabilities must sum to 1.0")
        self.n_ports = len(self.probs)
        self.outs = [None for i in range(self.n_ports)]  # Create and initialize output ports
        self.parts_rec = 0

    def put(self, pkt):
        self.parts_rec += 1
        rand = random.random()
        for i in range(self.n_ports):
            if rand < self.ranges[i]:
                if self.outs[i]:  # A check to make sure the output has been assigned before we put to it
                    self.outs[i].put(pkt)
                return