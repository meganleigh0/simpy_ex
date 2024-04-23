import simpy
from collections import deque, OrderedDict

class DataframePart(object):

    def __init__(self, time, block_data, id, src="a", dst="z", flow_id=0):
        self.time = time
        self.project = block_data[0]  # Project to which the block belongs
        self.location_code = block_data[1]  # Location code of the block
        self.activity_data = block_data[2]  # Activity information for the block
        self.simulation_data = OrderedDict()  # Simulation results for the block
        self.id = id
        self.src = src
        self.dst = dst
        self.flow_id = flow_id

    def __repr__(self):
        return "id: {}, src: {}, time: {}".format(self.id, self.src, self.time)


class DataframeSource(object):

    def __init__(self, env, id, IAT, block_data, process_dict, initial_delay=0, finish=float("inf"), flow_id=0):
        self.id = id
        self.env = env
        self.IAT = IAT  # Inter-arrival time between blocks
        self.block_data = block_data  # Project, location code, and activity information for the block
        self.process_dict = process_dict  # Dictionary storing Process objects corresponding to each activity
        self.initial_delay = initial_delay
        self.finish = finish
        self.parts_sent = 0
        self.action = env.process(self.run())
        self.flow_id = flow_id

    def run(self):
        yield self.env.timeout(self.initial_delay)
        while self.env.now < self.finish:
            try:
                # Wait for inter-arrival time before creating a block
                yield self.env.timeout(next(self.IAT))
                self.parts_sent += 1
                p = DataframePart(self.env.now, next(self.block_data), self.parts_sent, src=self.id,
                                  flow_id=self.flow_id)

                idx = list(p.activity_data.keys())[0]  # Activity code of the first activity in the block
                # If the number of blocks in the first Process is equal to qlimit, trigger an event for waiting
                if self.process_dict[idx].inventory + self.process_dict[idx].busy >= self.process_dict[idx].qlimit:
                    stop = self.env.event()
                    self.process_dict[idx].wait_for_next_process.append(stop)
                    yield stop
                self.process_dict[idx].put(p)
            except StopIteration:
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
        self.block_project_sim = {}  # Dictionary to store simulation results for all blocks

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
            # Integrate simulation results for each block
            # If the dictionary storing simulation result data does not have the block's project as a key, add it
            if not part.project in list(self.block_project_sim.keys()):
                self.block_project_sim[part.project] = {}
            # Add a dictionary storing simulation results for the block with the block's location code as the key
            self.block_project_sim[part.project][part.location_code] = part.simulation_data
            if self.debug:
                print(part)

class Process(object):

    def __init__(self, env, name, subprocess_num, process_dict, qlimit=None, limit_bytes=True, debug=False):
        self.name = name
        self.store = simpy.Store(env)
        self.env = env
        self.subprocess_num = subprocess_num
        self.wait_for_next_process = deque([])  # Store waiting events triggered by the previous Process
        self.wait_for_subprocess = self.env.event()  # Event for waiting when all subprocesses are running
        self.parts_rec = 0
        self.parts_drop = 0
        self.qlimit = qlimit
        self.process_dict = process_dict  # Dictionary storing Process objects corresponding to each activity
        self.limit_bytes = limit_bytes
        self.debug = debug
        self.inventory = 0
        self.busy = 0
        self.working_time = 0
        self.working_time_list = []
        self.action = env.process(self.run())

    def run(self):
        while True:
            # Create a subprocess if the number of blocks being processed is less than subprocess_num
            # Trigger a waiting event if the number of blocks being processed equals subprocess_num
            if self.busy < self.subprocess_num:
                msg = (yield self.store.get())
                self.inventory -= 1
                self.busy += 1
                self.env.process(self.subrun(msg))
            else:
                yield self.wait_for_subprocess

    def subrun(self, msg):
        proc_time = msg.activity_data[self.name][1]  # Store the current Process's work time for the block
        start = self.env.now  # Start time of the block's work
        msg.simulation_data[self.name] = [start]  # Store the block's work start time in simulation results
        yield self.env.timeout(proc_time)  # Trigger an event for the block's work time
        finish = self.env.now  # End time of the block's work
        msg.simulation_data[self.name].append(finish - start)  # Store the block's work time in simulation results

        idx = list(msg.activity_data.keys()).index(self.name)  # Store the order of the current Process among the block's activities
        if idx != len(msg.activity_data) - 1:  # If the current Process is not the last activity, put the block into the next Process
            next_process = list(msg.activity_data.keys())[idx + 1]  # Store the activity_code of the block's next Process
            # Calculate lag based on the planned start time of the next Process and the end time of the current Process's work
            # Trigger a waiting event for the lag time
            lag = msg.activity_data[next_process][0] - finish
            if lag > 0:
                yield self.env.timeout(lag)
            # Trigger a waiting event if the number of blocks in the next Process equals qlimit
            if self.process_dict[next_process].inventory + self.process_dict[next_process].busy >= self.process_dict[next_process].qlimit:
                stop = self.env.event()
                self.process_dict[next_process].wait_for_next_process.append(stop)
                yield stop
            self.process_dict[next_process].put(msg)
        else:  # If the current Process is the last activity, put the block into Sink
            self.process_dict['Sink'].put(msg)

        # End the event triggered for waiting when all subprocesses are running
        self.busy -= 1
        self.wait_for_subprocess.succeed()
        self.wait_for_subprocess = self.env.event()

        if self.debug:
            print(msg)

        # End the waiting events triggered by the previous Process if the number of blocks in the current Process is less than qlimit
        if self.inventory + self.busy < self.qlimit and len(self.wait_for_next_process) > 0:
            temp = self.wait_for_next_process.popleft()
            temp.succeed()

    def put(self, part):
        self.inventory += 1
        self.parts_rec += 1
        
        if self.qlimit is None:
            return self.store.put(part)
        elif len(self.store.items) >= self.qlimit:
            self.parts_drop += 1
        else:
            return self.store.put(part)
        #print(f"Putting part {part.id} into {self.name}, current inventory: {self.inventory}")



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