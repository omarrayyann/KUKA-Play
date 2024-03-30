import numpy as np
import time

def move_joints_position(viz, q):
    viz.display(np.array(q))

def move_trajectory(viz, frequency, trajectory):
    while not trajectory.done():
        move_joints_position(viz, trajectory.current_q)
        trajectory.step(1/frequency)
        time.sleep(1/frequency)

class Trajectory:
    
    def __init__(self, from_q, to_q, total_time):
        self.from_q = from_q
        self.to_q = to_q
        self.total_time = total_time
        self.dq = (to_q-from_q)/total_time
        self.current_q = from_q
        self.current_time = 0.0

    def step(self, time_step):

        self.current_time = self.current_time + time_step
        
        if(self.current_time > self.total_time):
            self.current_q = self.to_q
            return self.current_q
        
        self.current_q = self.current_q + self.dq*time_step
        
        return self.current_q
    
    def reset(self):
        self.current_q = self.from_q
        self.current_time = 0.0

    def done(self):
        return self.current_time >= self.total_time

    



        