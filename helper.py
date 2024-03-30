import numpy as np
import time
import scipy

def move_joints_position(viz, q):
    viz.display(np.array(q))

def move_trajectory(viz, frequency, trajectory):
    while not trajectory.done():
        move_joints_position(viz, trajectory.current_q)
        trajectory.step(1/frequency)
        time.sleep(1/frequency)

def translate(vector):
    # we allocate a 4x4 array (as identity since this corresponds to no motion)
    transform = np.eye(4)
    
    # here you can fill the rest of the transform
    transform[0:3,3] = vector

    ### we return the object
    return transform

def rotateX(angle):
    # we allocate a 4x4 array (as identity since this corresponds to no motion)
    transform = np.eye(4)
    
    # here you can fill the rest of the transform
    transform[1:3,1:3] = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    ### we return the object
    return transform

def rotateY(angle):
    # we allocate a 4x4 array (as identity since this corresponds to no motion)
    transform = np.eye(4)
    
    # here you can fill the rest of the transform
    transform[0,0] = np.cos(angle)
    transform[0,2] = np.sin(angle)
    transform[2,0] = -np.sin(angle)
    transform[2,2] = np.cos(angle)

    ### we return the object
    return transform

def rotateZ(angle):
    # we allocate a 4x4 array (as identity since this corresponds to no motion)
    transform = np.eye(4)
    
    # here you can fill the rest of the transform
    transform[0:2,0:2] = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    
    ### we return the object
    return transform

def forward_kinematics_KUKA(theta):
    T_BJ1 = translate([0,0,0.1575]) @ rotateZ(theta[0])
    T_J1J2 = translate([0,0,0.2025]) @ rotateY(theta[1])
    T_J2J3 = translate([0,0,0.2045]) @ rotateZ(theta[2])
    T_J3J4 = translate([0,0,0.2155]) @ rotateY(-theta[3])
    T_J4J5 = translate([0,0,0.1845]) @ rotateZ(theta[4])
    T_J5J6 = translate([0,-0.0607, 0.2155]) @rotateY(theta[5])
    T_J6J7 = translate([0,0.0607, 0.0810])@ rotateZ(theta[6])
    T_J7E = translate([0,0,0.04])

    T_SF = T_BJ1 @ T_J1J2 @ T_J2J3 @ T_J3J4 @ T_J4J5 @ T_J5J6 @ T_J6J7 @ T_J7E
    
    ### we return the pose of the end-effector
    return T_SF

def euclidean_distance(pos1, pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    d_pos = pos2-pos1
    return np.sqrt(np.sum(d_pos*d_pos))

def get_error(joint_angles, desired_position, initial_joint_angles):
    current_position = forward_kinematics_KUKA(joint_angles)[0:3,3]
    error = euclidean_distance(current_position, desired_position)**2
    if initial_joint_angles != None:
        error  = error + np.sum((initial_joint_angles-joint_angles)**2)
    return error

def inverse_kinematics(desired_position, current_q):
    q0 = np.zeros(7)
    res = scipy.optimize.minimize(get_error, q0, args = (desired_position,current_q))
    return res.x, res.fun

def reset(viz):
    move_joints_position(viz,np.zeros(7))

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

    



        