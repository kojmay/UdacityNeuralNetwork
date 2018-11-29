import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.old_pos = self.sim.pose

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #位置
#         reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

#         reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)
#         if self.sim.pose[2] >= self.target_pos[2] :
#             reward += 10.0

#         # 水平面距离
        xy_distance = np.sqrt(((self.target_pos[:2] - self.sim.pose[:2])**2).sum())
#         水平面速度
#         xy_velocity = np.sqrt((self.sim.v[:2]**2).sum())
        # 垂直距离
        z_distance = abs(self.target_pos[2] - self.sim.pose[2])
        # 垂直速度
        z_velocity = self.sim.v[2]
#         # xy平面，相对于上一位置移动的距离
#         xy_move = np.sqrt(((self.old_pos[:2] - self.sim.pose[:2])**2).sum())
        #垂直平面相对于上一位置移动的距离
        z_move = self.sim.pose[2] - self.old_pos[2]
        
        # 角速度
        xyz_angular_v = (abs(self.sim.angular_v[:3])).sum()
        
        reward = -2.0*z_distance + 4.0 * z_velocity - 4.0*xyz_angular_v - xy_distance

#         reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()


        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.old_pos = self.sim.pose
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.old_pos = None
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state