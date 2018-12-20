import numpy as np 
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from tf import TransformListener
from math import pi 
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys, tf
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from copy import deepcopy

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
DURATION = 0.01
GOAL = [-0.5,-0.5,1.5,-1.57,1.57,0]
INIT = [0,-pi/2,0,-pi/2,0,0]

class Ur5(object):
    get_goal = False
    get_counter = 0

    def __init__(self, init_joints=INIT, goal_pose=GOAL):
        rospy.init_node('ur5_env', anonymous=True)
        parameters = rospy.get_param(None)
        index = str(parameters).find('prefix')
        if (index > 0):
            prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
            for i, name in enumerate(JOINT_NAMES):
                JOINT_NAMES[i] = prefix + name
        
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
                                                                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.initial= FollowJointTrajectoryGoal()
        self.initial.trajectory = JointTrajectory()
        self.initial.trajectory.joint_names = JOINT_NAMES
        self.initial.trajectory.points = [JointTrajectoryPoint(positions=INIT, velocities=[0]*6, 
                                                                        time_from_start=rospy.Duration(DURATION))]                                                                
        self.tf = TransformListener()                            
        self.goal_pose = np.array(goal_pose)
        self.action_p = np.array(INIT)
        self.action_delta = np.zeros(6)
    
    def step(self,action):
        #Execute action 
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        #set action joint limit
        action[2] = action[2] * 7 / 9 
        action[3] = (action[3]-1)/2
        action[4] = action[4] / 2
        action *= pi
        action = np.concatenate((action,0),axis=None)
        goal.trajectory.points = [JointTrajectoryPoint(positions=action, velocities=[0]*6, 
                                                                    time_from_start=rospy.Duration(DURATION))]
        self.client.send_goal(goal)
        self.client.wait_for_result()
        position, rpy = self.get_pos(link_name='wrist_3_link')
        self.action_delta = action - self.action_p
        self.action_p = action

        state = self.get_state()
        reward, terminal = self.get_reward(position)

        return state, reward, terminal

    def reset(self):
        self.get_point = False
        self.grab_counter, self.action_delta = 0, np.zeros(6)
        self.action_p = np.zeros(6)
        self.client.send_goal(self.initial)
        self.client.wait_for_result()
        rand_x, rand_y, rand_z= np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(-0.4,0)
        self.goal_pose = np.array(GOAL)
        self.goal_pose[0] += rand_x
        self.goal_pose[1] += rand_y
        self.goal_pose[2] += rand_z
        self.target_vis(self.goal_pose)

        return self.get_state()
        #return np.array(position)

    def get_state(self):
        goal_pose = self.goal_pose[:3]
        joint_3 = self.get_pos(link_name='forearm_link')[0]
        pose_3 = joint_3-goal_pose
        dis_3 = np.linalg.norm(pose_3)
        joint_6 = self.get_pos(link_name='wrist_3_link')[0]
        pose_6 = joint_6-goal_pose
        dis_6 = np.linalg.norm(pose_6)

        in_point = 1 if self.get_counter > 0 else 0
        
        
        state = np.concatenate((pose_3,pose_6,dis_6),axis=None)
        #state = state / np.linalg.norm(state)

        return np.concatenate((state,self.action_delta[:5],in_point),axis=None)

    def get_reward(self,pos):
        target = self.goal_pose
        threshold = 20
        #Compute reward based on distance
        dis = np.linalg.norm(target[:3]-pos)

        #add regularization term
        reward = -1*dis / 1 - np.linalg.norm(self.action_delta) / 10
        t = False
        
        if dis < 0.15 and (not self.get_goal):
            reward += 1
            self.get_counter += 1
            print 'reach goal'
            if self.get_counter > threshold:
                reward += 10.
                self.get_goal = True
                print 'reach completed'
        elif dis > 0.15:
            self.get_counter = 0
            self.get_goal = False
        #change buffer

        return reward, t
    
    def get_pos(self,link_name='ee_link',ref_link='world'):
        position = None
        while position is None:
            try:
                if self.tf.frameExists('wrist_2_link') and self.tf.frameExists(link_name):
                    t = self.tf.getLatestCommonTime(ref_link, link_name)
                    position, quaternion = self.tf.lookupTransform(ref_link, link_name, t)
                    rpy = euler_from_quaternion(quaternion)
            except:
                pass

        return np.array(position), np.array(rpy)
        
    def target_vis(self,goal):
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        
        s = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        
        orient = Quaternion(*tf.transformations.quaternion_from_euler(1.571, 0, 0))
        origin_pose = Pose(Point(goal[0],goal[1],goal[2]), orient)

        with open('/home/waiyang/pana_ws/src/Panasonic_UR5/pana_gazebo/worlds/reel_simple.sdf',"r") as f:
            reel_xml = f.read()
        
        #for row in [2,4,5]:
        #  for	col in xrange(0,1):
        for row in [1]:
            for	col in xrange(1):
                reel_name = "reel_%d_%d" % (row,col)
                delete_model(reel_name)
                pose = deepcopy(origin_pose)
                pose.position.x = origin_pose.position.x #- 3.5 * unit + col * unit
                pose.position.y = origin_pose.position.y #- 3.5 * unit + row * unit
                pose.position.z = origin_pose.position.z
                s(reel_name, reel_xml, "", pose, "world")

if __name__ == '__main__':
    arm = Ur5()
    while True:
        arm.reset()
        arm.step([2.2,0,-1.57,0,0,0])
