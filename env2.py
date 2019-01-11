'''
This environment is used for gripping object with obstacle
command for launch the gazebo world: roslaunch pana_gazebo pana_ur5_joint_limited.launch 
'''
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
from std_srvs.srv import Empty
from std_msgs.msg import UInt16
from copy import deepcopy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
DURATION = 0.01
GOAL = [0.5,0,1.4,-1.57,1.57,0] 
INIT = [0,-pi/2,0,-pi/2,0,0]

class Ur5_vision(object):
    get_counter = 0
    get_rotation = 0

    def __init__(self, init_joints=INIT, goal_pose=GOAL, duration=DURATION):
        rospy.init_node('ur5_env', anonymous=True)
        parameters = rospy.get_param(None)
        index = str(parameters).find('prefix')
        if (index > 0):
            prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
            for i, name in enumerate(JOINT_NAMES):
                JOINT_NAMES[i] = prefix + name

        self.spawn = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.collision = rospy.Subscriber('/contact_sensor_plugin/contact_info',UInt16,self.callback)
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
                                                                FollowJointTrajectoryAction)
        self.client_gripper = actionlib.SimpleActionClient('/robotiq_controller/follow_joint_trajectory',
                                                                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.client_gripper.wait_for_server()
        self.initial= FollowJointTrajectoryGoal()
        self.initial.trajectory = JointTrajectory()
        self.initial.trajectory.joint_names = JOINT_NAMES
        self.current_joints = init_joints
        self.initial.trajectory.points = [JointTrajectoryPoint(positions=INIT, velocities=[0]*6, 
                                                                        time_from_start=rospy.Duration(duration))]                                                                
        self.tf = TransformListener()                            
        self.goal_pose = np.array(goal_pose)
        self.base_pos = self.get_pos(link_name='base_link')
        self.duration = duration
        self.termination = 0
        self.state_dim = 16
        self.action_dim = 5
        self.receive = False
        #initilize frame buffer
        self.frame_buffer = np.empty([4,64,64], dtype=np.float32)
        self.gripper(action=[1.0])
        
    def callback(self, msg):
        if self.receive:
            self.termination = msg.data

    def step(self,action):
        #Execute action 
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        #set action joint limit
        action_ = np.concatenate((action,0),axis=None)
        self.current_joints += action_
        action_sent = np.zeros(6)
        #adjust joints degree  which has bound [-pi,pi]
        for i in range(5):
            #self.current_joints[i] = self.current_joints[i] % np.pi if self.current_joints[i] > 0 elif self.current_joints[i] % -np.pi
            if self.current_joints[i] > np.pi:
                action_sent[i] = self.current_joints[i] % -np.pi
            elif self.current_joints[i] < -np.pi:
                action_sent[i] = self.current_joints[i] % np.pi
            else:
                action_sent[i] = self.current_joints[i]
        #set constraint for joints
        action_sent[2] = 0.7 * action_sent[2]
        action_sent[3] = 0.5 * (action_sent[3] - np.pi)
        #action_sent[4] = 0.7 * action_sent[4]

        goal.trajectory.points = [JointTrajectoryPoint(positions=action_sent, velocities=[0]*6, 
                                                                    time_from_start=rospy.Duration(self.duration))]
        self.client.send_goal(goal)
        self.client.wait_for_result()
        position, rpy = self.get_pos(link_name='wrist_3_link')
        vision_frames = self.get_vision_frames()
        state = self.get_state(action,position,action_sent[:5])
        reward, terminal = self.get_reward(position,rpy,action)

        return vision_frames, state, reward, terminal

    def reset(self):
        self.current_joints = INIT
        self.get_counter, self.get_rotation, self.termination = 0, 0, 0
        self.delete_model("object")
        self.delete_model("obstacle")
        self.client.send_goal(self.initial)
        self.client.wait_for_result()
        self.receive = True
        self.target_generate()
        vision_frames = self.get_vision_frames(work=False)
        position, rpy = self.get_pos(link_name='wrist_3_link')

        return vision_frames, self.get_state([0,0,0,0,0],position,self.current_joints[:5])

    def get_state(self,action,position,current_joints):
        #x, y, z of goal
        goal_pose = self.goal_pose[:3]
        #goal_dis = np.linalg.norm(goal_pose-self.base_pos)
        pose = position - goal_pose
        dis = np.linalg.norm(pose)
        #get flag for reaching the expected condition
        in_point = 1 if self.get_counter > 0 else 0
        in_rpy = 1 if self.get_rotation > 0 else 0

        state = np.concatenate((current_joints,pose,dis,action,in_point,in_rpy),axis=None)
        #state normalization
        #state = state / np.linalg.norm(state)

        return state

    def get_reward(self,pos,rpy,action):
        threshold = 10
        t = False
        #Compute reward based on distance
        dis = np.linalg.norm(self.goal_pose[:3]-pos)
        #add regularization term
        reward = -0.1 * dis - 0.01 * np.linalg.norm(action)
        #compute reward based on rotation
        dis_a = np.linalg.norm(self.goal_pose[3]-rpy[0])
        r_a = -0.1 * dis_a 
        
        if dis < 0.05:
            reward += 1 + r_a
            self.get_counter += 1
            print 'reach target'
            if dis_a < 0.05:
                reward += 1
                print 'reach rotation'
                self.get_rotation += 1
            else:
                self.get_rotation = 0

            if self.get_rotation > threshold:
                reward += 10
                t = True
                print 'successfully complete task'
                print '############################'
        else:
            self.get_counter = 0
            self.get_rotation = 0
   
        if self.termination == 1:
            reward += -1
            t = True
            self.receive = False
            self.termination = 0

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
                print 'fail to get data from tf'

        return np.array(position), np.array(rpy)
    
    def get_vision_frames(self,work=True):
        '''
        This function returns image frames comprised by last three frames and current frame
        '''
        img = None
        while img is None:
            try:
                img = rospy.wait_for_message('/camera1/color/image_raw', Image)
            except:
                print 'fail to get data from camera'

        img = CvBridge().imgmsg_to_cv2(img, "mono8")
        img = cv2.resize(img,(64,64))
        #normolizae pixel into [0,1]
        img = img / 255.0
        #group 4 frames as observation
        if work:
            for i in range(self.frame_buffer.shape[0]-1):
                self.frame_buffer[i] = self.frame_buffer[i+1]
            self.frame_buffer[-1] = img
        #fullfill the frame buffer at the beginning
        else:
            for i in range(self.frame_buffer.shape[0]):
                 self.frame_buffer[i] = img
        vision = np.copy(self.frame_buffer)
        return vision

    def target_vis(self,goal):
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        #orient of both object and obstacle
        orient = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, 0))
        origin_pose1 = Pose(Point(goal[0],goal[1],goal[2]), orient)
        origin_pose2 = Pose(Point(0.4,np.random.uniform(-0.2,0.2) ,1.2), orient)

        with open('/home/waiyang/pana_ws/src/Panasonic_UR5/pana_gazebo/worlds/reel_simple.sdf',"r") as f:
            xml1 = f.read()
        with open('/home/waiyang/pana_ws/src/Panasonic_UR5/pana_gazebo/worlds/wall.sdf',"r") as f:
            xml2 = f.read()
        
        name1 = "object"
        pose1 = deepcopy(origin_pose1)
        pose1.position.x = origin_pose1.position.x #- 3.5 * unit + col * unit
        pose1.position.y = origin_pose1.position.y #- 3.5 * unit + row * unit
        pose1.position.z = origin_pose1.position.z
        self.spawn(name1, xml1, "", pose1, "world")

        name2 = "obstacle"
        pose2 = deepcopy(origin_pose2)
        pose2.position.x = origin_pose2.position.x #- 3.5 * unit + col * unit
        pose2.position.y = origin_pose2.position.y #- 3.5 * unit + row * unit
        pose2.position.z = origin_pose2.position.z
        self.spawn(name2, xml2, "", pose2, "world")
        
    def target_generate(self):
        self.goal_pose = np.array(GOAL)
        self.goal_pose[0] += np.random.uniform(0,0.2)
        self.goal_pose[1] += np.random.uniform(-0.2,0.2)
        self.goal_pose[2] += np.random.uniform(-0.1,0.1)
        self.target_vis(self.goal_pose)

    def gripper(self, action):
        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = ['finger_joint']
        try:
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*1, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=action, velocities=[0]*1, time_from_start=rospy.Duration(DURATION)),]
            self.client_gripper.send_goal(g)
            self.client_gripper.wait_for_result()
        except KeyboardInterrupt:
            self.client_gripper.cancel_goal()
            raise
        

if __name__ == '__main__':
    arm = Ur5_vision()
    arm.target_generate()
