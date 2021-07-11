#!/usr/bin/env python
"""
IBA Module example.
This module will run eight times during every CLE loop and will output a log message at every iteration.
"""

__author__ = 'Omer Yilmaz'

import time
import random
import traceback
import rospy
from external_module_interface.external_module import ExternalModule
import tf
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
import pfrl
from pfrl.nn import BoundByTanh
from pfrl.policies import DeterministicHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import cv2
import numpy
import torchvision
from torchvision.transforms.transforms import Lambda, ToTensor
#from gazebo_msgs.msgs import WheelSpeeds
import vonenet
from sensor_msgs.msg import Image
import numpy as np
import math
from gazebo_msgs.msg import ModelState

# EXPERIMENT CONFIGURATION
EXPERIENCE_NAME = 'mobilenetv3_RGB'
EPOCH_NUMBER = 0

# Reward configurations
R_COLLISION = -100 # reward when collision
R_ARRIVED = 100 # reward when robot arrived
C_R = 100 # judge arrival
C_D = 1.1 # DISTANCE MAX FOR THE ARRIVAL (if the robot2goal distance is less than this, the episode is over)
C_P = -0.05   # time step penalty
MAX_TIMESTEPS = 100000 # Max number of timesteps allowed before finishing the episode

# Action configuration

AGENT_WEIGHT_FILE = None # path of a weight file to load before launching the training
N_ACTIONS = 5 # Number of actions. =2 for continuous values and =1 for discrete mode
USE_GPU = -1 # -1 stands for "no"
IMAGE_DIMENSION = (3, 64, 64) # (C, H, W) format for the RGB images
VISUAL_ARCH = None # architecture of the visual extractor. can be: 'cornets', 'resnet50' or 'alexnet' or None for a basic mobilenetv3 network
VISUAL_PRETRAINED = True # flag to indicate the pretraining of visual extractor?
LR = 1e-2 # Learning rate


# ARCHITECTURE OF Q_FUNCTION (for the DoubleDQN agent)
def QFunction(visual_model: nn.Module, n_actions: int):
    """
    visual_model (nn.Module): model of the visual feature extractor (must be the first module of the QFunction)
    n_actions (int): number of action (must be the output of the QFunction).
    """
    return torch.nn.Sequential(
        visual_model,
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 250),
        torch.nn.ReLU(),
        torch.nn.Linear(250, n_actions),
        pfrl.q_functions.DiscreteActionValueHead(), # Must end with this (pfrl thing) 
    )

class Module2(ExternalModule):

    def __init__(self, module_name=None, steps=1):
        super(Module2, self).__init__(module_name, steps)

    def initialize(self):
        try:
            rospy.logwarn('BEGIN RL AGENT')

            # Sub to the camera of the husky robot
            self.sub = rospy.Subscriber('/husky/husky/camera',
                                        Image,
                                        self.camera_callback)

            # sub to the collision rostopic
            self.sub_collision = rospy.Subscriber('/gazebo/contact_point_data', ContactsState, self.collision_callback)

            # keeps the last image available
            self.image = None
            self.capture_image = True

            # TorchVision Transforms to adapt the cv2 image
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x/255.
                ),
                transforms.Resize((IMAGE_DIMENSION[1], IMAGE_DIMENSION[2])),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # initialize agent
            self.agent = self._initialize_agent(
                vonenet_arch=VISUAL_ARCH
            )

            # initialize subscribers etc
            self._initialize_rospy()

            # get goal position
            self.goal = self.get_state("cylinder_0", "world").pose.position

            # get current distance from goal (will be used for the reward computation)
            robot_state = self._get_robot_state()
            position = robot_state.pose.position
            orientation = robot_state.pose.orientation
            self.previous_distance, _ = self._distance_from_goal(position, orientation)


            # Set episode state
            self.episode_reward = 0.0 # total reward calculated over the episode
            self.timestep = 0 # current number of timesteps (incremented after each run_step() call from the NRP experiment)

            # vel_cmd
            self.vel_cmd = [0.0, 0.0]

            # measure time
            self.tick = time.time()

            # collision flag
            self.has_collided = False

            # Episode Number (begins at 0 of course)
            self.episode = 0

            self._log_file('\nEnd INITIALIZE')
        except Exception as exc:
            self._log_file(f"\n{exc}\n{traceback.format_exc()}", file="error.txt")

    def _initialize_agent(self, vonenet_arch='cornets', eps=1e-2):
        rospy.logwarn('Visual model get architecture')
        # get vonenet visual feature extractor
        if VISUAL_ARCH is not None:
            visual_model = vonenet.get_model(
                model_arch=vonenet_arch,
                pretrained=VISUAL_PRETRAINED
            )
        else:
            visual_model = torchvision.models.mobilenet.mobilenet_v3_small(pretrained=VISUAL_PRETRAINED)
        self.visual_model = visual_model

        rospy.logwarn('Define Q-Function')

        network = QFunction(visual_model, n_actions=N_ACTIONS)

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(network.parameters(), lr=LR, eps=eps)

        # Set the discount factor that discounts future rewards.
        gamma = 0.9
        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=lambda: random.uniform(0., 5.)) #env.action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10000)

        # Set the device id to use GPU. To use CPU only, set it to -1.
        gpu = USE_GPU

        agent = None
        # Now create an agent that will interact with the environment.
        agent = pfrl.agents.DoubleDQN(
            network,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=32,#500,
            update_interval=1,
            target_update_interval=100,
            gpu=gpu,
            )
        
        rospy.logwarn('Agent initialized!')

        if AGENT_WEIGHT_FILE is not None:
            agent.load(AGENT_WEIGHT_FILE)

        return agent

    def _initialize_rospy(self):
        """
        Initializes all rospy proxies/subscribers/publishers to interact with the environment
        """
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.vel_pub = rospy.Publisher('/husky/husky/cmd_vel', Twist)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def camera_callback(self, img_msg):
        # check if the callback is still allowed to capture images
        if not self.capture_image:
            return

        # TIMESTEP EVERY 0.5 SECONDS
        tick = time.time()
        if (tick - self.tick) < 0.5:
            return

        self.tick = tick

        try:
            image = np.frombuffer(img_msg.data, np.uint8).reshape((img_msg.height, img_msg.width, 3))
            # from np array to Tensor using the transform
            self.image = self.transform_image(image)

            # OBSERVATIONS HANDLING FROM PREVIOUS ACTIONS
            is_episode_end = self._handle_observation()
            if is_episode_end:
                self._reinitialize()
                return

            # AGENT PRODUCES ITS ACTION
            action = self.agent.act(self.image)

            # ACTION HANDLING FOR CURRENT TIMESTEP
            self._handle_action(action)
        except Exception as exc:
            self._log_file(f"\n{exc}\n{traceback.format_exc()}", file="error.txt")
            self.capture_image = False # stop from capturing image if this code does not work

    def _log_file(self, message, file='log.txt'):
        """
        Short function to write some logs in a text file to debug the IBA module
        """
        text_file = open("/home/bbpnrsoa/.opt/nrpStorage/telluride_maze_husky_0_0_0/" + file, "a")
        text_file.write(message)
        text_file.close()

    # DON'T USE THIS: it doesn't work anywhere for some unknown reason...
    def run_step(self):
        pass

    def share_module_data(self):
        self.module_data = [0, 0, 5.1]

    def _handle_observation(self):
        """
        Handles the reward computation from the current observations (i.e. image from previous timestep).
        Function called by "run_step"
        """
        # GET robot state
        robot_state = self._get_robot_state()
        position = robot_state.pose.position
        orientation = robot_state.pose.orientation

        # GET distance FROM objective
        distance, _ = self._distance_from_goal(position, orientation)

        # COMPUTE reward AND end_of_episode
        reward, has_arrived = self._compute_reward(
            distance=distance
        )
        self.previous_distance = distance

        # ADD reward TO episode_reward
        self.episode_reward += reward

        if self.timestep > 0:
            # AGENT OBSERVES
            self.agent.observe(
                obs=self.image,
                reward=reward,
                done=has_arrived,
                reset= self.has_collided or self.timestep >= MAX_TIMESTEPS
            )

        # INCREMENT TIMESTEP
        self._log_file(f"\nTimestep={self.timestep}, Reward={reward}, Distance2Goal={distance}, Position=({position.x, position.y}), cmd=({self.vel_cmd[0]},{self.vel_cmd[1]})")
        self.timestep += 1

        # CHECK end of episode
        is_episode_end = has_arrived or self.timestep >= MAX_TIMESTEPS or self.has_collided

        if is_episode_end:
            end_episode_reason = ""
            if has_arrived:
                end_episode_reason = 'success'
            elif self.has_collided:
                end_episode_reason = 'collision'
            else:
                end_episode_reason = 'timesteps'
            self.end_episode(end_episode_reason=end_episode_reason)

        return is_episode_end

    def _get_robot_state(self):
        robot_state = None
        # use the handle just like a normal function, "robot" relative to "world"
        robot_state = self.get_state("husky", "world")
        assert robot_state.success is True
        return robot_state

    def _distance_from_goal(self, position, orientation):
        """
        Computes the distance between the robot and the goal object (i.e. cylinder_0).
        Function called by _handle_observation
        """
        d_x = self.goal.x - position.x
        d_y = self.goal.y - position.y

        _, _, theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])

        d = math.sqrt(d_x * d_x + d_y * d_y) # Compute distance (euler)        
        alpha = math.atan2(d_y, d_x) - theta # compute angle diff

        return d, alpha

    

    def _compute_reward(self, distance):
        """
        Computes and returns the reward value from the observations in parameters.
        Function called by "_handle_observations".
        """
        has_arrived = distance <= C_D

        # CASE: robot has arrived (game is over)
        if distance <= C_D:
            return R_ARRIVED, has_arrived

        # CASE: robot has collided
        if self.has_collided:
            return R_COLLISION, has_arrived

        # CASE: compute proportional distance depending on the previous distance
        delta_d = self.previous_distance - distance
        return C_R*delta_d + C_P, has_arrived

    def _handle_action(self, action):
        """
        Applies the action (in parameter) produced by the agent.
        """
        vel_cmd = Twist()
        action = max(0, min(int(action), 4)) # clamp between 0 and 4
        # 3 actions
        if action == 0:  # Left
            vel_cmd.linear.x = 0.25
            vel_cmd.angular.z = 1.0
        elif action == 1:  # H-LEFT
            vel_cmd.linear.x = 0.25
            vel_cmd.angular.z = 0.4
        elif action == 2:  # Straight
            vel_cmd.linear.x = 0.25
            vel_cmd.angular.z = 0
        elif action == 3:  # H-Right
            vel_cmd.linear.x = 0.25
            vel_cmd.angular.z = -0.4
        elif action == 4:  # Right
            vel_cmd.linear.x = 0.25
            vel_cmd.angular.z = -1.0
        else:
            raise Exception('Error discrete action: {}'.format(action))

        self.vel_cmd = [vel_cmd.linear.x * 1.5, vel_cmd.angular.z * 1.5]
        self.vel_pub.publish(vel_cmd)


    def end_episode(self, end_episode_reason):
        """
        Handles the end of the episode
        """
        # LOG the episode's stats
        self._log_file(
            f"\nreward={self.episode_reward}\ntimestep={self.timestep}\nend_episod_reason={end_episode_reason}\n{self.agent.get_statistics()}",
            file=f"agent_logs/{EXPERIENCE_NAME}_epoch{EPOCH_NUMBER}_episode{self.episode}.txt"
        )

        # save the weights every 50 episodes
        if self.episode%50==0:
            self.agent.save(f"/home/bbpnrsoa/.opt/nrpStorage/telluride_maze_husky_0_0_0/agent/{EXPERIENCE_NAME}_epoch{EPOCH_NUMBER}_episode{self.episode}")


    def _reinitialize(self):
        """
        Reinitializes the simulation for a next episode
        """
        # PAUSE
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause()
        
        self.capture_image = False # prevent from entering camera callback
        #time.sleep(0.5) # sleep for 3 seconds to avoid synchronization problems with callbacks (UGLY)

        # MOVES THE ROBOT TO THE INITIAL POSE
        state = ModelState()
        state.model_name = 'husky'
        state.reference_frame = 'world'  # ''ground_plane'
        state.pose.position.x = 1.338 # hardcoded values from experiment
        state.pose.position.y = -2.528
        state.pose.position.z = 1.084
        state.scale.x = 1
        state.scale.y = 1
        state.scale.z = 1
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        result = self.set_state(state)
        assert result.success is True # ensure it works

        # MOVES THE CYLINDER GOAL TO ITS INITIAL POSE
        state = ModelState()
        state.model_name = 'cylinder_0'
        state.reference_frame = 'world'  # ''ground_plane'
        state.pose.position.x = 6.589 # hardcoded values from experiment
        state.pose.position.y = 0.511
        state.pose.position.z = 1.208
        state.scale.x = 1
        state.scale.y = 1
        state.scale.z = 1
        result = self.set_state(state)
        assert result.success is True # ensure it works

        # increase episode number
        self.episode += 1

        # reset attributes
        self.image = None
        self.capture_image = True
        self.episode_reward = 0.0 
        self.timestep = 0 # current number of timesteps
        self.vel_cmd = [0.0, 0.0] # no velocity command

        # DISTANCE FROM GOAL
        robot_state = self._get_robot_state()
        position = robot_state.pose.position
        orientation = robot_state.pose.orientation
        self.previous_distance, _ = self._distance_from_goal(position, orientation)


        self.tick = time.time() # new time

        # collision flag reset
        self.has_collided = False

        # unpause physics
        result = self.unpause()


    def collision_callback(self, data):
        """
        Callback from a "contact_point" ROS Subscriber to detect if the Husky robot has collided with a wall
        """
        for state in data.states:
            self.has_collided = self.has_collided or 'husky' in state.collision1_name and 'concretewall' in state.collision2_name or 'concretewall' in state.collision1_name and 'husky' in state.collision2_name


if __name__ == "__main__":
    m = Module2(module_name='module_cerebellum', steps=1)
    rospy.spin()
