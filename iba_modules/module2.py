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
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import cv2
import numpy
from torchvision.transforms.transforms import Lambda, ToTensor
#from gazebo_msgs.msgs import WheelSpeeds
import vonenet
from sensor_msgs.msg import Image
import numpy as np
import math

# EPISODE CONFIGURATION
EPISODE_NUMBER = 0

# Reward configurations
R_COLLISION = -10 # reward when collision
R_ARRIVED = 100 # reward when robot arrived
C_R = 100 # judge arrival
C_D = 1.1 # DISTANCE MAX FOR THE ARRIVAL (if the robot2goal distance is less than this, the episode is over)
C_P = -0.05   # time step penalty
MAX_TIMESTEPS = 100000 # Max number of timesteps allowed before finishing the episode

# Action configuration
IS_CONTINUOUS = False
V_MAX = 0.3  # max linear velocity (for continuous)
W_MAX = 1.0  # max angular velocity (for continuous)

AGENT_WEIGHT_FILE = None
OBS_SIZE = 10
N_ACTIONS = 1
USE_GPU = -1 # -1 stands for "no"
IMAGE_DIMENSION = (3, 64, 64) # (C, H, W) format
VISUAL_ARCH = 'cornets'
LR = 1e-2




# from gazebo_msgs.msgs import WheelSpeeds

class ContinuousModel(nn.Module):
    """The RL model to predict two continuous values (velocity and angular)"""
    def __init__(self, visual_model):
        super(ContinuousModel, self).__init__()
        self.visual_model = visual_model
        

    def forward(self, x):

        return x

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
            self.agent = self._initialize_agent(N_ACTIONS, VISUAL_ARCH, image_dimension=IMAGE_DIMENSION)

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

            self._log_file('\nEnd INITIALIZE')
        except Exception as exc:
            self._log_file(f"\n{exc}\n{traceback.format_exc()}", file="error.txt")

    def _initialize_agent(self, n_actions, vonenet_arch='cornets', image_dimension=(224, 224), eps=1e-2):
        rospy.logwarn('Visual model get architecture')
        # get vonenet visual feature extractor
        visual_model = vonenet.get_model(
            model_arch=vonenet_arch,
            pretrained=True
        )
        self.visual_model = visual_model

        rospy.logwarn('Define Q-Function')
        # Q-Function definition

        network = None
        if IS_CONTINUOUS:
            pass
        else:
            network = torch.nn.Sequential(
                visual_model,
                torch.nn.Linear(1000, 500),
                torch.nn.ReLU(),
                torch.nn.Linear(500, 250),
                torch.nn.ReLU(),
                torch.nn.Linear(250, n_actions),
                pfrl.q_functions.DiscreteActionValueHead(),
            )

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(network.parameters(), lr=LR, eps=eps)

        # Set the discount factor that discounts future rewards.
        gamma = 0.9
        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=lambda: random.uniform(0., 4.)) #env.action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10000)

        phi = lambda x: x.astype(numpy.float32, copy=False)

        # Set the device id to use GPU. To use CPU only, set it to -1.
        gpu = USE_GPU

        agent = None
        if IS_CONTINUOUS:
            pass
        else:
            # Now create an agent that will interact with the environment.
            agent = pfrl.agents.DoubleDQN(
                network,
                optimizer,
                replay_buffer,
                gamma,
                explorer,
                replay_start_size=500,
                update_interval=1,
                target_update_interval=100,
                #phi=phi,
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
        self.vel_pub = rospy.Publisher('/husky/husky/cmd_vel', Twist)

    def camera_callback(self, img_msg):
        # check if the callback is still allowed to capture images
        if not self.capture_image:
            return

        tick = time.time()
        if (tick - self.tick) < 1.0:
            return

        self.tick = tick

        try:
            image = np.frombuffer(img_msg.data, np.uint8).reshape((img_msg.height, img_msg.width, 3))
            # from np array to Tensor using the transform
            self.image = self.transform_image(image)
            #self.image = self.image.unsqueeze(0)

            self._log_file('Wait for contact', file='log.txt')
            contact_data = rospy.wait_for_message('/contact_state', ContactsState)
            self._log_file(f"\n\n\n{contact_data}", file='log.txt')

            # OBSERVATIONS HANDLING FROM PREVIOUS ACTIONS
            is_episode_end = self._handle_observation()
            if is_episode_end:
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

    def run_step(self):
        self._log_file('\nRUN_STEP')
        # try:
        #     self._log_file('Wait for contact', file='log.txt')
        #     contact_data = rospy.wait_for_message('/contact_state', ContactsState)
        #     self._log_file(f"\n\n\n{contact_data}", file='log.txt')
        #     # don't do anything if image has not been captured
        #     if self.image is None:
        #         return

        #     # OBSERVATIONS HANDLING FROM PREVIOUS ACTIONS
        #     is_episode_end = self._handle_observation()
        #     if is_episode_end:
        #         return

        #     # AGENT PRODUCES ITS ACTION
        #     action = self.agent.act(self.image)

        #     # ACTION HANDLING FOR CURRENT TIMESTEP
        #     self._handle_action(action)
        # except Exception as exc:
        #     self._log_file(f"\n{exc}\n{traceback.format_exc()}", file="error.txt")

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

        # GET COLLISION TODO

        # GET distance FROM objective
        distance, _ = self._distance_from_goal(position, orientation)

        # COMPUTE reward AND end_of_episode
        reward, has_arrived = self._compute_reward(
            is_collision=False,
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
                reset= self.timestep >= MAX_TIMESTEPS
            )

        # INCREMENT TIMESTEP
        self._log_file(f"\nTimestep={self.timestep}, Reward={reward}, Distance2Goal={distance}, Position=({position.x, position.y}), cmd=({self.vel_cmd[0]},{self.vel_cmd[1]})")
        self.timestep += 1

        # CHECK end of episode
        is_episode_end = has_arrived or self.timestep >= MAX_TIMESTEPS

        if is_episode_end:
            self.end_episode(is_episode_end)

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

    

    def _compute_reward(self, is_collision, distance):
        """
        Computes and returns the reward value from the observations in parameters.
        Function called by "_handle_observations".
        """
        has_arrived = distance <= C_D

        # CASE: robot has arrived (game is over)
        if distance <= C_D:
            return R_ARRIVED, has_arrived

        # CASE: robot has collided
        if is_collision:
            return R_COLLISION, has_arrived

        # CASE: compute proportional distance depending on the previous distance
        delta_d = self.previous_distance - distance
        return C_R*delta_d + C_P, has_arrived

    def _handle_action(self, action):
        """
        Applies the action (in parameter) produced by the agent.
        """
        self._log_file(f'\naction={action}')
        vel_cmd = Twist()
        if IS_CONTINUOUS:
            vel_cmd.linear.x = V_MAX*action['linear_vel']
            vel_cmd.angular.z = W_MAX*action['angular_vel']
        else:
            action = round(action)
            # 3 actions
            if action == 0:  # Left
                vel_cmd.linear.x = 0.25 *4
                vel_cmd.angular.z = 1.0 *4
            elif action == 1:  # H-LEFT
                vel_cmd.linear.x = 0.25 *4
                vel_cmd.angular.z = 0.4 *4
            elif action == 2:  # Straight
                vel_cmd.linear.x = 0.25 *4
                vel_cmd.angular.z = 0 *4
            elif action == 3:  # H-Right
                vel_cmd.linear.x = 0.25 *4
                vel_cmd.angular.z = -0.4 *4
            elif action == 4:  # Right
                vel_cmd.linear.x = 0.25 *4
                vel_cmd.angular.z = -1.0 *4
            else:
                raise Exception('Error discrete action: {}'.format(action))

        self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]
        self.vel_pub.publish(vel_cmd)

    def end_episode(self, is_episode_end):
        """
        Handles the end of the episode
        """
        # LOG the episode's stats
        self._log_file(
            f"\n{self.episode_reward},{self.timestep},{is_episode_end}\n{self.agent.get_statistics()}",
            file=f"episode_{EPISODE_NUMBER}.txt"
        )

        # save the weights
        self.agent.save(f"agent_{'PPO' if IS_CONTINUOUS else 'DQN'}_episode_{EPISODE_NUMBER}")

        # shutdown experiment
        self.capture_image = False
        rospy.signal_shutdown(
            'Experiment done!'
        )

if __name__ == "__main__":
    m = Module2(module_name='module_cerebellum', steps=1)
    rospy.spin()
