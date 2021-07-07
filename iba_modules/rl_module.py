#!/usr/bin/env python
"""
IBA Module example.
This module will run four times during every CLE loop and will output a log message at every iteration.
"""

__author__ = 'Sami BARCHID'

import rospy
from external_module_interface.external_module import ExternalModule
import sys
import pfrl
import torch
import torch.nn
import gym
import numpy
from gazebo_msgs.msgs import WheelSpeeds
import vonenet
from sensor_msgs.msg import Image

OBS_SIZE = 10
N_ACTIONS = 3
USE_GPU = -1 # -1 stands for "no"

class RlModule(ExternalModule):
    
    def __init__(self, module_name=None, steps=1):
        super(RlModule, self).__init__(module_name, steps)
    
    def initialize(self):
        rospy.logwarn('BEGIN RL AGENT')

        # Sub to the camera of the husky robot
        self.sub = rospy.Subscriber("/husky/husky/camera", self.sub_callback)

        # Create the QFunction network
        self.agent = self._initialize_agent(
            n_actions=N_ACTIONS
        )

        # keeps the last image available
        self.image = None

        

    def _initialize_agent(n_actions, vonenet_arch='cornets', image_dimension=(224, 224), eps=1e-2):
        # get vonenet visual feature extractor
        visual_model = vonenet.get_model(
            model_arch=vonenet_arch,
            pretrained=True
        )

        # Q-Function definition
        q_func = torch.nn.Sequential(
            visual_model,
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, n_actions),
            pfrl.q_functions.DiscreteActionValueHead(),
        )

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(q_func.parameters(), eps=eps)

        # Set the discount factor that discounts future rewards.
        gamma = 0.9
        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=env.action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

        # Since observations from CartPole-v0 is numpy.float64 while
        # As PyTorch only accepts numpy.float32 by default, specify
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(numpy.float32, copy=False)

        # Set the device id to use GPU. To use CPU only, set it to -1.
        gpu = -1

        # Now create an agent that will interact with the environment.
        agent = pfrl.agents.DoubleDQN(
            q_func,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=500,
            update_interval=1,
            target_update_interval=100,
            phi=phi,
            gpu=gpu,
        )

        return agent


    
    def camera_callback(img_msg):
        self.image = np.frombuffer(img_msg.data, np.uint8).reshape((img_msg.height, img_msg.width, self.num_channels[img_msg.encoding]))


    def run_step(self): 
        if self.image is not None:
            pass
        pass
        #rospy.logwarn("Module 1 called")

    def shutdown(self):
        self.

    def share_module_data(self):
        self.module_data = [1, 2.50, -3.7]


if __name__ == "__main__":
    # m = Module1(module_name='module1', steps=1)
    # rospy.spin()

