# Q Function
class QFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QFunction, self).__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

        def forward(self, x):
            h = x
            h = torch.nn.functional.relu(self.l1(h))
            h = torch.nn.functional.relu(self.l2(h))
            h = self.l3(h)
            return pfrl.action_value.DiscreteActionValue(h)

@nrp.MapVariable("rl_agent", initial_value=None)
@nrp.Robot2Neuron()
def init_rl_agent(t, rl_agent):
    if rl_agent.value is None:
        clientLogger.info('INITIALIZATION OF THE VARIABLE rl_agent')
        
        import site, os
        # WARNING: the path can change according to the python version you chose when initializing the virtualenv
        site.addsitedir(os.path.expanduser('~/.opt/pytorch/lib/python3.8/site-packages'))
        import pfrl
        import torch
        import torch.nn

        # BUILD RL AGENT
        ###############################################################

        q_func = QFunction(10, 3) # TODO: change that

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

        # Set the discount factor that discounts future rewards.
        gamma = 0.9

        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=None) # TODO

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

        rl_agent.value = agent
        ###############################################################
        clientLogger.info(rl_agent.value)
        clientLogger.info('rl_agent VARIABLE CORRECTLY INITIALIZED')