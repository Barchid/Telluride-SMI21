@nrp.MapVariable("rl_agent", initial_value=None)
@nrp.Robot2Neuron()
def init_rl_agent(t, rl_agent):
    if rl_agent.value is None:
        clientLogger.info('INITIALIZATION OF THE VARIABLE rl_agent')
        
        # IMPORTS
        ###############################################################
        import site, os
        # WARNING: the path can change according to the python version you chose when initializing the virtualenv
        site.addsitedir(os.path.expanduser('~/.opt/tensorflow/lib/python3.8/site-packages'))
        import tensorflow as tf
        import tensorflow.keras as keras
        from tensorflow.tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.layers import Dense, Activation, Flatten, Input, concatenate
        from tensorflow.keras.optimizers import Adam, RMSprop
        from rl.agents import DDPGAgent
        from rl.memory import SequentialMemory
        from rl.random import OrnsteinUhlenbeckProcess
        from tensorflow.keras import backend as K

        # PARAMETERS
        ###############################################################
        obs_shape = (6,) # TODO
        nb_actions = 4 # TODO
        PATH_WEIGHT = 'rl_weights.h5'

        clientLogger.info('INIT AGENT')
        clientLogger.info('obs_shape', obs_shape)

        # ACTOR NETWORK
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + obs_shape))
        actor.add(Dense(32))
        actor.add(Activation('relu'))
        actor.add(Dense(32))
        actor.add(Activation('relu'))
        actor.add(Dense(32))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('sigmoid'))

        # CRITIC NETWORK
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + obs_shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = concatenate([action_input, flattened_observation])
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)

        # INSTANTIATE THE AGENT
        memory = SequentialMemory(limit=1000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=nb_actions)
        rl_agent.value = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10, random_process=random_process, gamma=.99, batch_size=5, target_model_update=1e-3, delta_clip=1.)
        rl_agent.value.training = True
        ###############################################################

        # load weights if exist
        if os.path.exists(PATH_WEIGHT) and os.path.isfile(PATH_WEIGHT):
            clientLogger.info('Loading weight file for the rl agent.')
            rl_agent.value.load_weights(PATH_WEIGHT)

        rl_agent.value.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        