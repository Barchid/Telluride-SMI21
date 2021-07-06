@nrp.Robot2Neuron()
def transferFunction (t):
    #log the first timestep (20ms), each couple of seconds
    if t % 2 < 0.02:
        clientLogger.info('Time: ', t)