@nrp.MapVariable("visual_extractor", initial_value=None)
@nrp.Robot2Neuron()
def init_vonenet(t, visual_extractor):
    if visual_extractor.value is None:
        clientLogger.info(f'INITIALIZATION OF THE VARIABLE {visual_extractor}')
        
        import site, os
        # WARNING: the path can change according to the python version you chose when initializing the virtualenv
        site.addsitedir(os.path.expanduser('~/.opt/pytorch/lib/python3.8/site-packages'))
        
        # VONENET ARCHITECTURE
        import vonenet
        visual_extractor.value = vonenet.VOneNet(model_arch='cornets')
        clientLogger.info(visual_extractor.value)
        