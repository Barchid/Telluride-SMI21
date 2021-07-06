@nrp.MapVariable("visual_extractor", initial_value=None, scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def init_vonenet(t, visual_extractor):
    if visual_extractor.value is None:
        clientLogger.info('INITIALIZATION OF THE VARIABLE visual_extractor')
        
        import site, os
        # WARNING: the path can change according to the python version you chose when initializing the virtualenv
        site.addsitedir(os.path.expanduser('~/.opt/tensorflow/lib/python3.8/site-packages'))
        
        # ONNX ARCHITECTURE
        import onnx
        from onnx_tf.backend import prepare

        visual_extractor.value = prepare(onnx.load('visual_extractor.onnx'))
        clientLogger.info(visual_extractor.value)
        clientLogger.info('visual_extractor VARIABLE CORRECTLY INITIALIZED')
        