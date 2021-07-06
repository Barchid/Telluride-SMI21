# This TF converts robot sensor data (camera images) to spikes
# In the TF body, camera.value, when available, holds a sensor_msgs.msg.Image message 
@nrp.MapVariable("visual_extractor", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapRobotSubscriber("camera", Topic('/husky/husky/camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("red_left_eye", nrp.brain.sensors[slice(0, 3, 2)], nrp.poisson)
@nrp.MapSpikeSource("red_right_eye", nrp.brain.sensors[slice(1, 4, 2)], nrp.poisson)
@nrp.MapSpikeSource("green_blue_eye", nrp.brain.sensors[4], nrp.poisson)
@nrp.Robot2Neuron()
def eye_sensor_transmit(t, visual_extractor, camera, red_left_eye, red_right_eye, green_blue_eye):
    bridge = CvBridge()
    if not isinstance(camera.value, type(None)) and visual_extractor.value is not None:
        cv_image = bridge.imgmsg_to_cv2(camera.value, "rgb8")
        #clientLogger.info(cv_image.shape)
        test_im = visual_extractor.value(cv_image)
        clientLogger.info(test_im)

    image_results = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera.value)
    red_left_eye.rate = 40000.0 * image_results.left
    red_right_eye.rate = 40000.0 * image_results.right
    green_blue_eye.rate = 75.0 * image_results.go_on

