#!/usr/bin/env python

# Check Pytorch installation
from logging import debug
import string
from mmcv import image
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmcls
print(mmcls.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmcls.apis import inference_model, init_model, show_result_pyplot

import os
import sys
import cv2
import numpy as np

# ROS related imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image

# NOTE: 
# CvBridge meet problems since we are using python3 env
# We can do the data transformation manually
# from cv_bridge import CvBridge, CvBridgeError

from vision_msgs.msg import Classification2D, \
                            ObjectHypothesis

from mmclassification_ros.srv import *

from mmcls.models import build_classifier

import threading

# Choose to use a config and initialize the detector
CONFIG_NAME = 'configs/resnet/resnet50_8xb32_in1k.py'
CONFIG_PATH = os.path.join(os.path.dirname(sys.path[0]),'mmclassification', CONFIG_NAME)

# Setup a checkpoint file to load
MODEL_NAME =  'resnet50_8xb32_in1k_20210831-ea4938fc.pth'
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'scripts', MODEL_NAME)

class Classifier:

    def __init__(self, model):
        self.image_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.object_pub = rospy.Publisher("~objects", Classification2D, queue_size=1)
        # self.bridge = CvBridge()
        self.model = model

        self._last_msg = None
        self._msg_lock = threading.Lock()
        
        self._publish_rate = rospy.get_param('~publish_rate', 1)
        self._is_service = rospy.get_param('~is_service', False)
        self._visualization = rospy.get_param('~visualization', True)
        

    def run(self):

        if not self._is_service:
            rospy.loginfo('RUNNING MMDETECTOR AS PUBLISHER NODE')
            image_sub = rospy.Subscriber("~image", Image, self._image_callback, queue_size=1)
        else:
            rospy.loginfo('RUNNING MMDETECTOR AS SERVICE')
            rospy.loginfo('SETTING UP SRV')
            srv = rospy.Service('~image', mmclassSrv, self.service_handler)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                rospy.loginfo('RECEIVED MESSAGE')
                class_obj = Classification2D()
                # try:
                #     cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # except CvBridgeError as e:
                #     print(e)
                # NOTE: This is a way using numpy to convert manually
                im = np.frombuffer(msg.data, dtype = np.uint8).reshape(msg.height, msg.width, -1)
                # image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                image_np = np.asarray(im)

                # Use the inference_model to do inference
                # NOTE: inference_model() is able to receive both str and ndarray
                results = inference_model(self.model, image_np)

                # convert inference results to ros message
                class_obj.header = msg.header
                class_obj.source_img = msg
                obj_hypothesis = ObjectHypothesis()
                obj_hypothesis.id = results['pred_label']
                obj_hypothesis.score = results['pred_score']
                class_obj.results.append(obj_hypothesis)



                if not self._is_service:
                    self.object_pub.publish(class_obj)
                else:
                    rospy.loginfo('RESPONSING SERVICE')
                    return mmdetSrvResponse(class_obj)

                # Visualize results
                if self._visualization:
                    # NOTE: Hack the provided visualization function by mmdetection
                    # Let's plot the result
                    # show_result_pyplot(self.model, image_np, results, score_thr=0.3)
                    # if hasattr(self.model, 'module'):
                    #     m = self.model.module
                    debug_image = self.model.show_result(
                                    image_np,
                                    results,
                                    show=False,
                                    wait_time=0,
                                    win_name='result')
                    # img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    # image_out = Image()
                    # try:
                        # image_out = self.bridge.cv2_to_imgmsg(img,"bgr8")
                    # except CvBridgeError as e:
                    #     print(e)
                    # image_out.header = msg.header
                    image_out = msg
                    # NOTE: Copy other fields from msg, modify the data field manually
                    # (check the source code of cvbridge)
                    image_out.data = debug_image.tostring()

                    self.image_pub.publish(image_out)

            rate.sleep()

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

    def service_handler(self, request):
        return self._image_callback(request.image)


def main(args):
    rospy.loginfo('Init node successfully')
    rospy.init_node('mmclassification')
    model = init_model(CONFIG_PATH, MODEL_PATH, device='cuda:0')
    obj = Classifier(model)
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("ShutDown")
    obj.run()
    # cv2.destroyAllWindows()

if __name__=='__main__':
    main(sys.argv)
