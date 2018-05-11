from thread_basic import FileVideoStream
import argparse
import logging
import time
import copy

import cv2
import numpy as np
import pandas as pd
from collections import Counter

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

path = 'C:/Users/BIT-USER/Desktop/HUN.mp4'
scaling_factor = 0.25

fvs = FileVideoStream(path).start()

time.sleep(1)

e1 = cv2.getTickCount()

def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 1)  # 뒤집기

    elif degrees == 180:
        dst = cv2.flip(src, 0)  # 뒤집기

    elif degrees == 270:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 0)  # 뒤집기
    else:
        dst = None
    return dst

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    #logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/HUN.mp4')
    #ret_val, image = cap.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    ret_val, image = cap.read()
    image = Rotate(image, 90)
    pre_L_x, pre_L_y, pre_R_x, pre_R_y = 0, 0, 0, 0
    curr_R_x, curr_R_y, curr_L_x, curr_L_y = 0, 0, 0, 0
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    pre_R_x = humans[0].body_parts[2].x / 2
    pre_R_y = humans[0].body_parts[2].y
    pre_L_x = humans[0].body_parts[5].x/2
    pre_L_y = humans[0].body_parts[5].y

    while fvs.has_more():
        ret_val, image = cap.read()
        image = Rotate(image, 90)

        humans = e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        curr_R_x = humans[0].body_parts[2].x/2
        curr_R_y = humans[0].body_parts[2].y
        curr_L_x = humans[0].body_parts[5].x/2
        curr_L_y = humans[0].body_parts[5].y

        print(pre_R_x - curr_R_x)
        if ((pre_R_x - curr_R_x) > 0.01):
            print('정신차리세요')
        elif ((pre_R_x - curr_R_x) < 0):
            print('정신차리세요')
        else:
            print('올바른 자세입니다.')

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
    fvs.stop()
    cv2.destroyAllWindows()
logger.debug('finished+')
