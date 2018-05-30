from threading import Thread, Lock
import cv2
import argparse
import logging
import time

import cv2
import numpy as np
import pandas as pd
from collections import Counter

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import sys

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

ch.setFormatter(formatter)
logger.addHandler(ch)


def rotate(src, degrees):
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
# update_cnt = 0
read_cnt = 0


class WebcamVideoStream:
    def __init__(self, src='', width=320, height=240):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        # global update_cnt
        while self.started:
            self.read_lock.acquire()
            (grabbed, frame) = self.stream.read()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

            # update_cnt = update_cnt + 1
            # print("update_cnt", update_cnt)
            time.sleep(0.01)

    def read(self):
        global read_cnt
        self.read_lock.acquire()

        grabbed = self.grabbed
        frame = self.frame
        self.read_lock.release()
        read_cnt += 1
        print("read_cnt", read_cnt)
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


# def do_pose_estimation(video_info, result_queue):
if __name__ == "__main__":
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

    cap = 'C:/Users/BIT-USER/Desktop/HUN.mp4'
    # cap = 'C:/Users/feb29/PycharmProjects/OpenCV_Ex/HUN2.mp4'
    # cap = video_info['videoPath']
    vs = WebcamVideoStream(src=cap).start()
    v_cap = cv2.VideoCapture(cap)

    # if (cap.start() == False):
    #     print("Error opening video stream or file")

    ret, image = vs.read()
    image = rotate(image, 90)
    pre_L_x, pre_L_y, pre_R_x, pre_R_y = 0, 0, 0, 0
    curr_R_x, curr_R_y, curr_L_x, curr_L_y = 0, 0, 0, 0
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # 어깨
    s_pre_R_x = humans[0].body_parts[2].x
    s_pre_R_y = humans[0].body_parts[2].y
    s_pre_L_x = humans[0].body_parts[5].x
    s_pre_L_y = humans[0].body_parts[5].y

    # 손목
    w_pre_R_x = humans[0].body_parts[4].x
    w_pre_R_y = humans[0].body_parts[4].y
    w_pre_L_x = humans[0].body_parts[7].x
    w_pre_L_y = humans[0].body_parts[7].y

    # 무릎
    k_pre_R_x = humans[0].body_parts[9].x
    k_pre_R_y = humans[0].body_parts[9].y
    k_pre_L_x = humans[0].body_parts[12].x
    k_pre_L_y = humans[0].body_parts[12].y

    print('처음좌표', s_pre_R_x, s_pre_R_y)
    e1 = cv2.getTickCount()
    i = True

    # frame_cnt = 0
    # print("pose_estimation_frame ::", v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while i:
        ret, image = vs.read()
        if not ret:
            break
        else:
            # print("frame_cnt", frame_cnt)
            # frame_cnt += 1
            image = rotate(image, 90)
            humans = e.inference(image)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            chk_move = 0
            frame_cnt = 0
            shoulder_move_cnt = 0
            wrist_move_cnt = 0
            miss_location = []
            move_direction = [0, 0, 0, 0]  # 상, 하, 좌, 우

            # R 어깨
            if 2 in humans[0].body_parts:
                s_curr_R_x = humans[0].body_parts[2].x
                s_curr_R_y = humans[0].body_parts[2].y
            # L 어깨
            if 5 in humans[0].body_parts:
                s_curr_L_x = humans[0].body_parts[5].x
                s_curr_L_y = humans[0].body_parts[5].y
            # R 손목
            if 4 in humans[0].body_parts:
                w_curr_R_x = humans[0].body_parts[4].x
                w_curr_R_y = humans[0].body_parts[4].y
            # L 손목
            if 7 in humans[0].body_parts:
                w_curr_L_x = humans[0].body_parts[7].x
                w_curr_L_y = humans[0].body_parts[7].y
            # R 무릎
            if 9 in humans[0].body_parts:
                k_curr_R_x = humans[0].body_parts[9].x
                k_curr_R_y = humans[0].body_parts[9].y
            # L 무릎
            if 12 in humans[0].body_parts:
                k_curr_L_x = humans[0].body_parts[12].x
                k_curr_L_y = humans[0].body_parts[12].y

            # print('어깨x', s_pre_R_x - s_curr_R_x)
            # print('어깨y', s_pre_R_y - s_curr_R_y)
            print('손목x', w_pre_R_x - w_curr_R_x)
            print('손목y', w_pre_R_y - w_curr_R_y)
            print('손목x', w_pre_L_x - w_curr_L_x)
            print('손목y', w_pre_L_y - w_curr_L_y)
            # print('무릎x', k_pre_R_x - k_curr_R_x)
            # print('무릎y', k_pre_R_y - k_curr_R_y)

            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)

            # 어깨
            if (s_pre_R_x - s_curr_R_x) > 0.01 or (s_pre_R_y - s_pre_R_y) > 0.01:
                if chk_move == 0:
                    shoulder_move_cnt = shoulder_move_cnt + 1
                    chk_move = 1

                    # if abs(s_pre_R_y - s_curr_R_y) > 0.02:  # 상
                    #     move_direction[0] += 1
                    #     print('위로 움직였습니다')
                    # if abs(s_pre_R_y - s_curr_R_y) < 0.03:  # 하
                    #     move_direction[1] += 1
                    #     print('아래로 움직였습니다')
                    # if abs(s_pre_R_x - s_curr_R_x) < 0.02:  # 좌
                    #     move_direction[2] += 1
                    #     print('왼쪽으로 움직였습니다')
                    # if abs(s_pre_R_x - s_curr_R_x) > 0.02:  # 우
                    #     move_direction[3] += 1
                    #     print('오른쪽으로 움직였습니다')

                    # print(shoulder_move_cnt)
            if abs(w_pre_R_x - w_curr_R_x) > 0.01 or abs(w_pre_R_y - w_curr_R_y) > 0.01:
                if chk_move == 0:
                    wrist_move_cnt = wrist_move_cnt + 1
                    chk_move = 1

                    if abs(w_pre_R_x - w_curr_R_x) < 0.03 or abs(w_pre_R_y - w_curr_R_y) > 0.02:  # 우
                        move_direction[0] += 1
                        print('오른손이 움직였습니다')
                    if abs(w_pre_L_x - w_curr_L_x) > 0.01 or abs(w_pre_L_y - w_curr_L_y) > 0.02:  # 좌
                        move_direction[1] += 1
                        print('왼손이 움직였습니다')
                    # if w_pre_R_x - w_curr_R_x < -0.02:  # 좌
                    #     move_direction[2] += 1
                    #     print('왼쪽으로 움직였습니다')
                    # if w_pre_R_x - w_curr_R_x > 0.02:  # 우
                    #     move_direction[3] += 1
                    #     print('오른쪽으로 움직였습니다')

                    print(wrist_move_cnt)
            else:
                # if chk_move == 1:
                    # end = int(ceil(frame_cnt / fps))
                    # miss_location.append(end)
                    # print("frame_cnt, end::::", frame_cnt, end)
                chk_move = 0


            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            # cap.stop()
    i = False
    vs.stop()
    cv2.destroyAllWindows()

    e2 = cv2.getTickCount()
    print("correcting time :: ", (e2 - e1) / cv2.getTickFrequency())

logger.debug('finished+ pose_estimation')