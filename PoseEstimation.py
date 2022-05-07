from PoseModule import *
from ast import excepthandler
import cv2
import mediapipe as mp
import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
import random

inputFile = "PoseVideos/1.mp4"
cap = cv2.VideoCapture(inputFile)
detector = poseDetector(cap)

for i in range(detector.fc)[:100]:
    success, img = cap.read()
    if success:
        img = detector.resizeImageByHeight(img, 500)
        img = detector.findPose(img)
        img, lm_list = detector.findPosition(img)
        img = detector.findAngle(img,[26,30,32],'ankle')
        img = detector.findAngle(img,[12,24,26],'hip')
        img = detector.findAngle(img,[24,26,28],'knee')

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        cap.release()
        cv2.destroyAllWindows()

