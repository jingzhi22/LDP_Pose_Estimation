from PoseModule import *

import cv2
import mediapipe as mp

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import imageio
import io

inputFile = "PoseVideos/4.mp4"
cap = cv2.VideoCapture(inputFile)
detector = poseDetector(cap)
print("frame count: ", detector.fc)
print("FPS: ", detector.fps)
angle_list = []
img_list = []
plot_list = []
final_list = []

for i in range(detector.fc):
    if i % int(detector.fc/10) == 0:
        print('{0} / {1}'.format(i, detector.fc))
    success, img = cap.read()
    if success:
        img = detector.resizeImageByHeight(img, 500)
        img = detector.findPose(img)
        img, lm_list = detector.findPosition(img)
        img, angle_24 = detector.findAngle(img,[12,24,26],'hip')
        img, angle_26 = detector.findAngle(img,[24,26,28],'knee')
        img, angle_30 = detector.findAngle(img,[26,30,32],'ankle')
        angle_list.append([angle_24,angle_26,angle_30])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

imageio.mimsave('test.gif',img_list,fps=detector.fps)

def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

for i in range(detector.fc):
    if i % int(detector.fc/10) == 0:
        print('{0} / {1}'.format(i, len(angle_list)))
    try:
        fig, ax = plt.subplots(figsize=(3,5), dpi = 100)
        plt.xlim([-45, 45])
        plt.ylim([0, detector.fc / detector.fps])
        plt.xlabel("Angle / degrees")
        plt.ylabel("Time / seconds")
        df = pd.DataFrame(angle_list[:i],columns = ['hip','knee','ankle'])
        y = [j / detector.fps for j in range(i)]
        ax.plot(df.hip.rolling(window=4).mean(),y, c = 'coral',label = 'hip')
        ax.plot(df.knee.rolling(window=4).mean(),y, c = 'turquoise',label = 'knee')
        ax.plot(df.ankle.rolling(window=4).mean(),y, c = 'goldenrod',label = 'ankle')
        ax.legend(loc = 'upper right')
        plot_list.append(get_img_from_fig(fig))
    except:
        pass

imageio.mimsave('test2.gif',plot_list,fps=detector.fps)

for i in range(detector.fc):
    if i % int(detector.fc/10) == 0:
        print('{0} / {1}'.format(i, len(angle_list)))
    try:
        final_list.append(np.concatenate((img_list[i],plot_list[i]), axis = 1))
    except:
        pass

imageio.mimsave('test3.gif',final_list,fps=detector.fps)
