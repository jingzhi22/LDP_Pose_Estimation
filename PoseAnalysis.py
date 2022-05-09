import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import io


class poseDetector():
    def __init__(self,cap):
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.cap = cap
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
    
    def resizeImageByHeight(self, img, height):
        dim = (int(self.width*height/self.height), height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lm_list = []
        coor_list = []
        h,w,c = img.shape
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy, cz = int(lm.x*w), int(lm.y*h), int(lm.z*w)
                lm_list.append([cx,cy,cz])
                if draw:
                    cv2.circle(img,(cx,cy),1,(200,0,0),cv2.FILLED)
        self.lm_list = lm_list
        return img, lm_list
    
    def findAngle(self,img,poses,joint,draw=True):
        x1, y1, z1 = self.lm_list[poses[0]]
        x2, y2, z2 = self.lm_list[poses[1]]
        x3, y3, z3 = self.lm_list[poses[2]]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        
        if angle<0:
            angle += 360
        if joint == 'ankle':
            if angle>180:
                angle-=180
            angle = 90 - angle
        if joint == 'knee' or joint == 'hip':
            angle = 180 - angle

        if draw:
            cv2.putText(img,str(int(angle)),(x2,y2+20), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        return img, int(angle)

def GetPose(detector, cap):
    angle_list = []
    img_list = []
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
    return angle_list, img_list

def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def GetDiagram(angle_list1, angle_list2):
    plot_list = []
    for i in range(0,len(angle_list1),2):
        if i % int(len(angle_list1)/10) == 0:
            print('{0} / {1}'.format(i, len(angle_list1)))
        try:
            fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
            fig.set_size_inches(6, 5)
            fig.set_dpi(100)
            ax1.set_xlim([-45, 45])
            ax1.set_ylim([0, len(angle_list1) / 25])
            ax1.set_xlabel("Angle / degrees")
            ax1.set_ylabel("Time / seconds")
            ax2.set_xlim([-45, 45])
            ax2.set_xlabel("Angle / degrees")


            df1 = pd.DataFrame(angle_list1[:i],columns = ['hip','knee','ankle'])
            df2 = pd.DataFrame(angle_list2[:i],columns = ['hip','knee','ankle'])
            y = [j / 25 for j in range(i)]
            ax1.plot(df1.hip.rolling(window=4).mean(),y, c = 'r',label = 'hip')
            ax1.plot(df1.knee.rolling(window=4).mean(),y, c = 'g',label = 'knee')
            ax1.plot(df1.ankle.rolling(window=4).mean(),y, c = 'b',label = 'ankle')

            ax2.plot(df2.hip.rolling(window=4).mean(),y, c = 'r',label = 'hip', linestyle='dashed')
            ax2.plot(df2.knee.rolling(window=4).mean(),y, c = 'g',label = 'knee', linestyle='dashed')
            ax2.plot(df2.ankle.rolling(window=4).mean(),y, c = 'b',label = 'ankle', linestyle='dashed')
            ax1.legend(loc = 'upper right')
            ax2.legend(loc = 'upper right')
            plot_list.append(get_img_from_fig(fig))
            plot_list.append(get_img_from_fig(fig))

        except:
            pass
    return plot_list

# link to first video file
inputFile1 = "1.mp4"
cap1 = cv2.VideoCapture(inputFile1)
cap1.set(cv2.CAP_PROP_FPS, 25)
detector1 = poseDetector(cap1)
print("frame count: ", detector1.fc)
print("FPS: ", detector1.fps)

# link to second video file
inputFile2 = "2.mp4"
cap2 = cv2.VideoCapture(inputFile2)
detector2 = poseDetector(cap2)
print("frame count: ", detector2.fc)
print("FPS: ", detector2.fps)

angle_list1, img_list1 = GetPose(detector1, cap1)
angle_list2, img_list2 = GetPose(detector2, cap2)

if detector1.fc > detector2.fc:
    angle_list1 = angle_list1[:detector2.fc]
else:
    angle_list2 = angle_list2[:detector1.fc]
plot_list = GetDiagram(angle_list1, angle_list2)

final_list = []
for i in range(len(angle_list1)):
    final_list.append(np.concatenate((img_list1[i],plot_list[i],img_list2[i]), axis = 1))

imageio.mimsave('test.gif',final_list,fps=25)
