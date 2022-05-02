from ast import excepthandler
import cv2
import mediapipe as mp
import time
import math
# https://google.github.io/mediapipe/solutions/pose

class poseDetector():
    def __init__(self):
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
    
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
            cv2.putText(img,str(int(angle)),(x2,y2+20), cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),5)
        return img
    
def main():
    cap = cv2.VideoCapture("PoseVideos/5.mp4")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    videoWriter = cv2.VideoWriter("PoseResults/5.mp4", cv2.VideoWriter_fourcc('P','I','M','1'), int(fps*1), (width,height))

    count = 0
    detector = poseDetector()
    while True: 
        success, img = cap.read()
        if success:
            try:
                img = detector.findPose(img)
                img, lm_list = detector.findPosition(img)
                # right / left ankle
                img = detector.findAngle(img,[26,30,32],'ankle')
                #img = detector.findAngle(img,[25,29,31],'ankle')
                # right / left hip
                img = detector.findAngle(img,[12,24,26],'hip')
                #img = detector.findAngle(img,[11,23,25],'hip')
                # right / left knee
                img = detector.findAngle(img,[24,26,28],'knee')
                #img = detector.findAngle(img,[23,25,27],'knee')
            except:
                pass
            cv2.imshow("Image",img)
        videoWriter.write(img)

        if count == length-1:
            cap.release()
            cv2.destroyAllWindows()
            videoWriter.release()
        count += 1
        print(count)

if __name__ == "__main__":
    main()
    print('done')