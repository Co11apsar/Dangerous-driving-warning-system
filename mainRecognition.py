import cv2
import cvzone
import time
import joblib
#import settings 
#from sklearn import svm
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
clf = joblib.load("svm1.m")

cap=cv2.VideoCapture(0)

face_detector=FaceMeshDetector(maxFaces=1)
plotY=LivePlot(640,360,[20,50])

eyeRatioList=[]
mouthRatioList=[]

hand_detector=HandDetector(detectionCon=0.8)

blinking_frequency=0
yawning_frequency=0
count=0
blink_times=0
yawning_times=0

preTime=0

VECTOR_SIZE = 3
ratio_vector=[]
def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

while True: 
    count+=1
    
    success,img=cap.read()
    img ,faces=face_detector.findFaceMesh(img,draw=True)
    cv2.putText(img,(str('blink frequency  ')+str(blinking_frequency)),(70,70),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    cv2.putText(img,(str('yawning frequency  ')+str(yawning_frequency)),(70,90),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    
    curTime=time.time()
    interval=curTime-preTime
    fps=1/interval
    preTime=curTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)#输出帧率
    
    if faces:
        face=faces[0]
        eyePoint=[face[160],face[158],face[144],face[153],face[33],face[133]]    
        
        mouthUp=face[13]
        mouthDown=face[14]
        mouthLeft=face[62]
        mouthRight=face[308]
        
        lengthVer1,_=face_detector.findDistance(eyePoint[0],eyePoint[2])#垂直
        lengthVer2,_=face_detector.findDistance(eyePoint[1],eyePoint[3])    
        lengthHor,_=face_detector.findDistance(eyePoint[4],eyePoint[5])#水平
        
        mouthLengthVer,_=face_detector.findDistance(mouthUp,mouthDown)#垂直
        mouthLengthHor,_=face_detector.findDistance(mouthLeft,mouthRight)#水平
        
        leftEyeRatio=(((lengthVer1+lengthVer2)/(2*lengthHor)))
        mouthRatio=int((mouthLengthVer/mouthLengthHor)*100)
        eyeRatioList.append(leftEyeRatio)
        mouthRatioList.append(mouthRatio)
        if len(mouthRatioList)>20:
            mouthRatioList.pop(0)
        if len(eyeRatioList)>20:
            eyeRatioList.pop(0)
        eyeRatioAvg=sum(eyeRatioList)/len(eyeRatioList)
        mouthRatioAvg=sum(mouthRatioList)/len(mouthRatioList)
        print(leftEyeRatio)
        ret, ratio_vector = queue_in(ratio_vector, leftEyeRatio)
        if(len(ratio_vector) == VECTOR_SIZE):
            #print(ratio_vector)
            input_vector = []
            input_vector.append(ratio_vector)
            res = clf.predict(input_vector)
            print(res)

            if res == 1:
                blink_times+=1
                cvzone.putTextRect(img,str('eyes are closed'),(100,100),scale=1,thickness=2)
                

        #if leftEyeRatio<25:#阈值判断闭眼的方式并不精确，不同的人阈值也不同，训练SVM
        #    cvzone.putTextRect(img,str('eyes are closed'),(100,100),scale=1,thickness=2)
        #    blink_times+=1 
        if eyeRatioAvg<=25:
            cvzone.putTextRect(img,str('you are tired!!!'),(100,150),scale=1,thickness=2)
        if mouthRatioAvg>=30:
            cvzone.putTextRect(img,str('yawning'),(300,100),scale=1,thickness=2)
            yawning_times+=1
        imgPlot=plotY.update(leftEyeRatio)
        cv2.imshow("imagePlot",imgPlot)
    else:
        cvzone.putTextRect(img,str('naping!!!'),(100,100),scale=1,thickness=2)
        
        
    hands,img=hand_detector.findHands(img)
    
    cv2.circle(img,(300,512),200,(255,255,255))
    cv2.circle(img,(300,512),150,(255,255,255))
    
    if len(hands)==1:
        hand=hands[0]

        if hand["type"]=="Left":
            cvzone.putTextRect(img,str('right hand missed'),(400,100),scale=1,thickness=2)
        elif hand["type"]=="Right":
            cvzone.putTextRect(img,str('left hand missed'),(400,150),scale=1,thickness=2)
    elif len(hands)==0:
        cvzone.putTextRect(img,str('both hands missed'),(400,100),scale=1,thickness=2)
    else:
        coord=[]
        for hand in hands:
            cx,cy=hand["center"]
            coord.append(cx)
            coord.append(cy)
            if pow(cx-200,2)+pow(cy-512,2)>pow(250,2):
                if hand["type"]=="Left":
                    cvzone.putTextRect(img,str('left hands off the wheel'),(400,100),scale=1,thickness=2)
                elif hand["type"]=="Right":
                    cvzone.putTextRect(img,str('right hands off the wheel'),(400,100),scale=1,thickness=2)
        if pow(coord[0]-200,2)+pow(coord[1]-512,2)>pow(250,2) and pow(coord[2]-200,2)+pow(coord[3]-512,2)>pow(250,2):
            cvzone.putTextRect(img,str('both hands off the wheel'),(400,100),scale=1,thickness=2)
    if hands:
        for hand in hands:
            hand_detector.fingers
            fingers=hand_detector.fingersUp(hand)
            if fingers==[0,1,1,0,0] or fingers==[0,0,0,1,1]:
                cvzone.putTextRect(img,str('smoking'),(100,250),scale=1,thickness=2)
    img=cv2.resize(img,(1024,768))
    
    cv2.imshow("closure",img)
    if(count==100):
        blinking_frequency=blink_times/10
        yawning_frequency=yawning_times/10
        blink_times=0
        yawning_times=0
        count=0
    

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
