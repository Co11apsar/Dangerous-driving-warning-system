import cv2
import cvzone
import time
import joblib
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from ciga_recog import CigaRecog
from SVMTraining import queue_in

clf = joblib.load("svmTraining.m")#SVM训练集

cap=cv2.VideoCapture(0)#摄像头

faceDetector=FaceMeshDetector(maxFaces=1)#面部识别
handDetector=HandDetector()#手部识别
ciga_detector = CigaRecog()

eyeRatioList=[]
eyeList=[]
mouthRatioList=[]

blinkingFrequency=0
yawningFrequency=0
count=0
blinkTimes=0
yawningTimes=0

preTime=0

VECTOR_SIZE = 3
ratio_vector=[]

tmieF=300

while True: 
    count+=1
    
    success,img=cap.read()
    img ,faces=faceDetector.findFaceMesh(img,draw=False)#识别并绘制出面部关键点
    ###
    curTime=time.time()
    interval=curTime-preTime
    fps=1/interval
    preTime=curTime
    cv2.putText(img,str('FPS:')+str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)#输出帧率
    ###
    if faces:
        face=faces[0]
        eyePoint=[face[160],face[158],face[144],face[153],face[33],face[133]]#眼睛的关键点
        
        mouthUp=face[13]
        mouthDown=face[14]
        mouthLeft=face[62]
        mouthRight=face[308]#嘴的关键点
        mouth_box=mouthLeft,mouthUp,mouthRight,mouthDown
        
        lengthVer1,_=faceDetector.findDistance(eyePoint[0],eyePoint[2])#眼睛的关键点的垂直连线
        lengthVer2,_=faceDetector.findDistance(eyePoint[1],eyePoint[3])    
        lengthHor,_=faceDetector.findDistance(eyePoint[4],eyePoint[5])#眼睛的关键点的水平连线
        
        mouthLengthVer,_=faceDetector.findDistance(mouthUp,mouthDown)#嘴的关键点的垂直连线
        mouthLengthHor,_=faceDetector.findDistance(mouthLeft,mouthRight)#嘴的关键点的水平连线
        
        
        leftEyeRatio=(((lengthVer1+lengthVer2)/(2*lengthHor)))#计算眼睛的闭合度 这里只计算了左眼的
        mouthRatio=int((mouthLengthVer/mouthLengthHor)*100)#计算嘴的闭合度
        eyeRatioList.append(leftEyeRatio)
        mouthRatioList.append(mouthRatio)
        if len(mouthRatioList)>20:#只存储大约0.05s的数据,大概是一次眨眼的最短时间
            mouthRatioList.pop(0)
        if len(eyeRatioList)>20:
            eyeRatioList.pop(0)
        if len(eyeList)>20:
            eyeList.pop(0)
        eyeRatioAvg=sum(eyeRatioList)/len(eyeRatioList)#计算眼睛的平均闭合度
        mouthRatioAvg=sum(mouthRatioList)/len(mouthRatioList)#计算嘴的平均闭合度
        ret, ratio_vector = queue_in(ratio_vector, leftEyeRatio)
        
        if(len(ratio_vector) == VECTOR_SIZE):
            input_vector = []
            input_vector.append(ratio_vector)
            res = clf.predict(input_vector)
            if res == 1:#闭眼检测
                blinkTimes+=1
                #cvzone.putTextRect(img,str('eyes are closed'),(100,100),scale=1,thickness=2)
                eyeList.append(1)
            else:
                eyeList.append(0)
                
        if sum(eyeList)>=10:
            cvzone.putTextRect(img,str('you are tired!!!'),(100,150),scale=1,thickness=2)
        if mouthRatioAvg>=30:#打哈欠动作很明显，设定的阈值完全可以检测出来，不用训练
            cvzone.putTextRect(img,str('yawning'),(300,100),scale=1,thickness=2)
            cvzone.putTextRect(img,str('you are tired!!!'),(100,150),scale=1,thickness=2)
            yawningTimes+=1
    else:#没检测到面部就提示打盹
        cvzone.putTextRect(img,str('naping!!!'),(100,100),scale=1,thickness=2)
        
    hands=handDetector.findHands(img,draw=False)
    # 画一个圆环表示方向盘
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
            # 手的坐标在圆外就提示离开了方向盘
            if pow(cx-200,2)+pow(cy-512,2)>pow(300,2):
                if hand["type"]=="Left":
                    cvzone.putTextRect(img,str('left hands off the wheel'),(400,100),scale=1,thickness=2)
                elif hand["type"]=="Right":
                    cvzone.putTextRect(img,str('right hands off the wheel'),(400,100),scale=1,thickness=2)
        if pow(coord[0]-200,2)+pow(coord[1]-512,2)>pow(300,2) and pow(coord[2]-200,2)+pow(coord[3]-512,2)>pow(300,2):
            cvzone.putTextRect(img,str('both hands off the wheel'),(400,100),scale=1,thickness=2)
    
    ###
    ret, frame = cap.read()# 识别香烟
    if not ret or frame is None:
                break
    frame = cv2.flip(frame, 1)
    
    ciga_dets = ciga_detector.predict(frame[:, :, ::-1])
    if ciga_dets:
        # 只处理一支香烟情况
        ciga_det = ciga_dets[0]
        l, t, r, b = ciga_box = ciga_det[0]
        conf = ciga_det[1]

        cvzone.putTextRect(img,str('Cigarette detected'),(400,300),scale=1,thickness=2)
    else:
        ciga_box = None
    ###   
    img=cv2.resize(img,(1024,768))
    cv2.imshow("closure",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
