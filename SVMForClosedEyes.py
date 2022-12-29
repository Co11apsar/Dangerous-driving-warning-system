from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import codecs
VECTOR_SIZE = 3
cap=cv2.VideoCapture(0)
face_detector=FaceMeshDetector(maxFaces=1)
eyeRatioList=[]

flag = 0
txt = codecs.open("train_close.txt",'w','utf-8')
data_counter = 0
ratio_vector = []

def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

while True:
    
    ret,img=cap.read()
    key = cv2.waitKey(1)
    if key & 0xFF == ord("b"):
        print('Start collecting images.')
        flag = 1
    elif key & 0xFF == ord("s"):
        print('Stop collecting images.')
        flag = 0
    elif key & 0xFF == ord("q"):
        print('quit')
        break
    img ,faces=face_detector.findFaceMesh(img,draw=True)
    cv2.imshow("closure",img)
    if flag == 1:
        face=faces[0]
        eyePoint=[face[160],face[158],face[144],face[153],face[33],face[133]]    

        lengthVer1,_=face_detector.findDistance(eyePoint[0],eyePoint[2])#垂直
        lengthVer2,_=face_detector.findDistance(eyePoint[1],eyePoint[3])    
        lengthHor,_=face_detector.findDistance(eyePoint[4],eyePoint[5])#水平
        
        leftEyeRatio=(((lengthVer1+lengthVer2)/(2*lengthHor)))
        eyeRatioList.append(leftEyeRatio)
        #print(leftEyeRatio)
        ret, ratio_vector = queue_in(ratio_vector, leftEyeRatio)
        if(len(ratio_vector) == VECTOR_SIZE):
            txt.write(str(ratio_vector))
            txt.write('\n')
            data_counter += 1
            print(data_counter)
txt.close()