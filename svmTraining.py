from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import codecs
import codecs
import joblib 
from sklearn import svm

VECTOR_SIZE=3

def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

class SVMTraining:
    def __init__(self) :
        self.flag=0
        self.dataCounter=0
        self.ratioVector=[]
        self.eyeRatioList=[]
        self.cap=cv2.VideoCapture(0)
        self.faceDetector=FaceMeshDetector(maxFaces=1)
    def SVMForClosedEyes(self):
        txt=codecs.open(str("train_close.txt"),'w','utf-8')
        print("现在进行闭眼数据收集\n")
        print("按b开始，按s暂停，按q退出\n")
        print("为了模型的准确性，请尽可能收集较长时间的数据\n")
        self.getVideoData(txt)
        
    def SVMForOpenedEyes(self):
        txt=codecs.open(str("train_open.txt"),'w','utf-8')
        print("现在进行睁眼眼数据收集\n")
        print("按b开始，按s暂停，按q退出\n")
        print("为了模型的准确性，请尽可能收集较长时间的数据\n")
        self.getVideoData(txt)
        
    def getVideoData(self,txt):
        while True:
    
            ret,img=self.cap.read()
            key = cv2.waitKey(1)
            if key & 0xFF == ord("b"):
                print('Start collecting images.')
                self.flag = 1
            elif key & 0xFF == ord("s"):
                print('Stop collecting images.')
                self.flag = 0
            elif key & 0xFF == ord("q"):
                print('quit')
                break
            img ,faces=self.faceDetector.findFaceMesh(img,draw=True)
            cv2.imshow("closure",img)
            if self.flag == 1:
                face=faces[0]
                eyePoint=[face[160],face[158],face[144],face[153],face[33],face[133]]    

                lengthVer1,_=self.faceDetector.findDistance(eyePoint[0],eyePoint[2])#垂直
                lengthVer2,_=self.faceDetector.findDistance(eyePoint[1],eyePoint[3])    
                lengthHor,_=self.faceDetector.findDistance(eyePoint[4],eyePoint[5])#水平
        
                leftEyeRatio=(((lengthVer1+lengthVer2)/(2*lengthHor)))
                self.eyeRatioList.append(leftEyeRatio)
                ret, self.ratioVector = queue_in(self.ratioVector, leftEyeRatio)
                if(len(self.ratioVector) == VECTOR_SIZE):
                    txt.write(str(self.ratioVector))
                    txt.write('\n')
                    self.dataCounter += 1
                    print(self.dataCounter)
        self.dataCounter=0
        self.ratioVector=[]
        txt.close()
    def svmTraining(self):
        train = []
        labels = []

        print('Reading train_open.txt...')
        train_open_txt=codecs.open("train_open.txt",'r','utf-8')
        line_ctr = 0
        for txt_str in train_open_txt.readlines():
            temp = []
            # print(txt_str)
            datas = txt_str.strip()
            datas = datas.replace('[', '')
            datas = datas.replace(']', '')
            datas = datas.split(',')
            print(datas)
            for data in datas:
                # print(data)
                data = float(data)
                temp.append(data)
            # print(temp)
            train.append(temp)
            labels.append(0)

        print('Reading train_close.txt...')
        train_close_txt=codecs.open("train_close.txt",'r','utf-8')
        line_ctr = 0
        temp = []
        for txt_str in train_close_txt.readlines():
            temp = []
            # print(txt_str)
            datas = txt_str.strip()
            datas = datas.replace('[', '')
            datas = datas.replace(']', '')
            datas = datas.split(',')
            print(datas)
            for data in datas:
                # print(data)
                data = float(data)
                temp.append(data)
            # print(temp)
            train.append(temp)
            labels.append(1)

        for i in range(len(labels)):
            print("{0} --> {1}".format(train[i], labels[i]))

        train_close_txt.close()
        train_open_txt.close()

        clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
        clf.fit(train, labels)
        joblib.dump(clf, "svmTraining.m")

if __name__=='__main__':
    SVMTraining().SVMForClosedEyes()
    SVMTraining().SVMForOpenedEyes()
    SVMTraining().svmTraining()