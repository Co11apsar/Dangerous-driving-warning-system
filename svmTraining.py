import numpy as np
import codecs
import joblib 
from sklearn import svm


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
joblib.dump(clf, "svmTraning.m")

print('predicting [[0.34, 0.34, 0.31]]')
res = clf.predict([[0.34, 0.34, 0.31]])
print(res)

print('predicting [[0.24, 0.24, 0.24]]')
res = clf.predict([[0.19, 0.18, 0.18]])
print(res)
