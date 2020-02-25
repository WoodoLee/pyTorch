import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib._color_data as mcd
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import moduleRNN
import printColor as pc
import math
#데이터 불러오기
color = pc.bcolors()

file1 = "~/data/eDepPhiVid/g2wd10k_1.root"
file2 = "~/data/eDepPhiVid/g2wd10k_2.root"

rCut = 0

momCutmin = 0
momCutmax = 300

def selectData(file, momCutmin, momCutmax, rCut, labelID):
    LabelData = modules.momLabel(file, momCutmin, momCutmax, rCut)
    LabelDataSel = LabelData[(LabelData['label'] == labelID)]
    print(color.GREEN + "..", len(LabelData["label"]), "- s track is in this label", color.ENDC )
    return LabelDataSel

dataTrk1, dataTrk2 = moduleRNN.momRCut(file1, file2, momCutmin, momCutmax, rCut)

def selectRanSample(data, number):
    eveid = data.drop_duplicates('eventID')['eventID']
    dataTrkSingleMin = eveid.min()
    dataTrkSingleMax = eveid.max()
    dataTrkSingleRange = range(int(dataTrkSingleMin) , int(dataTrkSingleMax)+1)
    dataTrkSampleLabel = random.sample(dataTrkSingleRange , number)
    return dataTrkSampleLabel

sampleN = 2
sampleTr = selectRanSample(dataTrk1, sampleN)
#sampleTe = selectRanSample(dataTrkTeS, sampleN)
dfHitCut1, dfHitCut2= moduleRNN.momRCut(file1, file2 ,momCutmin, momCutmax, rCut)

#Selecting Eve ID
eveId = dfHitCut1.drop_duplicates('eventID')['eventID']
#eveidTe = dfHitCutTe.drop_duplicates('eventID')['eventID']
print("event ID = ", eveId )

print("Total of eventID  =", len(eveId))
eID1 = eveId.values[1]
eID2 = eveId.values[2]
eID3 = eveId.values[3]
eID4 = eveId.values[4]

sampleeID = (int(eID1) , int(eID2), int(eID3))

momRange = range(momCutmin , momCutmax)

print(momRange)
LabelData = moduleRNN.momLabel(file1, momCutmin, momCutmax, rCut)
    
dataTrkTr1 = dfHitCut1[(dfHitCut1['eventID'] == eID1)]
dataTrkTr2 = dfHitCut1[(dfHitCut1['eventID'] == eID2)]
dataTrkTr3 = dfHitCut1[(dfHitCut1['eventID'] == eID3)]

def dataPre(Track1, Track2, Track3):
    TrackSum = pd.concat([Track1, Track2,Track3])
    #class_le = LabelEncoder()
    z = TrackSum['hitPosZ'].values
    t = TrackSum['hitTime'].values
    
    z = z.tolist()
    t = t.tolist()
    #TrackSum = TrackSum[TrackSum.columns.difference(['eventID'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitMag'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitAngle'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitR'])]
    size = len(TrackSum.index)
    return TrackSum, t , z, size

def dataTZ(Track):
    z = Track['hitPosZ'].values
    t = Track['hitTime'].values
    z = z.tolist()
    t = t.tolist()
    size = len(Track.columns)
    return t, z, size

def dataVZ(Track):
    v = Track['VolID'].values
    z = Track['hitPosZ'].values
    v = v.tolist()
    z = z.tolist()
    size = len(Track.columns)
    return v, z, size

trackGraph, t_trainSum , z_trainSum , size_trackGraph = dataPre(dataTrkTr1, dataTrkTr2, dataTrkTr3)   

trackT1 , trackZ1, trackTZSize1 = dataTZ(dataTrkTr1)
trackT2 , trackZ2, trackTZSize2 = dataTZ(dataTrkTr2)
trackT3 , trackZ3, trackTZSize3 = dataTZ(dataTrkTr3)

trackV1 , trackZ1, trackVZSize1 = dataVZ(dataTrkTr1)
trackV2 , trackZ2, trackVZSize2 = dataVZ(dataTrkTr2)
trackV3 , trackZ3, trackVZSize3 = dataVZ(dataTrkTr3)

#data_frame = pd.read_excel("Data/Month_data.xlsx", sheet = 1)

#최고기온, 최저기온 변수 제거 

#del data_frame["Max_temperature"]
#del data_frame["Min_temperature"]

#하이퍼 파라미터 설정

timesteps = seq_length = 6
data_dim = 14
hidden_dim = 14
output_dim = 1
learing_rate = 0.0005
iterations = 50_000

#데이터 조절

#data_frame["Population"] /= 1e5
#data_frame["Supply"] /= 1e7

#Framework 제작

#x = data_frame.values
#y = data_frame["Supply"].values  

print(dataTrkTr1)

x = dataTrkTr1.values
y = dataTrkTr1["hitPosZ"].values

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = np.copy(x[i:i + seq_length + 1])
    _x[timesteps-2][data_dim-1] = 0
    _x[timesteps-1][data_dim-1] = 0
    _x[timesteps][data_dim-1] = 0
    _y = [y[i + seq_length]]
    dataX.append(_x)
    dataY.append(_y)

#학습데이터와 테스트데이터 분류

train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size 

trainX = np.array(dataX[:train_size])
testX = np.array(dataX[train_size : ])

trainY = np.array(dataY[:train_size])
testY = np.array(dataY[train_size : ])


#LSTM모델 구축

X = tf.placeholder(tf.float32, [None, seq_length+1, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

def lstm_cell(): 
    cell = rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True) 
    return cell 


multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(9)], state_is_tuple=True)


outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred - Y))  
train = tf.train.RMSPropOptimizer(learing_rate).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(iterations):
    _  , cost = sess.run([train ,loss], feed_dict={X: trainX, Y: trainY})
    if (i+1) % (iterations/10) == 0:
        print("[step: {}] loss: {}".format(i+1, cost))

train_predict = sess.run(Y_pred, feed_dict={X: trainX})
test_predict = sess.run(Y_pred, feed_dict={X: testX})


print(train_predict)
print(test_predict)


