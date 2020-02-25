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
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import modules
color = pc.bcolors()

file1 = "~/data/eDepPhi/g2wdEdep10k_1.root"
file2 = "~/data/eDepPhi/g2wdEdep10k_2.root"

rCut = 0

momCutmin = 200
momCutmax = 275

def selectData(file, momCutmin, momCutmax, rCut, labelID):
    LabelData = modules.momLabel(file, momCutmin, momCutmax, rCut)
    LabelDataSel = LabelData[(LabelData['label'] == labelID)]
    print(color.GREEN + "..", len(LabelData["label"]), "- s track is in this label", color.ENDC )
    return LabelDataSel


dataTrkTrS,dataTrkTeS = moduleRNN.momRCut(file1, file2,  momCutmin, momCutmax, rCut)

def selectRanSample(data, number):
    eveid = data.drop_duplicates('eventID')['eventID']
    dataTrkSingleMin = eveid.min()
    dataTrkSingleMax = eveid.max()
    dataTrkSingleRange = range(int(dataTrkSingleMin) , int(dataTrkSingleMax)+1)
    dataTrkSampleLabel = random.sample(dataTrkSingleRange , number)
    return dataTrkSampleLabel

sampleN = 2
sampleTr = selectRanSample(dataTrkTrS, sampleN)
#sampleTe = selectRanSample(dataTrkTeS, sampleN)
dfHitCut, eveid = modules.momRCut(file1, momCutmin, momCutmax, rCut)

eveid = dfHitCut.drop_duplicates('eventID')['eventID']
#eveidTe = dfHitCutTe.drop_duplicates('eventID')['eventID']
print("event ID = ", eveid )
print("Total of eventID  =", len(eveid))
eID1 = eveid.values[11]
eID2 = eveid.values[22]

eID3 = eveid.values[33]
eID4 = eveid.values[44]

sampleeID = (int(eID1) , int(eID2), int(eID3))



#dataTrkTr1 = dataTrkTrS[(dataTrkTrS['eventID'] == sampleTr[0])]
#dataTrkTr2 = dataTrkTrS[(dataTrkTrS['eventID'] == sampleTr[1])]
#dataTrkTr3 = dataTrkTrS[(dataTrkTrS['eventID'] == sampleTr[2])]

#dataTrkTe1 = dataTrkTeS[(dataTrkTeS['eventID'] == sampleTe[0])]
#dataTrkTe2 = dataTrkTeS[(dataTrkTeS['eventID'] == sampleTe[1])]
#dataTrkTe3 = dataTrkTeS[(dataTrkTeS['eventID'] == sampleTe[2])]

    
dataTrkTr1 = dfHitCut[(dfHitCut['eventID'] == eID1)]
dataTrkTr2 = dfHitCut[(dfHitCut['eventID'] == eID2)]
dataTrkTr3 = dfHitCut[(dfHitCut['eventID'] == eID3)]

#print(dataTrMultiSample)
#print(dataTeMultiSample)
#plt.show()
def dataPre(Track1, Track2, Track3):
    TrackSum = pd.concat([Track1, Track2,Track3])
    #class_le = LabelEncoder()
    y = TrackSum['hitPosZ'].values
    x = TrackSum['hitTime'].values
    
    y = y.tolist()
    x = x.tolist()
    #TrackSum = TrackSum[TrackSum.columns.difference(['eventID'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitMag'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitAngle'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitR'])]
    size = len(TrackSum.index)
    print(y)
    print(TrackSum)
    return TrackSum, y , x, size

def dataTZ(Track):
    y = Track['hitPosZ'].values
    x = Track['hitTime'].values
    y = y.tolist()
    x = x.tolist()
    size = len(Track.columns)
    return y, x, size


trackGraph, X_train_all , y_train_all , size_train_all = dataPre(dataTrkTr1, dataTrkTr2, dataTrkTr3)   

X_train , y_train, size_train = dataTZ(dataTrkTr1)

X_test , y_test, size_test = dataTZ(dataTrkTr2)

n_all = len(X_train_all)
print(X_train_all)

lr = 0.01
n_hidden = 35
epochs = 100

def track_to_onehot(track):
    start = np.zeros(shape = len(X_train_all), dtype = int)
    end = np.zeros(shape = len(X_train_all), dtype = int)
    start[-2] = 1
    end[-1] = 1
    for i in track:
        idx = X_train_all.index(i)
        zero = np.zeros(shape= n_all, dtype = int)
        zero[idx] = 1
        start = np.vstack([start,zero])
    output  = np.vstack([start,end])
    return output
def onehot_to_track(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return X_train_all[onehot.argmax()]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.act_fn = nn.Tanh()
    def forward(self, input, hidden):
        hidden = self.act_fn(self.i2h(input) + self.h2h(hidden))
        output = self.i2o(hidden)
        return output, hidden
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)



rnn = RNN(size_train_all , n_hidden , size_train_all)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters() , lr = lr)
print(X_train)
one_hot = torch.from_numpy(track_to_onehot(X_train)).type_as(torch.FloatTensor())

for i in range(epochs):
    rnn.zero_grad()
    total_loss = 0
    hidden = rnn.init_hidden()
    for j in range(one_hot.size()[0]-1):
        input_ = one_hot[j:j+1,:]
        target = one_hot[j+1]

        output, hidden = rnn.forward(input_, hidden)
        loss = loss_func(output.view(-1), target.view(-1))
        total_loss +=loss
        input_ = output

    total_loss.backward()
    optimizer.step()

    if i % 10 ==0:
        print(total_loss)

start = torch.zeros(1, n_all)
start[:,-2] = 1


with torch.no_grad():
    hidden = rnn.init_hidden()
    input_ = start
    output_track = []
    for i in range(len(X_train)):
    #for i in range(len(X_train)):
        output, hidden = rnn.forward(input_, hidden)
        output_track.append(onehot_to_track(output.data))
        print(i)
        print(onehot_to_track(output.data))
        input_ = output
#print(output_track)
output_track = np.array(output_track)

X_train = np.array(X_train)
print("=================================================")
accuracy = sum(X_train == output_track)/  len(X_train)
#accuracy = sum(X_train == output_track)/ 5
print(accuracy)
print("=================================================")
#print(output_track)

def plottingMerging(fig3dN, fig2dN , data, sample):
    sampleN = len(sample)
    fig1 = plt.figure(fig3dN, figsize =(8 , 8))
    fig2 = plt.figure(fig2dN, figsize =(8 , 8))
    pos3D  = fig1.add_subplot(111, projection='3d')
    pos2DXY = fig2.add_subplot(3,2,1)
    pos2DTZ = fig2.add_subplot(3,2,2)
    pos2DTR = fig2.add_subplot(3,2,3)
    pos2DTA = fig2.add_subplot(3,2,4)
    pos2DRA = fig2.add_subplot(3,2,5)
    pos2DRZ = fig2.add_subplot(3,2,6)
    print(sample)
    #print(sampleN)
    dataMulti = pd.DataFrame([])
    for i in sample:
        print(i)
        dataTemp = data[(data['eventID'] == i)]
        print(i, "-th event Track will be plotted..")
        c3D = pos3D.scatter(dataTemp["hitPosX"],dataTemp["hitPosY"],dataTemp["hitPosZ"], cmap=plt.cm.get_cmap('rainbow', sampleN), s=10)
        pos2DXY. scatter(dataTemp["hitPosX"],dataTemp["hitPosY"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DTZ. scatter(dataTemp["hitTime"],dataTemp["hitPosZ"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DTR. scatter(dataTemp["hitTime"],dataTemp["hitR"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DTA. scatter(dataTemp["hitTime"],dataTemp["hitAngle"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DRA. scatter(dataTemp["hitR"],dataTemp["hitAngle"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DRZ. scatter(dataTemp["hitR"],dataTemp["hitPosZ"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 )
        dataMulti  = pd.concat([dataMulti,dataTemp])
    fig1.colorbar(c3D)
    #fig2.colorbar(ticks=range(sampleN), format='color: %d', label='color')
    pos3D.set_xlabel('x [mm]')
    pos3D.set_ylabel('y [mm]')
    pos3D.set_zlabel('z [mm]')
    pos3D.set_xlim( -400,400)
    pos3D.set_ylim( -400,400)
    pos3D.set_zlim(-400,400)

    pos2DXY.set_xlabel('X [mm]')
    pos2DXY.set_ylabel('Y [mm]')
    pos2DXY.set_xlim(-400,400)
    pos2DXY.set_ylim(-400,400)
    
    pos2DTZ.set_xlabel('T [us]')
    pos2DTZ.set_ylabel('Z [mm]')
    pos2DTZ.set_xlim(0,60)
    pos2DTZ.set_ylim(-400,400)
    
    pos2DTR.set_xlabel('T [us]')
    pos2DTR.set_ylabel('R [mm]')
    pos2DTR.set_xlim(0,60)
    pos2DTR.set_ylim(0,400)
    
    pos2DTA.set_xlabel('T [us]')
    pos2DTA.set_ylabel('A [degree]')
    pos2DTA.set_xlim(0,60)
    pos2DTA.set_ylim(0,180)
    
    pos2DRA.set_xlabel('R [mm]')
    pos2DRA.set_ylabel('A [degree]')
    pos2DRA.set_xlim(0,400)
    pos2DRA.set_ylim(0,180)
    
    pos2DRZ.set_xlabel('R [mm]')
    pos2DRZ.set_ylabel('Z [mm]')
    pos2DRZ.set_xlim(0,400)
    pos2DRZ.set_ylim(-400,400)
    return dataMulti

#print(sampleeID)
#print(type(sampleeID))
#print(sampleTr)
#print(type(sampleTr))

dataTrMultiSample = plottingMerging(1,2, trackGraph, sampleeID) 

    
plt.show()