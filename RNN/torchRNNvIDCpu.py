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

color = pc.bcolors()

file1 = "~/data/g2wd10k_1.root"
file2 = "~/data/g2wd10k_2.root"

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
eID1 = eveId.values[11]
eID2 = eveId.values[22]
eID3 = eveId.values[33]
eID4 = eveId.values[44]

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

volumeID = range(40)


n_all = len(volumeID)
print(n_all)

lr = 0.01
n_hidden = 35
epochs = 100

def track_to_onehot(track):
    start = np.zeros(shape = len(volumeID), dtype = int)
    end = np.zeros(shape = len(volumeID), dtype = int)
    start[-2] = 1
    end[-1] = 1
    for i in track:
        idx = volumeID.index(i)
        zero = np.zeros(shape= n_all, dtype = int)
        zero[idx] = 1
        start = np.vstack([start,zero])
    output  = np.vstack([start,end])
    return output
def onehot_to_track(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return volumeID[onehot.argmax()]

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



rnn = RNN(len(volumeID) , n_hidden , len(volumeID))

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters() , lr = lr)

print(trackV1)

one_hot = torch.from_numpy(track_to_onehot(trackV1)).type_as(torch.FloatTensor())

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
    for i in range(len(trackV2)):
    #for i in range(len(X_train)):
        output, hidden = rnn.forward(input_, hidden)
        output_track.append(onehot_to_track(output.data))
        print(i)
        print(onehot_to_track(output.data))
        input_ = output

#print(output_track)
output_track = np.array(output_track)

trackV2 = np.array(trackV2)
print("=================================================")
accuracy = sum(trackV2 == output_track)/  len(trackV2)
#accuracy = sum(X_train == output_track)/ 5
print(accuracy)
print("=================================================")
#print(output_track)


def plottingMerging(fig3dN, fig2dN , data, sample):
    sampleN = len(sample)
    fig1 = plt.figure(fig3dN, figsize =(10 , 10))
    fig2 = plt.figure(fig2dN, figsize =(10 , 10))
    fig3 = plt.figure(3, figsize =(10 , 10))
    pos3DXYZ  = fig1.add_subplot(111, projection='3d')
    pos3DVTZ  = fig3.add_subplot(111, projection='3d')
    pos2DXY = fig2.add_subplot(3,3,1)
    pos2DTZ = fig2.add_subplot(3,3,2)
    pos2DTR = fig2.add_subplot(3,3,3)
    pos2DTA = fig2.add_subplot(3,3,4)
    pos2DRA = fig2.add_subplot(3,3,5)
    pos2DRZ = fig2.add_subplot(3,3,6)
    pos2DVZ = fig2.add_subplot(3,3,7)
    pos2DVR = fig2.add_subplot(3,3,8)
    pos2DTV = fig2.add_subplot(3,3,9)
    print(sample)
    #print(sampleN)
    dataMulti = pd.DataFrame([])
    for i in sample:
        print(i)
        dataTemp = data[(data['eventID'] == i)]
        print(i, "-th event Track will be plotted..")
        c3D1 = pos3DXYZ.scatter(dataTemp["hitPosX"],dataTemp["hitPosY"],dataTemp["hitPosZ"], cmap=plt.cm.get_cmap('rainbow', sampleN), s=10)
        c3D3 = pos3DVTZ.scatter(dataTemp["VolID"],dataTemp["hitTime"],dataTemp["hitPosZ"], cmap=plt.cm.get_cmap('rainbow', sampleN), s=10)
        pos2DXY. scatter(dataTemp["hitPosX"],dataTemp["hitPosY"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DTZ. scatter(dataTemp["hitTime"],dataTemp["hitPosZ"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DTR. scatter(dataTemp["hitTime"],dataTemp["hitR"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DTA. scatter(dataTemp["hitTime"],dataTemp["hitAngle"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DRA. scatter(dataTemp["hitR"],dataTemp["hitAngle"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 ) 
        pos2DRZ. scatter(dataTemp["hitR"],dataTemp["hitPosZ"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 )
        pos2DVZ. scatter(dataTemp["VolID"],dataTemp["hitPosZ"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 )
        pos2DVR. scatter(dataTemp["VolID"],dataTemp["hitR"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 )
        pos2DTV. scatter(dataTemp["hitTime"],dataTemp["VolID"] , cmap=plt.cm.get_cmap('rainbow', sampleN), s=10 )
        dataMulti  = pd.concat([dataMulti,dataTemp])
    fig1.colorbar(c3D1)
    fig3.colorbar(c3D3)
    #fig2.colorbar(ticks=range(sampleN), format='color: %d', label='color')
    pos3DXYZ.set_xlabel('x [mm]')
    pos3DXYZ.set_ylabel('y [mm]')
    pos3DXYZ.set_zlabel('z [mm]')
    pos3DXYZ.set_xlim( -400,400)
    pos3DXYZ.set_ylim( -400,400)
    pos3DXYZ.set_zlim(-400,400)

    pos3DVTZ.set_xlabel('vID [vaneID]')
    pos3DVTZ.set_ylabel('t[us]')
    pos3DVTZ.set_zlabel('z [mm]')
    pos3DVTZ.set_xlim( 0,40)
    pos3DVTZ.set_ylim( 0,30)
    pos3DVTZ.set_zlim(-400,400)


    pos2DXY.set_xlabel('X [mm]')
    pos2DXY.set_ylabel('Y [mm]')
    pos2DXY.set_xlim(-400,400)
    pos2DXY.set_ylim(-400,400)
    
    pos2DTZ.set_xlabel('T [us]')
    pos2DTZ.set_ylabel('Z [mm]')
    pos2DTZ.set_xlim(0,30)
    pos2DTZ.set_ylim(-400,400)
    
    pos2DTR.set_xlabel('T [us]')
    pos2DTR.set_ylabel('R [mm]')
    pos2DTR.set_xlim(0,30)
    pos2DTR.set_ylim(0,400)
    
    pos2DTA.set_xlabel('T [us]')
    pos2DTA.set_ylabel('A [degree]')
    pos2DTA.set_xlim(0,30)
    pos2DTA.set_ylim(0,180)
    
    pos2DRA.set_xlabel('R [mm]')
    pos2DRA.set_ylabel('A [degree]')
    pos2DRA.set_xlim(0,400)
    pos2DRA.set_ylim(0,180)
    
    pos2DRZ.set_xlabel('R [mm]')
    pos2DRZ.set_ylabel('Z [mm]')
    pos2DRZ.set_xlim(0,400)
    pos2DRZ.set_ylim(-400,400)

    pos2DVZ.set_xlabel('vID [vaneID]')
    pos2DVZ.set_ylabel('Z [mm]')
    pos2DVZ.set_xlim(0,40)
    pos2DVZ.set_ylim(-400,400)

    pos2DVR.set_xlabel('vID [vaneID]')
    pos2DVR.set_ylabel('R [mm]')
    pos2DVR.set_xlim(0,40)
    pos2DVR.set_ylim(0,400)

    pos2DTV.set_xlabel('T [us]')
    pos2DTV.set_ylabel('vID [vaneID]')
    pos2DTV.set_xlim(0,30)
    pos2DTV.set_ylim(0,40)


    return dataMulti

dataTrMultiSample = plottingMerging(1,2, trackGraph, sampleeID) 

    
plt.show()
