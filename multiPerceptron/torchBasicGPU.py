import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib._color_data as mcd
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modules
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

file1 = "../../../data/eDepPhi/g2wdEdep10k_1.root"
file2 = "../../../data/eDepPhi/g2wdEdep10k_2.root"

#file2 = "../../../data/eDepPhi/g2wdEdepCut10k_2.root"
rCut = 100

momCutmin = 0
momCutmax = 300

def selectData(file, momCutmin, momCutmax, rCut, labelID):
    LabelData = modules.momLabel(file, momCutmin, momCutmax, rCut)
    LabelDataSel = LabelData[(LabelData['label'] == labelID)]
    print(color.GREEN + "..", len(LabelData["label"]), "- s track is in this label", color.ENDC )
    return LabelDataSel




label1 = 100
label2 = 120
label3 = 140



data1_1 = selectData(file1, 0, 300, 100, label1)
data1_2 = selectData(file1, 0, 300, 100, label2)
data1_3 = selectData(file1, 0, 300, 100, label3)


data2_1 = selectData(file2, 0, 300, 100, label1)
data2_2 = selectData(file2, 0, 300, 100, label2)
data2_3 = selectData(file2, 0, 300, 100, label3)


print('data1_1')
print(data1_1)
print('data1_2')
print(data1_2)
print('data1_3')
print(data1_3)

print('data2_1')
print(data2_1)
print('data2_2')
print(data2_2)
print('data2_3')
print(data2_3)
#data2_1 = selectData(file2, 0, 300, 100, label1)
#data2_2 = selectData(file2, 0, 300, 100, label2)
#data2_3 = selectData(file2, 0, 300, 100, label3)


def dataPre(Track1, Track2, Track3):
    #TrackSum = pd.concat([Track1, Track2, Track3, Track4])
    TrackSum = pd.concat([Track1, Track2, Track3])
    #TrackSum = pd.concat([TrackSum, Track3])
    class_le = LabelEncoder()
    y = class_le.fit_transform(TrackSum['label'].values)
    TrackSum = TrackSum[TrackSum.columns.difference(['label'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['eDep'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitAngle'])]
    #TrackSum = TrackSum[TrackSum.columns.difference(['hitR'])]
    size = len(TrackSum.columns)
    #print(y)
    #print(size)
    print(y)
    #y = np.where( y == label1, -1, 1)
    return TrackSum , y, size



#X,y,size = dataPre(data1, data2, data3, data4)    
X_train , y_train , size_train = dataPre(data1_1, data1_2, data1_3)   
X_test , y_test , size_test = dataPre(data2_1, data2_2, data2_3)

X_train = X_train.values
print(X_train.ndim)
print(X_train)

X_test = X_test.values
print(X_test.ndim)
print(X_test)


#wine = load_wine()
#wine_data = wine.data[0:130]
#wine_target = wine.target[0:130]
#print("wine_target")
#print(wine_target)
#print("y1")
#print(y1)
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=1)
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2)
#X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)
#X_trainW, X_testW, y_trainW, y_testW = train_test_split(wine_data, wine_target, test_size=0.2)
#X_train = X1
#y_train = y1
#X_test = X2
#y_test = y2

sc = StandardScaler()
#X_train1 = sc.fit_transform(X_train1)
#X_test1 = sc.fit_transform(X_test1)
#X_train2 = sc.fit_transform(X_train2)
#X_test2 = sc.fit_transform(X_test2)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)




#X_train1 = torch.from_numpy(X_train1).float() 
#y_train1 = torch.from_numpy(y_train1).long() 
#X_test1 = torch.from_numpy(X_test1).float() 
#y_test1 = torch.from_numpy(y_test1).long()


#X_train2 = torch.from_numpy(X_train2).float() 
#y_train2 = torch.from_numpy(y_train2).long() 
#X_test2 = torch.from_numpy(X_test2).float() 
#y_test2 = torch.from_numpy(y_test2).long()


#X_trainW = torch.from_numpy(X_trainW).float() 
#y_trainW = torch.from_numpy(y_trainW).long() 
#X_testW  = torch.from_numpy(X_testW).float() 
#y_testW  = torch.from_numpy(y_testW).long()
#device = torch.device("cuda:0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = torch.from_numpy(X_train).float().to(device) 
y_train = torch.from_numpy(y_train).long().to(device) 
X_test = torch.from_numpy(X_test).float().to(device) 
y_test = torch.from_numpy(y_test).long().to(device)



#print("X_train1.shape, y_train1.shape") 
#print(X_train1.shape) 
#print(y_train1.shape) 
#print("X_train2.shape, y_train2.shape") 
#print(X_train2.shape) 
#print(y_train2.shape) 

print("X_train.shape, y_train.shape") 
print(X_train.shape) 
print(y_train.shape) 


print("X_test.shape, y_test.shape") 
print(X_test.shape) 
print(y_test.shape)


#print("X_trainW.shape, y_trainW.shape") 
#print(X_trainW.shape) 
#print(y_trainW.shape) 


#train1 = TensorDataset(X_train1, y_train1)
#train2 = TensorDataset(X_train2, y_train2)
#trainW = TensorDataset(X_trainW, y_trainW)
train = TensorDataset(X_train, y_train)
test = TensorDataset(X_test, y_test)



print(train[0])
print(test[0])
#print(trainW[0])

#train_loader = DataLoader(train1, batch_size=16, shuffle=True)
train_loader = DataLoader(train, batch_size=16, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(7,98)
        self.fc2 = nn.Linear(98,98)
        self.fc3 = nn.Linear(98,98)
        self.fc4 = nn.Linear(98,98)
        self.fc5 = nn.Linear(98,98)
        self.fc6 = nn.Linear(98,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x)

model = Net()
model = model.cuda()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(300):
    total_loss = 0
    for X_train, y_train in train_loader:
        X_train, y_train = Variable(X_train), Variable(y_train)
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if (epoch+1) % 50 ==0:
            print(epoch+1, total_loss)

X_test, y_test = Variable(X_test), Variable(y_test)
#X_test1, y_test1 = Variable(X_train2), Variable(y_train2)
result = torch.max(model(X_test).data, 1)[1]

accuracy = sum(y_test.cpu().data.numpy() == result.cpu().numpy()) / len(y_test.cpu().data.numpy())
#accuracy = sum(y_test1.data.numpy() / len(y_test1.data.numpy()))
#print(y_test1.data.numpy())

#accuracy = sum(y_test.data.numpy() == result.numpy()) / len(y_test.data.numpy())

#accuracy = sum( y_test1.data.numpy() / len(y_test1.data.numpy()) )

print(accuracy)