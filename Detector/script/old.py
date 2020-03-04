import ROOT
import numpy as np
import pandas as pd
import time
import printColor as pc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from celluloid import Camera
from moviepy.editor import *
from matplotlib import gridspec
import scipy.stats as stats
import itertools
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
color = pc.bcolors()
import detectRead as root
import detectWinPlot as winPlot
import pickle



filename = "~/data/g2wd10k_1.root"

dfData   = root.importROOT(filename)
dfPosi   = root.dataWindowing(dfData)
dfMomCut = root.dataMomCut(dfData,200,275)
dfPosiMom   = root.dataWindowing(dfMomCut)

pkData = open("pkData.pkl", "wb")
pkPosi = open("pkPosi.pkl", "wb")
pkMomCut = open("pkMomCut.pkl", "wb")
pkPosiMom = open("pkMomCut.pkl", "wb")


pickle.dump(dfData, pkData)
pickle.dump(dfPosi, pkPosi)
pickle.dump(dfMomCut, pkMomCut)
pickle.dump(dfPosiMom, pkPosiMom)

pkData.close()
pkPosi.close()
pkMomCut.close()
pkPosiMom.close()

dfData=[]
dfDataSum = pd.DataFrame([])
timeBinMin = 35
timeBinMax = 40

dfDataSum = dfPosi[(dfPosi["timeBin"] >= timeBinMin) & (dfPosi["timeBin"] < timeBinMax)]
dfDataSumMom = dfPosiMom[(dfPosiMom["timeBin"] >= timeBinMin) & (dfPosiMom["timeBin"] < timeBinMax)]
print(dfDataSum)

fig1 = plt.figure(1, figsize =(10 , 10))
fig2 = plt.figure(2, figsize =(10 , 10))
fig3 = plt.figure(3, figsize =(10 , 10))

#fig7 = plt.figure(7, figsize =(10 , 10))

pos3DXYZ = fig1.add_subplot(1,2,1, projection='3d')
pos3DVTZ = fig1.add_subplot(1,2,2, projection='3d')
pos2DVZ = fig2.add_subplot(2,2,1)
pos2DVT = fig2.add_subplot(2,2,2)
pos2DXY = fig2.add_subplot(2,2,3)
pos2DTZ = fig2.add_subplot(2,2,4)
histAll = fig3.add_subplot(1,2,1)
histMom = fig3.add_subplot(1,2,2)

pos3DXYZ.set_xlabel('x [mm]')
pos3DXYZ.set_ylabel('y [mm]')
pos3DXYZ.set_zlabel('z [mm]')
pos3DXYZ.set_xlim(-400,400)
pos3DXYZ.set_ylim(-400,400)
pos3DXYZ.set_zlim(-400,400)

pos2DVZ.set_xlabel('vID [vaneID]')
pos2DVZ.set_ylabel('z [mm]')
pos2DVZ.set_xlim(-1,41)
pos2DVZ.set_ylim(-400,400)


# Change minor ticks to show every 5. (20/4 = 5)


timeMin = dfDataSum["hitTime"].min()
timeMax = dfDataSum["hitTime"].max()

pos2DVT.set_xlabel('vID [vaneID]')
pos2DVT.set_ylabel('t [us]')
pos2DVT.set_xlim(-1,41)
pos2DVT.set_ylim(timeMin,timeMax)


winTimeRange = range(timeBinMin,timeBinMax)


# Change minor ticks to show every 5. (20/4 = 5)
pos2DXY.set_xlabel('x [mm]')
pos2DXY.set_ylabel('y [mm]')
pos2DXY.set_xlim(-400,400)
pos2DXY.set_ylim(-400,400)

pos2DTZ.set_xlabel('t [us]')
pos2DTZ.set_ylabel('z [mm]')
pos2DTZ.set_xlim(timeMin,timeMax)
pos2DTZ.set_ylim(-400,400)


# Change minor ticks to show every 5. (20/4 = 5)
#pos2DVZ.grid(which='major', color='#CCCCCC', linestyle='--')
#pos2DVZ.grid(which='minor', color='#CCCCCC', linestyle=':')
#pos2DVT.grid(which='major', color='#CCCCCC', linestyle='--')
#pos2DVT.grid(which='minor', color='#CCCCCC', linestyle=':')

pos3DVTZ.set_xlabel('vID [vaneID]')
pos3DVTZ.set_ylabel('t [us]')
pos3DVTZ.set_zlabel('z [mm]')
pos3DVTZ.set_xlim(0,40)
pos3DVTZ.set_ylim(timeMin,timeMax)
pos3DVTZ.set_zlim(-400,400)


def plotting(data, c):
    eveID = data["eventID"]
    pos3DXYZ.scatter(data["hitPosX"],data["hitPosY"],data["hitPosZ"], color=c)
    pos2DXY.scatter(data["hitPosX"],data["hitPosY"], color=c)
    pos2DVZ.scatter(data["VolID"],data["hitPosZ"], color=c)
    pos2DVT.scatter(data["VolID"],data["hitTime"], color=c)
    pos2DTZ.scatter(data["hitTime"],data["hitPosZ"], color=c)
    pos3DVTZ.scatter(data["VolID"],data["hitTime"], data["hitPosZ"], color=c)

def histHits(histogram, data, hitCut):
    print(data)
    eveID = data["eventID"]
    print("=======================================================")
    print(eveID)
    hist = histogram.hist(eveID, bins = 10000, range =(0,10000), edgecolor = 'black')
    yHist, xHist, patches = hist
    eveSel = []
    #print(yHist)
    for i, j in zip(yHist, xHist):
        if(i >= hitCut):
            #print(int(j))
            eveSel.append(int(j))
            #dfSel = data[data['eventID'].isin(eveSel)]
        #print(dfSel)
    #print(eveSel)
    return eveSel

plotting(dfDataSum, "black")
plotting(dfDataSumMom, "blue")

def winLine(data, winRange):
	for i in winRange:
	    winL = data[data["timeBin"] == i]
	    winL = data["timeBin"]
	    winL = winL.drop_duplicates()
	    print(winL)
	    print(winL.values)
	    pos2DVT.axhline(y=i*5e-2, color='r', linewidth=1)
winLine(dfDataSum, winTimeRange)
dfeveSel = histHits(histAll, dfDataSum, 5)
dfeveSelMom = histHits(histMom ,dfDataSumMom, 4)

#print("===================== dfeveSel =====================")
#print(dfeveSel)

def plotSel(eveSel, data, c, strC):  
    #print(eveSel)
    for i in eveSel: 
        #print(i)
        dfSel = data[data["eventID"] == i ]
        pos3DXYZ.plot(dfSel["hitPosX"],dfSel["hitPosY"],dfSel["hitPosZ"], c)
        pos2DXY.plot(dfSel["hitPosX"],dfSel["hitPosY"],c)
        pos2DVZ.plot(dfSel["VolID"],dfSel["hitPosZ"], c)
        pos2DVT.plot(dfSel["VolID"],dfSel["hitTime"],c)
        pos2DTZ.plot(dfSel["hitTime"],dfSel["hitPosZ"],c)
        pos3DVTZ.plot(dfSel["VolID"],dfSel["hitTime"], dfSel["hitPosZ"], c)

        eveID = dfSel["eventID"].values
        hitTimes = dfSel["hitTime"].values
        volIDs = dfSel["VolID"].values
        zS = dfSel["hitPosZ"].values
        yS = dfSel["hitPosY"].values
        xS = dfSel["hitPosX"].values
        
        for j, k in enumerate(eveID):
        	track = str(k)
        	pos2DXY.annotate(track,color= strC, xy=(xS[j], yS[j]))
        for j, k in enumerate(eveID):
        	track = str(k)
        	pos2DVT.annotate(track,color= strC, xy=(volIDs[j], hitTimes[j]))
        for j, k in enumerate(eveID):
        	track = str(k)
        	pos2DVZ.annotate(track,color= strC, xy=(volIDs[j], zS[j]))
        for j, k in enumerate(eveID):
            track = str(k)
            pos2DTZ.annotate(track,color= strC, xy=(hitTimes[j], zS[j]))    
#print(dfeveSel)

plotSel(dfeveSel, dfDataSum, "b-", "blue")
plotSel(dfeveSelMom, dfDataSumMom, "r-", "red")

plt.show()