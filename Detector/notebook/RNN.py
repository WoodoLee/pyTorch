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

filename = "~/data/g2wd10k_1.root"

dfData   = root.importROOT(filename)
dfPosi   = root.dataWindowing(dfData)

dfMomCut = root.dataMomCut(dfData,200,275)
dfPosiMom   = root.dataWindowing(dfMomCut)

dfData=[]
dfDataSum = pd.DataFrame([])
timeBinMin = 500
timeBinMax = 515

dfDataSum = dfPosi[(dfPosi["timeBin"] >= timeBinMin) & (dfPosi["timeBin"] < timeBinMax)]
dfDataSumMom = dfPosiMom[(dfPosiMom["timeBin"] >= timeBinMin) & (dfPosiMom["timeBin"] < timeBinMax)]
print(dfDataSum)

fig1 = plt.figure(1, figsize =(10 , 10))
fig2 = plt.figure(2, figsize =(10 , 10))
fig3 = plt.figure(3, figsize =(10 , 10))
fig4 = plt.figure(4, figsize =(10 , 10))
fig5 = plt.figure(5, figsize =(10 , 10))
fig6 = plt.figure(6, figsize =(10 , 10))

pos3D = fig1.add_subplot(111, projection='3d')
pos2DVZ = fig2.add_subplot()
pos2DVT = fig3.add_subplot()
pos2DXY = fig4.add_subplot()
histAll = fig5.add_subplot()
histMom = fig6.add_subplot()

pos3D.set_xlabel('x [mm]')
pos3D.set_ylabel('y [mm]')
pos3D.set_zlabel('z [mm]')
pos3D.set_xlim(-400,400)
pos3D.set_ylim(-400,400)
pos3D.set_zlim(-400,400)

pos2DVZ.set_xlabel('vID [vaneID]')
pos2DVZ.set_ylabel('z [mm]')
pos2DVZ.set_xlim(0,40)
pos2DVZ.set_ylim(-100,100)
pos2DVZ.xaxis.set_major_locator(MultipleLocator(5))
pos2DVZ.yaxis.set_major_locator(MultipleLocator(20))
# Change minor ticks to show every 5. (20/4 = 5)
pos2DVZ.xaxis.set_minor_locator(AutoMinorLocator(5))
pos2DVZ.yaxis.set_minor_locator(AutoMinorLocator(4))

timeMin = dfDataSum["hitTime"].min()
timeMax = dfDataSum["hitTime"].max()

pos2DVT.set_xlabel('vID [vaneID]')
pos2DVT.set_ylabel('t [us]')
pos2DVT.set_xlim(0,40)
pos2DVT.set_ylim(timeMin,timeMax)
winTimeRange = range(timeBinMin,timeBinMax)
print(winTimeRange)
pos2DVT.set_xticks([winTimeRange])
pos2DVT.xaxis.set_major_locator(MultipleLocator(5))
pos2DVT.yaxis.set_major_locator(MultipleLocator(20))
# Change minor ticks to show every 5. (20/4 = 5)
pos2DVT.xaxis.set_minor_locator(AutoMinorLocator(5))
pos2DVT.yaxis.set_minor_locator(AutoMinorLocator(4))
pos2DXY.set_xlabel('x [mm]')
pos2DXY.set_ylabel('y [mm]')
pos2DXY.set_xlim(-400,400)
pos2DXY.set_ylim(-400,400)
pos2DXY.xaxis.set_major_locator(MultipleLocator(5))
pos2DXY.yaxis.set_major_locator(MultipleLocator(20))
# Change minor ticks to show every 5. (20/4 = 5)
pos2DXY.xaxis.set_minor_locator(AutoMinorLocator(5))
pos2DXY.yaxis.set_minor_locator(AutoMinorLocator(4))
pos2DVZ.grid(which='major', color='#CCCCCC', linestyle='--')
pos2DVZ.grid(which='minor', color='#CCCCCC', linestyle=':')
pos2DVT.grid(which='major', color='#CCCCCC', linestyle='--')
pos2DVT.grid(which='minor', color='#CCCCCC', linestyle=':')

def plotting(data, c):
    eveID = data["eventID"]
    pos3D.scatter(data["hitPosX"],data["hitPosY"],data["hitPosZ"], color=c)
    pos2DXY.scatter(data["hitPosX"],data["hitPosY"], color=c)
    pos2DVZ.scatter(data["VolID"],data["hitPosZ"], color=c)
    pos2DVT.scatter(data["VolID"],data["hitTime"], color=c)
     
    #eveID = eveID.values
    #hitTimes = data["hitTime"].values
    #volIDs = data["VolID"].values
    #zS = data["hitPosZ"].values
    #yS = data["hitPosY"].values
    #xS = data["hitPosX"].values 
    #for i, j in enumerate(eveID):
    #    track = str(j)
    #    pos2DXY.annotate(track,color=c, xy=(xS[i], yS[i]))
    #for i, j in enumerate(eveID):
    #    track = str(j)
    #    pos2DVT.annotate(track,color=c, xy=(volIDs[i], hitTimes[i]))
    #for i, j in enumerate(eveID):
    #    track = str(j)
    #    pos2DVZ.annotate(track,color=c, xy=(volIDs[i], zS[i]))

def histHits(histogram, data):
    print(data)
    eveID = data["eventID"]
    print("=======================================================")
    print(eveID)
    hist = histogram.hist(eveID, bins = 10000, range =(0,10000), edgecolor = 'black')
    yHist, xHist, patches = hist
    eveSel = []
    #print(yHist)
    for i, j in zip(yHist, xHist):
        if(i >= 10):
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

dfeveSel = histHits(histAll, dfDataSum)
dfeveSelMom = histHits(histMom ,dfDataSumMom)

#print("===================== dfeveSel =====================")
#print(dfeveSel)

def plotSel(eveSel, data, c):  
    #print(eveSel)
    for i in eveSel: 
        #print(i)
        dfSel = data[data["eventID"] == i ]
        pos3D.plot(dfSel["hitPosX"],dfSel["hitPosY"],dfSel["hitPosZ"], c)
        pos2DXY.plot(dfSel["hitPosX"],dfSel["hitPosY"],c)
        pos2DVZ.plot(dfSel["VolID"],dfSel["hitPosZ"], c)
        pos2DVT.plot(dfSel["VolID"],dfSel["hitTime"],c)
        
        eveID = dfSel["eventID"].values
        hitTimes = dfSel["hitTime"].values
        volIDs = dfSel["VolID"].values
        zS = dfSel["hitPosZ"].values
        yS = dfSel["hitPosY"].values
        xS = dfSel["hitPosX"].values
        
        for j, k in enumerate(eveID):
        	track = str(k)
        	pos2DXY.annotate(track,color= "red", xy=(xS[j], yS[j]))
        for j, k in enumerate(eveID):
        	track = str(k)
        	pos2DVT.annotate(track,color= "red", xy=(volIDs[j], hitTimes[j]))
        for j, k in enumerate(eveID):
        	track = str(k)
        	pos2DVZ.annotate(track,color="red", xy=(volIDs[j], zS[j]))

#print(dfeveSel)

plotSel(dfeveSel, dfDataSum, "b-")
plotSel(dfeveSelMom, dfDataSumMom, "r-")

plt.show()