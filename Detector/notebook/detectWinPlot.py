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

def colorList(sample_size):
   #sample_size = 100
    x = np.vstack([
        np.random.normal(0, 1, sample_size).reshape(sample_size//2, 2), 
        np.random.normal(2, 1, sample_size).reshape(sample_size//2, 2), 
        np.random.normal(4, 1, sample_size).reshape(sample_size//2, 2)
    ])
    y = np.array(list(itertools.chain.from_iterable([ [i+1 for j in range(0, sample_size//2)] for i in range(0, 3)])))
    y = y.reshape(-1, 1)
    dfColor = pd.DataFrame(np.hstack([x, y]), columns=['x1', 'x2', 'y'])
    c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(set(dfColor['y'])))]
    return c_list


def windowsPlotSel(dfMuonPos, winMin):
    winMax = winMin + 10
    testPlot = dfMuonPos[(dfMuonPos["timeBin"] >= winMin) & (dfMuonPos["timeBin"] < winMax)]
    return testPlot

def windowsPlotting(dfDataSum):
    fig1 = plt.figure(1, figsize =(10 , 10))
    fig2 = plt.figure(2, figsize =(10 , 10))
    fig3 = plt.figure(3, figsize =(10 , 10))
    fig4 = plt.figure(4, figsize =(10 , 10))
    fig5 = plt.figure(5, figsize =(10 , 10))
    pos3D = fig1.add_subplot(111, projection='3d')
    pos2DVZ = fig2.add_subplot()
    pos2DVT = fig3.add_subplot()
    pos2DXY = fig4.add_subplot()
    hist = fig5.add_subplot()
    
    pos3D.set_xlabel('x [mm]')
    pos3D.set_ylabel('y [mm]')
    pos3D.set_zlabel('z [mm]')
    pos3D.set_xlim( -400,400)
    pos3D.set_ylim( -400,400)
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

    def plotting(data):
        eveID = data["eventID"]
        pos3D.scatter(data["hitPosX"],data["hitPosY"],data["hitPosZ"], color="black")
        pos2DXY.scatter(data["hitPosX"],data["hitPosY"], color="black")
        pos2DVZ.scatter(data["VolID"],data["hitPosZ"], color="black")
        pos2DVT.scatter(data["VolID"],data["hitTime"], color="black")
        eveID = eveID.values
        hitTimes = data["hitTime"].values
        volIDs = data["VolID"].values
        zS = data["hitPosZ"].values
        yS = data["hitPosY"].values
        xS = data["hitPosX"].values
        
        for i, j in enumerate(eveID):
            track = str(j)
            pos2DXY.annotate(track,color='black', xy=(xS[i], yS[i]))
        for i, j in enumerate(eveID):
            track = str(j)
            pos2DVT.annotate(track,color='black', xy=(volIDs[i], hitTimes[i]))
        for i, j in enumerate(eveID):
            track = str(j)
            pos2DVZ.annotate(track,color='black', xy=(volIDs[i], zS[i]))
    
    def histHits(data):
        eveID = data["eventID"]
        hist = plt.hist(eveID, bins = 10000, range =(0,10000), edgecolor = 'black')
        yHist, xHist, patches = hist
        eveSel = []
        #print(yHist)
        for i, j in zip(yHist, xHist):
            if(i >= 2):
                print(int(j))
                eveSel.append(int(j))
                dfSel = data[data['eventID'].isin(eveSel)]
            #print(dfSel)
        #print(eveSel)
        return eveSel
    for i in range(0,len(timeBinRange)):
        plotting(dfData[i])
        winL = dfData[i]["timeBin"]
        winL = winL.drop_duplicates()
        print(winL)
        print(winL.values)
        pos2DVT.axhline(y=(winL.values)*5e-3, color='r', linewidth=1)
    dfeveSel = histHits(dfDataSum)
    print("===================== dfeveSel =====================")
    print(dfeveSel)
    def plotSel(eveSel):  
        #print(eveSel)
        for i in eveSel: 
            print(i)
            dfSel = dfDataSum[dfDataSum["eventID"] == i ]
            pos3D.plot(dfSel["hitPosX"],dfSel["hitPosY"],dfSel["hitPosZ"], '.r-', color='red', linestyle= 'dashed')
            pos2DXY.plot(dfSel["hitPosX"],dfSel["hitPosY"],'.r-', color='red', linestyle= 'dashed')
            pos2DVZ.plot(dfSel["VolID"],dfSel["hitPosZ"],'.r-', color='red', linestyle= 'dashed')
            pos2DVT.plot(dfSel["VolID"],dfSel["hitTime"],'.r-', color='red', linestyle= 'dashed')
    #print(dfeveSel)
    plotSel(dfeveSel)
