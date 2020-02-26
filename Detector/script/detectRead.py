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

ROOT.ROOT.EnableImplicitMT()

def importROOT(filename):
	f = ROOT.TFile.Open(filename, "read")
	Hit = f.Get("Hit")
	dataTrack, columnsTrack = Hit.AsMatrix(return_labels=True)
	track = pd.DataFrame(data=dataTrack, columns=columnsTrack)
	#print(track)
	return(track)

track = importROOT("~/data/eDepPhiVid/g2wd10k_1.root")
eID = track['eventID']
eID = eID.drop_duplicates()
eIDNum = len(eID)

print(len(track))
eIDNum = range(eIDNum)
timeWinTotal = 60e-6 # s
binNum = 5e-9 # s
timeWinNum = int(timeWinTotal / binNum)



print(timeWinNum)
timeNum = range(timeWinNum)
dfMuonPos = pd.DataFrame([])
BinRange = range(timeWinNum)
for j in timeNum:
    dfMuonPosTemp = track[(track['hitTime'] >= j*5e-3) & (track['hitTime'] < j*5e-3+5e-3)] 
    dfMuonPosTemp['timeBin'] = j
    if (j % 1000 ==0):
        print(j)
    dfMuonPosTemp = dfMuonPosTemp[['eventID', 'hitTime' , 'hitPosX', 'hitPosY' ,'hitPosZ','hitPMag','hitR','eDep','hitAngle','VolID','timeBin']]
    dfMuonPos = pd.concat([dfMuonPos, dfMuonPosTemp])

winMin = 50
winMax = 100
timeMin = winMin*5e-3
timeMax = winMax*5e-3
n = 400000
cmin, cmax = 0, 2
color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])
print(color)
testPlot = dfMuonPos[(dfMuonPos["timeBin"] > winMin) & (dfMuonPos["timeBin"] < winMax)]
testPlotEveID = testPlot["eventID"]
testPlotEveID = testPlotEveID.drop_duplicates()
trEID = range(len(testPlotEveID))
print(trEID)
dfPosiPlot = pd.DataFrame([])
y = np.array(list(itertools.chain.from_iterable([ [i+1 for j in range(0, 303//2)] for i in range(0, 3)])))
y = y.reshape(-1, 1)
#c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(set(testPlot['hitPosX'])))]
c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, 12000)]

fig1 = plt.figure(1, figsize =(10 , 10))
fig2 = plt.figure(2, figsize =(10 , 10))
fig3 = plt.figure(3, figsize =(10 , 10))
pos3D = fig1.add_subplot(111, projection='3d')
pos2DVZ = fig2.add_subplot()
pos2DVT = fig3.add_subplot()

pos3D.set_xlabel('x [mm]')
pos3D.set_ylabel('y [mm]')
pos3D.set_zlabel('z [mm]')
pos3D.set_xlim( -400,400)
pos3D.set_ylim( -400,400)
pos3D.set_zlim(-400,400)

pos2DVZ.set_xlabel('vID [vaneID]')
pos2DVZ.set_ylabel('z [mm]')
pos2DVZ.set_xlim(0,40)
pos2DVZ.set_ylim(-400,400)
pos2DVZ.xaxis.set_major_locator(MultipleLocator(5))
pos2DVZ.yaxis.set_major_locator(MultipleLocator(20))
# Change minor ticks to show every 5. (20/4 = 5)
pos2DVZ.xaxis.set_minor_locator(AutoMinorLocator(5))
pos2DVZ.yaxis.set_minor_locator(AutoMinorLocator(4))


pos2DVT.set_xlabel('vID [vaneID]')
pos2DVT.set_ylabel('t [us]')
pos2DVT.set_xlim(0,40)
pos2DVT.set_ylim(timeMin,timeMax)
pos2DVT.xaxis.set_major_locator(MultipleLocator(5))
pos2DVT.yaxis.set_major_locator(MultipleLocator(20))
# Change minor ticks to show every 5. (20/4 = 5)
pos2DVT.xaxis.set_minor_locator(AutoMinorLocator(5))
pos2DVT.yaxis.set_minor_locator(AutoMinorLocator(4))


# Turn grid on for both major and minor ticks and style minor slightly
# differently.
pos2DVZ.grid(which='major', color='#CCCCCC', linestyle='--')
pos2DVZ.grid(which='minor', color='#CCCCCC', linestyle=':')
pos2DVT.grid(which='major', color='#CCCCCC', linestyle='--')
pos2DVT.grid(which='minor', color='#CCCCCC', linestyle=':')

for i,j in enumerate(testPlotEveID):
    #print(i)
    #print(j)
    testPlotTemp = testPlot[(testPlot["eventID"] == j)]
    color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])
    testPlotTemp = testPlotTemp[['eventID', 'hitPosX', 'hitPosY', 'hitPosZ','VolID','timeBin','hitTime']]
    testPlotTemp["c"] = i
    #print(j)
    #print (testPlotTemp["hitTime"])
    pos3D.scatter(testPlotTemp["hitPosX"],testPlotTemp["hitPosY"],testPlotTemp["hitPosZ"], color=c_lst[int(j/5)])
    pos2DVZ.scatter(testPlotTemp["VolID"],testPlotTemp["hitPosZ"], color=c_lst[int(j/5)])
    pos2DVT.scatter(testPlotTemp["VolID"],testPlotTemp["hitTime"], color=c_lst[int(j/5)])
    #plt.scatter(g[1]['x1'], g[1]['x2'], color=c_lst[i], label='group {}'.format(int(g[0])), alpha=0.5)
    dfPosiPlot = pd.concat([dfPosiPlot, testPlotTemp])
    #print(dfPosiPlot)
    #print(testPlotTemp)
print(dfPosiPlot)
n = dfPosiPlot["eventID"].values
hitTimes = dfPosiPlot["hitTime"].values
volIDs = dfPosiPlot["VolID"].values
for i, txt in enumerate(n):
    pos2DVT.annotate(txt, (volIDs[i], hitTimes[i]))
trackS = dfPosiPlot["c"]
hitS = dfPosiPlot["hitPosX"]
trackS = trackS.drop_duplicates()
print(len(trackS))
print(len(hitS))
#print(testPlot)
#pos.scatter(dfPosiPlot["hitPosX"],dfPosiPlot["hitPosY"],dfPosiPlot["hitPosZ"],  c=color, cmap='hot' )

plt.show()