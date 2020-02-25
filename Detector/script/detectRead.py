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


color = pc.bcolors()

ROOT.ROOT.EnableImplicitMT()

def importROOT(filename):
	f = ROOT.TFile.Open(filename, "read")
	Hit = f.Get("Hit")
	dataTrack, columnsTrack = Hit.AsMatrix(return_labels=True)
	track = pd.DataFrame(data=dataTrack, columns=columnsTrack)
	#print(track)
	return(track)

track = importROOT("~/data/g2wd10k_1.root")
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
    dfMuonPosTemp = dfMuonPosTemp[['eventID', 'hitTime' , 'hitPosX', 'hitPosY' ,'hitPosZ','hitPMag','hitTime','hitR','eDep','hitAngle','VolID','timeBin']]
    dfMuonPos = pd.concat([dfMuonPos, dfMuonPosTemp])
#print(dfMuonPos)
winMin = 90
winMax = 95
n = 303
cmin, cmax = 0, 2
color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])
import itertools

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
pos = fig1.add_subplot(111, projection='3d')
pos.set_xlabel('x [mm]')
pos.set_ylabel('y [mm]')
pos.set_zlabel('z [mm]')
pos.set_xlim( -400,400)
pos.set_ylim( -400,400)
pos.set_zlim(-400,400)

for i,j in enumerate(testPlotEveID):
    #print(i)
    #print(j)
    testPlotTemp = testPlot[(testPlot["eventID"] == j)]
    color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])
    testPlotTemp = testPlotTemp[['eventID', 'hitPosX', 'hitPosY', 'hitPosZ']]
    testPlotTemp["c"] = i
    pos.scatter(testPlotTemp["hitPosX"],testPlotTemp["hitPosY"],testPlotTemp["hitPosZ"], color=c_lst[int(j)])
    #plt.scatter(g[1]['x1'], g[1]['x2'], color=c_lst[i], label='group {}'.format(int(g[0])), alpha=0.5)
    dfPosiPlot = pd.concat([dfPosiPlot, testPlotTemp])
    #print(testPlotTemp)
print(dfPosiPlot)
#print(testPlot)
#pos.scatter(dfPosiPlot["hitPosX"],dfPosiPlot["hitPosY"],dfPosiPlot["hitPosZ"],  c=color, cmap='hot' )

plt.show()




