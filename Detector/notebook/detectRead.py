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

def dataWindowing(track):
    #track = importROOT("~/data/eDepPhiVid/g2wd10k_1.root")
    eID = track['eventID']
    eID = eID.drop_duplicates()
    eIDNum = len(eID)
    #print(len(track))
    eIDNum = range(eIDNum)
    timeWinTotal = 60e-6 # s
    binNum = 5e-9 # s
    timeWinNum = int(timeWinTotal / binNum)
    #print(timeWinNum)
    timeNum = range(timeWinNum)
    dfMuonPos = pd.DataFrame([])
    BinRange = range(timeWinNum)
    for j in timeNum:
        dfMuonPosTemp = track[(track['hitTime'] >= j*5e-3) & (track['hitTime'] < j*5e-3+5e-3)] 
        dfMuonPosTemp['timeBin'] = j
        if (j % 1000 ==0):
            print(j, "th window:", j*5e-3, " ~ ", j*5e-3+5e-3 ,  "[us] is creating..")
        dfMuonPosTemp = dfMuonPosTemp[['eventID', 'hitTime' , 'hitPosX', 'hitPosY' ,'hitPosZ','hitPMag','hitR','eDep','hitAngle','VolID','timeBin']]
        dfMuonPos = pd.concat([dfMuonPos, dfMuonPosTemp])
    return dfMuonPos
#print(dfMuonPos)

